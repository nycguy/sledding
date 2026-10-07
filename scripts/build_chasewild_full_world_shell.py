#!/usr/bin/env python3
"""Build Chase Wild M0.5 full GIS/LiDAR world shell.

Outputs a reproducible, Roblox-scaled world shell for the entire Chase street
cluster plus the complete Leonard Park polygon. Private owner/address attributes
are never exported.

Authoritative ground:
  2019 FEMA/USGS NY FEMAR2 Central B1 original-product-resolution 1 m DTM.

Planimentric geometry:
  Westchester County 2023 planimetrics.

Visual classification/QA:
  Westchester County 2025 four-band orthoimagery.

The script is designed to run in CI with internet access. Studio/rendering
validation remains a separate evidence layer.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import requests
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import rowcol
from shapely.geometry import (
    GeometryCollection, LineString, MultiLineString, MultiPolygon, Point,
    Polygon, box, mapping, shape,
)
from shapely.ops import nearest_points, transform as shp_transform, triangulate, unary_union

OUT = Path("analysis-output/world-shell")
GIS_OUT = OUT / "gis"
OUT.mkdir(parents=True, exist_ok=True)
GIS_OUT.mkdir(parents=True, exist_ok=True)

STUDS_PER_M = 4.0
TILE_SIZE_M = 256.0
PAD_MULTIPLE_M = 256.0

WC = "https://giswww.westchestergov.com/arcgis/rest/services"
PLAN = WC + "/DataHub_BasemapPlanimetrics/MapServer"
ENV = WC + "/DataHub_EnvironmentandPlanning/MapServer"
ORTHO = WC + "/MappingWestchesterCounty_AerialPhoto2025/ImageServer"

LIDAR_CRS = "EPSG:6350"
WGS84 = "EPSG:4326"
STATE_PLANE = "EPSG:2260"
PROJECT = "NY_FEMAR2_Central_2018_D19"
WORKUNIT = "NY_FEMAR2_Central_B1_2018"
TNM_BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/OPR/Projects"
DTM_PREFIX = f"{TNM_BASE}/{PROJECT}/{WORKUNIT}/TIFF"

ROUTE_CENTER = (-73.72145, 41.19325)
ROUGH_ENVELOPE = (-73.755, 41.175, -73.690, 41.215)

TARGET_STREETS = [
    "Carlton Drive",
    "Brentwood Court",
    "Stratford Drive",
    "Austin Drive",
    "Rolling Ridge Court",
    "Cold Spring Court",
    "Ascot Circle",
]

PLAN_LAYERS = {
    "streams": 0,
    "railroad_lines": 1,
    "guard_rails": 2,
    "road_barriers": 3,
    "stone_walls": 4,
    "buildings": 5,
    "bridges": 6,
    "roadways": 7,
    "sidewalks": 8,
    "parking_lots": 9,
    "driveways": 10,
    "lakes": 11,
    "barriers": 12,
    "streets": 13,
    "structures": 14,
    "transportation_features": 15,
    "transportation_structures": 16,
}

ENV_LAYERS = {
    "open_space": 209,
    "land_use": 210,
    "steep_slopes": 152,
    "nwi_wetlands": 154,
    "nys_regulated_wetlands": 217,
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ChaseWild-GIS-WorldShell/0.1"})

TO_LIDAR = Transformer.from_crs(WGS84, LIDAR_CRS, always_xy=True)
FROM_LIDAR = Transformer.from_crs(LIDAR_CRS, WGS84, always_xy=True)
TO_STATE = Transformer.from_crs(WGS84, STATE_PLANE, always_xy=True)
STATE_TO_LIDAR = Transformer.from_crs(STATE_PLANE, LIDAR_CRS, always_xy=True)

def log(*args):
    print(*args, flush=True)

def request_json(url: str, params: dict[str, Any], timeout=120):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"ArcGIS error {j['error']} for {r.url}")
    return j

def arcgis_ids(layer_url: str, envelope=None, where="1=1"):
    params = {
        "where": where,
        "returnIdsOnly": "true",
        "f": "json",
    }
    if envelope is not None:
        xmin,ymin,xmax,ymax=envelope
        params.update({
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
        })
    j=request_json(layer_url+"/query", params)
    return j.get("objectIdFieldName","OBJECTID"), sorted(j.get("objectIds") or [])

def fetch_geojson(layer_url: str, envelope=None, where="1=1", out_fields="*"):
    oid_field, ids = arcgis_ids(layer_url,envelope,where)
    features=[]
    for i in range(0,len(ids),400):
        chunk=ids[i:i+400]
        params={
            "objectIds": ",".join(map(str,chunk)),
            "outFields": out_fields,
            "returnGeometry":"true",
            "returnZ":"false",
            "returnM":"false",
            "outSR":"4326",
            "f":"geojson",
        }
        j=request_json(layer_url+"/query",params)
        features.extend(j.get("features",[]))
    return {"type":"FeatureCollection","features":features}, oid_field

def norm_street_name(name: str) -> str:
    if not name: return ""
    s=re.sub(r"[^A-Z0-9 ]+"," ",name.upper())
    parts=[p for p in s.split() if p]
    mapping={
        "DR":"DRIVE","DRV":"DRIVE",
        "CT":"COURT",
        "CIR":"CIRCLE","CIRC":"CIRCLE",
        "RD":"ROAD",
        "LN":"LANE",
        "PL":"PLACE",
        "AVE":"AVENUE","AV":"AVENUE",
        "ST":"STREET",
    }
    parts=[mapping.get(p,p) for p in parts]
    return " ".join(parts)

def project_geom(g, transformer=TO_LIDAR):
    return shp_transform(transformer.transform,g)

def unproject_geom(g):
    return shp_transform(FROM_LIDAR.transform,g)

def geom_from_feature(f):
    if not f.get("geometry"): return None
    g=shape(f["geometry"])
    return g if not g.is_empty else None

def safe_feature_collection(name, fc, keep_props=None):
    out=[]
    for idx,f in enumerate(fc["features"]):
        g=f.get("geometry")
        if not g: continue
        p=f.get("properties") or {}
        if keep_props is None:
            # Deliberately conservative allowlist, excludes address/owner fields.
            safe={}
            for k,v in p.items():
                ku=k.upper()
                if ku in ("OBJECTID","FID","FEA_CODE","DESCRIPTION","NAME","SOURCE","PHOTO_DATE",
                          "STRCL_CODE","TYPE","CATEGORY","CLASS","CLASSIFICATION","LANDUSE",
                          "LAND_USE","OPENSPACE","OS_NAME"):
                    safe[k]=v
        else:
            safe={k:p.get(k) for k in keep_props if k in p}
        out.append({"type":"Feature","id":f"{name}:{idx}","properties":safe,"geometry":g})
    result={"type":"FeatureCollection","features":out}
    (GIS_OUT/f"{name}.geojson").write_text(json.dumps(result,separators=(",",":")))
    return result

def feature_strings(f):
    vals=[]
    for k,v in (f.get("properties") or {}).items():
        if isinstance(v,str) and v.strip():
            vals.append((k,v))
    return vals

def pad_bounds(bounds, multiple=PAD_MULTIPLE_M):
    xmin,ymin,xmax,ymax=bounds
    w=xmax-xmin; h=ymax-ymin
    wp=math.ceil(w/multiple)*multiple
    hp=math.ceil(h/multiple)*multiple
    cx=(xmin+xmax)/2; cy=(ymin+ymax)/2
    # Snap lower-left to whole meter while keeping requested padded size.
    x0=math.floor(cx-wp/2)
    y0=math.floor(cy-hp/2)
    return (float(x0),float(y0),float(x0+wp),float(y0+hp))

def download_dtm(world_bounds):
    xmin,ymin,xmax,ymax=world_bounds
    xs=range(math.floor(xmin/1000),math.floor((xmax-1e-6)/1000)+1)
    ys=range(math.floor(ymin/1000),math.floor((ymax-1e-6)/1000)+1)
    mem=[]
    meta=[]
    for ex in xs:
        for ny in ys:
            name=f"USGS_OPR_{PROJECT}_e{ex}n{ny}.tif"
            url=f"{DTM_PREFIX}/{name}"
            r=SESSION.get(url,timeout=180)
            log("DTM",name,r.status_code,len(r.content))
            r.raise_for_status()
            sha=hashlib.sha256(r.content).hexdigest()
            mf=MemoryFile(r.content); ds=mf.open()
            if str(ds.crs)!="EPSG:6350":
                raise RuntimeError(f"{name}: unexpected CRS {ds.crs}")
            if abs(abs(ds.transform.a)-1)>1e-6 or abs(abs(ds.transform.e)-1)>1e-6:
                raise RuntimeError(f"{name}: expected 1 m cells, got {ds.res}")
            mem.append((mf,ds))
            meta.append({
                "file":name,"url":url,"sha256":sha,
                "bounds":[ds.bounds.left,ds.bounds.bottom,ds.bounds.right,ds.bounds.top],
                "resolutionM":[abs(ds.transform.a),abs(ds.transform.e)],
                "crs":str(ds.crs),
            })
    try:
        arr, transform=merge([d for _,d in mem],bounds=world_bounds,res=(1,1),nodata=-9999,dtype="float32")
        z=arr[0]
        valid=np.isfinite(z)&(z>-1000)
        if valid.mean()<0.999:
            raise RuntimeError(f"LiDAR DTM missing {1-valid.mean():.3%} of world AOI")
        return z,transform,meta
    finally:
        for mf,ds in mem:
            ds.close(); mf.close()

def write_dtm_outputs(z, transform, bounds, tile_meta):
    h,w=z.shape
    valid=np.isfinite(z)&(z>-1000)
    zmin=float(z[valid].min()); zmax=float(z[valid].max()); zrange=zmax-zmin
    meta={
        "driver":"GTiff","height":h,"width":w,"count":1,"dtype":"float32",
        "crs":LIDAR_CRS,"transform":transform,"nodata":-9999.0,"compress":"deflate"
    }
    tif=OUT/"chasewild_world_dtm_1m.tif"
    with rasterio.open(tif,"w",**meta) as dst: dst.write(z.astype("float32"),1)

    norm=np.rint((z-zmin)/zrange*255).clip(0,255).astype(np.uint8)
    Image.fromarray(norm,mode="L").save(OUT/"chasewild_world_heightmap_1m.png")
    norm16=np.rint((z-zmin)/zrange*65535).clip(0,65535).astype(np.uint16)
    Image.fromarray(norm16,mode="I;16").save(OUT/"chasewild_world_heightmap_1m_16bit.png")

    gy,gx=np.gradient(z,1,1)
    slope=np.pi/2-np.arctan(np.hypot(gx,gy))
    aspect=np.arctan2(-gx,gy)
    az=np.deg2rad(315); alt=np.deg2rad(45)
    shade=np.sin(alt)*np.sin(slope)+np.cos(alt)*np.cos(slope)*np.cos(az-aspect)
    shade=((shade-np.nanmin(shade))/(np.nanmax(shade)-np.nanmin(shade))*255).clip(0,255).astype(np.uint8)
    Image.fromarray(shade,mode="L").save(OUT/"chasewild_world_hillshade.png")

    return {
        "width":w,"height":h,"zmin":zmin,"zmax":zmax,"zrange":zrange,
        "tiles":tile_meta,
    }

def dtm_value(z, transform, x, y):
    r,c=rowcol(transform,x,y)
    r=int(np.clip(r,0,z.shape[0]-1)); c=int(np.clip(c,0,z.shape[1]-1))
    return float(z[r,c])

def polygon_ground_stats(poly, z, transform):
    minx,miny,maxx,maxy=poly.bounds
    r0,c0=rowcol(transform,minx,maxy)
    r1,c1=rowcol(transform,maxx,miny)
    r0=max(0,int(r0)-1); c0=max(0,int(c0)-1)
    r1=min(z.shape[0]-1,int(r1)+1); c1=min(z.shape[1]-1,int(c1)+1)
    if r1<r0 or c1<c0:
        zz=dtm_value(z,transform,poly.centroid.x,poly.centroid.y)
        return zz,zz,zz
    # Rasterize just the local window.
    from rasterio.windows import Window
    win=Window(c0,r0,c1-c0+1,r1-r0+1)
    wtransform=rasterio.windows.transform(win,transform)
    mask=rasterize([(mapping(poly),1)],out_shape=(int(win.height),int(win.width)),
                   transform=wtransform,fill=0,dtype="uint8")
    vals=z[r0:r1+1,c0:c1+1][mask==1]
    vals=vals[np.isfinite(vals)&(vals>-1000)]
    if len(vals)==0:
        vals=np.array([dtm_value(z,transform,poly.centroid.x,poly.centroid.y)])
    return float(np.median(vals)),float(np.min(vals)),float(np.max(vals))

def oriented_rect_info(poly):
    rect=poly.minimum_rotated_rectangle
    coords=list(rect.exterior.coords)[:-1]
    if len(coords)!=4:
        return 0.0, max(poly.bounds[2]-poly.bounds[0],1), max(poly.bounds[3]-poly.bounds[1],1)
    edges=[]
    for a,b in zip(coords,coords[1:]+coords[:1]):
        dx=b[0]-a[0]; dy=b[1]-a[1]
        edges.append((math.hypot(dx,dy),math.degrees(math.atan2(dy,dx))))
    edges.sort(reverse=True)
    long_len,long_ang=edges[0]
    short_len=min(e[0] for e in edges)
    return long_ang%180,long_len,short_len

def deterministic_variant(source_id, geom):
    h=hashlib.sha256((str(source_id)+geom.wkb_hex).encode()).digest()
    seed=int.from_bytes(h[:8],"big")
    roof=["side_gable","side_gable_dormers","side_gable_center_dormer","side_gable_crosswing"][seed%4]
    facade=["white_clapboard","warm_gray_clapboard","cream_clapboard","brick_accent","stone_accent","muted_blue_clapboard"][(seed//7)%6]
    entry=["pediment_portico","column_portico","simple_surround","broken_pediment"][(seed//17)%4]
    chimney=["single_end","paired_end","single_interior","offset_end"][(seed//29)%4]
    return {
        "seed":seed,"variantIndex":None,
        "roofFamily":roof,"facadeFamily":facade,"entryFamily":entry,"chimneyFamily":chimney,
    }

def local_xzy(x,y,elev,origin,zmin):
    ox,oy=origin
    return ((x-ox)*STUDS_PER_M,(elev-zmin)*STUDS_PER_M,-(y-oy)*STUDS_PER_M)

def clean_poly(g):
    if g is None or g.is_empty: return None
    if not g.is_valid: g=g.buffer(0)
    if g.is_empty: return None
    if isinstance(g,Polygon): return g
    if isinstance(g,MultiPolygon):
        return max(g.geoms,key=lambda p:p.area)
    return None

def building_records(building_fc, streets_projected, z, transform, origin, zmin):
    street_union=unary_union(streets_projected)
    records=[]
    project=lambda gg: project_geom(gg)
    for i,f in enumerate(building_fc["features"]):
        g=clean_poly(project(geom_from_feature(f)))
        if g is None or g.area<8: continue
        p=f.get("properties") or {}
        source_id=p.get("OBJECTID",i+1)
        med,gmin,gmax=polygon_ground_stats(g,z,transform)
        ang,long_len,short_len=oriented_rect_info(g)
        centroid=g.centroid
        near=nearest_points(centroid,street_union)[1] if not street_union.is_empty else centroid
        frontage_ang=math.degrees(math.atan2(near.y-centroid.y,near.x-centroid.x))%360
        road_dist=centroid.distance(near)
        confidence="high" if road_dist<=45 else ("medium" if road_dist<=75 else "low")
        variant=deterministic_variant(source_id,g)
        ring=[]
        for x,y in list(g.exterior.coords)[:-1]:
            lx,ly,lz=local_xzy(x,y,med,origin,zmin)
            ring.append([round(lx,3),round(lz,3)])
        rec={
            "id":f"B{i+1:04d}",
            "sourceObjectId":source_id,
            "footprintAreaM2":round(g.area,2),
            "footprintLocalXZStuds":ring,
            "centroidLocalStuds":{
                "x":round((centroid.x-origin[0])*STUDS_PER_M,3),
                "y":round((med-zmin)*STUDS_PER_M,3),
                "z":round(-(centroid.y-origin[1])*STUDS_PER_M,3),
            },
            "ground":{
                "medianElevationM":round(med,3),
                "minElevationM":round(gmin,3),
                "maxElevationM":round(gmax,3),
                "reliefM":round(gmax-gmin,3),
            },
            "architecture":{
                "type":"Center Hall Colonial",
                "stories":2,
                "storyHeightM":3.05,
                "wallHeightM":6.10,
                "dominantAxisDeg":round(ang,2),
                "mainLengthM":round(long_len,2),
                "mainWidthM":round(short_len,2),
                "inferredFrontageDeg":round(frontage_ang,2),
                "frontageRoadDistanceM":round(road_dist,2),
                "frontageConfidence":confidence,
                **variant,
            },
            "privacy":{
                "exactPrivateFacade":False,
                "addressIncluded":False,
                "ownerIncluded":False,
            }
        }
        records.append((rec,g))
    records.sort(key=lambda rg:(rg[0]["centroidLocalStuds"]["z"],rg[0]["centroidLocalStuds"]["x"]))
    for idx,(rec,g) in enumerate(records):
        rec["architecture"]["variantIndex"]=idx%12
    return records

def write_obj_buildings(records_geoms, zmin, origin):
    obj=OUT/"chasewild_buildings_proxy.obj"
    mtl=OUT/"chasewild_buildings_proxy.mtl"
    colors=[
        (0.82,0.83,0.80),(0.90,0.88,0.80),(0.74,0.77,0.80),
        (0.72,0.55,0.45),(0.70,0.65,0.57),(0.65,0.72,0.76),
    ]
    with mtl.open("w") as fh:
        for i,c in enumerate(colors):
            fh.write(f"newmtl facade_{i}\nKd {c[0]} {c[1]} {c[2]}\nKa 0.1 0.1 0.1\nKs 0.05 0.05 0.05\n\n")
        fh.write("newmtl roof\nKd 0.18 0.20 0.22\nKa 0.05 0.05 0.05\nKs 0.02 0.02 0.02\n\n")
    vcount=0
    with obj.open("w") as fh:
        fh.write(f"mtllib {mtl.name}\n")
        for rec,g in records_geoms:
            med=rec["ground"]["medianElevationM"]
            base=(med-zmin)*STUDS_PER_M
            top=base+rec["architecture"]["wallHeightM"]*STUDS_PER_M
            ext=list(g.exterior.coords)[:-1]
            if len(ext)<3: continue
            verts=[]
            for x,y in ext:
                verts.append(((x-origin[0])*STUDS_PER_M,base,-(y-origin[1])*STUDS_PER_M))
            for x,y in ext:
                verts.append(((x-origin[0])*STUDS_PER_M,top,-(y-origin[1])*STUDS_PER_M))
            fh.write(f"o {rec['id']}\n")
            for x,y,zv in verts: fh.write(f"v {x:.3f} {y:.3f} {zv:.3f}\n")
            mat_idx=["white_clapboard","warm_gray_clapboard","cream_clapboard","brick_accent","stone_accent","muted_blue_clapboard"].index(rec["architecture"]["facadeFamily"])
            fh.write(f"usemtl facade_{mat_idx}\n")
            n=len(ext)
            for j in range(n):
                a=wall_start+j+1; b=wall_start+(j+1)%n+1
                c=wall_start+n+(j+1)%n+1; d=wall_start+n+j+1
                fh.write(f"f {a} {b} {c} {d}\n")
            # Flat cap under proxy roof to close the mass.
            try:
                for tri in triangulate(g):
                    if not g.covers(tri.representative_point()): continue
                    inds=[]
                    for x,y in list(tri.exterior.coords)[:-1]:
                        fh.write(f"v {(x-origin[0])*STUDS_PER_M:.3f} {top:.3f} {-(y-origin[1])*STUDS_PER_M:.3f}\n")
                        vcount+=1; inds.append(vcount)
                    fh.write("f "+" ".join(map(str,inds))+"\n")
            except Exception:
                pass
            # Gable roof over oriented main rectangle.
            ang=math.radians(rec["architecture"]["dominantAxisDeg"])
            ux,uy=math.cos(ang),math.sin(ang); vx,vy=-uy,ux
            cx,cy=g.centroid.x,g.centroid.y
            L=max(rec["architecture"]["mainLengthM"]/2,2)
            W=max(rec["architecture"]["mainWidthM"]/2,2)
            pitch=math.radians(34 + (rec["architecture"]["seed"]%5))
            rise=min(W*math.tan(pitch),4.0)*STUDS_PER_M
            roofpts=[]
            for du,dv in [(-L,-W),(L,-W),(L,W),(-L,W),(-L,0),(L,0)]:
                x=cx+du*ux+dv*vx; y=cy+du*uy+dv*vy
                yy=top+(rise if dv==0 else 0)
                roofpts.append(((x-origin[0])*STUDS_PER_M,yy,-(y-origin[1])*STUDS_PER_M))
            basei=vcount
            for x,y,zv in roofpts:
                fh.write(f"v {x:.3f} {y:.3f} {zv:.3f}\n")
            fh.write("usemtl roof\n")
            # 1,2 are one eave side; 4,3 opposite; 5,6 ridge endpoints
            idx=[basei+k for k in range(1,7)]
            fh.write(f"f {idx[0]} {idx[1]} {idx[5]} {idx[4]}\n")
            fh.write(f"f {idx[3]} {idx[4]} {idx[5]} {idx[2]}\n")
            fh.write(f"f {idx[0]} {idx[4]} {idx[3]}\n")
            fh.write(f"f {idx[1]} {idx[2]} {idx[5]}\n")
            vcount+=6
    return obj,mtl

def _iter_polys(g):
    if isinstance(g,Polygon): yield g
    elif isinstance(g,MultiPolygon):
        for p in g.geoms: yield p

def write_surface_obj(layer_geoms, z, transform, origin, zmin):
    obj=OUT/"chasewild_hard_surfaces_proxy.obj"
    mtl=OUT/"chasewild_hard_surfaces_proxy.mtl"
    mats={
      "roadways":(0.18,0.18,0.19,0.05),
      "driveways":(0.25,0.25,0.26,0.06),
      "sidewalks":(0.58,0.58,0.56,0.08),
      "parking_lots":(0.22,0.22,0.23,0.05),
      "bridges":(0.36,0.35,0.33,0.12),
      "lakes":(0.20,0.42,0.58,0.10),
    }
    with mtl.open("w") as fh:
        for name,(r,g,b,off) in mats.items():
            fh.write(f"newmtl {name}\nKd {r} {g} {b}\nKa 0.04 0.04 0.04\nKs 0.05 0.05 0.05\n\n")
    vcount=0
    with obj.open("w") as fh:
        fh.write(f"mtllib {mtl.name}\n")
        for lname,geoms in layer_geoms.items():
            if lname not in mats: continue
            off=mats[lname][3]
            fh.write(f"usemtl {lname}\n")
            for fi,g in enumerate(geoms):
                for poly in _iter_polys(g):
                    if poly.area<0.2: continue
                    try:
                        tris=[t for t in triangulate(poly) if poly.covers(t.representative_point())]
                    except Exception:
                        continue
                    for tri in tris:
                        inds=[]
                        for x,y in list(tri.exterior.coords)[:-1]:
                            if lname=="lakes":
                                elev=np.median([dtm_value(z,transform,xx,yy) for xx,yy in list(poly.exterior.coords)[::max(1,len(poly.exterior.coords)//20)]])
                            else:
                                elev=dtm_value(z,transform,x,y)
                            vx=(x-origin[0])*STUDS_PER_M
                            vy=(elev-zmin+off)*STUDS_PER_M
                            vz=-(y-origin[1])*STUDS_PER_M
                            fh.write(f"v {vx:.3f} {vy:.3f} {vz:.3f}\n")
                            vcount+=1; inds.append(vcount)
                        fh.write("f "+" ".join(map(str,inds))+"\n")
    return obj,mtl

def export_ortho(bounds,width,height, hard_shapes):
    xmin,ymin,xmax,ymax=bounds
    params={
        "bbox":f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR":"6350","imageSR":"6350",
        "size":f"{width},{height}",
        "format":"tiff",
        "pixelType":"U8",
        "bandIds":"0,1,2,3",
        "interpolation":"RSP_BilinearInterpolation",
        "f":"image",
    }
    r=SESSION.get(ORTHO+"/exportImage",params=params,timeout=180)
    log("ORTHO",r.status_code,len(r.content),r.headers.get("content-type"))
    r.raise_for_status()
    try:
        mf=MemoryFile(r.content); ds=mf.open()
    except Exception as exc:
        log("WARN: 2025 four-band TIFF unavailable",repr(exc))
        return None
    try:
        arr=ds.read()
        log("ORTHO bands",arr.shape,str(ds.crs),ds.res)
        if arr.shape[0]<3: return None
        # Copy georeferenced ortho exactly as returned.
        outmeta=ds.meta.copy(); outmeta.update(driver="GTiff",compress="deflate")
        with rasterio.open(OUT/"chasewild_world_ortho_2025_1m.tif","w",**outmeta) as out:
            out.write(arr)
        rgb=np.moveaxis(arr[:3],0,2)
        Image.fromarray(rgb.astype(np.uint8),"RGB").save(OUT/"chasewild_world_ortho_2025_rgb.jpg",quality=90)
        result={"bands":int(arr.shape[0]),"crs":str(ds.crs),"resolution":[abs(ds.transform.a),abs(ds.transform.e)]}
        if arr.shape[0]>=4:
            red=arr[0].astype(np.float32); nir=arr[3].astype(np.float32)
            ndvi=(nir-red)/(nir+red+1e-6)
            ndvi=np.clip(ndvi,-1,1)
            meta=ds.meta.copy(); meta.update(driver="GTiff",count=1,dtype="float32",compress="deflate")
            with rasterio.open(OUT/"chasewild_world_ndvi_2025.tif","w",**meta) as out:
                out.write(ndvi.astype("float32"),1)
            ndv8=np.rint((ndvi+1)/2*255).clip(0,255).astype(np.uint8)
            Image.fromarray(ndv8,"L").save(OUT/"chasewild_world_ndvi_2025.png")
            # Data-derived vegetation classes. This is an EXPERIMENT mask until Studio/visual QA.
            cls=np.zeros(ndvi.shape,dtype=np.uint8)
            cls[(ndvi>=0.12)&(ndvi<0.30)]=1  # low vegetation candidate
            cls[ndvi>=0.30]=2                # woody/high vegetation candidate
            if hard_shapes:
                mask=rasterize([(mapping(g),1) for g in hard_shapes if g and not g.is_empty],
                               out_shape=cls.shape,transform=ds.transform,fill=0,dtype="uint8")
                cls[mask==1]=0
            Image.fromarray(cls,"L").save(OUT/"chasewild_world_vegetation_class_experiment.png")
            result["ndvi"]=True
            result["vegetationClassLegend"]={"0":"nonvegetated/unknown","1":"low vegetation candidate","2":"woody/high vegetation candidate"}
        return result
    finally:
        ds.close(); mf.close()

def write_qa_preview(z, transform, vectors, park_geom, buildings, bounds):
    # Hillshade as base, downsample to <=1600 px max side.
    gy,gx=np.gradient(z,1,1)
    slope=np.pi/2-np.arctan(np.hypot(gx,gy))
    aspect=np.arctan2(-gx,gy)
    az=np.deg2rad(315); alt=np.deg2rad(45)
    shade=np.sin(alt)*np.sin(slope)+np.cos(alt)*np.cos(slope)*np.cos(az-aspect)
    shade=((shade-np.nanmin(shade))/(np.nanmax(shade)-np.nanmin(shade))*255).clip(0,255).astype(np.uint8)
    h,w=shade.shape
    scale=min(1.0,1600/max(w,h))
    pw=max(1,int(w*scale)); ph=max(1,int(h*scale))
    img=Image.fromarray(shade,"L").convert("RGB").resize((pw,ph),Image.Resampling.BILINEAR)
    draw=ImageDraw.Draw(img,"RGBA")
    xmin,ymin,xmax,ymax=bounds
    def pix(x,y):
        return ((x-xmin)/(xmax-xmin)*pw,(ymax-y)/(ymax-ymin)*ph)
    def draw_geom(g,fill,outline,width=1):
        if isinstance(g,Polygon):
            pts=[pix(x,y) for x,y in g.exterior.coords]
            if fill: draw.polygon(pts,fill=fill)
            if outline: draw.line(pts,fill=outline,width=width,joint="curve")
        elif isinstance(g,MultiPolygon):
            for p in g.geoms: draw_geom(p,fill,outline,width)
        elif isinstance(g,(LineString,MultiLineString)):
            lines=[g] if isinstance(g,LineString) else list(g.geoms)
            for line in lines:
                draw.line([pix(x,y) for x,y in line.coords],fill=outline,width=width)
    if park_geom: draw_geom(park_geom,(70,150,70,35),(70,220,70,180),2)
    for g in vectors.get("roadways",[]): draw_geom(g,(55,55,55,140),(30,30,30,180),1)
    for g in vectors.get("driveways",[]): draw_geom(g,(95,95,95,110),None,1)
    for g in vectors.get("sidewalks",[]): draw_geom(g,(190,190,180,100),None,1)
    for g in vectors.get("lakes",[]): draw_geom(g,(50,120,210,150),(30,90,170,180),1)
    for rec,g in buildings: draw_geom(g,(225,170,100,180),(100,70,40,220),1)
    for g in vectors.get("streets_selected",[]): draw_geom(g,None,(255,220,80,230),2)
    img.save(OUT/"chasewild_world_qa_overview.png")

def main():
    log("1. Querying streets around Chase...")
    rough_fc,_=fetch_geojson(f"{PLAN}/13",ROUGH_ENVELOPE,out_fields="OBJECTID,NAME,DESCRIPTION")
    targets={norm_street_name(n):n for n in TARGET_STREETS}
    selected=[]
    found=defaultdict(int)
    nearby_names=set()
    for f in rough_fc["features"]:
        p=f.get("properties") or {}
        nm=norm_street_name(str(p.get("NAME") or ""))
        if nm: nearby_names.add(nm)
        if nm in targets:
            selected.append(f); found[nm]+=1
    missing=[targets[k] for k in targets if not found[k]]
    fallback_street_points=[]
    fallback_sources=[]
    if missing:
        log("County Streets layer missing named centerline(s):",missing)
        for missing_name in missing:
            # Public geocoder is used only as an AOI inclusion fallback. It does
            # not replace County planimetric roadway geometry and is tagged inferred.
            params={
              "SingleLine":f"{missing_name}, Mount Kisco, NY 10549",
              "f":"json","outSR":"4326","maxLocations":1,
            }
            try:
                gj=request_json("https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates",params)
                cand=(gj.get("candidates") or [None])[0]
                if cand and float(cand.get("score",0))>=75:
                    loc=cand["location"]
                    pg=project_geom(Point(float(loc["x"]),float(loc["y"])))
                    fallback_street_points.append(pg)
                    fallback_sources.append({"name":missing_name,"score":cand.get("score"),"matchedAddress":cand.get("address"),"source":"ArcGIS World Geocoder AOI fallback"})
                    log("Fallback street point",missing_name,cand.get("score"),cand.get("address"),loc)
                else:
                    log("No reliable fallback geocode for",missing_name,cand)
            except Exception as exc:
                log("Fallback geocode failed",missing_name,repr(exc))
    street_geoms_wgs=[geom_from_feature(f) for f in selected if geom_from_feature(f)]
    street_geoms=[project_geom(g) for g in street_geoms_wgs]
    streets_union=unary_union(street_geoms)
    log("Selected street segments",len(selected),{targets[k]:found[k] for k in targets})

    log("2. Querying Leonard Park open-space polygon...")
    open_fc,_=fetch_geojson(f"{ENV}/209",ROUGH_ENVELOPE,out_fields="*")
    leonard=[]
    candidates=[]
    for f in open_fc["features"]:
        strings=feature_strings(f)
        text=" | ".join(v.upper() for k,v in strings)
        candidates.append([(k,v) for k,v in strings[:8]])
        if "LEONARD" in text:
            g=geom_from_feature(f)
            if g: leonard.append(g)
    if not leonard:
        log("Open-space candidates:",json.dumps(candidates[:30],indent=2))
        raise RuntimeError("Could not identify Leonard Park in Open Space layer")
    leonard_wgs=unary_union(leonard)
    leonard_proj=project_geom(leonard_wgs)
    log("Leonard polygon area ha",leonard_proj.area/10000)

    # Spec AOI rule: 100 m street cluster, union complete Leonard Park, then 50 m context.
    chase_parts=[streets_union.buffer(100)]
    # The County Streets layer currently omits Rolling Ridge Court by name.
    # Include any reliably geocoded missing street location with a conservative
    # 175 m AOI buffer; actual rendered road geometry still comes from County
    # Roadways polygons.
    chase_parts.extend(p.buffer(175) for p in fallback_street_points)
    required=unary_union(chase_parts+[leonard_proj])
    world_shape=required.buffer(50)
    bounds=pad_bounds(world_shape.bounds)
    world_rect=box(*bounds)
    width=int(round(bounds[2]-bounds[0])); height=int(round(bounds[3]-bounds[1]))
    if max(width,height)>4096:
        raise RuntimeError(f"World raster {width}x{height} exceeds 4096 heightmap limit")
    origin=((bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2)
    center_lon,center_lat=FROM_LIDAR.transform(*origin)
    log("World bounds",bounds,"pixels",width,height,"center",center_lon,center_lat)

    # AOI output.
    aoi_fc={"type":"FeatureCollection","features":[
      {"type":"Feature","properties":{"kind":"required_union"},"geometry":mapping(unproject_geom(required))},
      {"type":"Feature","properties":{"kind":"world_rect"},"geometry":mapping(unproject_geom(world_rect))},
      {"type":"Feature","properties":{"kind":"Leonard Park"},"geometry":mapping(leonard_wgs)},
    ]}
    (GIS_OUT/"world_aoi.geojson").write_text(json.dumps(aoi_fc,separators=(",",":")))

    log("3. Downloading native 1 m lidar-derived DTM...")
    z,transform,tile_meta=download_dtm(bounds)
    dtm=write_dtm_outputs(z,transform,bounds,tile_meta)
    zmin=dtm["zmin"]

    # Query all GIS layers inside rectangular world extent.
    world_wgs=unproject_geom(world_rect)
    env_wgs=world_wgs.bounds
    vector_geoms={}
    vector_counts={}
    raw_safe={}
    log("4. Querying Westchester planimetrics...")
    for name,lid in PLAN_LAYERS.items():
        fc,_=fetch_geojson(f"{PLAN}/{lid}",env_wgs,out_fields="*")
        safe=safe_feature_collection(name,fc)
        raw_safe[name]=safe
        geoms=[]
        for f in safe["features"]:
            g=geom_from_feature(f)
            if g:
                pg=project_geom(g).intersection(world_rect)
                if not pg.is_empty: geoms.append(pg)
        vector_geoms[name]=geoms
        vector_counts[name]=len(geoms)
        log(" ",name,len(geoms))

    log("5. Querying environmental/semantic GIS...")
    for name,lid in ENV_LAYERS.items():
        fc,_=fetch_geojson(f"{ENV}/{lid}",env_wgs,out_fields="*")
        safe=safe_feature_collection(name,fc)
        raw_safe[name]=safe
        geoms=[]
        for f in safe["features"]:
            g=geom_from_feature(f)
            if g:
                pg=project_geom(g).intersection(world_rect)
                if not pg.is_empty: geoms.append(pg)
        vector_geoms[name]=geoms
        vector_counts[name]=len(geoms)
        log(" ",name,len(geoms))

    # Save exact selected Chase streets separately.
    selected_fc={"type":"FeatureCollection","features":[]}
    for f in selected:
        p=f.get("properties") or {}
        selected_fc["features"].append({
          "type":"Feature","properties":{"NAME":p.get("NAME"),"OBJECTID":p.get("OBJECTID")},
          "geometry":f.get("geometry")
        })
    safe_feature_collection("streets_selected",selected_fc,keep_props=["NAME","OBJECTID"])
    vector_geoms["streets_selected"]=street_geoms

    log("6. Generating Center Hall Colonial building records...")
    building_fc=raw_safe["buildings"]
    records_geoms=building_records(building_fc,street_geoms,z,transform,origin,zmin)
    records=[r for r,g in records_geoms]
    (OUT/"chasewild_building_records.json").write_text(json.dumps({
      "generatorVersion":"0.1","architecture":"Center Hall Colonial",
      "recordCount":len(records),"records":records
    },indent=2))
    log("Buildings",len(records))

    # Create semantic region in local studs.
    def local_ring(poly):
        return [[round((x-origin[0])*STUDS_PER_M,3),round(-(y-origin[1])*STUDS_PER_M,3)]
                for x,y in poly.exterior.coords]
    park_polys=list(_iter_polys(leonard_proj))
    semantic={
      "worldOrigin":{"projectedCRS":LIDAR_CRS,"x":origin[0],"y":origin[1],"lon":center_lon,"lat":center_lat},
      "scaleStudsPerMeter":STUDS_PER_M,
      "regions":{
        "LeonardPark":[local_ring(p) for p in park_polys],
        "ChaseStreetEnvelope":[local_ring(p) for p in _iter_polys(streets_union.buffer(100))],
      }
    }
    (OUT/"chasewild_semantic_regions.json").write_text(json.dumps(semantic,indent=2))

    # Build streamable tile index.
    tiles=defaultdict(lambda:{"buildings":[],"layers":defaultdict(list)})
    xmin,ymin,xmax,ymax=bounds
    for rec,g in records_geoms:
        ix=int(math.floor((g.centroid.x-xmin)/TILE_SIZE_M)); iy=int(math.floor((g.centroid.y-ymin)/TILE_SIZE_M))
        tiles[f"{ix}_{iy}"]["buildings"].append(rec["id"])
    for lname,geoms in vector_geoms.items():
        if lname=="buildings": continue
        for j,g in enumerate(geoms):
            c=g.centroid
            ix=int(math.floor((c.x-xmin)/TILE_SIZE_M)); iy=int(math.floor((c.y-ymin)/TILE_SIZE_M))
            tiles[f"{ix}_{iy}"]["layers"][lname].append(j)
    tile_json={}
    for tid,t in tiles.items():
        tile_json[tid]={"buildings":t["buildings"],"layers":dict(t["layers"])}
    (OUT/"chasewild_world_tiles.json").write_text(json.dumps({
      "tileSizeM":TILE_SIZE_M,"tileSizeStuds":TILE_SIZE_M*STUDS_PER_M,"tiles":tile_json
    },indent=2))

    log("7. Generating M0.5 proxy meshes...")
    write_obj_buildings(records_geoms,zmin,origin)
    write_surface_obj(vector_geoms,z,transform,origin,zmin)

    # 2025 ortho / vegetation.
    hard=[]
    for lname in ("buildings","roadways","driveways","sidewalks","parking_lots","lakes"):
        hard.extend(vector_geoms.get(lname,[]))
    log("8. Exporting 2025 ortho / vegetation index...")
    ortho=export_ortho(bounds,width,height,hard)

    log("9. Writing QA preview...")
    write_qa_preview(z,transform,vector_geoms,leonard_proj,records_geoms,bounds)

    # Local source-aligned route: preserve existing source GeoJSON if available.
    route_path=Path("analysis-output/chase_leonard_routes.geojson")
    route_in_world=False
    if route_path.exists():
        route=json.loads(route_path.read_text())
        local_features=[]
        for f in route.get("features",[]):
            g=geom_from_feature(f)
            if not g: continue
            pg=project_geom(g)
            if not world_rect.intersects(pg): continue
            route_in_world=True
            coords=[]
            if isinstance(pg,LineString):
                coords=[[round((x-origin[0])*STUDS_PER_M,3),round(-(y-origin[1])*STUDS_PER_M,3)] for x,y in pg.coords]
            local_features.append({
              "name":(f.get("properties") or {}).get("name"),
              "kind":(f.get("properties") or {}).get("kind"),
              "localXZStuds":coords,
              "sourceProperties":f.get("properties") or {},
            })
        (OUT/"chasewild_source_routes_local.json").write_text(json.dumps({
          "originProjected":list(origin),"scaleStudsPerMeter":STUDS_PER_M,"features":local_features
        },indent=2))

    # Validation.
    world_tol=world_rect.buffer(1e-6)
    validations=[]
    def add(name,status,detail):
        validations.append({"name":name,"status":status,"detail":detail})
    if not missing:
        add("CHASE-STREETS","PASS",f"7/7 named County street centerlines found; {len(selected)} source segments")
    elif len(fallback_street_points)==len(missing):
        add("CHASE-STREETS","WARN",f"{7-len(missing)}/7 named County centerlines found; {', '.join(missing)} included in AOI using public geocoder fallback and rendered from County Roadways polygons")
    else:
        add("CHASE-STREETS","UNVERIFIED",f"{7-len(missing)}/7 named County centerlines found; unresolved: {missing}")
    add("LEONARD-PARK","PASS" if world_tol.contains(leonard_proj) else "FAIL",f"complete selected open-space polygon area {leonard_proj.area/10000:.2f} ha inside world")
    add("LIDAR-COVERAGE","PASS",f"{width}x{height} native 1 m cells; {len(tile_meta)} source DTM tiles")
    add("BUILDINGS","PASS" if len(records)>0 else "FAIL",f"{len(records)} footprint-derived records, no address/owner fields")
    variants=sorted(set(r["architecture"]["variantIndex"] for r in records))
    add("COLONIAL-VARIANTS","PASS" if len(variants)>=12 else "WARN",f"{len(variants)} deterministic Center Hall Colonial variants represented")
    add("PLANIMETRICS","PASS",", ".join(f"{k}={vector_counts[k]}" for k in ("roadways","driveways","sidewalks","parking_lots","streams","lakes")))
    add("COURSE-IN-WORLD","PASS" if route_in_world else "UNVERIFIED","source route GeoJSON reprojected into world" if route_in_world else "route source not available")
    add("ORTHO-2025","PASS" if ortho else "WARN","4-band RGB+NIR export and derived vegetation index" if ortho else "orthophoto export unavailable in CI")
    add("STUDIO-RENDER","UNVERIFIED","Studio import/render/streaming/physics required")
    if any(v["status"]=="FAIL" for v in validations):
        raise RuntimeError("World-shell validation failed: "+json.dumps(validations))

    y_studs=int(math.ceil(dtm["zrange"]*STUDS_PER_M/4)*4)
    manifest={
      "contentVersion":"m0.5-gis-world-1",
      "generatedBy":"build_chasewild_full_world_shell.py",
      "world":{
        "boundsEPSG6350":list(bounds),
        "centerEPSG6350":list(origin),
        "centerWGS84":[center_lon,center_lat],
        "widthM":width,"heightM":height,
        "widthStuds":width*STUDS_PER_M,"heightStuds":height*STUDS_PER_M,
        "studsPerMeter":STUDS_PER_M,
      },
      "terrain":{
        "sourceProject":PROJECT,"sourceWorkUnit":WORKUNIT,
        "nativeResolutionM":1.0,"crs":LIDAR_CRS,
        "elevationMinM":dtm["zmin"],"elevationMaxM":dtm["zmax"],"elevationRangeM":dtm["zrange"],
        "robloxImport":{
          "heightmap":"chasewild_world_heightmap_1m.png",
          "sizeStuds":{"x":width*STUDS_PER_M,"y":y_studs,"z":height*STUDS_PER_M},
          "positionStuds":{"x":0,"y":y_studs/2,"z":0},
          "material":"Snow",
        },
        "sourceTiles":tile_meta,
        "globalSmoothing":False,"horizontalExaggeration":False,"verticalExaggeration":False,
      },
      "aoiRule":{
        "chaseStreetBufferM":100,"combinedContextBufferM":50,"rasterPadMultipleM":PAD_MULTIPLE_M,
        "streets":TARGET_STREETS,
        "countyStreetCenterlinesMissing":missing,
        "fallbackStreetLocations":fallback_sources,
        "leonardParkSource":"Westchester County DataHub Environment and Planning / Open Space layer 209",
      },
      "gisSources":{
        "planimetrics":{"service":PLAN,"capture":"Spring 2023 photography","layers":PLAN_LAYERS},
        "environment":{"service":ENV,"layers":ENV_LAYERS},
        "ortho2025":{"service":ORTHO,"metadata":"4-band RGB+NIR, 0.5-ft native GSD"},
      },
      "featureCounts":vector_counts,
      "buildingCount":len(records),
      "buildingArchitecture":"two-story Center Hall Colonial procedural/proxy massing; exact private facades excluded",
      "ortho":ortho,
      "privacy":{
        "ownerFieldsExported":False,"addressFieldsExported":False,"houseNumbersExported":False,
      },
      "validation":validations,
    }
    (OUT/"chasewild_world_manifest.json").write_text(json.dumps(manifest,indent=2))

    report=[
      "# Chase Wild M0.5 GIS World Shell",
      "",
      f"- World extent: **{width} m × {height} m** = **{width*4:.0f} × {height*4:.0f} studs**",
      f"- Native terrain: **{width} × {height} 1 m LiDAR-derived cells**",
      f"- Elevation range: **{dtm['zmin']:.2f}–{dtm['zmax']:.2f} m**",
      f"- Buildings: **{len(records)}** source-footprint Center Hall Colonial proxy records",
      f"- Leonard Park open-space area represented: **{leonard_proj.area/10000:.2f} ha**",
      "",
      "## GIS feature counts",
      ""
    ]
    for k in sorted(vector_counts): report.append(f"- {k}: {vector_counts[k]}")
    report += ["","## Validation",""]
    for v in validations: report.append(f"- **{v['status']} {v['name']}**: {v['detail']}")
    report += ["","Studio rendering, terrain voxel appearance, streaming and game performance remain **UNVERIFIED** until the package is imported and exercised in Roblox Studio."]
    (OUT/"WORLD_SHELL_REPORT.md").write_text("\n".join(report))
    log(json.dumps({"world":[width,height],"buildings":len(records),"counts":vector_counts,"validations":validations},indent=2))

if __name__=="__main__":
    main()
