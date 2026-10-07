#!/usr/bin/env python3
"""Build the Chase Wild Roblox M0 base terrain from the original 2019 FEMA/USGS
lidar-derived 1 m hydro-flattened source DEM, not the seamless 3DEP service.

The source DEM is the official OPR raster generated from classified Class 2
(ground) lidar plus hydro breaklines. The Roblox import is 4 studs per meter so
one 1 m DEM cell maps to one 4x4-stud Terrain voxel in X/Z.
"""
from __future__ import annotations
import io, json, math, hashlib
from pathlib import Path
import requests, numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from pyproj import Transformer

OUT=Path("analysis-output"); OUT.mkdir(exist_ok=True)
CENTER_LON=-73.72145145835725
CENTER_LAT=41.19325125520854
SIZE_M=512
STUDS_PER_M=4.0
SRC_CRS="EPSG:6350"
PROJECT="NY_FEMAR2_Central_2018_D19"
WORKUNIT="NY_FEMAR2_Central_B1_2018"
BASE="https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/OPR/Projects"
PREFIX=f"{BASE}/{PROJECT}/{WORKUNIT}/TIFF"
T=Transformer.from_crs("EPSG:4326",SRC_CRS,always_xy=True)

cx,cy=T.transform(CENTER_LON,CENTER_LAT)
# Snap to whole-meter coordinates so the 512 x 512 output stays at exact 1 m GSD.
cx=round(cx); cy=round(cy)
half=SIZE_M/2
bounds=(cx-half,cy-half,cx+half,cy+half)

xs=range(math.floor(bounds[0]/1000),math.floor((bounds[2]-1e-6)/1000)+1)
ys=range(math.floor(bounds[1]/1000),math.floor((bounds[3]-1e-6)/1000)+1)
urls=[]
for ex in xs:
  for ny in ys:
    name=f"USGS_OPR_{PROJECT}_e{ex}n{ny}.tif"
    urls.append((name,f"{PREFIX}/{name}"))

datasets=[]; source_meta=[]
session=requests.Session(); session.headers["User-Agent"]="ChaseWild-LidarTerrain/1.0"
try:
  for name,url in urls:
    r=session.get(url,timeout=120)
    print(name,r.status_code,len(r.content))
    r.raise_for_status()
    sha=hashlib.sha256(r.content).hexdigest()
    mf=MemoryFile(r.content)
    ds=mf.open()
    if ds.crs is None:
      raise RuntimeError(f"{name}: missing CRS")
    if abs(abs(ds.transform.a)-1.0)>1e-6 or abs(abs(ds.transform.e)-1.0)>1e-6:
      raise RuntimeError(f"{name}: expected 1 m source DEM, got {ds.res}")
    datasets.append((mf,ds))
    source_meta.append({
      "file":name,"url":url,"sha256":sha,"crs":str(ds.crs),
      "resolutionM":[abs(ds.transform.a),abs(ds.transform.e)],
      "bounds":[ds.bounds.left,ds.bounds.bottom,ds.bounds.right,ds.bounds.top],
      "dtype":ds.dtypes[0],"nodata":ds.nodata
    })

  srcs=[d for _,d in datasets]
  arr, transform = merge(srcs,bounds=bounds,res=(1.0,1.0),nodata=-9999.0,dtype="float32")
  z=arr[0]
  if z.shape!=(SIZE_M,SIZE_M):
    raise RuntimeError(f"Expected {(SIZE_M,SIZE_M)}, got {z.shape}")
  valid=np.isfinite(z)&(z>-1000)
  if valid.mean()<0.999:
    raise RuntimeError(f"Unexpected nodata coverage: {1-valid.mean():.3%}")
  zmin=float(z[valid].min()); zmax=float(z[valid].max()); zrange=zmax-zmin

  # Preserve the float32 DTM as the archival/source artifact.
  tif_meta=srcs[0].meta.copy()
  tif_meta.update(driver="GTiff",height=SIZE_M,width=SIZE_M,count=1,dtype="float32",
                  crs=SRC_CRS,transform=transform,nodata=-9999.0,compress="deflate")
  with rasterio.open(OUT/"chasewild_m0_lidar_dtm_1m.tif","w",**tif_meta) as dst:
    dst.write(z.astype("float32"),1)

  # Roblox-import PNG. 8-bit vertical quantization is finer than Roblox's 4-stud
  # vertical voxel at 4 studs/m; the float32 GeoTIFF remains the evidence source.
  norm=np.rint((z-zmin)/zrange*255.0).clip(0,255).astype(np.uint8)
  Image.fromarray(norm,mode="L").save(OUT/"chasewild_m0_heightmap_lidar_1m.png")

  # 16-bit archival height image as an additional non-lossy-ish normalized raster.
  norm16=np.rint((z-zmin)/zrange*65535.0).clip(0,65535).astype(np.uint16)
  Image.fromarray(norm16,mode="I;16").save(OUT/"chasewild_m0_heightmap_lidar_1m_16bit.png")

  # Hillshade-like preview for human review, derived only from source DTM.
  gy,gx=np.gradient(z,1.0,1.0)
  slope=np.pi/2-np.arctan(np.hypot(gx,gy))
  aspect=np.arctan2(-gx,gy)
  az=np.deg2rad(315); alt=np.deg2rad(45)
  shade=np.sin(alt)*np.sin(slope)+np.cos(alt)*np.cos(slope)*np.cos(az-aspect)
  shade=((shade-shade.min())/(shade.max()-shade.min())*255).astype(np.uint8)
  Image.fromarray(shade,mode="L").save(OUT/"chasewild_m0_lidar_hillshade.png")

  y_studs=math.ceil((zrange*STUDS_PER_M)/4)*4
  meta={
    "contentVersion":"m0-terrain-lidar-2",
    "sourceClass":"PLATFORM/DATA",
    "source":"2019 FEMA/USGS NY FEMA Region 2 Central original-product-resolution 1 m hydro-flattened DEM derived from classified lidar Class 2 ground returns and hydro breaklines",
    "sourceProject":PROJECT,
    "sourceWorkUnit":WORKUNIT,
    "sourceCRS":SRC_CRS,
    "verticalDatum":"NAVD88 (GEOID12B), meters per project metadata",
    "nativeGroundSampleDistanceM":1.0,
    "aoi":{"centerLon":CENTER_LON,"centerLat":CENTER_LAT,"centerProjected":[cx,cy],
           "boundsProjected":list(bounds),"widthM":SIZE_M,"heightM":SIZE_M,
           "elevationMinM":zmin,"elevationMaxM":zmax,"elevationRangeM":zrange},
    "robloxImport":{
      "heightmapFile":"chasewild_m0_heightmap_lidar_1m.png",
      "sizeStuds":{"x":SIZE_M*STUDS_PER_M,"y":y_studs,"z":SIZE_M*STUDS_PER_M},
      "positionStuds":{"x":0,"y":y_studs/2,"z":0},
      "horizontalStudsPerMeter":STUDS_PER_M,
      "verticalStudsPerMeter":STUDS_PER_M,
      "oneMeterSourceCellToOneTerrainVoxel":True,
      "defaultMaterial":"Snow"
    },
    "fidelityPolicy":{
      "globalSmoothing":False,
      "verticalExaggeration":False,
      "horizontalExaggeration":False,
      "baseTerrainEditsAllowed":False,
      "gameplayEdits":"Only explicit, versioned local corridor edits after natural-terrain Studio test."
    },
    "sourceTiles":source_meta
  }
  (OUT/"chasewild_m0_lidar_terrain_import.json").write_text(json.dumps(meta,indent=2))
  print(json.dumps({
    "shape":list(z.shape),"centerProjected":[cx,cy],"bounds":list(bounds),
    "zmin":zmin,"zmax":zmax,"zrange":zrange,
    "robloxXZStuds":SIZE_M*STUDS_PER_M,"robloxYStuds":y_studs,
    "tiles":[m["file"] for m in source_meta]
  },indent=2))
finally:
  for mf,ds in datasets:
    ds.close(); mf.close()
