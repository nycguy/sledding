#!/usr/bin/env python3
import io, json, math, os, sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

WC = "https://giswww.westchestergov.com/arcgis/rest/services"
PLAN = WC + "/DataHub_BasemapPlanimetrics/MapServer"
ENV = WC + "/DataHub_EnvironmentandPlanning/MapServer"
DEP = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
FT = 3.28084
R_EARTH = 6378137.0

CENTER_LAT = float(os.getenv("CENTER_LAT", "41.1935"))
CENTER_LON = float(os.getenv("CENTER_LON", "-73.7230"))
SIZE_MI = float(os.getenv("SIZE_MI", "1.0"))
TARGET_SPEED_MPH = 24.0
TARGET_LEN_FT = 550.0
TARGET_DROP_FT = 41.0
OUT = Path("analysis-output")
OUT.mkdir(exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "ChaseRobloxRouteAnalysis/1.0"})

def merc(lon, lat):
    x = R_EARTH * math.radians(lon)
    y = R_EARTH * math.log(math.tan(math.pi/4 + math.radians(lat)/2))
    return x, y

def lonlat(x, y):
    lon = math.degrees(x / R_EARTH)
    lat = math.degrees(2 * math.atan(math.exp(y / R_EARTH)) - math.pi/2)
    return lon, lat

def get_png(url, params, size):
    r = session.get(url, params=params, timeout=90)
    r.raise_for_status()
    im = Image.open(io.BytesIO(r.content)).convert("RGBA")
    if im.size != size:
        im = im.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(im)

def stretch_rule(vmin, vmax):
    return json.dumps({
        "rasterFunction":"Stretch",
        "rasterFunctionArguments":{
            "StretchType":5,"Min":0,"Max":255,
            "Statistics":[[vmin,vmax,(vmin+vmax)/2,(vmax-vmin)/4]],
            "DRA":False,"UseGamma":False
        },
        "outputPixelType":"U8"
    }, separators=(",",":"))

def dem_png(bbox, w, h, vmin, vmax):
    return get_png(DEP + "/exportImage", {
        "bbox": ",".join(map(str,bbox)), "bboxSR":"3857", "imageSR":"3857",
        "size":f"{w},{h}", "format":"png", "interpolation":"RSP_BilinearInterpolation",
        "renderingRule":stretch_rule(vmin,vmax), "f":"image"
    }, (w,h))

def fetch_dem(ext, n):
    bbox = (ext["xmin"], ext["ymin"], ext["xmax"], ext["ymax"])
    cmin, cmax = -20.0, 420.0
    coarse = dem_png(bbox,n,n,cmin,cmax)
    cs = (ext["xmax"]-ext["xmin"])/n
    z = np.full((n,n), np.nan, dtype=np.float32)
    t=3
    tn=n//t
    for ty in range(t):
        for tx in range(t):
            r0,r1=ty*tn,(ty+1)*tn
            c0,c1=tx*tn,(tx+1)*tn
            tile=coarse[r0:r1,c0:c1]
            valid=tile[:,:,3] >= 250
            if not valid.any():
                continue
            vals=cmin + tile[:,:,0].astype(np.float32)/255.0*(cmax-cmin)
            lo=float(vals[valid].min())-4
            hi=float(vals[valid].max())+4
            x0=ext["xmin"]+c0*cs; x1=ext["xmin"]+c1*cs
            y1=ext["ymax"]-r0*cs; y0=ext["ymax"]-r1*cs
            px=dem_png((x0,y0,x1,y1),tn,tn,lo,hi)
            ok=px[:,:,3]>=250
            zv=lo + px[:,:,0].astype(np.float32)/255.0*(hi-lo)
            sub=z[r0:r1,c0:c1]
            sub[ok]=zv[ok]
    return z

def fetch_mask(service, ids, ext, n, line=False):
    dyn=[]
    for i,mid in enumerate(ids):
        symbol = ({"type":"esriSLS","style":"esriSLSSolid","color":[0,0,0,255],"width":2}
                  if line else
                  {"type":"esriSFS","style":"esriSFSSolid","color":[0,0,0,255],
                   "outline":{"type":"esriSLS","style":"esriSLSSolid","color":[0,0,0,255],"width":1}})
        dyn.append({"id":900+i,"source":{"type":"mapLayer","mapLayerId":mid},
                    "drawingInfo":{"renderer":{"type":"simple","symbol":symbol}}})
    base={
        "bbox":f'{ext["xmin"]},{ext["ymin"]},{ext["xmax"]},{ext["ymax"]}',
        "bboxSR":"3857","imageSR":"3857","size":f"{n},{n}",
        "transparent":"true","format":"png32","f":"image"
    }
    try:
        px=get_png(service+"/export", {**base,"dynamicLayers":json.dumps(dyn,separators=(",",":"))}, (n,n))
    except Exception:
        px=get_png(service+"/export", {**base,"layers":"show:"+",".join(map(str,ids))}, (n,n))
    return (px[:,:,3] > 40).astype(np.uint8)

def dilate(src, radius):
    if radius<=0: return src.copy()
    h,w=src.shape
    out=src.copy()
    ys,xs=np.where(src>0)
    offs=[(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)
          if dx*dx+dy*dy<=radius*radius]
    for y,x in zip(ys,xs):
        for dy,dx in offs:
            yy=y+dy; xx=x+dx
            if 0<=yy<h and 0<=xx<w: out[yy,xx]=1
    return out

def smooth3(z):
    out=np.full_like(z,np.nan)
    h,w=z.shape
    for r in range(h):
        r0=max(0,r-1); r1=min(h,r+2)
        for c in range(w):
            c0=max(0,c-1); c1=min(w,c+2)
            a=z[r0:r1,c0:c1]
            good=a[np.isfinite(a)]
            if len(good): out[r,c]=good.mean()
    return out

def bilinear(a,x,y):
    h,w=a.shape
    c=int(math.floor(x)); r=int(math.floor(y))
    if c<1 or r<1 or c>=w-2 or r>=h-2: return None
    fx=x-c; fy=y-r
    vals=[a[r,c],a[r,c+1],a[r+1,c],a[r+1,c+1]]
    if any(not math.isfinite(float(v)) for v in vals): return None
    return ((vals[0]*(1-fx)+vals[1]*fx)*(1-fy)+
            (vals[2]*(1-fx)+vals[3]*fx)*fy)

def analyze():
    cx,cy=merc(CENTER_LON,CENTER_LAT)
    half=SIZE_MI*1609.34/2/math.cos(math.radians(CENTER_LAT))
    ext={"xmin":cx-half,"xmax":cx+half,"ymin":cy-half,"ymax":cy+half}
    n=402
    cs=(ext["xmax"]-ext["xmin"])/n
    cell=cs*math.cos(math.radians(CENTER_LAT))
    print("AOI", ext, "cell_m", cell)

    z=fetch_dem(ext,n)
    valid=np.isfinite(z)
    print("elevation valid", float(valid.mean()))
    if valid.mean()<0.3: raise RuntimeError("Elevation mostly empty")

    road=fetch_mask(PLAN,[7],ext,n)
    drive=fetch_mask(PLAN,[9,10],ext,n)
    bldg=fetch_mask(PLAN,[5],ext,n)
    water=fetch_mask(PLAN,[11],ext,n)
    stream=fetch_mask(PLAN,[0],ext,n,line=True)
    openm=fetch_mask(ENV,[209],ext,n)

    R=lambda m: max(0,round(m/cell))
    road_near=dilate(road,R(10/FT))
    drive=dilate(drive,R(30/FT))
    bldg=dilate(bldg,R(40/FT))
    water=dilate(water,R(10/FT))
    stream=dilate(stream,R(15/FT))
    haz=((road|drive|bldg|water|stream)>0).astype(np.uint8)
    haz[~valid]=1

    zs=smooth3(z)
    gx=np.zeros_like(zs); gy=np.zeros_like(zs)
    gx[:,1:-1]=(zs[:,2:]-zs[:,:-2])/(2*cell)
    gy[1:-1,:]=(zs[2:,:]-zs[:-2,:])/(2*cell)
    slope=np.degrees(np.arctan(np.hypot(gx,gy)))

    g=9.81; mu=0.05; kd=0.0036; latMax=0.3*g; dt=.08
    dirs=[(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

    def at(x,y):
        zv=bilinear(zs,x,y); gxx=bilinear(gx,x,y); gyy=bilinear(gy,x,y)
        if zv is None or gxx is None or gyy is None: return None
        c=int(math.floor(x)); r=int(math.floor(y))
        return zv,gxx,gyy,r,c

    def highest(rc):
        r,c=rc; best=None; bz=zs[r,c]
        for dx,dy in dirs:
            rr=r+dy; cc=c+dx
            if 0<=rr<n and 0<=cc<n and not haz[rr,cc] and zs[rr,cc]>bz:
                bz=zs[rr,cc]; best=(rr,cc)
        return best

    def ride(top):
        r0,c0=top; x=c0+.5; y=r0+.5; vx=vy=0.0; length=t=vmax=0.0
        gd=at(x,y)
        if gd is None:return None
        samp=[(x,y,gd[0],0.0,0.0)]; cells=[top]; stopped=False
        lastd=0
        for st in range(3000):
            gd=at(x,y)
            if gd is None:return None
            zz,gxx,gyy,r,c=gd
            if haz[r,c]: return None
            g2=gxx*gxx+gyy*gyy; cosT=1/math.sqrt(1+g2); c2=cosT*cosT
            ax=-g*gxx*c2; ay=-g*gyy*c2
            sp=math.hypot(vx,vy)
            if sp>0.05:
                fr=mu*g*cosT+kd*sp*sp
                ax-=fr*vx/sp; ay-=fr*vy/sp
                ux=vx/sp; uy=vy/sp
                along=ax*ux+ay*uy; lat=-ax*uy+ay*ux
                lat=max(-latMax,min(latMax,lat))
                ax=along*ux-lat*uy; ay=along*uy+lat*ux
            else:
                drivef=math.hypot(ax,ay)
                if drivef<=mu*g*cosT:
                    if st>0: stopped=True
                    break
                kk=(drivef-mu*g*cosT)/drivef; ax*=kk; ay*=kk
            nvx=vx+ax*dt; nvy=vy+ay*dt
            if sp>0.05 and nvx*vx+nvy*vy<0 and sp<1:
                stopped=True; break
            vx,vy=nvx,nvy
            dx=vx*dt; dy=vy*dt
            x+=dx/cell; y+=dy/cell
            length+=math.hypot(dx,dy)
            vv=math.hypot(vx,vy); vmax=max(vmax,vv); t+=dt
            rr=int(math.floor(y)); cc=int(math.floor(x))
            if not (0<=rr<n and 0<=cc<n): return None
            if (rr,cc)!=cells[-1]: cells.append((rr,cc))
            if length-lastd>=2:
                q=at(x,y)
                if q: samp.append((x,y,q[0],length,vv)); lastd=length
            if t>150:return None
        if not stopped:return None
        q=at(x,y)
        if q: samp.append((x,y,q[0],length,0.0))
        er,ec=cells[-1]
        if road_near[er,ec]:return None
        if len(samp)<2:return None
        max_pitch=0.0
        for k in range(3,len(samp)):
            dd=samp[k][3]-samp[k-3][3]
            if dd>0:
                pp=math.degrees(math.atan2(samp[k-3][2]-samp[k][2],dd))
                max_pitch=max(max_pitch,pp)
        return {
            "samp":samp,"cells":cells,"top":top,"end":cells[-1],
            "lenFt":length*FT,"dropFt":(samp[0][2]-samp[-1][2])*FT,
            "vmax":vmax*2.23694,"maxPitch":max_pitch,"secs":t,
            "open":bool(openm[top])
        }

    seen=set(); found=[]
    for r in range(2,n-2,3):
        for c in range(2,n-2,3):
            if haz[r,c] or slope[r,c]<7: continue
            top=(r,c)
            for _ in range(300):
                j=highest(top)
                if j is None or slope[j]<3.5: break
                top=j
            if top in seen or haz[top]: continue
            seen.add(top)
            f=ride(top)
            if not f or f["maxPitch"]>30 or f["lenFt"]<200 or f["dropFt"]<12: continue
            f["score"]=f["vmax"]*f["lenFt"]/1000
            found.append(f)
    found.sort(key=lambda f:f["score"], reverse=True)

    occ=np.zeros((n,n),dtype=np.uint8); keep=[]
    for f in found:
        hits=sum(1 for r,c in f["cells"] if occ[r,c])
        if hits>len(f["cells"])*.25: continue
        keep.append(f)
        m=np.zeros((n,n),dtype=np.uint8)
        for r,c in f["cells"]:m[r,c]=1
        occ=np.maximum(occ,dilate(m,R(12)))
        if len(keep)>=10:break

    def point_ll(x,y):
        mx=ext["xmin"]+x*cs
        my=ext["ymax"]-y*cs
        return lonlat(mx,my)

    for idx,f in enumerate(keep,1):
        f["rank"]=idx
        f["coords"]=[point_ll(p[0],p[1]) for p in f["samp"]]
        f["startLonLat"]=f["coords"][0]
        f["endLonLat"]=f["coords"][-1]

    def anchor_error(f):
        return ((f["vmax"]-TARGET_SPEED_MPH)/5)**2 + ((f["lenFt"]-TARGET_LEN_FT)/150)**2 + ((f["dropFt"]-TARGET_DROP_FT)/20)**2
    anchor=min(keep,key=anchor_error)
    anchor_idx=anchor["rank"]
    ar,ac=anchor["top"]
    az=zs[ar,ac]

    # Search higher/easterly safe cells whose natural sled trajectory merges near the screenshot #4 start.
    # These are game-course candidates, not claims that the real woods are currently safe for sledding.
    candidates=[]
    for r in range(2,n-2,4):
        for c in range(2,n-2,4):
            if haz[r,c] or slope[r,c]<5: continue
            dx=(c-ac)*cell; dy=(r-ar)*cell
            dist=math.hypot(dx,dy)
            if dist<120/FT or dist>1400/FT: continue
            if c <= ac: continue  # neighborhood/wooded side visible east of the open-hill anchor
            if zs[r,c] < az + 8/FT: continue
            f=ride((r,c))
            if not f or f["maxPitch"]>35 or f["lenFt"]<150: continue
            # closest approach to anchor top
            min_cells=min(math.hypot(rr-ar,cc-ac) for rr,cc in f["cells"])
            if min_cells*cell > 120/FT: continue
            # merge point: first cell within 120 ft
            mi=next((i for i,(rr,cc) in enumerate(f["cells"]) if math.hypot(rr-ar,cc-ac)*cell<=120/FT),None)
            if mi is None: continue
            # only retain substantial wooded pre-run
            pre_len=min(f["lenFt"], dist*FT)
            extra_drop=(zs[r,c]-az)*FT
            score=pre_len + extra_drop*10 - min_cells*cell*FT*2
            candidates.append((score,f,min_cells*cell*FT,extra_drop))
    candidates.sort(key=lambda x:x[0], reverse=True)

    ext_keep=[]
    start_seen=[]
    for score,f,nearft,extra_drop in candidates:
        sr,sc=f["top"]
        if any(math.hypot(sr-r0,sc-c0)*cell<150/FT for r0,c0 in start_seen): continue
        start_seen.append((sr,sc))
        ext_keep.append((score,f,nearft,extra_drop))
        if len(ext_keep)>=3:break

    features=[]
    def add_line(name,coords,props):
        features.append({"type":"Feature","properties":{"name":name,**props},
                         "geometry":{"type":"LineString","coordinates":[list(x) for x in coords]}})
    for f in keep:
        add_line(f'Fall Line {f["rank"]}',f["coords"],{
            "kind":"fall_line","rank":f["rank"],"speed_mph":round(f["vmax"],1),
            "length_ft":round(f["lenFt"]), "drop_ft":round(f["dropFt"]),
            "max_pitch_deg":round(f["maxPitch"],1)
        })
    add_line("Open-hill anchor (screenshot #4 match)",anchor["coords"],{
        "kind":"anchor","matched_rank":anchor_idx,"speed_mph":round(anchor["vmax"],1),
        "length_ft":round(anchor["lenFt"]), "drop_ft":round(anchor["dropFt"]),
        "target_speed_mph":TARGET_SPEED_MPH,"target_length_ft":TARGET_LEN_FT,"target_drop_ft":TARGET_DROP_FT
    })
    ext_summ=[]
    for i,(score,f,nearft,extra_drop) in enumerate(ext_keep,1):
        f["coords"]=[point_ll(p[0],p[1]) for p in f["samp"]]
        f["startLonLat"]=f["coords"][0]
        f["endLonLat"]=f["coords"][-1]
        add_line(f"Wooded extension candidate {i}",f["coords"],{
            "kind":"wooded_extension_candidate","candidate":i,
            "speed_mph":round(f["vmax"],1),"length_ft":round(f["lenFt"]),
            "drop_ft":round(f["dropFt"]),"closest_to_anchor_ft":round(nearft),
            "extra_start_elevation_ft":round(extra_drop)
        })
        ext_summ.append({
            "candidate":i,"start_lon":f["startLonLat"][0],"start_lat":f["startLonLat"][1],
            "speed_mph":round(f["vmax"],1),"natural_run_length_ft":round(f["lenFt"]),
            "natural_drop_ft":round(f["dropFt"]),"closest_to_anchor_ft":round(nearft),
            "extra_start_elevation_ft":round(extra_drop)
        })

    gj={"type":"FeatureCollection","properties":{
        "center":[CENTER_LON,CENTER_LAT],"size_mi":SIZE_MI,
        "note":"Game-design screening only. Wooded extensions use bare-earth terrain and Westchester hazard masks; tree-level clearance requires canopy/visual QA."
    },"features":features}
    (OUT/"chase_leonard_routes.geojson").write_text(json.dumps(gj,indent=2))

    summary={
        "aoi":{"center_lat":CENTER_LAT,"center_lon":CENTER_LON,"size_mi":SIZE_MI,"cell_ground_m":cell},
        "anchor_match":{
            "rank_in_rerun":anchor_idx,"speed_mph":round(anchor["vmax"],1),
            "length_ft":round(anchor["lenFt"]),"drop_ft":round(anchor["dropFt"]),
            "start_lon":anchor["startLonLat"][0],"start_lat":anchor["startLonLat"][1],
            "end_lon":anchor["endLonLat"][0],"end_lat":anchor["endLonLat"][1]
        },
        "wooded_extension_candidates":ext_summ,
        "warnings":[
            "UNVERIFIED: tree trunks/canopy are not in the bare-earth 3DEP surface.",
            "UNVERIFIED: candidates are for a fictional Roblox course, not a recommendation for real-world sledding.",
            "Private-property ownership is intentionally not exported."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    md=["# Chase → Leonard Park Roblox sled-route screening","",
        "This reruns the Fall Line terrain model against USGS 3DEP elevation and Westchester County hazard layers, then finds higher east-side runs that approach the open-hill anchor represented by the screenshot's Course #4 metrics.","",
        "## Open-hill anchor",
        f'- Rerun match: Fall Line rank **{anchor_idx}**',
        f'- {anchor["vmax"]:.1f} mph, {anchor["lenFt"]:.0f} ft, {anchor["dropFt"]:.0f} ft drop',
        f'- Start: {anchor["startLonLat"][1]:.6f}, {anchor["startLonLat"][0]:.6f}',
        f'- End: {anchor["endLonLat"][1]:.6f}, {anchor["endLonLat"][0]:.6f}',"",
        "## Wooded extension candidates"]
    if ext_summ:
        for x in ext_summ:
            md += [f'### Candidate {x["candidate"]}',
                   f'- Start: {x["start_lat"]:.6f}, {x["start_lon"]:.6f}',
                   f'- Natural terrain run: {x["natural_run_length_ft"]} ft, {x["natural_drop_ft"]} ft drop, {x["speed_mph"]} mph',
                   f'- Comes within {x["closest_to_anchor_ft"]} ft of the open-hill anchor start',
                   f'- Starts about {x["extra_start_elevation_ft"]} ft above the anchor start',""] 
    else:
        md += ["No natural fall-line extension met the merge criteria. The game route should use a designed connector corridor rather than pretending the terrain naturally merges.",""]
    md += ["## Evidence limits",
           "- **PASS:** elevation and GIS obstacle data were fetched and analyzed.",
           "- **UNVERIFIED:** tree-level clearance and visual route quality require 2025 imagery/canopy inspection.",
           "- **GAME-SPECIFIC:** the Roblox version can widen/grade the woodland corridor while preserving the recognizable terrain.",
           "- This is not a real-world sledding safety assessment."]
    (OUT/"REPORT.md").write_text("\n".join(md))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    analyze()
