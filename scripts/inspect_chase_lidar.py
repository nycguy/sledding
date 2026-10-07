#!/usr/bin/env python3
import json, requests, re
from urllib.parse import quote

SOURCES = [
  "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/NY_FEMAR2_Central_B1_2018/ept.json",
  "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/NY_FEMAR2_Central_2_2018/ept.json",
]
for url in SOURCES:
    r=requests.get(url,timeout=60)
    print("\nEPT",url,r.status_code)
    if r.ok:
        j=r.json()
        print(json.dumps({k:j.get(k) for k in ("bounds","boundsConforming","points","srs","schema","span")},indent=2)[:12000])

prefixes=[
 "StagedProducts/Elevation/OPR/Projects/NY_FEMAR2_Central_2018_D19/NY_FEMAR2_Central_B1_2018/TIFF/",
 "StagedProducts/Elevation/OPR/Projects/NY_FEMAR2_Central_2018_D19/NY_FEMAR2_Central_2_2018/TIFF/",
]
for prefix in prefixes:
    url="https://prd-tnm.s3.amazonaws.com/?list-type=2&prefix="+quote(prefix,safe="/")
    r=requests.get(url,timeout=60)
    print("\nS3",prefix,r.status_code,"bytes",len(r.content))
    keys=re.findall(r"<Key>(.*?)</Key>",r.text)
    print("keys",len(keys))
    for k in keys[:15]: print(k)
    if len(keys)>15:
        print("...")
        for k in keys[-5:]: print(k)
