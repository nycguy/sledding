#!/usr/bin/env python3
"""Export Chase Wild M0 Roblox terrain inputs from the verified GIS/LiDAR analysis.

This runs only on the temporary Fall Line analysis branch. It does not make the
Fall Line repository authoritative for Chase Wild game code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import chase_roblox_route_analysis as base

OUT = Path("analysis-output")
ROUTE_FILE = OUT / "chase_leonard_routes.geojson"

HORIZONTAL_STUDS_PER_METER = 3.0
HEIGHTMAP_PIXELS = 512
ROBLOX_STUDS_PER_HEIGHTMAP_PIXEL = 4.0
GROUND_METERS = HEIGHTMAP_PIXELS * ROBLOX_STUDS_PER_HEIGHTMAP_PIXEL / HORIZONTAL_STUDS_PER_METER
ROUTE_SAMPLE_M = 4.0


def local_meters(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    """Small-area equirectangular east/north meters."""
    east = math.radians(lon - lon0) * 6378137.0 * math.cos(math.radians(lat0))
    north = math.radians(lat - lat0) * 6378137.0
    return east, north


def distance_m(a, b, lat0):
    ax, ay = local_meters(a[0], a[1], 0.0, lat0)
    bx, by = local_meters(b[0], b[1], 0.0, lat0)
    return math.hypot(ax - bx, ay - by)


def cumulative(coords):
    lat0 = sum(p[1] for p in coords) / len(coords)
    out = [0.0]
    for a, b in zip(coords, coords[1:]):
        out.append(out[-1] + distance_m(a, b, lat0))
    return out


def interpolate(a, b, t):
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def resample(coords, step_m):
    cum = cumulative(coords)
    total = cum[-1]
    targets = list(np.arange(0.0, total, step_m))
    if not targets or abs(targets[-1] - total) > 0.01:
        targets.append(total)
    out = []
    seg = 0
    for d in targets:
        while seg < len(cum) - 2 and cum[seg + 1] < d:
            seg += 1
        a_d, b_d = cum[seg], cum[seg + 1]
        t = 0.0 if b_d <= a_d else (d - a_d) / (b_d - a_d)
        out.append(interpolate(coords[seg], coords[seg + 1], t))
    return out, total


def point_at_distance(coords, d):
    cum = cumulative(coords)
    d = max(0.0, min(d, cum[-1]))
    seg = 0
    while seg < len(cum) - 2 and cum[seg + 1] < d:
        seg += 1
    span = cum[seg + 1] - cum[seg]
    t = 0.0 if span <= 0 else (d - cum[seg]) / span
    return interpolate(coords[seg], coords[seg + 1], t)


def main():
    gj = json.loads(ROUTE_FILE.read_text())
    anchor = next(f for f in gj["features"] if f["properties"].get("kind") == "anchor")
    wooded = next(f for f in gj["features"] if f["properties"].get("kind") == "wooded_extension_candidate")

    acoords = anchor["geometry"]["coordinates"]
    wcoords = wooded["geometry"]["coordinates"]
    anchor_start = acoords[0]

    wcum = cumulative(wcoords)
    lat0 = sum(p[1] for p in wcoords + acoords) / (len(wcoords) + len(acoords))
    closest_i = min(range(len(wcoords)), key=lambda i: distance_m(wcoords[i], anchor_start, lat0))
    wooded_to_merge = wcoords[: closest_i + 1]
    merge_start = wooded_to_merge[-1]
    connector_m = distance_m(merge_start, anchor_start, lat0)

    # Straight connector is intentional M0 GAME-SPECIFIC authored terrain.
    combined = wooded_to_merge + [anchor_start] + acoords[1:]
    route, route_total_m = resample(combined, ROUTE_SAMPLE_M)

    lon_min = min(p[0] for p in combined)
    lon_max = max(p[0] for p in combined)
    lat_min = min(p[1] for p in combined)
    lat_max = max(p[1] for p in combined)
    center_lon = (lon_min + lon_max) / 2
    center_lat = (lat_min + lat_max) / 2

    cx, cy = base.merc(center_lon, center_lat)
    ground_half = GROUND_METERS / 2
    projected_half = ground_half / math.cos(math.radians(center_lat))
    ext = {
        "xmin": cx - projected_half,
        "xmax": cx + projected_half,
        "ymin": cy - projected_half,
        "ymax": cy + projected_half,
    }

    print("Fetching 512x512 M0 terrain...")
    z = base.fetch_dem(ext, HEIGHTMAP_PIXELS)
    valid = np.isfinite(z)
    if valid.mean() < 0.95:
        raise RuntimeError(f"DEM coverage only {valid.mean():.1%}")

    zmin = float(z[valid].min())
    zmax = float(z[valid].max())
    zrange = zmax - zmin
    if zrange <= 0:
        raise RuntimeError("DEM range is zero")

    normalized = np.zeros_like(z, dtype=np.uint8)
    normalized[valid] = np.clip(np.rint((z[valid] - zmin) / zrange * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(normalized, mode="L").save(OUT / "chasewild_m0_heightmap.png")

    # Roblox: one heightmap pixel = four studs.
    xz_studs = HEIGHTMAP_PIXELS * ROBLOX_STUDS_PER_HEIGHTMAP_PIXEL
    y_studs = math.ceil((zrange * HORIZONTAL_STUDS_PER_METER) / 4.0) * 4
    vertical_spm = y_studs / zrange
    import_meta = {
        "contentVersion": "m0-terrain-1",
        "heightmapFile": "chasewild_m0_heightmap.png",
        "source": "USGS 3DEP bare-earth elevation via Fall Line analysis pipeline",
        "pixels": [HEIGHTMAP_PIXELS, HEIGHTMAP_PIXELS],
        "robloxImport": {
            "sizeStuds": {"x": xz_studs, "y": y_studs, "z": xz_studs},
            "positionStuds": {"x": 0, "y": y_studs / 2, "z": 0},
            "defaultMaterial": "Snow",
            "horizontalStudsPerMeter": HORIZONTAL_STUDS_PER_METER,
            "verticalStudsPerMeterActual": vertical_spm,
        },
        "aoi": {
            "centerLon": center_lon,
            "centerLat": center_lat,
            "groundWidthMeters": GROUND_METERS,
            "elevationMinMeters": zmin,
            "elevationMaxMeters": zmax,
            "elevationRangeMeters": zrange,
        },
        "notes": [
            "Heightmap terrain is source-derived baseline, not final game corridor.",
            "The 84-ft merge connector is GAME-SPECIFIC and may be graded/widened in Studio.",
            "Private parcel-owner attributes are not exported.",
        ],
    }
    (OUT / "chasewild_m0_terrain_import.json").write_text(json.dumps(import_meta, indent=2))

    # Local Roblox coordinates: +X east, +Z south, world center at AOI center.
    def to_studs(p):
        east, north = local_meters(p[0], p[1], center_lon, center_lat)
        return {"x": east * HORIZONTAL_STUDS_PER_METER, "z": -north * HORIZONTAL_STUDS_PER_METER}

    route_points = []
    rcum = cumulative(route)
    for p, d in zip(route, rcum):
        q = to_studs(p)
        route_points.append({"x": round(q["x"], 3), "z": round(q["z"], 3), "distanceM": round(d, 3)})

    wooded_m = wcum[closest_i]
    connector_end_m = wooded_m + connector_m
    checkpoints_def = [
        ("Start", 0.0),
        ("UpperChaseWoods", min(55.0, wooded_m * 0.25)),
        ("LeonardWoods", wooded_m * 0.62),
        ("WoodsExitMergeGlade", wooded_m),
        ("OpenHill", min(route_total_m - 35.0, connector_end_m + 80.0)),
        ("Finish", route_total_m),
    ]
    checkpoints = []
    for order, (name, d) in enumerate(checkpoints_def, start=1):
        p = point_at_distance(combined, d)
        q = to_studs(p)
        checkpoints.append({
            "id": f"CP{order}",
            "name": name,
            "order": order,
            "distanceM": round(d, 2),
            "x": round(q["x"], 3),
            "z": round(q["z"], 3),
            "radiusStuds": 54 if order not in (1, 6) else 60,
        })

    course = {
        "courseId": "chase-signature",
        "courseVersion": "m0-1",
        "terrainContentVersion": "m0-terrain-1",
        "coordinateSystem": {
            "worldOriginLon": center_lon,
            "worldOriginLat": center_lat,
            "xAxis": "east",
            "zAxis": "south",
            "horizontalStudsPerMeter": HORIZONTAL_STUDS_PER_METER,
        },
        "sourceEvidence": {
            "woodedNaturalRunFt": wooded["properties"]["length_ft"],
            "woodedNaturalDropFt": wooded["properties"]["drop_ft"],
            "openHillAnchorFt": anchor["properties"]["length_ft"],
            "openHillAnchorDropFt": anchor["properties"]["drop_ft"],
            "designedConnectorFt": round(connector_m * base.FT),
        },
        "designedRoute": {
            "approxLengthM": round(route_total_m, 2),
            "approxLengthFt": round(route_total_m * base.FT),
            "mainPackedSnowWidthStuds": 36,
            "recoveryCorridorWidthStuds": 54,
            "points": route_points,
        },
        "checkpoints": checkpoints,
        "validation": {
            "maxPlausibleSpeedStudsPerSec": 135,
            "checkpointGraceStuds": 12,
            "minFinishSeconds": 16,
            "maxRunSeconds": 180,
        },
    }
    (OUT / "chasewild_m0_course.json").write_text(json.dumps(course, indent=2))

    # Quick route-on-heightmap preview.
    preview = Image.fromarray(normalized, mode="L").convert("RGB")
    draw = ImageDraw.Draw(preview)
    def to_px(p):
        mx, my = base.merc(p[0], p[1])
        return (
            (mx - ext["xmin"]) / (ext["xmax"] - ext["xmin"]) * HEIGHTMAP_PIXELS,
            (ext["ymax"] - my) / (ext["ymax"] - ext["ymin"]) * HEIGHTMAP_PIXELS,
        )
    pts = [to_px(p) for p in combined]
    draw.line(pts, fill=(255, 120, 0), width=4)
    for cp in checkpoints:
        # find geographic point again by cp distance
        p = point_at_distance(combined, cp["distanceM"])
        x, y = to_px(p)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(0, 170, 255), outline=(255, 255, 255), width=1)
    preview.resize((1024, 1024), Image.Resampling.NEAREST).save(OUT / "chasewild_m0_heightmap_route_preview.png")

    summary = {
        "routeLengthFt": round(route_total_m * base.FT),
        "woodedToMergeFt": round(wooded_m * base.FT),
        "connectorFt": round(connector_m * base.FT),
        "openHillFt": anchor["properties"]["length_ft"],
        "heightmapPixels": HEIGHTMAP_PIXELS,
        "terrainXZStuds": xz_studs,
        "terrainYStuds": y_studs,
        "elevationRangeM": round(zrange, 2),
        "center": [center_lon, center_lat],
    }
    (OUT / "chasewild_m0_export_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
