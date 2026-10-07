# Chase → Leonard Park Roblox signature-run analysis

Status: **GIS/LiDAR screening complete**

## Purpose

Identify a terrain-supported Roblox sledding course that:

1. begins on the wooded Mount Kisco Chase side;
2. descends through Leonard Park woodland;
3. connects to the existing open sledding hillside represented by the user's Fall Line Course #4 screenshot;
4. preserves real terrain character while allowing game-specific widening/grading and wildlife gameplay.

This is a game-design analysis, not a recommendation for real-world sledding.

## Data and method

- USGS 3DEP bare-earth elevation, using the same export/physics method as the existing Fall Line app.
- Westchester County GIS road, driveway, parking, building, water, stream, and open-space masks.
- Westchester County 2025 aerial imagery for visual woodland/open-ground review.
- 1-mile study square centered at 41.1935, -73.7230.
- Approximate terrain-analysis cell size: 4.0 m.

Private property-owner attributes are intentionally not exported.

## Open-hill anchor

The rerun produced a course whose geometry and metrics closely match the screenshot's Course #4:

- Rerun rank: **5** (rank is AOI/result-set dependent)
- Top speed: **23.1 mph**
- Length: **602 ft**
- Drop: **41 ft**
- Start: **41.193770, -73.721256**
- End: **41.194495, -73.722842**

Screenshot target: 24 mph, 550 ft, 41 ft drop.

**PASS:** this is a strong terrain/physics anchor for the current open-hill finishing section.

## Wooded extension candidate

The analysis found one especially strong higher/east-side terrain line:

- Start: **41.192008, -73.720061**
- Natural terrain run: **1,259 ft**
- Natural drop: **94 ft**
- Modeled peak speed: **22.6 mph**
- Starts approximately **51 ft above** the open-hill anchor start.
- Comes within approximately **84 ft** of the open-hill anchor start.

Along the candidate trajectory, the closest approach occurs after approximately **763 ft** of travel. A game-designed connector of about **84 ft** can join that point to the open-hill anchor.

### Recommended signature-run geometry

Approximate first-pass course:

- wooded descent to merge: **763 ft**
- designed merge/connector: **84 ft**
- open-hill finish: **602 ft**
- total: **~1,449 ft**
- total vertical drop: **~92 ft**

The combined-course peak speed has **not** yet been validated with a continuous forced-route physics simulation.

## 2025 aerial review

The County 2025 aerial overlay shows the proposed upper candidate traveling through a substantially wooded corridor before reaching the edge of the open recreation/sledding area. The final anchor continues across open ground.

**PASS:** broad wooded-to-open spatial sequence is supported by current aerial imagery.

**UNVERIFIED:** individual tree-trunk clearance, understory, rocks, fallen timber, and whether a real continuous sled corridor exists on the ground. Bare-earth LiDAR and aerial canopy cannot prove those conditions.

For Roblox this is acceptable because the course is deliberately fictionalized: preserve the recognizable terrain, woodland character, and open-hill transition, while widening/clearing the virtual corridor to achieve readable, fair gameplay.

## Game-design interpretation

Use the upper route as the signature woodland descent, with:
- a forgiving main line for younger/mobile players;
- optional faster side lines;
- wildlife encounters that never require striking animals;
- deer/turkey/rabbit as common sightings, fox uncommon, bear/bobcat rare and generally distant;
- a dramatic visual transition from enclosed woods to the open hill;
- immediate restart/return after the finish rather than a forced uphill walk.

## Evidence status

- **PASS:** elevation service fetched and analyzed.
- **PASS:** Westchester GIS obstacle masks fetched and analyzed.
- **PASS:** open-hill anchor reproduced closely from Course #4 metrics.
- **PASS:** a higher wooded terrain candidate approaches the open-hill anchor closely enough for a short designed connector.
- **PASS:** 2025 aerial imagery supports the broad woodland-to-open sequence.
- **GAME-SPECIFIC:** virtual corridor widening, grading, jumps, route branches, and animal placements.
- **UNVERIFIED:** tree-level clearance and human visual/readability/fun until Roblox Studio and playtesting.
- **UNVERIFIED:** continuous combined-route speed until the final designed centerline is simulated.

## Next development artifact

Use this route as the world backbone in the first authoritative GameSpec. Do not begin large-scale Roblox implementation until the GameSpec defines audience, sled controls/physics, wildlife behavior, route difficulty bands, FTUE, performance/streaming targets, and privacy/fidelity rules.
