# trail-hazard-map (demo build)

Runs on a laptop, CPU only, no network, no API keys:

    pip install -e .[dev]
    trail-hazard-map demo

Outputs land in `out/demo/`: `birdseye.png`, `score_strip.png`, `frame_overlays.png`, `flythrough.gif`, `scene_heat.ply`, `viewer.html` (three.js, needs internet for the CDN scripts), `report.md`, `metrics.json`, `hazards.json`, `segments.json`.

What is real: the quality filter, Harris + ratio-test matching + 8-point RANSAC baseline, multi-view log-odds fusion with k-view hazard verification, RANSAC ground plane, kNN slope/roughness, Zhang-Suen centerline, width / drop-off / step / clearance / sight-distance, percentile-normalized weighted scoring with per-segment explanations, heat recoloring and PLY export.

What is stubbed: Mapillary, VGGT, gsplat, SAM 2, SegFormer and Grounded-SAM-2 are replaced by a procedural 163k-Gaussian trail with injected hazards and simulated noisy per-view predictions. Metrics in `metrics.json` are against that injected ground truth, not real trails. `trail-hazard-map run` exits with an explanation until the GPU stack is wired in.

## 3D trail map

Open `web/trail3d.html` in a browser (needs internet). It shows real terrain with satellite imagery and OpenStreetMap trails draped on it for a few Providence-area parks. Trails and place lookup come live from Overpass and Nominatim; terrain is AWS terrain tiles. This is display only and has nothing to do with the hazard score.
