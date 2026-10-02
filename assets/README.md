# Assets

## Panoramic multi-agent system header

| Asset | Duration | Resolution | FPS | Purpose |
| --- | --- | --- | ---: | --- |
| [pmas-semantic-map.webp](pmas-semantic-map.webp) | 24 s | 960 × 540 | 10 | Automatically looping README header preview |
| [pmas-semantic-map.mp4](pmas-semantic-map.mp4) | 24 s | 1920 × 1080 | 24 | H.264 video with fast-start playback |

The camera rotates around the measured COLMAP point cloud, recorded three-UAV
trajectories, and semantic observation anchors. This is not a synchronized flight
replay. Labels mark supporting UAV observation poses, not reconstructed object
centers; the COLMAP reconstruction has arbitrary scale, not calibrated meters.

The display uses UAV1 from Pose 50 onward, excludes the isolated UAV2 Pose 105,
and retains UAV3 unchanged (56, 175, and 123 trajectory points). The 58 semantic
labels share exactly the same display mask. These are visualization-only filters;
raw observations and the QA benchmark evidence are unchanged. The map is enlarged
1.4×, and the only footer reads `Panoramic multi-agent system` (32 px at 1080p).

Both files derive from the same final 24-second render; the WebP reduces frame
rate and resolution for the README without changing playback duration. The MP4
retains the original resolution and frame rate with web-oriented compression.

## Captioned demos

| Animation | Duration | Resolution | FPS | Size |
| --- | --- | --- | ---: | ---: |
| [Town04, K=4](town04_4uavs.gif) | 15 s | 960 × 540 | 10 | 25.02 MB |
| [Town05, K=10](town05_10uavs.webp) | 15 s | 1600 × 360 | 10 | 8.28 MB |

Both captioned animations loop automatically and are 15 seconds long at 10 FPS.
Town04 uses a 2 × 2 GIF layout; Town05 uses a 5 × 2 animated WebP layout.

## MemNTN paper figures

Figures from *Memory-Native Non-Terrestrial Networks for Embodied Intelligence*,
IEEE Communications Standards Magazine, 2026.

| Asset | Figure |
| --- | --- |
| [memntn_fig1.png](memntn_fig1.png) | Satellite-enabled perception and remote question answering |
| [memntn_fig2.png](memntn_fig2.png) | System framework |
| [memntn_fig3.png](memntn_fig3.png) | Physical and digital memory |
| [memntn_fig4.png](memntn_fig4.png) | Simulation setup and evaluation |

Figure 1 geographic data sources: OpenStreetMap, LEOPath, and Mega-NeRF/Mill-19.
