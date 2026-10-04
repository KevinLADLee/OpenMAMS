# Single-frame CARLA samples

These images were collected in CARLA 0.9.16, Town05, with this repository's
recorders. Each directory contains **one synchronized frame per UAV**, not a
video-derived screenshot, captioned composite, or full flight dataset.

| Sample | Images | Resolution per image | Image bytes | Format |
| --- | ---: | --- | ---: | --- |
| [Panoramic, 3 UAVs](panorama_3uav/) | 3 | 2048 × 1024 | 7,326,276 (7.33 MB) | Lossless PNG of stitched RGB |
| [Perspective RGB, 10 UAVs](rgb_10uav/) | 10 | 1280 × 720 | 1,630,600 (1.63 MB) | Recorder-native JPEG |

Sizes use decimal MB. Together the images are 8.96 MB; metadata brings the
download to approximately 9 MB. Actual filesystem allocation and Git transfer
sizes differ. Full video recordings are intentionally not included.

## Files and provenance

- `panorama_3uav/panorama_frames/uav1/000000.png` through `uav3/000000.png`:
  six co-located 90° camera faces stitched into a 2:1 equirectangular image.
  No captions, rectified FPV panels, resizing, or re-encoding were applied.
- `rgb_10uav/images/uav_01/000000.jpg` through `uav_10/000000.jpg`:
  ordinary perspective cameras, 75° horizontal FOV, −45° pitch.
- Each directory retains the recorder's `capture.json`, `frames.jsonl`, and
  `route.json`. Image paths are relative to that directory. Coordinates are
  CARLA world meters; rotations are degrees.

All UAV records **within each sample** share a world frame and simulation
timestamp. The three-UAV and ten-UAV samples are **separate captures**, not a
cross-mode synchronized pair. Both use equally spaced starting points on the
included Town05 loop and kinematic camera rigs, not UAV flight dynamics.
No additional benchmark objects were spawned for these examples.

The panorama sample is an image-only subset of a successful one-frame capture.
Its original manifests retain `panorama_video` paths and encoding information,
but those MP4s and auxiliary previews are deliberately omitted from Git.
`completed: true` describes the source capture, not inclusion of every artifact.
The PNGs are lossless stitched panoramas; they are not the six original sensor
buffers. Subtle cubemap seams from rendering/interpolation can remain.

## Reproduce

Install from the repository root:

```bash
pip install -e './uav_data_recorder[carla,panorama]'
```

Start a dedicated CARLA 0.9.16 server on port 2010. Run the following commands
**sequentially**, using fresh output directories:

```bash
openmams-record-panorama --port 2010 --map Town05 \
  --route uav_data_recorder/routes/town05_loop.json \
  --uavs 3 --seconds 0.06666666666666667 --fps 15 \
  --speed 4.5 --altitude 12 --face-size 768 --pano-width 2048 \
  --output runs/sample_panorama_3uav

openmams-record --port 2010 --map Town05 \
  --route uav_data_recorder/routes/town05_loop.json \
  --uavs 10 --seconds 0.1 --fps 10 --speed 4.5 --altitude 17.5 \
  --pitch -45 --fov 75 --width 1280 --height 720 --objects 0 \
  --output runs/sample_rgb_10uav
```

The durations request exactly one output frame per UAV. Panorama capture also
writes one-frame MP4 playback copies and previews; only the PNGs and metadata are
distributed here. Absolute simulation timestamps/frame IDs and rendered pixels
may differ across simulator sessions, GPUs, or rendering configurations.

These files illustrate acquisition formats and camera geometry. A single frame
is not evidence of temporal coverage or downstream QA performance.
