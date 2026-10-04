# Single-frame CARLA samples

CARLA 0.9.16, Town05: **one synchronized frame per UAV**, collected with this
repository's recorders.

| Sample | Images | Resolution per image | Image size | Format |
| --- | ---: | --- | ---: | --- |
| [Panoramic, 3 UAVs](panorama_3uav/) | 3 | 2048 × 1024 | 7.33 MB | Lossless stitched PNG |
| [Perspective RGB, 10 UAVs](rgb_10uav/) | 10 | 1280 × 720 | 1.63 MB | JPEG |

Total including metadata: approximately **9 MB** (decimal).

## Contents

- `panorama_3uav/panorama_frames/uav1/000000.png` through `uav3/000000.png`:
  2:1 equirectangular panoramas.
- `rgb_10uav/images/uav_01/000000.jpg` through `uav_10/000000.jpg`:
  perspective images with 75° horizontal FOV and −45° pitch.
- Each directory retains the recorder's `capture.json`, `frames.jsonl`, and
  `route.json`. Image paths are relative to that directory. Coordinates are
  CARLA world meters; rotations are degrees.

UAVs within each sample share a world frame and timestamp; the two samples were
captured separately. Both use the Town05 loop without additional spawned objects.
Panorama manifests retain the original MP4 references, but videos and previews
are omitted from this image-only sample.

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

These durations request exactly one frame per UAV. Panorama capture also writes
MP4 playback copies and previews.
