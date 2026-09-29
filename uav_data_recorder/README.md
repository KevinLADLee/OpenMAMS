# OpenMAMS · UAV data recorder

Collect synchronized RGB images, camera poses, and object ground truth from CARLA.
The recorder uses virtual UAV cameras that follow road waypoints.

## Quick start

Use Linux, Python 3.11, and matching CARLA 0.9.16 client/server versions.
Run these commands from this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[carla]'
```

Start CARLA in a separate terminal:

```bash
/path/to/CARLA/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000
```

Record four UAVs in Town04 for 30 seconds at 10 FPS per UAV:

```bash
openmams-record --map Town04 --uavs 4 --seconds 30 --fps 10 \
  --speed 4.5 --altitude 17.5 --pitch -45 --objects 10 --output runs/town04
```

This produces 1,200 images. Record the ten-UAV Town05 demo at 10 FPS:

```bash
openmams-record --map Town05 --route routes/town05_loop.json \
  --uavs 10 --seconds 15 --fps 10 --objects 10 --output runs/town05_k10_10fps/recording
```

## Routes and objects

Automatic routes follow roads from separated starting points.
`--seed` controls route choices and object placement. Use `--spawns 12 35 80 110`
to select starting points explicitly. Spawn indices depend on the map.

`--route routes/town05_loop.json` uses the included 1,005.44 m Town05 loop. UAVs
start at equal distances along the loop. Use either `--route` or `--spawns`.

The ten-UAV, 15-second recording at 10 FPS produces 1,500 images.

Default speed is 4.5 m/s, altitude is 17.5 m above the road, and camera pitch is −45°.
The recorder uses virtual cameras without flight dynamics.

`--objects 10` places five colored cars, a fire truck, motorcycle, bus, taxi,
and traffic cone along the routes. Their world coordinates are saved for evaluation.

Use a dedicated CARLA instance and a new output directory.

## Output

```text
runs/town04/
  capture.json                 # Recording parameters and completion status
  frames.jsonl                 # Frame IDs, UAV IDs, timestamps, and poses
  images/uav_01/000000.jpg
  images/uav_02/000000.jpg
  ...
  objects.json                 # Object ground truth when --objects 10 is used
```

Each frame contains `id, uav_id, time_s, timestamp, world_frame, x, y, z, yaw, pitch, image`.
Coordinates use CARLA world meters; yaw and pitch use degrees. `time_s` starts at zero
for the recording, and `timestamp` is CARLA elapsed simulation time. Image paths are
relative to the recording directory. `capture.json` marks a finished recording with
`completed: true`.

Pass the output directory to [NTN replay](../ntn/README.md#replay).

[Captioned K=4 demo](../assets/town04_4uavs.gif) ·
[Captioned K=10 demo](../assets/town05_10uavs.webp) · [MIT license](LICENSE)
