# OpenMAMS · Satellite backhaul

Simulate satellite backhaul capacity and image delivery.

## Setup

Use Python 3.11. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the included 400-satellite example:

```bash
bash scripts/run_backhaul.sh runs/backhaul_400
```

Results are saved in `uplink/`, `downlink/`, and `effective/`. The `uid=1` rows in
`effective/mac_user_qos.csv` give the backhaul rate for PF and Max-C/I scheduling.
Backhaul capacity is the smaller endpoint mean rate, held constant during replay;
ISL capacity is assumed sufficient.

## Generate a constellation

```bash
bash scripts/setup_dependencies.sh --leopath
openmams-ntn topology --planes 20 --sats-per-plane 20 --output runs/topology_400
bash scripts/run_backhaul.sh runs/new_backhaul runs/topology_400
```

Use `--minute` to select a snapshot time. The generated files include
`constellation.tle`, `hong_kong.json`, `istanbul.json`, and `topology.json`.
To fetch both upstream repositories without installing them:

```bash
bash scripts/setup_dependencies.sh --fetch
```

## Replay a recording

```bash
openmams-ntn replay --data ../uav_data_recorder/runs/town04 \
  --backhaul runs/backhaul_400/effective --scheduler 'Proportional Fair' \
  --deadline-s 30 --output runs/received
```

Payload size defaults to the image file size. Use `--frame-bytes` to set a fixed
size, or `--propagation-ms` to add propagation delay. The deadline is measured
from the start of the recording.

Delivered images and their original poses are saved in `images/` and `frames.jsonl`.
`delivery.json` summarizes delivery; `transmissions.jsonl` contains per-frame results.
Use a new output directory for each run.

## Commands

| Command | Purpose |
| --- | --- |
| `openmams-ntn topology` | Generate satellite geometry |
| `openmams-ntn endpoint` | Simulate an uplink or downlink endpoint |
| `openmams-ntn combine` | Combine endpoint capacities |
| `openmams-ntn replay` | Replay image transmission |

Append `--help` to any command for its options.

## Tests

```bash
pip install -e '.[test]'
python -m pytest -q
```

## Sources

[LEOPath](https://github.com/Fundacio-i2CAT/LEOPath) (AGPL-3.0) ·
[OpenNTN](https://github.com/ant-uni-bremen/OpenNTN) (MIT).
Fetched repositories retain their original license and copyright notices.

Building geometry: © OpenStreetMap contributors,
[ODbL](https://www.openstreetmap.org/copyright). Some heights use default estimates.
