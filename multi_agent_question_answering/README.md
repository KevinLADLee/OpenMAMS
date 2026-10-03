# OpenMAMS · Multi-agent question answering

Generate image captions, store them in Milvus, and answer questions using ReMEmbR's
text, position, and time retrieval tools.

`multi_agent_qa/` provides data ingestion, UAV filtering, CLI commands, and evaluation.
`remembr/` contains the research agent, captioner, memory, tools, and prompts.

## Setup

Use Linux and Python 3.11. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Start `ollama serve` in a separate terminal, then download the models:

```bash
ollama pull qwen3-vl:8b-instruct
ollama pull qwen3:8b
```

The embedding model, `mixedbread-ai/mxbai-embed-large-v1`, is downloaded through
HuggingFace on first use and runs on CPU by default. Model settings are in
`multi_agent_qa/config.json`. To use a custom file:

```bash
openmams-qa --config my_config.json build --data examples/town04 --output runs/custom
```

## Caption, retrieve, and answer

Run the included ten-image, four-UAV example:

```bash
openmams-qa build --data examples/town04 --output runs/demo --interval 0
openmams-qa ask --db runs/demo/memory.db --uavs 1 2 3 4 \
  --question 'Where is the green car?' --output runs/where.json
openmams-qa evaluate --db runs/demo/memory.db --uavs 1 2 3 4 \
  --questions examples/town04/questions.json --output runs/evaluation.json
```

Use recorder output with one caption every three seconds per UAV:

```bash
openmams-qa build --data ../uav_data_recorder/runs/town04 --output runs/town04 --interval 3
openmams-qa ask --db runs/town04/memory.db --uavs 1 3 \
  --question 'Which drones saw a green car?' --output runs/uavs_1_3.json
```

Input consists of a completed `capture.json`, `frames.jsonl`, and relative image paths.
Each selected frame produces one caption with a `[UAV N]` prefix. UAV filtering happens
before retrieval: text search uses top-5, and position/time search uses top-4.
The agent can perform multiple retrieval steps before answering.

Build outputs include `memory.db`, `captions.jsonl`, and `memory.json` with configuration,
model information, and input hashes. Use a new output directory for each run.

The stored spatial vector is `[camera_x, camera_y, yaw_radians]`. QA uses capture times
mapped to a fixed UTC date; Where evaluation uses only XY coordinates.

## Captioned video

Render the Town05 recording as a 5 × 2 grid with one caption every three seconds
per UAV. The video uses the recording's frame rate, including 2 FPS:

```bash
pip install -e '.[video]'
openmams-caption-video --data ../uav_data_recorder/runs/town05_k10_2fps/recording \
  --output runs/town05_k10_captioned_2fps.mp4 \
  --work runs/town05_k10_2fps_video --interval 3 --columns 5 --tile-width 640
```

This command uses `qwen3-vl:8b-instruct` through Ollama and the system DejaVu Sans
Bold font. Use `--model`, `--ollama`, or `--font` to override them. Each tile is
640 × 360 for a 1280 × 720 recording. Captions, model responses, hashes, and preview
frames are saved under `--work`; interrupted caption generation resumes from these
checkpoints. Use a new video output path for each render.

## Evaluation

Question files are JSON lists:

```json
[
  {"id":"presence-green","question":"Is there a green car?","type":"exact","answer":"YES"},
  {"id":"where-green","question":"Where is the green car?","type":"where","answer":[300.49,-354.65]},
  {"id":"which-green","question":"Which drones saw a green car?","type":"uav_set","answer":[1]}
]
```

| Type | Scoring rule |
| --- | --- |
| `exact` | Structured yes/no match |
| `where` | XY distance to ground truth ≤ 50 m |
| `uav_set` | Exact UAV-ID set match |

Annotate presence and which-UAV answers from the observations available to the selected
UAVs. Where references must match the coordinate target requested in the question.
The included example asks for the observing camera's world XY coordinates and
contains 22 questions: 10 presence, 10 Where, and 2 which-UAV.

Reports contain overall and per-type accuracy, answers, retrieval traces, coordinate
errors, and model metadata. Each question has a 90-second default timeout, adjustable
with `--timeout`.

## Existing databases

Use the same embedding configuration and database filename as the original collection:

```bash
mkdir -p runs/legacy
cp /path/to/record_2026_0126_1600.db runs/legacy/
openmams-qa ask --db runs/legacy/record_2026_0126_1600.db --uavs 1 5 \
  --question 'Where is the yellow car?' --output runs/legacy_where.json
```

Databases without `memory.json` use the legacy time offset `1721761000`;
set `--time-offset` to override it.

## Tests and sources

```bash
pip install -e '.[test]'
pytest -q
```

Source: [remembr-invs](https://github.com/KevinLADLee/remembr-invs) and the retained local
research snapshot. Original source hashes, current file hashes, and maintenance edits
are recorded in [provenance.json](provenance.json).
See [validation results](../VALIDATION.md) and the [NVIDIA license](LICENSE.md).
