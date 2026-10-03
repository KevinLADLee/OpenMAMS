# Validation

## QA and data collection — 2026-09-29

The recorder was tested in a separate environment with CARLA 0.9.16, NumPy, and Pillow.
QA was installed as a wheel in a fresh Python 3.11 environment and exercised with
both an existing memory database and newly recorded images.

The model pipeline used Qwen3-VL-8B, HuggingFace Mxbai-Embed-Large-v1, and Qwen3-8B.
For the ten-image Town04 sample with UAVs `[1,2,3,4]`:

| Question type | Correct / total |
| --- | ---: |
| Presence | 7 / 10 |
| Where, XY distance ≤ 50 m | 7 / 10 |
| Which-UAV, exact ID set | 1 / 2 |
| Total | 15 / 22 (68.18%) |

Two additional images from a recorded scene completed captioning, memory creation,
and QA in the new environment.

Thirteen regression tests covered agent graph construction, retrieval filtering,
scoring, parsing, retries, incomplete inputs, and source integrity. Compatible
LangGraph components are pinned in `pyproject.toml`.

## Existing database comparison

The original research evaluator and the new QA entry point used the same 150-memory
database and UAVs `[1,5]`. Database SHA-256:

```text
4d2faa0cae4b581a227c5c05e2df1a94d16ea87c1da3d21bdd1a1de0b53e03f0
```

Both answered no to the presence query, returned UAV 5 for the yellow-car query,
and returned `[-62.786, 202.576, 139.459]` for its position. The third spatial
component is yaw. Two of the three compared answers and final retrieval records
matched exactly; the Where wording, orientation, and retrieval record differed.

## Evaluation settings

| Setting | Current package | Historical experiment |
| --- | --- | --- |
| Caption input | One selected frame | Up to five frames per window |
| Where threshold | 50 m | 150 m in the available replay |
| Questions | 22 in the sample | 20 in the available replay; 30 in the paper |

The sample results validate execution and interfaces. Paper-level accuracy requires
matching recordings, model versions, sampling, question sets, and scoring settings.
Full logs are stored in the source project's `output/openmams_remembr_validation/`.

## Package naming — 2026-09-30

The packages are `uav_data_recorder/recorder`,
`multi_agent_question_answering/question_answering`, and `ntn/satellite_backhaul`.
QA uses `question_answering/config.json`. The three `openmams-*` CLI names are retained.

After renaming, 13 QA tests and 11 NTN tests passed. Installed CLI entry points and
configuration loading worked outside the source directory. QA accepted all ten
frames from the NTN demo replay. All 81 original source and data files were retained.

## Documentation and assets — 2026-09-30

The four-UAV captioned Town04 demo was copied from `OpenMAMS_public/assets`:
30 seconds, 1280×720, 10 FPS, 3,662,592 bytes. The copy matches the source SHA-256.
README files and code comments use English. Comment edits in `milvus_memory.py`
are recorded alongside the original source hash in QA provenance.

## Town05 K=10 captioned video — 2026-09-30

CARLA 0.9.16 recorded ten virtual UAV cameras along the included Town05 loop,
with starting positions separated by 100.54 m. A 0.1-second simulation step and
every-fifth-frame sampling produced 600 images: 60 per UAV at 2 FPS. All ten camera
timestamps match at each sample, with 0.5-second spacing between samples.

Qwen3-VL-8B generated 100 captions at three-second intervals. The rendered video
contains 60 frames at 2 FPS, 3200 × 720 resolution, and 30-second duration. Full
decoding succeeded; all 600 input images are distinct and nonblank. Preview frames
were inspected for tile placement and caption clipping. Route checks covered equal
spacing, speed, heading, altitude removal, loop wrapping, and map matching.

Recording: `uav_data_recorder/runs/town05_k10_2fps/recording/`.
Caption responses, previews, model and input hashes, and validation report:
`multi_agent_question_answering/runs/town05_k10_2fps_video/`.

## Town05 K=10 at 10 FPS — 2026-09-30

CARLA 0.9.16 on an RTX 5090 completed a three-second smoke test followed by a
15-second recording with ten 1280 × 720 cameras at 10 FPS. Neither run crashed.
The full run saved 1,500 distinct, nonblank images with synchronized camera
timestamps and 0.1-second intervals; capture took 12.20 seconds wall time.
Sampled GPU memory peaked at 8,435 MiB in the smoke test and 6,186 MiB in the full
run. The captioned MP4 and looping WebP each contain 150 frames over 15 seconds.
Capture logs: `uav_data_recorder/runs/town05_k10_10fps/`. Caption responses and
validation: `multi_agent_question_answering/runs/town05_k10_10fps_video/`.

## Merged-checkout inference — 2026-10-03

Real local inference with Qwen3-VL-8B, Mxbai-Embed-Large-v1, and Qwen3-8B completed
captioning and persistence of all ten sample memories, all twenty-two questions,
and an additional retrieval query restricted to UAV 1. The persisted Milvus database
was reopened and verified to contain ten rows.

| Question type | Correct / total |
| --- | ---: |
| Presence | 6 / 10 |
| Where, observing camera XY distance ≤ 50 m | 7 / 10 |
| Which-UAV, exact ID set | 1 / 2 |
| Total | 14 / 22 (63.64%) |

Actual inference exposed duplicate insertion into shared tool definitions and
malformed model tool responses. The wrapper now copies the definitions, validates
response fields, permits one schema-correction attempt with model feedback, and
avoids entering an interactive debugger. Five regression cases fail before their
respective fixes and pass afterward; the complete QA suite has eighteen passing tests.
Maintenance edits and source hashes are recorded in QA provenance.

The final evaluation reuses the second run's completed ten-memory build. The final
run is `runs/verify_20261003_142031/`; its `REPORT.md`, `verification.json`,
`checks.json`, logs, captions, database, and answer traces document the execution.
The interrupted runs `verify_20261003_141005` and `verify_20261003_141443` remain
in `runs/` with incomplete reports and no accuracy metric. Runtime artifacts are
local and excluded from Git.

The example's Where references are observing camera positions. This validates the
sample QA workflow and its interfaces; CARLA collection, NTN delivery, fresh
installation, multi-machine deployment, and paper experiments were not rerun.
