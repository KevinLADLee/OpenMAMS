# Validation

## Sample inference — 2026-10-03

Real local inference used Qwen3-VL-8B, Mxbai-Embed-Large-v1 on CPU, and Qwen3-8B
with Python 3.11. All ten Town04 example images produced captions and persisted
memories; the Milvus database was reopened and verified to contain ten rows.
All 22 questions completed, and a separate UAV 1 query retrieved only UAV 1 memories.

| Question type | Correct / total |
| --- | ---: |
| Presence | 6 / 10 |
| Where, observing camera XY distance ≤ 50 m | 7 / 10 |
| Which-UAV, exact ID set | 1 / 2 |
| Total | 14 / 22 (63.64%) |

The sample checks the QA workflow and interfaces. It uses camera-coordinate
references and does not reproduce the paper experiments. CARLA collection,
NTN delivery, fresh installation, and multi-machine deployment were not rerun.

## Regression and packaging checks

- QA: 18 tests passed, including bounded parsing/retry and UAV-filtering checks.
- NTN: 12 tests passed.
- Following the upstream documentation merge, recorder and NTN wheels built;
  wheel-only imports, CLI help, package metadata, and documentation links passed.

The inference run is `runs/verify_20261003_142031/`. Its report, commands, model
and source hashes, answer traces, and logs remain local; generated runs are ignored
by Git. Source maintenance records are in
[provenance.json](multi_agent_question_answering/provenance.json).
[Earlier validation records](https://github.com/KevinLADLee/OpenMAMS/blob/2473913228f3805739b069ab61767e23faf345c2/VALIDATION.md)
remain in Git history.
