# OpenMAMS development

The modules install independently and share the following workflow:

**Recording → optional NTN delivery → captioning and memory → question answering.**

- `uav_data_recorder/`: synchronized CARLA images, camera poses, and object ground truth.
- `ntn/`: satellite geometry, scheduling, and image-delivery replay.
- `multi_agent_question_answering/`: captioning, memory construction, retrieval, and QA evaluation.

## Environment

Use Linux and Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ./uav_data_recorder -e './ntn[test]' -e './multi_agent_question_answering[test,video]'
```

CARLA, Ollama models, and optional satellite dependencies require additional setup.
Follow the [recorder guide](uav_data_recorder/README.md),
[NTN guide](ntn/README.md), and [QA guide](multi_agent_question_answering/README.md).

## Included QA example

After starting Ollama and downloading the models described in the QA guide:

```bash
openmams-qa build --data multi_agent_question_answering/examples/town04 --output runs/qa_demo --interval 0
openmams-qa ask --db runs/qa_demo/memory.db --uavs 1 2 3 4 --question 'Where is the green car?' --output runs/where.json
openmams-qa evaluate --db runs/qa_demo/memory.db --uavs 1 2 3 4 --questions multi_agent_question_answering/examples/town04/questions.json --output runs/evaluation.json
```

Use a new output directory for each run. The example can exercise memory
construction and QA without running CARLA or the optional NTN simulation.
For new recordings, pass the completed capture directory to `openmams-qa build`.
For NTN experiments, replay a completed capture first, then pass the replay output
directory to the same command. The module guides document the capture and replay
commands and configuration.

## Verification

Run the test suites separately from their module directories:

```bash
cd ntn
python -m pytest -q
```

```bash
cd multi_agent_question_answering
python -m pytest -q
```

The inner QA packages are `multi_agent_qa` and `remembr`. The CLI commands are
`openmams-qa` and `openmams-caption-video`.

[VALIDATION.md](VALIDATION.md) records historical experiments, package layouts,
and current sample inference checks. Each entry states its validation scope;
the sample checks do not reproduce the paper results. QA source
hashes and maintenance edits are recorded in
[provenance.json](multi_agent_question_answering/provenance.json).

The currently available modules cover collection, optional delivery replay,
memory construction, retrieval, and evaluation. The paper's GAE, MemCen, PSO,
and L2M methods are not fully implemented in this release.
