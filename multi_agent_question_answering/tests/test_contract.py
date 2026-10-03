import hashlib
import json
from pathlib import Path

import pytest

from multi_agent_qa.evaluation import score, evaluate
from multi_agent_qa.runtime import safe_literal, bounded_retry


def test_original_graph_compiles_without_models():
    # pip's dependency metadata alone misses incompatible prebuilt/runtime pairs.
    from remembr.agents.remembr_agent import ReMEmbRAgent
    agent = object.__new__(ReMEmbRAgent)
    agent.tool_list = []
    agent.build_graph()
    assert agent.graph is not None


def test_partial_capture_is_rejected(tmp_path):
    from multi_agent_qa.pipeline import read_frames
    (tmp_path / "frames.jsonl").write_text("")
    with pytest.raises(ValueError, match="completed capture.json"):
        read_frames(tmp_path)


@pytest.mark.parametrize("position,correct", [([30,40,9], True), ([30,40.01,9], False),
                                             (None, False), ([float('nan'),0,0], False)])
def test_where_50m_boundary(position, correct):
    assert score({"type":"where", "answer":[0,0]}, {"position":position})["correct"] is correct


def test_binary_does_not_use_yes_substring():
    answer = {"binary":"no", "text":"No. Yesterday was sunny."}
    assert score({"type":"exact", "answer":"NO"}, answer)["correct"]


def test_which_ignores_coordinates():
    q = {"type":"uav_set", "answer":[1,3]}
    assert score(q, {"text":"UAVs 1 and 3 saw it at (42, 7)."})["correct"]
    assert not score(q, {"text":"UAV 1 saw it at (3, 7)."})["correct"]


def test_safe_model_literals(tmp_path):
    marker = tmp_path / "executed"
    with pytest.raises(ValueError):
        safe_literal(f"__import__('pathlib').Path({str(marker)!r}).touch()")
    assert not marker.exists()
    assert safe_literal('{"text":"unknown", "position":"[x,y,z]"}')["position"] is None
    assert safe_literal('{"text":"here", "position":[1,2,3]}')["position"] == [1,2,3]


def test_retries_are_bounded():
    attempts = []
    def fail(state):
        attempts.append(1)
        raise RuntimeError("failed")
    with pytest.raises(RuntimeError):
        bounded_retry({}, fail)
    assert len(attempts) == 3


def test_retained_sources_match_provenance():
    root = Path(__file__).resolve().parents[1]
    for path, info in json.loads((root / "provenance.json").read_text())["files"].items():
        if "sha256" in info:
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == info["sha256"]


def test_failed_run_has_no_accuracy_and_truth_not_sent(tmp_path, monkeypatch):
    from multi_agent_qa import pipeline
    monkeypatch.setattr(pipeline, "model_digests", lambda c: {})
    monkeypatch.setattr(pipeline, "versions", lambda: {})
    class Config:
        def to_dict(self): return {}
    class FakeQA:
        config, uavs, db = Config(), [1], tmp_path / "memory.db"
        def ask(self, question, timeout):
            assert question == "Where is the car?"
            raise TimeoutError("test timeout")
    FakeQA.db.write_bytes(b"test")
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id":"q1", "type":"where", "question":"Where is the car?", "answer":[987,654]}]))
    output = tmp_path / "report.json"
    with pytest.raises(TimeoutError):
        evaluate(FakeQA(), questions, output)
    report = json.loads(output.read_text())
    assert report["completed"] is False and "accuracy" not in report
