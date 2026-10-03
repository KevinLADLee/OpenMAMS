"""Ground truth stays outside the agent; Where uses XY distance <= 50 m."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def numeric_xy(value):
    return (isinstance(value, (list, tuple)) and len(value) >= 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value[:2]))


def load_questions(path):
    items = json.loads(Path(path).read_text())
    if not isinstance(items, list) or not items:
        raise ValueError("Questions must be a nonempty JSON list")
    seen = set()
    for q in items:
        if not isinstance(q.get("id"), str) or q["id"] in seen or not q.get("question", "").strip():
            raise ValueError("Questions need distinct string IDs and nonempty question text")
        seen.add(q["id"])
        if q["type"] == "where":
            if not numeric_xy(q["answer"]):
                raise ValueError("Where answer must contain object ground-truth [x, y]")
        elif q["type"] == "exact":
            if q["answer"] not in ("YES", "NO"):
                raise ValueError("Presence answer must be YES or NO")
        elif q["type"] == "uav_set":
            if not isinstance(q["answer"], list) or not q["answer"] or any(
                type(i) is not int or i < 1 for i in q["answer"]):
                raise ValueError("UAV answers must contain positive IDs from actual observations")
        else:
            raise ValueError(f"Unknown question type {q['type']}")
    return items


def score(question, answer):
    kind, truth = question["type"], question["answer"]
    if kind == "where":
        position = answer.get("position")
        distance = math.hypot(position[0] - truth[0], position[1] - truth[1]) if numeric_xy(position) else None
        return {"correct": distance is not None and distance <= 50.0,
                "predicted_xy": list(position[:2]) if distance is not None else None,
                "distance_m": distance, "threshold_m": 50.0}
    if kind == "exact":
        binary = str(answer.get("binary") or "").strip().upper()
        if binary not in ("YES", "NO"):
            match = re.match(r"^\s*(YES|NO)\b", str(answer.get("text", "")), re.I)
            binary = match.group(1).upper() if match else "UNKNOWN"
        return {"correct": binary == truth, "predicted": binary}
    text = str(answer.get("text", ""))
    # Extract only explicit UAV/drone lists, not coordinates or arbitrary numerals.
    groups = re.findall(r"\b(?:UAVs?|drones?)\s*(?:IDs?\s*)?[:#]?\s*(\d+(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)\d+)*)", text, re.I)
    predicted = sorted({int(n) for group in groups for n in re.findall(r"\d+", group)})
    return {"correct": set(predicted) == set(truth), "predicted": predicted}


def evaluate(qa, path, output, timeout=90):
    from .pipeline import dump, model_digests, versions
    questions, output = load_questions(path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"completed": False, "allowed_uavs": qa.uavs, "location_threshold_m": 50.0,
              "uav_scoring": "exact set of explicitly named observed UAVs",
              "question_counts": dict(Counter(q["type"] for q in questions)),
              "db_sha256_before": hashlib.sha256(qa.db.read_bytes()).hexdigest(),
              "questions_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "config": qa.config.to_dict(), "model_digests": model_digests(qa.config),
              "versions": versions(), "results": []}
    dump(output, report)
    for question in questions:
        try:
            # Deliberately pass only the question string, never truth or object coordinates.
            prediction = qa.ask(question["question"], timeout)
        except Exception as exc:
            report["error"] = {"question_id": question["id"], "message": str(exc)}
            dump(output, report)
            raise
        report["results"].append({"id": question["id"], "type": question["type"],
                                  "ground_truth": question["answer"], **prediction,
                                  **score(question, prediction["answer"])})
        dump(output, report)
        print(f"Answered {len(report['results'])}/{len(questions)}", flush=True)
    by_type = {}
    for kind in report["question_counts"]:
        rows = [r for r in report["results"] if r["type"] == kind]
        correct = sum(r["correct"] for r in rows)
        by_type[kind] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}
    correct = sum(r["correct"] for r in report["results"])
    report.update(completed=True, correct=correct, total=len(questions),
                  accuracy=correct / len(questions), by_type=by_type)
    dump(output, report)
    return {key: report[key] for key in ("completed", "correct", "total", "accuracy", "by_type")}
