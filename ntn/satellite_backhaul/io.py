"""JSON output and input fingerprints."""
import hashlib
import json
from pathlib import Path


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source(path):
    path = Path(path)
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
