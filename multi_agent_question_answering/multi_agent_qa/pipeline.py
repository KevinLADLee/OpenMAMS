"""Capture manifest → original captioner → original Milvus memory."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def versions():
    names = ("langchain-core", "langchain-ollama", "langchain-milvus", "langgraph",
             "langgraph-prebuilt", "langgraph-checkpoint", "langgraph-sdk",
             "pymilvus", "milvus-lite", "sentence-transformers", "torch", "transformers")
    return {name: version(name) for name in names}


def model_digests(config):
    import ollama
    tags = ollama.Client(host=config.get("llm.api_base")).list()
    return {m.model: m.digest for m in tags.models
            if m.model in (config.get("llm.backend"), config.get("captioner.model"))}


def read_frames(data):
    data = Path(data).resolve()
    manifest = data / "frames.jsonl"
    capture = data / "capture.json"
    if not capture.exists() or not json.loads(capture.read_text()).get("completed"):
        raise ValueError("A completed capture.json is required; do not use partial captures")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Frames must be nonempty with unique IDs")
    for row in rows:
        if type(row["uav_id"]) is not int or row["uav_id"] < 1:
            raise ValueError("UAV IDs must be positive integers")
        for field in ("x", "y", "z", "yaw", "time_s"):
            if type(row[field]) not in (int, float) or not math.isfinite(row[field]):
                raise ValueError(f"Invalid frame field: {field}")
        if row["time_s"] < 0:
            raise ValueError("time_s must be nonnegative")
        image = (data / row["image"]).resolve()
        if not image.is_relative_to(data) or not image.is_file():
            raise ValueError(f"Missing image or image outside dataset: {row['image']}")
    return sorted(rows, key=lambda r: (r["uav_id"], r["time_s"], r["id"]))


def build(data, output, interval, config):
    from PIL import Image
    from remembr.captioners.ollama_captioner import OllamaCaptioner
    from remembr.memory.milvus_memory import MilvusMemory
    from remembr.memory.memory import MemoryItem

    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("Caption interval must be finite and >= 0")
    frames = read_frames(data)
    # A fixed UTC day gives the original HMS retrieval tool a valid time base.
    # The input timestamp remains simulation elapsed time, never a Unix timestamp.
    offset = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    output.mkdir(parents=True)
    info = {"completed": False, "collection": "memory", "time_offset": offset,
            "uavs": sorted({r["uav_id"] for r in frames}), "input_frames": len(frames),
            "caption_interval_s": interval, "caption_sampling": "one current frame per selected pose",
            "embedding": config.get("embedding"), "config": config.to_dict(),
            "model_digests": model_digests(config), "versions": versions(),
            "input_sha256": hashlib.sha256((data / "frames.jsonl").read_bytes()).hexdigest(),
            "position_schema": "[camera_x_m, camera_y_m, yaw_radians] (retained research schema)"}
    dump(output / "memory.json", info)
    captioner = OllamaCaptioner(model=config.get("captioner.model"), host=config.get("llm.api_base"),
                               temperature=config.get("captioner.temperature"),
                               num_ctx=config.get("captioner.num_ctx"))
    memory = MilvusMemory("memory", data_dir=str(output), time_offset=offset, config=config)
    count, last = 0, {}
    try:
        with (output / "captions.jsonl").open("w") as stream:
            for row in frames:
                uav = row["uav_id"]
                if row["time_s"] - last.get(uav, -math.inf) < interval - 1e-8:
                    continue
                with Image.open(data / row["image"]) as image:
                    caption = captioner.caption([image.convert("RGB")])
                if not caption.strip():
                    raise RuntimeError(f"Empty caption for {row['id']}")
                yaw = math.radians(row["yaw"])
                item = MemoryItem(caption=f"[UAV {uav}] {caption}", time=offset + row["time_s"],
                                  position=[row["x"], row["y"], yaw], theta=yaw)
                memory.insert(item)
                record = {**row, "caption": item.caption,
                          "image_sha256": hashlib.sha256((data / row["image"]).read_bytes()).hexdigest()}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                last[uav] = row["time_s"]
                count += 1
                print(f"Captioned {row['id']} ({count} memories)", flush=True)
        memory.milv_wrapper.collection.flush()
        info.update(completed=True, memories=count)
        dump(output / "memory.json", info)
    finally:
        memory.cleanup()
    return info
