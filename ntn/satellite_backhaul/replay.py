"""FIFO store-and-forward image replay over a constant snapshot backhaul."""
import json
import math
from pathlib import Path
import shutil

from .backhaul import hub_row
from .io import dump, new_output, source


def replay(args):
    data = Path(args.data).resolve()
    capture = json.loads((data / "capture.json").read_text())
    if capture.get("completed") is not True:
        raise ValueError("A completed capture.json is required")
    if not math.isfinite(args.deadline_s) or args.deadline_s < 0:
        raise ValueError("Deadline must be finite and nonnegative")
    if not math.isfinite(args.propagation_ms) or args.propagation_ms < 0:
        raise ValueError("Propagation delay must be finite and nonnegative")
    if args.frame_bytes is not None and args.frame_bytes <= 0:
        raise ValueError("Frame payload size must be positive")
    rate = float(hub_row(args.backhaul, args.scheduler)["throughput_mbps"])
    rows = [json.loads(line) for line in (data / "frames.jsonl").read_text().splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Input frames must be nonempty with unique IDs")
    images = {}
    for row in rows:
        if type(row["uav_id"]) is not int or row["uav_id"] < 1:
            raise ValueError("UAV IDs must be positive integers")
        for field in ("time_s", "x", "y", "z", "yaw"):
            if type(row[field]) not in (int, float) or not math.isfinite(row[field]):
                raise ValueError(f"Invalid frame field: {field}")
        if row["time_s"] < 0:
            raise ValueError("Frame time_s must be nonnegative")
        relative = Path(row["image"])
        image = (data / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not image.is_relative_to(data) or not image.is_file():
            raise ValueError(f"Image must be a relative file inside capture: {relative}")
        if image.stat().st_size <= 0:
            raise ValueError(f"Empty image: {relative}")
        images[row["id"]] = image
    rows.sort(key=lambda r: (r["time_s"], r["uav_id"], r["id"]))
    out = new_output(args.output)
    metadata = {**capture, "completed": False, "kind": "ntn_delivered_subset",
                "source_capture": source(data / "capture.json"), "source_frames": source(data / "frames.jsonl")}
    dump(out / "capture.json", metadata)
    info = {"completed": False, "scheduler": args.scheduler, "capacity_mbps": rate,
            "deadline_s": args.deadline_s, "propagation_ms": args.propagation_ms,
            "payload_bytes_per_frame": args.frame_bytes,
            "queue": "FIFO by (time_s, uav_id, id); frames arrive at hub at capture time_s",
            "assumptions": "constant snapshot service rate; ideal UAV access; no packet loss, headers, retries, buffer limit or memory-value selection",
            "sources": {"qos": source(Path(args.backhaul) / "mac_user_qos.csv"),
                        "summary": source(Path(args.backhaul) / "mac_ofdma_summary.json"),
                        "frames": source(data / "frames.jsonl")}}
    dump(out / "delivery.json", info)
    available_at, delivered_count, delivered_bytes = 0.0, 0, 0
    with (out / "frames.jsonl").open("w") as manifest, (out / "transmissions.jsonl").open("w") as log:
        for row in rows:
            image = images[row["id"]]
            size = args.frame_bytes if args.frame_bytes is not None else image.stat().st_size
            start = max(available_at, row["time_s"])
            finish = start + size * 8 / (rate * 1e6) if rate > 0 else None
            arrival = finish + args.propagation_ms / 1000 if finish is not None else None
            delivered = arrival is not None and arrival <= args.deadline_s
            if finish is not None:
                available_at = finish
            record = {"id": row["id"], "uav_id": row["uav_id"], "payload_bytes": size,
                      "start_s": start if rate > 0 else None, "finish_s": finish,
                      "arrival_s": arrival, "delivered": delivered,
                      "image_sha256": source(image)["sha256"]}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            if delivered:
                target = out / "images" / f"{delivered_count:08d}{image.suffix}"
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(image, target)
                manifest.write(json.dumps({**row, "image": str(target.relative_to(out)),
                                           "ntn_arrival_s": arrival}, allow_nan=False) + "\n")
                delivered_count += 1
                delivered_bytes += size
    # Ground truth is evaluation metadata, not a payload delivered over the simulated link.
    if (data / "objects.json").is_file():
        shutil.copyfile(data / "objects.json", out / "objects.json")
    metadata.update(completed=True, input_frames=len(rows), delivered_frames=delivered_count,
                    qa_ready=delivered_count > 0,
                    evaluation_metadata_note="objects.json, if present, is copied for offline scoring only")
    info.update(completed=True, input_frames=len(rows), delivered_frames=delivered_count,
                undelivered_frames=len(rows) - delivered_count, delivered_payload_bytes=delivered_bytes,
                qa_ready=delivered_count > 0)
    dump(out / "capture.json", metadata)
    dump(out / "delivery.json", info)
    return info
