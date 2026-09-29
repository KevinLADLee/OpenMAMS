"""Validate and combine the two endpoint scheduler artifacts."""
import csv
import json
import math
from pathlib import Path

from .io import dump, new_output, source

SCHEDULERS = ("Max-C/I", "Proportional Fair")


def hub_row(directory, scheduler):
    directory = Path(directory)
    info = json.loads((directory / "mac_ofdma_summary.json").read_text())
    if info.get("completed") is False:
        raise ValueError(f"Incomplete simulation: {directory}")
    with (directory / "mac_user_qos.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream)
                if int(r["uid"]) == 1 and r["scheduler"] == scheduler]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one uid=1/{scheduler} row in {directory}")
    row = rows[0]
    rate = float(row["throughput_mbps"])
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("Backhaul rate must be finite and nonnegative")
    return row


def combine(uplink, downlink, output):
    sources = {f"{name}_{kind}": source(Path(directory) / filename)
               for name, directory in (("uplink", uplink), ("downlink", downlink))
               for kind, filename in (("qos", "mac_user_qos.csv"), ("summary", "mac_ofdma_summary.json"))}
    for directory, direction in ((uplink, "uplink"), (downlink, "downlink")):
        info = json.loads((Path(directory) / "mac_ofdma_summary.json").read_text())
        if info.get("scenario", {}).get("link_direction") != direction:
            raise ValueError(f"Expected {direction} endpoint: {directory}")
    rows, summaries = [], {}
    for scheduler in SCHEDULERS:
        ul, dl = hub_row(uplink, scheduler), hub_row(downlink, scheduler)
        u, d = float(ul["throughput_mbps"]), float(dl["throughput_mbps"])
        selected = dict(ul if u <= d else dl)
        selected.update(throughput_mbps=min(u, d), role="backhaul_bottleneck")
        rows.append(selected)
        summaries[scheduler] = {"uplink_mbps": u, "downlink_mbps": d,
                                "effective_backhaul_mbps": min(u, d),
                                "bottleneck_direction": "uplink" if u <= d else "downlink"}
    out = new_output(output)
    info = {"completed": False, "model": "two-endpoint snapshot bottleneck",
            "isl_assumption": "sufficient capacity; no ISL routing, queueing or propagation simulated",
            "capacity_semantics": "minimum of endpoint mean rates, held constant during replay",
            "scheduler_summaries": summaries, "sources": sources}
    dump(out / "mac_ofdma_summary.json", info)
    with (out / "mac_user_qos.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    info["completed"] = True
    dump(out / "mac_ofdma_summary.json", info)
    return info
