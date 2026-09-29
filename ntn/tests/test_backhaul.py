from argparse import Namespace
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from satellite_backhaul import _mac
from satellite_backhaul.backhaul import combine, hub_row
from satellite_backhaul.endpoint import simulate
from satellite_backhaul.replay import replay

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "400sats"


def endpoint(tmp_path, direction, topology=None):
    return Namespace(topology=topology or EXAMPLES / ("hong_kong.json" if direction == "uplink" else "istanbul.json"),
                     link_direction=direction, buildings=EXAMPLES / "buildings.csv", half_width_m=1050,
                     random_seed=20260608, open_site_user1=direction == "downlink",
                     uplink_gs_eirp_dbm=58., uplink_sat_g_over_t_db_k=13., downlink_ground_rx_gain_dbi=40.,
                     output=tmp_path / direction)


def test_endpoint_rate_regression(tmp_path):
    # Frozen endpoint-rate regression references.
    expected = {"uplink": {"Max-C/I": 1.111346553936958, "Proportional Fair": 35.06533310144967},
                "downlink": {"Max-C/I": 114.52693229391348, "Proportional Fair": 49.57309287963167}}
    for direction in expected:
        args = endpoint(tmp_path, direction)
        simulate(args)
        assert str(tmp_path) not in (args.output / "mac_ofdma_summary.json").read_text()
        for name, rate in expected[direction].items():
            row = hub_row(args.output, name)
            assert float(row["throughput_mbps"]) == pytest.approx(rate, rel=1e-10)
        for tag, name in (("pf", "Proportional Fair"), ("maxci", "Max-C/I")):
            alloc = np.load(args.output / f"allocation_{tag}.npy")
            assert alloc.shape == (72, 48)
            assert np.all((alloc >= 0) & (alloc < 10))
            rates = np.load(args.output / f"hub_slot_rate_{tag}_mbps.npy")
            assert rates.mean() == pytest.approx(expected[direction][name])
    result = combine(tmp_path / "uplink", tmp_path / "downlink", tmp_path / "effective")
    assert result["scheduler_summaries"]["Proportional Fair"]["effective_backhaul_mbps"] == pytest.approx(35.06533310144967)
    with pytest.raises(FileExistsError):
        combine(tmp_path / "uplink", tmp_path / "downlink", tmp_path / "effective")


def test_no_visible_satellites_is_outage(tmp_path):
    topology = tmp_path / "empty.json"
    topology.write_text('{"visible_satellites": []}')
    args = endpoint(tmp_path, "uplink", topology)
    simulate(args)
    assert float(hub_row(args.output, "Proportional Fair")["throughput_mbps"]) == 0
    assert np.all(np.load(args.output / "allocation_pf.npy") == -1)


def make_qos(directory, rate, direction="uplink"):
    directory.mkdir()
    (directory / "mac_ofdma_summary.json").write_text(json.dumps({"completed": True, "scenario": {"link_direction": direction}}))
    with (directory / "mac_user_qos.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uid", "scheduler", "throughput_mbps", "role"])
        writer.writeheader()
        for name in ("Max-C/I", "Proportional Fair"):
            writer.writerow(dict(uid=1, scheduler=name, throughput_mbps=rate, role="hub"))


@pytest.mark.parametrize("ul,dl,expected", [(5, 3, 3), (0, 3, 0), (5, 0, 0)])
def test_bottleneck_both_directions(tmp_path, ul, dl, expected):
    make_qos(tmp_path / "ul", ul)
    make_qos(tmp_path / "dl", dl, "downlink")
    result = combine(tmp_path / "ul", tmp_path / "dl", tmp_path / "out")
    assert result["scheduler_summaries"]["Proportional Fair"]["effective_backhaul_mbps"] == expected


def replay_args(tmp_path, rate=8):
    data = tmp_path / "capture"
    data.mkdir()
    (data / "capture.json").write_text('{"completed": true}')
    (data / "image.jpg").write_bytes(b"payload")
    frames = [dict(id=str(i), uav_id=i+1, time_s=t, image="image.jpg", x=0, y=0, z=0, yaw=0)
              for i, t in enumerate([0, 0, 4])]
    (data / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in frames))
    make_qos(tmp_path / "qos", rate)
    return Namespace(data=data, backhaul=tmp_path / "qos", scheduler="Proportional Fair",
                     deadline_s=2.5, propagation_ms=500., frame_bytes=1000000, output=tmp_path / "received")


def test_fifo_deadline_and_propagation(tmp_path):
    args = replay_args(tmp_path)
    result = replay(args)
    assert str(tmp_path) not in (args.output / "capture.json").read_text()
    assert str(tmp_path) not in (args.output / "delivery.json").read_text()
    assert result["delivered_frames"] == 2  # 8 Mbit/s: completion 1.5, 2.5, 5.5 seconds
    logs = [json.loads(l) for l in (args.output / "transmissions.jsonl").read_text().splitlines()]
    assert [r["arrival_s"] for r in logs] == [1.5, 2.5, 5.5]
    frames = [json.loads(l) for l in (args.output / "frames.jsonl").read_text().splitlines()]
    assert all((args.output / r["image"]).is_file() for r in frames)
    assert [r["time_s"] for r in frames] == [0, 0]  # Preserve capture timestamps for QA.


def test_zero_rate_and_invalid_capture(tmp_path):
    args = replay_args(tmp_path, rate=0)
    assert replay(args)["qa_ready"] is False
    assert (args.output / "frames.jsonl").read_text() == ""
    args.output = tmp_path / "partial"
    (args.data / "capture.json").write_text('{"completed": false}')
    with pytest.raises(ValueError, match="completed"):
        replay(args)
    assert not args.output.exists()


@pytest.mark.parametrize("rate", [float("nan"), -1, float("inf")])
def test_invalid_capacity(tmp_path, rate):
    make_qos(tmp_path / "qos", rate)
    with pytest.raises(ValueError):
        hub_row(tmp_path / "qos", "Proportional Fair")


def test_pf_fairness_equal_channels():
    se = np.ones((10, 72, 48))
    allocation = _mac.run_pf(se)
    counts = np.bincount(allocation.ravel(), minlength=10)
    assert np.all(counts > 0)
    assert counts.max() - counts.min() <= 48


def test_source_fingerprint_omits_parent_directories(tmp_path):
    import hashlib
    from satellite_backhaul.io import source

    private = tmp_path / "private-user" / "internal-project"
    private.mkdir(parents=True)
    path = private / "input.json"
    path.write_bytes(b"example")
    assert source(path) == {"path": "input.json", "sha256": hashlib.sha256(b"example").hexdigest()}
