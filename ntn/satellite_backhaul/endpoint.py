"""Configure and run endpoint scheduling."""
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import _mac as mac
from .io import dump, new_output, source


def simulate(args):
    for key in ("half_width_m", "uplink_gs_eirp_dbm", "uplink_sat_g_over_t_db_k",
                "downlink_ground_rx_gain_dbi"):
        if not math.isfinite(getattr(args, key)):
            raise ValueError(f"{key} must be finite")
    if args.half_width_m <= 0 or args.random_seed < 0:
        raise ValueError("Scene width must be positive and seed nonnegative")
    mac.LINK_DIRECTION = args.link_direction
    mac.RNG = np.random.default_rng(args.random_seed)
    mac.PIN_USER1_TO_HUB = True
    mac.OPEN_SITE_USER1 = args.open_site_user1
    mac.CENTER_FREQ_HZ = 30e9 if args.link_direction == "uplink" else 20e9
    mac.TX_EIRP_DBM = args.uplink_gs_eirp_dbm if args.link_direction == "uplink" else 58.0
    mac.SAT_G_OVER_T_DB_K = args.uplink_sat_g_over_t_db_k
    mac.RX_GAIN_DBI = args.downlink_ground_rx_gain_dbi
    topology = Path(args.topology)
    data = json.loads(topology.read_text())
    visible = data.get("visible_satellites", data.get("hong_kong_visible_satellites"))
    if not isinstance(visible, list):
        raise ValueError("Topology must contain a visible_satellites list")
    ids = set()
    for sat in visible:
        if type(sat["sat_id"]) is not int or sat["sat_id"] in ids:
            raise ValueError("Satellite IDs must be unique integers")
        ids.add(sat["sat_id"])
        for key in ("elevation_deg", "azimuth_deg", "range_km", "delay_ms"):
            if not math.isfinite(float(sat[key])):
                raise ValueError(f"Invalid satellite {key}")
        if not 0 < sat["elevation_deg"] <= 90 or sat["range_km"] <= 0 or sat["delay_ms"] < 0:
            raise ValueError("Invalid satellite elevation, range or delay")
    sats, _ = mac.load_satellites(topology)
    buildings = (pd.read_csv(args.buildings) if args.buildings else
                 pd.DataFrame(columns=["centroid_x_m", "centroid_y_m", "height_m", "area_m2"]))
    fields = ["centroid_x_m", "centroid_y_m", "height_m", "area_m2"]
    if not np.isfinite(buildings[fields].to_numpy(dtype=float)).all():
        raise ValueError("Building geometry must be finite")
    if (buildings[["height_m", "area_m2"]] < 0).any().any():
        raise ValueError("Building height and area must be nonnegative")
    users = mac.generate_users(buildings, args.half_width_m)
    frames, summaries = [], {}
    out = new_output(args.output)
    info = {"completed": False,
            "scenario": {"link_direction": args.link_direction, "random_seed": args.random_seed,
                         "visible_satellites": len(sats), "users": mac.K,
                         "num_rb": mac.NUM_RB, "num_slots": mac.NUM_SLOTS,
                         "rb_bandwidth_hz": mac.RB_BW_HZ, "abstract_slot_duration_s": mac.SLOT_DURATION_S,
                         "half_width_m": args.half_width_m, "open_site_user1": args.open_site_user1},
            "channel_model": "OpenNTN-inspired channel abstraction",
            "link_budget": {"frequency_hz": mac.CENTER_FREQ_HZ, "eirp_dbm": mac.TX_EIRP_DBM,
                            "satellite_g_over_t_db_k": mac.SAT_G_OVER_T_DB_K if args.link_direction == "uplink" else None,
                            "ground_rx_gain_dbi": mac.RX_GAIN_DBI if args.link_direction == "downlink" else None,
                            "power_semantics": "per-RB link budget; no total-power redistribution"},
            "sources": {"topology": source(topology), "kernel": source(mac.__file__),
                        "buildings": source(args.buildings) if args.buildings else None},
            "users": [asdict(u) for u in users], "satellites": [asdict(s) for s in sats],
            "versions": {"numpy": np.__version__, "pandas": pd.__version__}}
    dump(out / "mac_ofdma_summary.json", info)
    if sats:
        cqi = mac.estimate_large_scale_cqi(sats, users)
        sinr, se = mac.generate_instantaneous_rates(cqi["base_sinr_db"])
    else:
        cqi = None
        se = np.zeros((mac.K, mac.NUM_SLOTS, mac.NUM_RB))
    for name, tag, scheduler in (("Max-C/I", "maxci", mac.run_max_ci),
                                  ("Proportional Fair", "pf", mac.run_pf)):
        if sats:
            allocation = scheduler(se)
            frame, summary = mac.compute_qos(name, allocation, sinr, se, users, sats, cqi)
        else:
            allocation = np.full((mac.NUM_SLOTS, mac.NUM_RB), -1, dtype=int)
            frame = pd.DataFrame([{"scheduler": name, "uid": u.uid, "user": f"U{u.uid:02d}",
                                   "role": u.role, "serving_sat": "", "allocated_rbs": 0,
                                   "service_share": 0.0, "throughput_mbps": 0.0} for u in users])
            summary = {"sum_throughput_mbps": 0.0, "starved_users_0_rbs": mac.K}
        frames.append(frame)
        summaries[name] = summary
        np.save(out / f"allocation_{tag}.npy", allocation)
        hub_rate = np.where(allocation == 0, se[0] * mac.RB_BW_HZ / 1e6, 0).sum(axis=1)
        np.save(out / f"hub_slot_rate_{tag}_mbps.npy", hub_rate)
    pd.concat(frames, ignore_index=True).to_csv(out / "mac_user_qos.csv", index=False)
    np.save(out / "instantaneous_se_users_slots_rb.npy", se)
    info.update(completed=True, scheduler_summaries=summaries)
    dump(out / "mac_ofdma_summary.json", info)
    return info
