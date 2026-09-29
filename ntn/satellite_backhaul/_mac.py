"""OFDMA channel and scheduling model."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

K = 10
NUM_RB = 48
NUM_SLOTS = 72
SUBCARRIERS_PER_RB = 12
OFDM_SYMBOLS_PER_SLOT = 14
SUBCARRIER_SPACING_HZ = 120e3
RB_BW_HZ = SUBCARRIERS_PER_RB * SUBCARRIER_SPACING_HZ
SLOT_DURATION_S = 1e-3
# Downlink parameters.
DOWNLINK_CENTER_FREQ_HZ = 20e9
DOWNLINK_SAT_EIRP_DBM = 58.0
DOWNLINK_GROUND_RX_GAIN_DBI = 28.0

# Ka-band uplink parameters; G/T includes receive gain and noise temperature.
UPLINK_CENTER_FREQ_HZ = 30e9
UPLINK_GS_EIRP_DBM = 58.0
UPLINK_SAT_G_OVER_T_DB_K = 13.0
BOLTZMANN_DBW_PER_K_HZ = -228.6

LINK_DIRECTION = "downlink"
CENTER_FREQ_HZ = DOWNLINK_CENTER_FREQ_HZ
TX_EIRP_DBM = DOWNLINK_SAT_EIRP_DBM
RX_GAIN_DBI = DOWNLINK_GROUND_RX_GAIN_DBI
SAT_G_OVER_T_DB_K = UPLINK_SAT_G_OVER_T_DB_K
PIN_USER1_TO_HUB = False
OPEN_SITE_USER1 = False
RANDOM_SEED = 20260608
ATMOSPHERIC_LOSS_DB = 1.2
RAIN_MARGIN_DB = 1.5
NOISE_FIGURE_DB = 5.0
THERMAL_NOISE_DBM_HZ = -174.0
NOISE_DBM = THERMAL_NOISE_DBM_HZ + 10 * math.log10(RB_BW_HZ) + NOISE_FIGURE_DB
REUSE_LEAKAGE_DB = -30.0
C_MPS = 299_792_458.0
RNG = np.random.default_rng(RANDOM_SEED)


@dataclass
class Satellite:
    sat_id: int
    elevation_deg: float
    azimuth_deg: float
    range_km: float
    delay_ms: float
    x_m: float
    y_m: float
    z_m: float
    doppler_hz: float


@dataclass
class User:
    uid: int
    x_m: float
    y_m: float
    environment: str
    density_score: float
    shadow_base_db: float
    role: str = "competing_terminal"


def dbm_to_mw(dbm: np.ndarray | float) -> np.ndarray | float:
    return 10.0 ** (np.asarray(dbm) / 10.0)


def mw_to_dbm(mw: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(np.asarray(mw), 1e-30))


def fspl_db(range_m: np.ndarray | float) -> np.ndarray | float:
    wavelength = C_MPS / CENTER_FREQ_HZ
    return 20.0 * np.log10(4.0 * np.pi * np.maximum(range_m, 1.0) / wavelength)


def sat_from_summary(item: dict, idx: int) -> Satellite:
    elev = math.radians(float(item["elevation_deg"]))
    az = math.radians(float(item["azimuth_deg"]))
    r_m = float(item["range_km"]) * 1000.0
    horizontal = r_m * math.cos(elev)
    x = horizontal * math.sin(az)
    y = horizontal * math.cos(az)
    z = r_m * math.sin(elev)
    v_radial = 7600.0 * math.cos(elev) * math.sin(0.81 * idx + 0.37)
    return Satellite(
        sat_id=int(item["sat_id"]),
        elevation_deg=float(item["elevation_deg"]),
        azimuth_deg=float(item["azimuth_deg"]),
        range_km=float(item["range_km"]),
        delay_ms=float(item["delay_ms"]),
        x_m=x,
        y_m=y,
        z_m=z,
        doppler_hz=float(CENTER_FREQ_HZ * v_radial / C_MPS),
    )


def load_satellites(input_path: Path) -> tuple[List[Satellite], int]:
    """Load satellites from topology JSON file and return (satellites, actual_M)."""
    data = json.loads(input_path.read_text(encoding="utf-8"))
    visible = data.get("visible_satellites", data.get("hong_kong_visible_satellites"))
    if visible is None:
        raise KeyError(
            f"{input_path} must contain 'visible_satellites' or "
            "'hong_kong_visible_satellites'"
        )
    sats = [sat_from_summary(v, i) for i, v in enumerate(visible)]
    actual_M = len(sats)  # Use actual number of visible satellites
    return sats, actual_M


def density_at(x: float, y: float, buildings: pd.DataFrame) -> float:
    if buildings.empty:
        return 0.0
    dx = buildings["centroid_x_m"].to_numpy() - x
    dy = buildings["centroid_y_m"].to_numpy() - y
    d2 = dx * dx + dy * dy
    h = np.clip(buildings["height_m"].to_numpy() / 80.0, 0.25, 4.0)
    area = np.clip(np.sqrt(np.maximum(buildings["area_m2"].to_numpy(), 1.0)) / 50.0, 0.3, 4.0)
    weights = np.exp(-d2 / (2 * 150.0**2)) * h * area
    return float(np.sum(weights))


def generate_users(buildings: pd.DataFrame, half_width_m: float) -> List[User]:
    # Stratify users so the scheduler sees open harbour, waterfront, and dense urban users.
    candidates: List[Tuple[float, float]] = []
    for _ in range(3000):
        x = RNG.uniform(-half_width_m * 0.96, half_width_m * 0.96)
        y = RNG.uniform(-half_width_m * 0.96, half_width_m * 0.96)
        candidates.append((x, y))

    densities = np.array([density_at(x, y, buildings) for x, y in candidates])
    d_norm = (densities - densities.min()) / max(float(densities.max() - densities.min()), 1e-9)

    selected: List[int] = []
    zones = [
        np.where(np.array([p[1] for p in candidates]) < -250.0)[0],  # harbour/open
        np.where((np.array([p[1] for p in candidates]) >= -250.0) & (d_norm < 0.35))[0],
        np.where(d_norm >= 0.35)[0],
    ]
    targets = [6, 6, 8]
    for zone, n in zip(zones, targets):
        zone = np.setdiff1d(zone, np.array(selected, dtype=int), assume_unique=False)
        chosen = RNG.choice(zone, size=n, replace=False) if len(zone) >= n else zone
        selected.extend(int(i) for i in chosen)
    while len(selected) < K:
        i = int(RNG.integers(0, len(candidates)))
        if i not in selected:
            selected.append(i)

    # Place the hub at the UAV-cluster origin.
    selected_points = [candidates[idx] for idx in selected[:K]]
    if PIN_USER1_TO_HUB:
        selected_points = [(0.0, 0.0), *selected_points[: K - 1]]

    density_min = float(densities.min())
    density_span = max(float(densities.max() - densities.min()), 1e-9)
    selected_norm = [
        float(np.clip((density_at(x, y, buildings) - density_min) / density_span, 0.0, 1.0))
        for x, y in selected_points
    ]

    users: List[User] = []
    for uid, ((x, y), dn) in enumerate(zip(selected_points, selected_norm), start=1):
        if y < -250 and dn < 0.45:
            env = "harbour/open"
        elif dn > 0.65:
            env = "dense urban"
        else:
            env = "urban edge"
        shadow = 2.0 + 19.0 * dn + RNG.normal(0.0, 2.2)
        if env == "harbour/open":
            shadow -= 4.0
        shadow = float(np.clip(shadow, 0.0, 28.0))
        if PIN_USER1_TO_HUB and uid == 1:
            role = "serving_memory_hub" if LINK_DIRECTION == "uplink" else "remote_command_gateway"
            if OPEN_SITE_USER1:
                # An open-site gateway has no building shadowing.
                env = "professional gateway/open site"
                dn = 0.0
                shadow = 0.0
        else:
            role = "competing_terminal"
        users.append(
            User(
                uid=uid,
                x_m=float(x),
                y_m=float(y),
                environment=env,
                density_score=float(dn),
                shadow_base_db=shadow,
                role=role,
            )
        )
    return users


def estimate_large_scale_cqi(sats: List[Satellite], users: List[User]) -> dict:
    """Estimate large-scale CQI for each user-satellite pair."""
    actual_M = len(sats)  # Use actual number of satellites
    rx_power = np.full((actual_M, K), np.nan, dtype=float)
    carrier_to_noise = np.zeros((actual_M, K), dtype=float)
    shadow_loss = np.zeros((actual_M, K), dtype=float)
    for mi, sat in enumerate(sats):
        elev = math.radians(max(sat.elevation_deg, 1.0))
        elevation_factor = 0.82 + 0.20 / max(math.sin(elev), 0.17)
        fspl = fspl_db(sat.range_km * 1000.0)
        for ki, user in enumerate(users):
            user_random = RNG.normal(0.0, 1.2)
            sat_random = 1.5 * math.sin(0.7 * mi + 0.31 * user.uid)
            loss = user.shadow_base_db * elevation_factor + user_random + sat_random
            loss = float(np.clip(loss, 0.0, 45.0))
            shadow_loss[mi, ki] = loss
            total_loss_db = fspl + ATMOSPHERIC_LOSS_DB + RAIN_MARGIN_DB + loss
            if LINK_DIRECTION == "uplink":
                # C/N0 = EIRP[dBW] + G/T[dB/K] - losses - k[dBW/K/Hz].
                cn0_db_hz = (
                    (TX_EIRP_DBM - 30.0)
                    + SAT_G_OVER_T_DB_K
                    - total_loss_db
                    - BOLTZMANN_DBW_PER_K_HZ
                )
                carrier_to_noise[mi, ki] = cn0_db_hz - 10.0 * math.log10(RB_BW_HZ)
            else:
                rx_power[mi, ki] = TX_EIRP_DBM + RX_GAIN_DBI - total_loss_db
                carrier_to_noise[mi, ki] = rx_power[mi, ki] - NOISE_DBM

    leakage = float(dbm_to_mw(REUSE_LEAKAGE_DB))
    carrier_to_noise_linear = 10.0 ** (carrier_to_noise / 10.0)
    sinr = np.zeros_like(carrier_to_noise_linear)
    if LINK_DIRECTION == "uplink":
        # At a satellite receiver, residual co-channel/adjacent-channel leakage
        # comes from the other simultaneously active ground terminals.
        for mi in range(actual_M):
            interference_to_noise = np.maximum(
                np.sum(carrier_to_noise_linear[mi]) - carrier_to_noise_linear[mi],
                0.0,
            ) * leakage
            sinr[mi] = carrier_to_noise_linear[mi] / (1.0 + interference_to_noise)
    else:
        # Other visible satellites contribute frequency-reuse leakage.
        for mi in range(actual_M):
            interference_to_noise = np.maximum(
                np.sum(carrier_to_noise_linear, axis=0) - carrier_to_noise_linear[mi],
                0.0,
            ) * leakage
            sinr[mi] = carrier_to_noise_linear[mi] / (1.0 + interference_to_noise)
    sinr_db = 10.0 * np.log10(np.maximum(sinr, 1e-12))
    best_sat_idx = np.argmax(sinr_db, axis=0)
    base_sinr_db = sinr_db[best_sat_idx, np.arange(K)]
    base_power_dbm = rx_power[best_sat_idx, np.arange(K)]
    base_cnr_db = carrier_to_noise[best_sat_idx, np.arange(K)]
    return {
        "rx_power_dbm": rx_power,
        "carrier_to_noise_db": carrier_to_noise,
        "shadow_loss_db": shadow_loss,
        "sinr_db": sinr_db,
        "best_sat_idx": best_sat_idx,
        "base_sinr_db": base_sinr_db,
        "base_power_dbm": base_power_dbm,
        "base_cnr_db": base_cnr_db,
    }


def generate_instantaneous_rates(base_sinr_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slow = np.zeros((K, NUM_SLOTS), dtype=float)
    slow[:, 0] = RNG.normal(0.0, 1.4, size=K)
    for t in range(1, NUM_SLOTS):
        slow[:, t] = 0.92 * slow[:, t - 1] + RNG.normal(0.0, 0.55, size=K)
    freq = RNG.normal(0.0, 1.1, size=(K, NUM_RB))
    rb_axis = np.linspace(0, 2 * np.pi, NUM_RB, endpoint=False)
    for k in range(K):
        freq[k] += 1.2 * np.sin(rb_axis * (1 + k % 4) + RNG.uniform(0, 2 * np.pi))
    fast = RNG.exponential(scale=1.0, size=(K, NUM_SLOTS, NUM_RB))
    fast_db = np.clip(10.0 * np.log10(np.maximum(fast, 1e-4)), -8.0, 5.0) * 0.45
    sinr_db = base_sinr_db[:, None, None] + slow[:, :, None] + freq[:, None, :] + fast_db
    sinr_linear = 10.0 ** (sinr_db / 10.0)
    se = 0.85 * np.log2(1.0 + sinr_linear)
    se = np.clip(se, 0.0, 7.4)
    return sinr_db, se


def run_max_ci(se: np.ndarray) -> np.ndarray:
    # allocation[t, rb] = user index
    return np.argmax(np.transpose(se, (1, 2, 0)), axis=2)


def run_pf(se: np.ndarray) -> np.ndarray:
    allocation = np.zeros((NUM_SLOTS, NUM_RB), dtype=int)
    avg_rate = np.full(K, 0.05, dtype=float)
    alpha = 0.06
    for t in range(NUM_SLOTS):
        slot_rate = np.zeros(K, dtype=float)
        for rb in range(NUM_RB):
            rates = se[:, t, rb] * RB_BW_HZ / 1e6
            metric = rates / np.maximum(avg_rate, 0.01)
            chosen = int(np.argmax(metric))
            allocation[t, rb] = chosen
            slot_rate[chosen] += rates[chosen]
        avg_rate = (1.0 - alpha) * avg_rate + alpha * slot_rate
    return allocation


def jain_index(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    denom = len(values) * np.sum(values**2)
    if denom <= 0:
        return 0.0
    return float(np.sum(values) ** 2 / denom)


def compute_qos(name: str, allocation: np.ndarray, sinr_db: np.ndarray, se: np.ndarray, users: List[User], sats: List[Satellite], cqi: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    for k, user in enumerate(users):
        mask = allocation == k
        allocated_rbs = int(mask.sum())
        slots_served = np.where(mask.any(axis=1))[0]
        if allocated_rbs:
            # se[k] has [slot, rb]
            bits_per_second_sum = float(np.sum(se[k][mask]) * RB_BW_HZ / NUM_SLOTS)
            throughput_mbps = bits_per_second_sum / 1e6
            mean_sinr = float(np.mean(sinr_db[k][mask]))
            mean_se = float(np.mean(se[k][mask]))
        else:
            throughput_mbps = 0.0
            mean_sinr = float("nan")
            mean_se = 0.0
        if len(slots_served) <= 1:
            mean_wait = float(NUM_SLOTS if len(slots_served) == 0 else NUM_SLOTS / 2)
            p95_wait = mean_wait
        else:
            gaps = np.diff(slots_served)
            mean_wait = float(np.mean(gaps))
            p95_wait = float(np.percentile(gaps, 95))
        best_sat = sats[int(cqi["best_sat_idx"][k])]
        rows.append(
            {
                "scheduler": name,
                "user": f"U{user.uid:02d}",
                "uid": user.uid,
                "x_m": user.x_m,
                "y_m": user.y_m,
                "environment": user.environment,
                "role": user.role,
                "density_score": user.density_score,
                "shadow_base_db": user.shadow_base_db,
                "serving_sat": f"S{best_sat.sat_id}",
                "long_term_sinr_db": float(cqi["base_sinr_db"][k]),
                "long_term_rx_power_dbm": float(cqi["base_power_dbm"][k]),
                "long_term_carrier_to_noise_db": float(cqi["base_cnr_db"][k]),
                "allocated_rbs": allocated_rbs,
                "service_share": allocated_rbs / float(NUM_RB * NUM_SLOTS),
                "slots_served": int(len(slots_served)),
                "slot_outage_fraction": 1.0 - len(slots_served) / float(NUM_SLOTS),
                "mean_wait_slots": mean_wait,
                "p95_wait_slots": p95_wait,
                "mean_scheduled_sinr_db": mean_sinr,
                "mean_spectral_eff_bps_hz": mean_se,
                "throughput_mbps": throughput_mbps,
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        "scheduler": name,
        "sum_throughput_mbps": float(df["throughput_mbps"].sum()),
        "mean_user_throughput_mbps": float(df["throughput_mbps"].mean()),
        "p05_user_throughput_mbps": float(df["throughput_mbps"].quantile(0.05)),
        "jain_fairness": jain_index(df["throughput_mbps"].to_numpy()),
        "starved_users_0_rbs": int((df["allocated_rbs"] == 0).sum()),
        "mean_slot_outage_fraction": float(df["slot_outage_fraction"].mean()),
    }
    return df, summary
