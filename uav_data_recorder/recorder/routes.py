"""Sample a closed road route at equally spaced UAV offsets."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_routes(path, map_name, count, frames, fps, speed):
    import carla

    data = json.loads(Path(path).read_text())
    if data["map"].rsplit("/", 1)[-1] != map_name or not data.get("closed"):
        raise ValueError("Route must be closed and match the selected map")
    points = np.array(data["route_waypoints"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Route must contain finite XYZ waypoints")
    points[:, 2] -= float(data.get("altitude_offset_m", 0))
    points = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    if np.any(lengths < 1e-6):
        raise ValueError("Route contains duplicate consecutive waypoints")
    cumulative = np.r_[0, np.cumsum(lengths)]
    total = float(cumulative[-1])
    offsets = np.arange(count) * total / count
    distances = (np.arange(frames)[:, None] * speed / fps + offsets) % total
    segments = np.searchsorted(cumulative, distances, side="right") - 1
    ratios = (distances - cumulative[segments]) / lengths[segments]
    positions = points[segments] + ratios[:, :, None] * (points[segments + 1] - points[segments])
    directions = points[segments + 1] - points[segments]
    headings = np.degrees(np.arctan2(directions[:, :, 1], directions[:, :, 0]))
    routes = [[SimpleNamespace(transform=carla.Transform(
        carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2])),
        carla.Rotation(yaw=float(headings[t, u]))))
        for t, p in enumerate(positions[:, u])] for u in range(count)]
    return routes, {"length_m": total, "offsets_m": offsets.tolist(), "source": data}
