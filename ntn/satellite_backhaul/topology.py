"""Portable LEOPath TLE generation and simultaneous two-endpoint visibility."""
from datetime import datetime, timedelta, timezone
import math

from .io import dump, new_output, source


def generate(args):
    # External source is installed by scripts/setup_dependencies.sh --leopath.
    import ephem
    from leopath.tles.generate_tles_from_scratch import generate_tles_from_scratch_with_sgp
    import inspect

    if args.planes < 1 or args.sats_per_plane < 1:
        raise ValueError("Constellation dimensions must be positive")
    if not math.isfinite(args.mean_motion) or args.mean_motion <= 0:
        raise ValueError("Mean motion must be finite and positive")
    if not math.isfinite(args.inclination) or not 0 <= args.inclination <= 180:
        raise ValueError("Inclination must lie in [0,180]")
    if not math.isfinite(args.min_elevation) or not 0 < args.min_elevation < 90:
        raise ValueError("Minimum elevation must lie in (0,90)")
    if args.minute is not None and (not math.isfinite(args.minute) or args.minute < 0):
        raise ValueError("Minute must be finite and nonnegative")
    out = new_output(args.output)
    dump(out / "topology.json", {"completed": False})
    tle = out / "constellation.tle"
    generate_tles_from_scratch_with_sgp(str(tle), "OpenMAMS-LEO", args.planes,
                                      args.sats_per_plane, True, args.inclination,
                                      0.0000001, 0.0, args.mean_motion)
    lines = tle.read_text().splitlines()[1:]
    bodies = [ephem.readtle(*lines[i:i+3]) for i in range(0, len(lines), 3)]
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)  # LEOPath synthetic TLE epoch
    stations = {"hong_kong": (22.3193, 114.1694, 30.0), "istanbul": (41.0082, 28.9784, 30.0)}

    def visible(station, minute):
        observer = ephem.Observer()
        observer.lat, observer.lon = str(station[0]), str(station[1])
        observer.elevation = station[2]
        observer.epoch = "2000/01/01 00:00:00"
        observer.date = (epoch + timedelta(minutes=minute)).strftime("%Y/%m/%d %H:%M:%S.%f")
        rows = []
        for sid, body in enumerate(bodies):
            body.compute(observer)
            elevation = math.degrees(float(body.alt))
            if elevation >= args.min_elevation:
                rows.append({"sat_id": sid, "elevation_deg": elevation,
                             "azimuth_deg": math.degrees(float(body.az)),
                             "range_km": float(body.range) / 1000,
                             "delay_ms": float(body.range) / 299792458 * 1000})
        return sorted(rows, key=lambda r: (-r["elevation_deg"], r["range_km"]))

    scan = []
    minute = args.minute
    if minute is None:
        best_count, minute = -1, 0
        for t in range(math.ceil(1440 / args.mean_motion) + 1):
            count = len(visible(stations["hong_kong"], t))
            scan.append({"minute": t, "hong_kong_visible_count": count})
            if count > best_count:
                best_count, minute = count, t
    scenario = {"planes": args.planes, "sats_per_plane": args.sats_per_plane,
                "total_satellites": len(bodies), "mean_motion_rev_per_day": args.mean_motion,
                "inclination_deg": args.inclination, "minimum_elevation_deg": args.min_elevation,
                "selected_minute": minute, "epoch_utc": epoch.isoformat(),
                "selection": "explicit minute" if args.minute is not None else "first maximum HK visibility over one orbit",
                "orbit_note": "Synthetic constellation with 360-degree RAAN and alternating half-slot phasing"}
    for name, station in stations.items():
        dump(out / f"{name}.json", {"scenario": {**scenario, "endpoint": dict(zip(("lat", "lon", "elevation_m"), station))},
                                   "visible_satellites": visible(station, minute)})
    dump(out / "topology.json", {"completed": True, "scenario": scenario, "visibility_scan": scan,
                                 "sources": {"tle": source(tle), "generator": source(inspect.getfile(generate_tles_from_scratch_with_sgp))}})
    return scenario
