"""Capture-only command; no QA, model or database dependency."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CARLA multi-UAV RGB capture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--uavs", type=int, default=4)
    parser.add_argument("--spawns", type=int, nargs="+", help="Map spawn indices; overrides --uavs")
    parser.add_argument("--route", type=Path, help="Closed route JSON; UAVs start at equal arc-length offsets")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--speed", type=float, default=4.5)
    parser.add_argument("--altitude", type=float, default=17.5)
    parser.add_argument("--pitch", type=float, default=-45)
    parser.add_argument("--fov", type=float, default=75)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objects", type=int, choices=[0, 10], default=0)
    args = parser.parse_args()
    if args.route and args.spawns:
        parser.error("Use either --route or --spawns")
    from .collect import record
    try:
        print(json.dumps(record(args), indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
