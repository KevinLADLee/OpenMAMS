"""Command-line entry points; heavy optional dependencies load only when used."""
import argparse
import json

from .backhaul import SCHEDULERS


def main():
    parser = argparse.ArgumentParser(description="OpenMAMS NTN backhaul simulation")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("topology", help="Generate LEOPath geometry at both endpoints")
    p.add_argument("--planes", type=int, default=20)
    p.add_argument("--sats-per-plane", type=int, default=20)
    p.add_argument("--mean-motion", type=float, default=15.19)
    p.add_argument("--inclination", type=float, default=53.0)
    p.add_argument("--min-elevation", type=float, default=10.0)
    p.add_argument("--minute", type=float, help="Minutes from synthetic TLE epoch; default scans one orbit")
    p.add_argument("--output", required=True)
    p = commands.add_parser("endpoint", help="Run PF and Max-C/I at one endpoint")
    p.add_argument("--topology", required=True)
    p.add_argument("--link-direction", choices=["uplink", "downlink"], required=True)
    p.add_argument("--buildings", help="Optional local building-density CSV; omitted means no buildings")
    p.add_argument("--half-width-m", type=float, default=1050.0)
    p.add_argument("--open-site-user1", action="store_true")
    p.add_argument("--uplink-gs-eirp-dbm", type=float, default=58.0)
    p.add_argument("--uplink-sat-g-over-t-db-k", type=float, default=13.0)
    p.add_argument("--downlink-ground-rx-gain-dbi", type=float, default=40.0)
    p.add_argument("--random-seed", type=int, default=20260608)
    p.add_argument("--output", required=True)
    p = commands.add_parser("combine", help="Take the minimum of endpoint mean rates")
    p.add_argument("--uplink-dir", required=True)
    p.add_argument("--downlink-dir", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("replay", help="Deliver captured frames over the snapshot rate")
    p.add_argument("--data", required=True)
    p.add_argument("--backhaul", required=True)
    p.add_argument("--scheduler", choices=SCHEDULERS, default="Proportional Fair")
    p.add_argument("--deadline-s", type=float, required=True)
    p.add_argument("--propagation-ms", type=float, default=0.0)
    p.add_argument("--frame-bytes", type=int, help="Override actual JPEG bytes, e.g. 200000 for paper load")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "topology":
            from .topology import generate
            result = generate(args)
        elif args.command == "endpoint":
            from .endpoint import simulate
            result = simulate(args)
        elif args.command == "combine":
            from .backhaul import combine
            result = combine(args.uplink_dir, args.downlink_dir, args.output)
        else:
            from .replay import replay
            result = replay(args)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
