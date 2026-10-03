"""Independent QA entry point. Capture need not be installed."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="Original ReMEmbR multi-UAV QA")
    parser.add_argument("--config", type=Path, help="Override the bundled model config JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Caption a capture directory and create memory")
    build.add_argument("--data", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--interval", type=float, default=3, help="Seconds between captions per UAV; 0 keeps all")
    for name in ("ask", "evaluate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--db", type=Path, required=True)
        cmd.add_argument("--uavs", type=int, nargs="+", required=True)
        cmd.add_argument("--timeout", type=float, default=90)
        cmd.add_argument("--time-offset", type=float, help="Legacy database time offset; omit for retained default")
        cmd.add_argument("--output", type=Path, required=True)
        if name == "ask":
            cmd.add_argument("--question", required=True)
        else:
            cmd.add_argument("--questions", type=Path, required=True)
    args = parser.parse_args()
    # Make the original localtime/HMS tools reproducible across hosts.
    os.environ["TZ"] = "UTC"
    time.tzset()
    from .runtime import load_config, QA
    from .pipeline import build as build_memory, dump
    from .evaluation import evaluate, load_questions
    try:
        config = load_config(args.config)
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.command == "build":
            result = build_memory(args.data, args.output, args.interval, config)
        else:
            if args.command == "evaluate":
                load_questions(args.questions)
            qa = QA(args.db, args.uavs, config, time_offset=args.time_offset)
            try:
                if args.command == "ask":
                    result = qa.ask(args.question, args.timeout)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    dump(args.output, result)
                else:
                    result = evaluate(qa, args.questions, args.output, args.timeout)
            finally:
                qa.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
