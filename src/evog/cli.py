import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(prog="evog", description="Explicit visual object grounding")
    parser.add_argument(
        "command", choices=["prepare", "train", "infer", "evaluate", "check", "pack", "download", "merge"]
    )
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    if args.command in ("prepare", "train", "infer", "evaluate", "check"):
        result = importlib.import_module(f"evog.{args.command}").main(remaining)
        if isinstance(result, int):
            raise SystemExit(result)
    else:
        from evog.utilities import main as utility_main

        utility_main(args.command, remaining)
