from __future__ import annotations

import argparse
from pathlib import Path

from routecode.controlled import run_controlled_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3 controlled ProbeRoute++ surrogate pilot.")
    parser.add_argument("--config", default="configs/proberoute_controlled.yaml")
    parser.add_argument("--stage", choices=["dry_run", "pilot"], default="pilot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_controlled_surrogate(Path(args.config), stage=args.stage)
    print(f"Wrote controlled outputs to {paths['output_dir']}")
    print(f"Run report: {paths['run_report']}")


if __name__ == "__main__":
    main()
