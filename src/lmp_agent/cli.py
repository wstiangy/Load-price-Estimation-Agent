from __future__ import annotations

import argparse
from pathlib import Path

from .agent import PricingWorkflowAgent
from .config import RunConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the IEEE 14-bus pricing agent prototype.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--training-days", type=int, default=16)
    parser.add_argument("--demand-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.08)
    parser.add_argument("--monte-carlo-samples", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/default-run"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RunConfig(
        seed=args.seed,
        training_days=args.training_days,
        demand_scale=args.demand_scale,
        noise_scale=args.noise_scale,
        monte_carlo_samples=args.monte_carlo_samples,
    )
    artifacts = PricingWorkflowAgent(config).run_pipeline(output_dir=args.output_dir)
    print(artifacts.report_markdown)


if __name__ == "__main__":
    main()
