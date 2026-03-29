from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import WorkflowArtifacts


def compute_metrics(
    estimated_bus_load: pd.DataFrame,
    truth_bus_load: pd.DataFrame,
    estimated_subbus_load: pd.DataFrame,
    truth_subbus_load: pd.DataFrame,
    forecast_p50: pd.DataFrame,
    forecast_truth: pd.DataFrame,
    lmp_p50: pd.DataFrame,
    lmp_truth: pd.DataFrame,
) -> dict[str, float]:
    return {
        "inverse_bus_mape": _mape(estimated_bus_load, truth_bus_load),
        "disagg_subbus_wape": _wape(estimated_subbus_load, truth_subbus_load),
        "forecast_subbus_p50_mape": _mape(forecast_p50, forecast_truth),
        "forecast_lmp_p50_mape": _mape(lmp_p50, lmp_truth),
    }


def render_markdown_report(artifacts: WorkflowArtifacts) -> str:
    lines = [
        f"# {artifacts.config.report_title}",
        "",
        "## Run Summary",
        f"- Case: {artifacts.config.case_id}",
        f"- Seed: {artifacts.config.seed}",
        f"- Horizon: {artifacts.config.horizon_hours} hours",
        f"- Training days: {artifacts.config.training_days}",
        f"- Monte Carlo samples: {artifacts.config.monte_carlo_samples}",
        "",
        "## Metrics",
    ]
    for key, value in artifacts.metrics.items():
        lines.append(f"- {key}: {value:.4f}")

    lines.extend(
        [
            "",
            "## Current Day OPF",
            f"- Feasible: {artifacts.current_opf.feasible}",
            f"- Objective: {artifacts.current_opf.objective:.2f}",
            "",
            "## Forecast Notes",
            "- Probabilistic price trajectories are obtained by sampling sub-bus forecast distributions and rerunning DC-OPF.",
            "- The prototype keeps the future generator costs and branch stress from the synthetic target day for evaluation.",
        ]
    )
    return "\n".join(lines)


def write_report(markdown: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


def _mape(left: pd.DataFrame, right: pd.DataFrame) -> float:
    denominator = right.abs().clip(lower=1e-3)
    return float((((left - right).abs() / denominator).to_numpy(dtype=float)).mean())


def _wape(left: pd.DataFrame, right: pd.DataFrame) -> float:
    numerator = (left - right).abs().to_numpy(dtype=float).sum()
    denominator = right.abs().to_numpy(dtype=float).sum()
    return float(numerator / max(denominator, 1e-3))
