from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import RunConfig, WorkflowArtifacts
from .data import SyntheticScenarioGenerator, load_ieee14_case
from .disaggregation import LoadDisaggregator
from .forecast import ProbabilisticLoadForecaster
from .inverse import InverseLoadEstimator
from .opf import OPFRunner
from .reporting import compute_metrics, render_markdown_report, write_report


@dataclass
class PricingWorkflowAgent:
    config: RunConfig

    def __post_init__(self) -> None:
        self.case_data = load_ieee14_case(seed=self.config.seed)
        self.generator = SyntheticScenarioGenerator(self.case_data, seed=self.config.seed)
        self.opf_runner = OPFRunner(self.case_data)
        self.inverse_tool = InverseLoadEstimator(surrogate_random_state=self.config.seed)
        self.disagg_tool = LoadDisaggregator(self.case_data)
        self.forecast_tool = ProbabilisticLoadForecaster(
            self.case_data,
            quantiles=tuple(self.config.quantiles),
            random_state=self.config.seed,
        )

    def plan_run(self) -> dict[str, int | str | float]:
        return {
            "case_id": self.config.case_id,
            "training_days": self.config.training_days,
            "seed": self.config.seed,
            "horizon_hours": self.config.horizon_hours,
            "quantiles": len(self.config.quantiles),
        }

    def run_opf(self, bus_loads: pd.DataFrame, generator_costs: pd.DataFrame, branch_capacity_factor: pd.Series):
        return self.opf_runner.run(bus_loads, generator_costs, branch_capacity_factor)

    def estimate_bus_load(self, observed_lmp: pd.DataFrame, prior_load: pd.DataFrame):
        return self.inverse_tool.estimate(observed_lmp=observed_lmp, prior_load=prior_load)

    def disaggregate_subbus(self, parent_bus_load: pd.DataFrame):
        return self.disagg_tool.disaggregate(parent_bus_load)

    def forecast_and_simulate_prices(
        self,
        current_subbus_loads: pd.DataFrame,
        current_parent_loads: pd.DataFrame,
        future_parent_proxy: pd.DataFrame,
        generator_costs: pd.DataFrame,
        branch_capacity_factor: pd.Series,
    ):
        quantile_prediction = self.forecast_tool.predict_subbus_quantiles(
            current_subbus_loads=current_subbus_loads,
            current_parent_loads=current_parent_loads,
            future_parent_proxy=future_parent_proxy,
        )
        return self.forecast_tool.simulate_price_distribution(
            quantile_loads=quantile_prediction,
            opf_runner=self.opf_runner,
            generator_costs=generator_costs,
            branch_capacity_factor=branch_capacity_factor,
            n_samples=self.config.monte_carlo_samples,
            seed=self.config.seed,
        )

    def render_report(self, artifacts: WorkflowArtifacts) -> str:
        return render_markdown_report(artifacts)

    def run_pipeline(self, output_dir: Path | None = None) -> WorkflowArtifacts:
        scenarios = self.generator.generate_history(
            days=self.config.training_days + 2,
            demand_scale=self.config.demand_scale,
            noise_scale=self.config.noise_scale,
        )
        history = scenarios[:-2]
        current = scenarios[-2]
        future = scenarios[-1]

        history_opf = [self.run_opf(s.bus_loads, s.generator_costs, s.branch_capacity_factor) for s in history]
        self.inverse_tool.fit(
            bus_load_history=[s.bus_loads for s in history],
            lmp_history=[opf.bus_lmp for opf in history_opf],
        )
        self.forecast_tool.fit(
            history_subbus=[s.subbus_loads for s in history] + [current.subbus_loads],
            history_parent=[s.bus_loads for s in history] + [current.bus_loads],
        )

        current_opf = self.run_opf(current.bus_loads, current.generator_costs, current.branch_capacity_factor)
        prior_load = pd.concat([history[-3].bus_loads, history[-2].bus_loads, history[-1].bus_loads]).groupby(level=0).mean()
        bus_estimate = self.estimate_bus_load(observed_lmp=current_opf.bus_lmp, prior_load=prior_load)
        subbus_estimate = self.disaggregate_subbus(bus_estimate.estimated_bus_load)

        future_parent_proxy = subbus_estimate.parent_bus_load.mul(1.015)
        forecast = self.forecast_and_simulate_prices(
            current_subbus_loads=subbus_estimate.subbus_loads,
            current_parent_loads=subbus_estimate.parent_bus_load,
            future_parent_proxy=future_parent_proxy,
            generator_costs=future.generator_costs,
            branch_capacity_factor=future.branch_capacity_factor,
        )
        future_truth_opf = self.run_opf(future.bus_loads, future.generator_costs, future.branch_capacity_factor)

        median_quantile = sorted(self.config.quantiles)[len(self.config.quantiles) // 2]
        metrics = compute_metrics(
            estimated_bus_load=bus_estimate.estimated_bus_load,
            truth_bus_load=current.bus_loads,
            estimated_subbus_load=subbus_estimate.subbus_loads,
            truth_subbus_load=current.subbus_loads,
            forecast_p50=forecast.subbus_load_quantiles[median_quantile],
            forecast_truth=future.subbus_loads,
            lmp_p50=forecast.bus_lmp_quantiles[median_quantile],
            lmp_truth=future_truth_opf.bus_lmp,
        )

        artifacts = WorkflowArtifacts(
            config=self.config,
            current_truth_bus_load=current.bus_loads,
            current_truth_subbus_load=current.subbus_loads,
            current_opf=current_opf,
            bus_estimate=bus_estimate,
            subbus_estimate=subbus_estimate,
            forecast=forecast,
            future_truth_bus_load=future.bus_loads,
            future_truth_subbus_load=future.subbus_loads,
            future_truth_opf=future_truth_opf,
            metrics=metrics,
            report_markdown="",
            output_dir=output_dir,
        )
        artifacts.report_markdown = self.render_report(artifacts)

        if output_dir is not None:
            write_report(artifacts.report_markdown, output_dir)
            self._write_csv_artifacts(artifacts, output_dir)

        return artifacts

    def _write_csv_artifacts(self, artifacts: WorkflowArtifacts, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts.current_opf.bus_lmp.to_csv(output_dir / "current_bus_lmp.csv")
        artifacts.bus_estimate.estimated_bus_load.to_csv(output_dir / "estimated_bus_load.csv")
        artifacts.subbus_estimate.subbus_loads.to_csv(output_dir / "estimated_subbus_load.csv")
        for quantile, frame in artifacts.forecast.subbus_load_quantiles.items():
            frame.to_csv(output_dir / f"forecast_subbus_q{int(quantile * 100)}.csv")
        for quantile, frame in artifacts.forecast.bus_lmp_quantiles.items():
            frame.to_csv(output_dir / f"forecast_lmp_q{int(quantile * 100)}.csv")
