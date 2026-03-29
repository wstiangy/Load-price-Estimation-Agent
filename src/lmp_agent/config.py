from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class RunConfig(BaseModel):
    case_id: str = "ieee14"
    horizon_hours: int = 24
    opf_model: str = "dc"
    inverse_mode: str = "hybrid"
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    seed: int = 7
    training_days: int = 16
    monte_carlo_samples: int = 20
    demand_scale: float = 1.0
    noise_scale: float = 0.08
    report_title: str = "IEEE 14-Bus Pricing Agent Report"

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("quantiles must not be empty")
        if sorted(value) != value:
            raise ValueError("quantiles must be sorted in ascending order")
        if any(q <= 0 or q >= 1 for q in value):
            raise ValueError("quantiles must be between 0 and 1")
        return value


@dataclass(slots=True)
class OPFResult:
    bus_lmp: pd.DataFrame
    gen_dispatch: pd.DataFrame
    line_flows: pd.DataFrame
    congestion_flags: pd.DataFrame
    feasible: bool
    objective: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class BusLoadEstimate:
    estimated_bus_load: pd.DataFrame
    surrogate_init: pd.DataFrame
    refined_solution: pd.DataFrame
    price_residual: pd.DataFrame


@dataclass(slots=True)
class SubBusAllocation:
    subbus_loads: pd.DataFrame
    parent_bus_load: pd.DataFrame
    balance_gap: pd.DataFrame


@dataclass(slots=True)
class ForecastBundle:
    subbus_load_quantiles: dict[float, pd.DataFrame]
    bus_load_scenarios: list[pd.DataFrame]
    bus_lmp_quantiles: dict[float, pd.DataFrame]
    scenario_metadata: dict[str, Any]


@dataclass(slots=True)
class WorkflowArtifacts:
    config: RunConfig
    current_truth_bus_load: pd.DataFrame
    current_truth_subbus_load: pd.DataFrame
    current_opf: OPFResult
    bus_estimate: BusLoadEstimate
    subbus_estimate: SubBusAllocation
    forecast: ForecastBundle
    future_truth_bus_load: pd.DataFrame
    future_truth_subbus_load: pd.DataFrame
    future_truth_opf: OPFResult
    metrics: dict[str, float]
    report_markdown: str
    output_dir: Path | None = None
