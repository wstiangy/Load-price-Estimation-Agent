from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .config import ForecastBundle
from .data import CaseData
from .opf import OPFRunner


@dataclass
class ProbabilisticLoadForecaster:
    case_data: CaseData
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    random_state: int = 7

    def __post_init__(self) -> None:
        self.models: dict[float, GradientBoostingRegressor] = {}
        self._fitted = False

    def fit(self, history_subbus: list[pd.DataFrame], history_parent: list[pd.DataFrame]) -> None:
        rows = []
        targets = []
        for previous, current, parent_previous, parent_current in zip(
            history_subbus[:-1],
            history_subbus[1:],
            history_parent[:-1],
            history_parent[1:],
            strict=True,
        ):
            for subbus_id in previous.columns:
                parent = subbus_id.split("-S", maxsplit=1)[0]
                spec = next(spec for spec in self.case_data.subbus_specs[parent] if spec.subbus_id == subbus_id)
                for hour in previous.index:
                    prev_hour = (hour - 1) % len(previous.index)
                    rows.append(
                        [
                            float(hour),
                            np.sin(2 * np.pi * hour / 24.0),
                            np.cos(2 * np.pi * hour / 24.0),
                            float(previous.loc[hour, subbus_id]),
                            float(previous.loc[prev_hour, subbus_id]),
                            float(parent_previous.loc[hour, parent]),
                            float(parent_current.loc[hour, parent]),
                            float(spec.base_share),
                            float(spec.capacity_mw),
                            float(("residential", "commercial", "industrial").index(spec.profile_name)),
                            float(int(parent.split()[-1])),
                        ]
                    )
                    targets.append(float(current.loc[hour, subbus_id]))

        x = np.asarray(rows, dtype=float)
        y = np.asarray(targets, dtype=float)
        for quantile in self.quantiles:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=quantile,
                n_estimators=220,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state,
            )
            model.fit(x, y)
            self.models[quantile] = model
        self._fitted = True

    def predict_subbus_quantiles(
        self,
        current_subbus_loads: pd.DataFrame,
        current_parent_loads: pd.DataFrame,
        future_parent_proxy: pd.DataFrame,
    ) -> dict[float, pd.DataFrame]:
        if not self._fitted:
            raise RuntimeError("forecaster must be fitted before calling predict")

        feature_rows = []
        row_keys: list[tuple[int, str]] = []
        for subbus_id in current_subbus_loads.columns:
            parent = subbus_id.split("-S", maxsplit=1)[0]
            spec = next(spec for spec in self.case_data.subbus_specs[parent] if spec.subbus_id == subbus_id)
            for hour in current_subbus_loads.index:
                prev_hour = (hour - 1) % len(current_subbus_loads.index)
                row_keys.append((hour, subbus_id))
                feature_rows.append(
                    [
                        float(hour),
                        np.sin(2 * np.pi * hour / 24.0),
                        np.cos(2 * np.pi * hour / 24.0),
                        float(current_subbus_loads.loc[hour, subbus_id]),
                        float(current_subbus_loads.loc[prev_hour, subbus_id]),
                        float(current_parent_loads.loc[hour, parent]),
                        float(future_parent_proxy.loc[hour, parent]),
                        float(spec.base_share),
                        float(spec.capacity_mw),
                        float(("residential", "commercial", "industrial").index(spec.profile_name)),
                        float(int(parent.split()[-1])),
                    ]
                )

        x = np.asarray(feature_rows, dtype=float)
        predictions: dict[float, pd.DataFrame] = {}
        for quantile, model in self.models.items():
            values = np.maximum(model.predict(x), 0.0)
            frame = pd.DataFrame(0.0, index=current_subbus_loads.index, columns=current_subbus_loads.columns)
            for (hour, subbus_id), value in zip(row_keys, values, strict=True):
                frame.loc[hour, subbus_id] = value
            predictions[quantile] = frame
        return enforce_monotone_quantiles(predictions)

    def simulate_price_distribution(
        self,
        quantile_loads: dict[float, pd.DataFrame],
        opf_runner: OPFRunner,
        generator_costs: pd.DataFrame,
        branch_capacity_factor: pd.Series,
        n_samples: int = 20,
        seed: int = 7,
    ) -> ForecastBundle:
        q10 = quantile_loads[min(quantile_loads)]
        q50 = quantile_loads[sorted(quantile_loads)[len(quantile_loads) // 2]]
        q90 = quantile_loads[max(quantile_loads)]

        rng = np.random.default_rng(seed)
        sigma = np.maximum((q90 - q10).to_numpy(dtype=float) / (2 * 1.28155), 1e-3)
        mu = q50.to_numpy(dtype=float)

        bus_load_scenarios: list[pd.DataFrame] = []
        lmp_paths = []
        for _ in range(n_samples):
            sample = np.maximum(rng.normal(mu, sigma), 0.0)
            subbus_sample = pd.DataFrame(sample, index=q50.index, columns=q50.columns)
            parent_load = aggregate_subbus_to_parent(subbus_sample)
            bus_load_scenarios.append(parent_load)
            opf = opf_runner.run(parent_load, generator_costs, branch_capacity_factor)
            lmp_paths.append(opf.bus_lmp.to_numpy(dtype=float))

        lmp_array = np.stack(lmp_paths, axis=0)
        bus_lmp_quantiles = {
            q: pd.DataFrame(np.quantile(lmp_array, q, axis=0), index=q50.index, columns=opf_runner.case_data.bus_names)
            for q in quantile_loads
        }

        return ForecastBundle(
            subbus_load_quantiles=quantile_loads,
            bus_load_scenarios=bus_load_scenarios,
            bus_lmp_quantiles=bus_lmp_quantiles,
            scenario_metadata={"n_samples": n_samples},
        )


def aggregate_subbus_to_parent(subbus_loads: pd.DataFrame) -> pd.DataFrame:
    parents = sorted({column.split("-S", maxsplit=1)[0] for column in subbus_loads.columns}, key=lambda name: int(name.split()[-1]))
    aggregated = pd.DataFrame(0.0, index=subbus_loads.index, columns=parents)
    for column in subbus_loads.columns:
        parent = column.split("-S", maxsplit=1)[0]
        aggregated[parent] += subbus_loads[column]
    return aggregated


def enforce_monotone_quantiles(predictions: dict[float, pd.DataFrame]) -> dict[float, pd.DataFrame]:
    ordered = sorted(predictions)
    stack = np.stack([predictions[q].to_numpy(dtype=float) for q in ordered], axis=0)
    stack.sort(axis=0)
    return {
        quantile: pd.DataFrame(stack[idx], index=predictions[quantile].index, columns=predictions[quantile].columns)
        for idx, quantile in enumerate(ordered)
    }
