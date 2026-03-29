from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import SubBusAllocation
from .data import PROFILE_LIBRARY, CaseData


@dataclass(slots=True)
class LoadDisaggregator:
    case_data: CaseData
    shape_penalty: float = 1.0
    smooth_penalty: float = 0.25
    share_penalty: float = 0.10

    def disaggregate(self, parent_bus_load: pd.DataFrame) -> SubBusAllocation:
        horizon = parent_bus_load.index
        columns = [spec.subbus_id for specs in self.case_data.subbus_specs.values() for spec in specs]
        estimates = pd.DataFrame(0.0, index=horizon, columns=columns)

        for bus_name, specs in self.case_data.subbus_specs.items():
            parent_series = parent_bus_load[bus_name].to_numpy(dtype=float)
            if np.allclose(parent_series, 0.0):
                continue
            template_matrix = np.vstack([PROFILE_LIBRARY[spec.profile_name] * spec.base_share for spec in specs])
            template_matrix = template_matrix / template_matrix.sum(axis=0, keepdims=True)
            prior = (template_matrix * parent_series).T

            x = cp.Variable((len(horizon), len(specs)), nonneg=True)
            constraints = [cp.sum(x, axis=1) == parent_series]
            for idx, spec in enumerate(specs):
                constraints.append(x[:, idx] <= spec.max_share * parent_series + 1e-6)
                constraints.append(x[:, idx] >= spec.min_share * parent_series - 1e-6)
                constraints.append(x[:, idx] <= spec.capacity_mw)

            share_prior = np.array([spec.base_share for spec in specs], dtype=float)
            current_share = cp.multiply(1.0 / np.maximum(parent_series[:, None], 1e-3), x)
            objective = self.shape_penalty * cp.sum_squares(x - prior) + self.share_penalty * cp.sum_squares(current_share - share_prior)
            if len(horizon) > 1:
                objective += self.smooth_penalty * cp.sum_squares(x[1:, :] - x[:-1, :])

            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            solution = x.value if x.value is not None else prior
            for idx, spec in enumerate(specs):
                estimates[spec.subbus_id] = np.maximum(solution[:, idx], 0.0)

        aggregated = self.aggregate_to_parent(estimates, parent_bus_load.columns.tolist())
        balance_gap = aggregated - parent_bus_load[parent_bus_load.columns]
        return SubBusAllocation(subbus_loads=estimates, parent_bus_load=parent_bus_load.copy(), balance_gap=balance_gap)

    def aggregate_to_parent(self, subbus_loads: pd.DataFrame, parent_order: list[str] | None = None) -> pd.DataFrame:
        parent_order = parent_order or list(self.case_data.subbus_specs.keys())
        aggregated = pd.DataFrame(0.0, index=subbus_loads.index, columns=parent_order)
        for parent, specs in self.case_data.subbus_specs.items():
            aggregated[parent] = subbus_loads[[spec.subbus_id for spec in specs]].sum(axis=1)
        return aggregated[parent_order]
