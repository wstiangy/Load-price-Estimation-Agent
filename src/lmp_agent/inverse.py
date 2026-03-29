from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from .config import BusLoadEstimate
from .data import build_hour_features


@dataclass
class InverseLoadEstimator:
    surrogate_random_state: int = 7
    price_penalty: float = 1.0
    init_penalty: float = 0.8
    prior_penalty: float = 0.35
    smooth_penalty: float = 0.20

    def __post_init__(self) -> None:
        self.bus_names: list[str] = []
        self.price_names: list[str] = []
        self.surrogate = MultiOutputRegressor(
            GradientBoostingRegressor(
                loss="squared_error",
                n_estimators=180,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.surrogate_random_state,
            )
        )
        self.sensitivity_model = Ridge(alpha=1.5, fit_intercept=True)

    def fit(self, bus_load_history: list[pd.DataFrame], lmp_history: list[pd.DataFrame]) -> None:
        self.bus_names = list(bus_load_history[0].columns)
        self.price_names = list(lmp_history[0].columns)
        feature_rows = []
        target_rows = []
        linear_x = []
        linear_y = []
        for bus_loads, lmps in zip(bus_load_history, lmp_history, strict=True):
            hour_features = build_hour_features(bus_loads.index)
            day_scale = bus_loads.sum(axis=1).mean()
            for hour in bus_loads.index:
                feature_rows.append(
                    np.concatenate(
                        [
                            lmps.loc[hour, self.price_names].to_numpy(dtype=float),
                            hour_features.loc[hour].to_numpy(dtype=float),
                            np.array([float(day_scale)]),
                        ]
                    )
                )
                target_rows.append(bus_loads.loc[hour, self.bus_names].to_numpy(dtype=float))
                linear_x.append(bus_loads.loc[hour, self.bus_names].to_numpy(dtype=float))
                linear_y.append(lmps.loc[hour, self.price_names].to_numpy(dtype=float))

        self.surrogate.fit(np.asarray(feature_rows, dtype=float), np.asarray(target_rows, dtype=float))
        self.sensitivity_model.fit(np.asarray(linear_x, dtype=float), np.asarray(linear_y, dtype=float))

    def estimate(self, observed_lmp: pd.DataFrame, prior_load: pd.DataFrame) -> BusLoadEstimate:
        feature_matrix = self._build_feature_matrix(observed_lmp, prior_load)
        surrogate_init = pd.DataFrame(
            self.surrogate.predict(feature_matrix),
            index=observed_lmp.index,
            columns=self.bus_names,
        ).clip(lower=0.0)
        refined = self._refine_with_inverse_optimization(observed_lmp, surrogate_init, prior_load)
        predicted_price = pd.DataFrame(
            self.sensitivity_model.predict(refined.to_numpy(dtype=float)),
            index=refined.index,
            columns=self.price_names,
        )
        residual = observed_lmp[self.price_names] - predicted_price
        return BusLoadEstimate(
            estimated_bus_load=refined,
            surrogate_init=surrogate_init,
            refined_solution=refined,
            price_residual=residual,
        )

    def _build_feature_matrix(self, observed_lmp: pd.DataFrame, prior_load: pd.DataFrame) -> np.ndarray:
        hour_features = build_hour_features(observed_lmp.index)
        prior_total = prior_load.sum(axis=1)
        rows = []
        for hour in observed_lmp.index:
            rows.append(
                np.concatenate(
                    [
                        observed_lmp.loc[hour, self.price_names].to_numpy(dtype=float),
                        hour_features.loc[hour].to_numpy(dtype=float),
                        np.array([float(prior_total.loc[hour])]),
                    ]
                )
            )
        return np.asarray(rows, dtype=float)

    def _refine_with_inverse_optimization(
        self,
        observed_lmp: pd.DataFrame,
        surrogate_init: pd.DataFrame,
        prior_load: pd.DataFrame,
    ) -> pd.DataFrame:
        y = observed_lmp[self.price_names].to_numpy(dtype=float)
        x0 = surrogate_init[self.bus_names].to_numpy(dtype=float)
        prior = prior_load[self.bus_names].to_numpy(dtype=float)
        coef = np.asarray(self.sensitivity_model.coef_, dtype=float)
        intercept = np.asarray(self.sensitivity_model.intercept_, dtype=float)

        horizon, n_bus = x0.shape
        x = cp.Variable((horizon, n_bus), nonneg=True)
        price_error = x @ coef.T + intercept - y
        objective = (
            self.price_penalty * cp.sum_squares(price_error)
            + self.init_penalty * cp.sum_squares(x - x0)
            + self.prior_penalty * cp.sum_squares(x - prior)
        )
        if horizon > 1:
            objective += self.smooth_penalty * cp.sum_squares(x[1:, :] - x[:-1, :])

        problem = cp.Problem(cp.Minimize(objective))
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if x.value is None:
            return surrogate_init
        return pd.DataFrame(np.maximum(x.value, 0.0), index=observed_lmp.index, columns=self.bus_names)
