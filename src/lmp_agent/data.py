from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from pypower.case14 import case14
from pypower.idx_brch import BR_R, BR_STATUS, BR_X, RATE_A, TAP, T_BUS, F_BUS
from pypower.idx_bus import BUS_I, PD
from pypower.idx_gen import GEN_BUS, PMAX, PMIN


DAILY_PROFILE = np.array(
    [
        0.63,
        0.60,
        0.58,
        0.57,
        0.58,
        0.64,
        0.72,
        0.81,
        0.88,
        0.91,
        0.95,
        0.98,
        1.00,
        0.99,
        0.96,
        0.94,
        0.97,
        1.04,
        1.09,
        1.12,
        1.06,
        0.96,
        0.82,
        0.70,
    ]
)

PROFILE_LIBRARY = {
    "residential": np.array(
        [0.56, 0.52, 0.49, 0.47, 0.50, 0.60, 0.73, 0.81, 0.84, 0.82, 0.79, 0.76, 0.75, 0.75, 0.78, 0.85, 0.98, 1.10, 1.18, 1.20, 1.12, 0.95, 0.77, 0.64]
    ),
    "commercial": np.array(
        [0.36, 0.34, 0.34, 0.35, 0.42, 0.58, 0.76, 0.92, 1.04, 1.12, 1.16, 1.19, 1.20, 1.18, 1.13, 1.07, 1.00, 0.92, 0.78, 0.63, 0.50, 0.43, 0.39, 0.37]
    ),
    "industrial": np.array(
        [0.86, 0.85, 0.84, 0.83, 0.84, 0.87, 0.92, 0.96, 1.00, 1.03, 1.06, 1.08, 1.09, 1.09, 1.08, 1.06, 1.03, 1.00, 0.97, 0.95, 0.93, 0.91, 0.89, 0.87]
    ),
}

SUBBUS_PROFILE_ORDER = ("residential", "commercial", "industrial")


@dataclass(slots=True)
class SubBusSpec:
    subbus_id: str
    parent_bus: str
    base_share: float
    profile_name: str
    min_share: float
    max_share: float
    capacity_mw: float


@dataclass(slots=True)
class CaseData:
    base_mva: float
    bus_table: pd.DataFrame
    generator_table: pd.DataFrame
    branch_table: pd.DataFrame
    gencost_table: pd.DataFrame
    bus_names: list[str]
    generator_names: list[str]
    branch_names: list[str]
    subbus_specs: dict[str, list[SubBusSpec]]


@dataclass(slots=True)
class SyntheticScenario:
    bus_loads: pd.DataFrame
    subbus_loads: pd.DataFrame
    generator_costs: pd.DataFrame
    branch_capacity_factor: pd.Series
    metadata: dict[str, float | int | str]


def _bus_name(bus_id: int) -> str:
    return f"Bus {bus_id}"


def _template_for(bus_idx: int, sub_idx: int) -> str:
    return SUBBUS_PROFILE_ORDER[(bus_idx + sub_idx) % len(SUBBUS_PROFILE_ORDER)]


def _derive_branch_limit(row: pd.Series) -> float:
    rate = float(row["rate_a"])
    if 0 < rate < 5000:
        return rate
    return 300.0


def load_ieee14_case(seed: int = 7) -> CaseData:
    raw = case14()
    rng = np.random.default_rng(seed)

    bus_table = pd.DataFrame(raw["bus"]).rename(columns={BUS_I: "bus_i", PD: "pd"})
    bus_table["bus_i"] = bus_table["bus_i"].astype(int)
    bus_table["name"] = bus_table["bus_i"].map(_bus_name)

    generator_table = pd.DataFrame(raw["gen"]).rename(columns={GEN_BUS: "gen_bus", PMAX: "pmax", PMIN: "pmin"})
    generator_table["gen_bus"] = generator_table["gen_bus"].astype(int)
    generator_table["name"] = [f"G{i+1}" for i in range(len(generator_table))]

    branch_table = pd.DataFrame(raw["branch"]).rename(
        columns={F_BUS: "f_bus", T_BUS: "t_bus", BR_R: "r", BR_X: "x", RATE_A: "rate_a", TAP: "tap", BR_STATUS: "status"}
    )
    branch_table["f_bus"] = branch_table["f_bus"].astype(int)
    branch_table["t_bus"] = branch_table["t_bus"].astype(int)
    branch_table["name"] = [f"BR{i+1}" for i in range(len(branch_table))]
    branch_table["is_transformer"] = False
    branch_table["base_limit"] = branch_table.apply(_derive_branch_limit, axis=1)

    gencost_table = pd.DataFrame(raw["gencost"])
    gencost_table["linear_cost"] = gencost_table.iloc[:, 5].astype(float)
    gencost_table["name"] = generator_table["name"]

    subbus_specs: dict[str, list[SubBusSpec]] = {}
    for bus_idx, row in bus_table.iterrows():
        parent = row["name"]
        count = 2 + (bus_idx % 3)
        if float(row["pd"]) <= 0:
            count = 2
        raw_weights = rng.uniform(0.7, 1.3, size=count)
        shares = raw_weights / raw_weights.sum()
        specs: list[SubBusSpec] = []
        for sub_idx in range(count):
            share = float(shares[sub_idx])
            specs.append(
                SubBusSpec(
                    subbus_id=f"{parent}-S{sub_idx+1}",
                    parent_bus=parent,
                    base_share=share,
                    profile_name=_template_for(bus_idx, sub_idx),
                    min_share=max(0.02, 0.45 * share),
                    max_share=min(0.95, 1.80 * share + 0.06),
                    capacity_mw=max(5.0, float(row["pd"]) * (share * 1.6 + 0.15)),
                )
            )
        subbus_specs[parent] = specs

    return CaseData(
        base_mva=float(raw["baseMVA"]),
        bus_table=bus_table,
        generator_table=generator_table,
        branch_table=branch_table,
        gencost_table=gencost_table,
        bus_names=bus_table["name"].tolist(),
        generator_names=generator_table["name"].tolist(),
        branch_names=branch_table["name"].tolist(),
        subbus_specs=subbus_specs,
    )


class SyntheticScenarioGenerator:
    def __init__(self, case_data: CaseData, seed: int = 7):
        self.case_data = case_data
        self.seed = seed

    def generate_history(self, days: int, demand_scale: float = 1.0, noise_scale: float = 0.08) -> list[SyntheticScenario]:
        return [self.generate_day(day_index=day, demand_scale=demand_scale, noise_scale=noise_scale) for day in range(days)]

    def generate_day(self, day_index: int, demand_scale: float = 1.0, noise_scale: float = 0.08) -> SyntheticScenario:
        rng = np.random.default_rng(self.seed + day_index * 97)
        hours = pd.RangeIndex(24, name="hour")
        daily_scale = demand_scale * (1.0 + 0.08 * np.sin(2 * np.pi * day_index / 14.0) + rng.normal(0.0, 0.03))
        stress = float(np.clip(0.55 + 0.35 * np.cos(2 * np.pi * day_index / 9.0) + rng.normal(0, 0.08), 0.15, 1.0))
        fuel_shock = 1.0 + 0.10 * np.sin(2 * np.pi * day_index / 10.0) + rng.normal(0.0, 0.03)

        bus_loads = pd.DataFrame(index=hours, columns=self.case_data.bus_names, dtype=float)
        subbus_loads = pd.DataFrame(
            index=hours,
            columns=[spec.subbus_id for specs in self.case_data.subbus_specs.values() for spec in specs],
            dtype=float,
        )

        per_bus_shape_noise = rng.normal(0.0, noise_scale * 0.35, size=(24, len(self.case_data.bus_names)))
        for bus_pos, bus_name in enumerate(self.case_data.bus_names):
            base_pd = float(self.case_data.bus_table.loc[self.case_data.bus_table["name"] == bus_name, "pd"].iloc[0])
            if base_pd <= 0:
                bus_loads[bus_name] = 0.0
                for spec in self.case_data.subbus_specs[bus_name]:
                    subbus_loads[spec.subbus_id] = 0.0
                continue

            bus_shape = DAILY_PROFILE * (1.0 + 0.05 * np.sin((bus_pos + 1) * np.linspace(0.0, 2.6, 24)))
            bus_shape = np.clip(bus_shape + per_bus_shape_noise[:, bus_pos], 0.35, None)
            if bus_name in {"Bus 4", "Bus 5", "Bus 9"}:
                bus_shape[17:21] *= 1.0 + 0.32 * stress
            if bus_name in {"Bus 10", "Bus 11", "Bus 13"}:
                bus_shape[10:15] *= 1.0 + 0.22 * stress
            hourly_load = np.clip(base_pd * daily_scale * bus_shape, 0.0, None)
            bus_loads[bus_name] = hourly_load

            parent_specs = self.case_data.subbus_specs[bus_name]
            raw_profiles = []
            for spec in parent_specs:
                profile = PROFILE_LIBRARY[spec.profile_name]
                volatility = rng.normal(0.0, noise_scale * 0.45, size=24)
                raw_profiles.append(np.clip(spec.base_share * profile * (1.0 + volatility), 0.01, None))
            raw_profiles_arr = np.vstack(raw_profiles)
            raw_profiles_arr = raw_profiles_arr / raw_profiles_arr.sum(axis=0, keepdims=True)
            for idx, spec in enumerate(parent_specs):
                subbus_loads[spec.subbus_id] = hourly_load * raw_profiles_arr[idx]

        generator_costs = pd.DataFrame(index=hours, columns=self.case_data.generator_names, dtype=float)
        for gen_name, base_cost in zip(self.case_data.generator_names, self.case_data.gencost_table["linear_cost"].astype(float).tolist(), strict=True):
            hourly = np.clip(
                base_cost * fuel_shock * (1.0 + 0.03 * np.sin(np.linspace(0, 2 * np.pi, 24)) + rng.normal(0, 0.01, 24)),
                0.1,
                None,
            )
            generator_costs[gen_name] = hourly

        branch_capacity_factor = pd.Series(1.0, index=self.case_data.branch_names, dtype=float)
        stressed = self.case_data.branch_table.loc[self.case_data.branch_table["name"].isin(["BR4", "BR5", "BR7", "BR9", "BR13"]), "name"]
        for branch in stressed:
            branch_capacity_factor.loc[branch] = float(np.clip(1.0 - 0.35 * stress + rng.normal(0.0, 0.03), 0.45, 1.0))

        return SyntheticScenario(
            bus_loads=bus_loads,
            subbus_loads=subbus_loads,
            generator_costs=generator_costs,
            branch_capacity_factor=branch_capacity_factor,
            metadata={"day_index": day_index, "daily_scale": float(daily_scale), "stress": float(stress), "fuel_shock": float(fuel_shock)},
        )


def build_hour_features(hours: Iterable[int]) -> pd.DataFrame:
    hour_index = pd.Index(list(hours), name="hour")
    radians = 2 * np.pi * hour_index.to_numpy(dtype=float) / 24.0
    return pd.DataFrame(
        {"hour": hour_index.to_numpy(dtype=float), "sin_hour": np.sin(radians), "cos_hour": np.cos(radians)},
        index=hour_index,
    )
