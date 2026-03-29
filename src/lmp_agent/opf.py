from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pypsa

from .config import OPFResult
from .data import CaseData


@dataclass(slots=True)
class OPFRunner:
    case_data: CaseData
    solver_name: str = "highs"

    def run(
        self,
        bus_loads: pd.DataFrame,
        generator_costs: pd.DataFrame,
        branch_capacity_factor: pd.Series | None = None,
    ) -> OPFResult:
        network = self._build_network(bus_loads=bus_loads, generator_costs=generator_costs, branch_capacity_factor=branch_capacity_factor)
        status, termination = network.optimize(
            solver_name=self.solver_name,
            assign_all_duals=True,
            include_objective_constant=False,
            log_to_console=False,
        )
        feasible = status == "ok" and termination == "optimal"

        line_flows = []
        if not network.lines.empty:
            line_component = network.lines_t.p0.copy()
            line_component.columns = [f"Line::{name}" for name in line_component.columns]
            line_flows.append(line_component)
        flow_frame = pd.concat(line_flows, axis=1) if line_flows else pd.DataFrame(index=bus_loads.index)

        return OPFResult(
            bus_lmp=network.buses_t.marginal_price.copy(),
            gen_dispatch=network.generators_t.p.copy(),
            line_flows=flow_frame,
            congestion_flags=self._congestion_flags(network, flow_frame),
            feasible=feasible,
            objective=float(network.objective if network.objective is not None else np.nan),
            metadata={"status": status, "termination": termination},
        )

    def _build_network(
        self,
        bus_loads: pd.DataFrame,
        generator_costs: pd.DataFrame,
        branch_capacity_factor: pd.Series | None = None,
    ) -> pypsa.Network:
        snapshots = bus_loads.index
        network = pypsa.Network()
        network.set_snapshots(snapshots)
        network.add("Carrier", "AC")

        for bus_name in self.case_data.bus_names:
            network.add("Bus", bus_name, carrier="AC")

        for _, row in self.case_data.generator_table.iterrows():
            pmax = float(row["pmax"])
            pmin = float(row["pmin"])
            network.add(
                "Generator",
                row["name"],
                bus=f"Bus {int(row['gen_bus'])}",
                p_nom=pmax,
                p_min_pu=max(0.0, pmin / pmax if pmax > 0 else 0.0),
                marginal_cost=generator_costs[row["name"]].to_numpy(dtype=float),
            )

        for bus_name in self.case_data.bus_names:
            network.add("Load", f"Load::{bus_name}", bus=bus_name, p_set=bus_loads[bus_name].to_numpy(dtype=float))

        capacity_factor = branch_capacity_factor if branch_capacity_factor is not None else pd.Series(1.0, index=self.case_data.branch_names)
        for _, row in self.case_data.branch_table.iterrows():
            name = row["name"]
            limit = float(row["base_limit"]) * float(capacity_factor.get(name, 1.0))
            common = {
                "bus0": f"Bus {int(row['f_bus'])}",
                "bus1": f"Bus {int(row['t_bus'])}",
                "x": max(abs(float(row["x"])), 1e-4),
                "r": max(abs(float(row["r"])), 1e-4),
                "s_nom": max(limit, 1.0),
                "carrier": "AC",
            }
            network.add("Line", name, **common)

        return network

    def _congestion_flags(self, network: pypsa.Network, flow_frame: pd.DataFrame) -> pd.DataFrame:
        limits = {}
        for name, row in network.lines.iterrows():
            limits[f"Line::{name}"] = float(row["s_nom"])
        if flow_frame.empty:
            return pd.DataFrame(index=network.snapshots)

        congestion = pd.DataFrame(index=flow_frame.index)
        for column in flow_frame.columns:
            limit = max(limits[column], 1.0)
            congestion[column] = (flow_frame[column].abs() / limit) >= 0.97
        return congestion
