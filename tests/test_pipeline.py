import pandas as pd

from lmp_agent.agent import PricingWorkflowAgent
from lmp_agent.config import RunConfig
from lmp_agent.data import SyntheticScenarioGenerator, load_ieee14_case
from lmp_agent.disaggregation import LoadDisaggregator
from lmp_agent.opf import OPFRunner


def test_case_loader_has_expected_shape():
    case_data = load_ieee14_case(seed=11)
    assert len(case_data.bus_names) == 14
    assert len(case_data.generator_names) == 5
    assert len(case_data.branch_names) == 20
    assert set(case_data.subbus_specs) == set(case_data.bus_names)


def test_subbus_aggregation_is_exact():
    case_data = load_ieee14_case(seed=7)
    generator = SyntheticScenarioGenerator(case_data, seed=7)
    scenario = generator.generate_day(day_index=0)
    disaggregator = LoadDisaggregator(case_data)
    aggregated = disaggregator.aggregate_to_parent(scenario.subbus_loads, case_data.bus_names)
    pd.testing.assert_frame_equal(aggregated, scenario.bus_loads, check_exact=False, atol=1e-8, rtol=1e-8)


def test_opf_returns_all_bus_lmps():
    case_data = load_ieee14_case(seed=7)
    generator = SyntheticScenarioGenerator(case_data, seed=7)
    scenario = generator.generate_day(day_index=1)
    opf = OPFRunner(case_data).run(
        bus_loads=scenario.bus_loads,
        generator_costs=scenario.generator_costs,
        branch_capacity_factor=scenario.branch_capacity_factor,
    )
    assert opf.feasible
    assert list(opf.bus_lmp.columns) == case_data.bus_names
    assert opf.bus_lmp.shape == (24, 14)


def test_end_to_end_pipeline_runs():
    config = RunConfig(training_days=6, monte_carlo_samples=6, seed=5, noise_scale=0.06)
    artifacts = PricingWorkflowAgent(config).run_pipeline()
    assert artifacts.current_opf.feasible
    assert artifacts.future_truth_opf.feasible
    assert artifacts.bus_estimate.estimated_bus_load.shape == (24, 14)
    assert artifacts.subbus_estimate.subbus_loads.shape[0] == 24
    assert 0.5 in artifacts.forecast.subbus_load_quantiles
    assert 0.5 in artifacts.forecast.bus_lmp_quantiles
    assert "inverse_bus_mape" in artifacts.metrics
