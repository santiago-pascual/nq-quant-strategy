import pytest
from paper_dashboard.alpha_monitor import MonitoringPlan, completed_looks, reference_distribution, drawdown


def test_one_trade_remains_insufficient_and_no_live_actions():
    result = completed_looks([-1.005])
    assert result['state'] == 'INSUFFICIENT_DATA' and not result['looks']
    assert result['trading_actions'] == [] and not result['formal_significance_enabled']


def test_future_appends_never_revise_completed_disjoint_looks():
    plan = MonitoringPlan(trades_per_look=5, maximum_looks=2, block_length=2)
    before = completed_looks([1,-1,1,0,-1], plan=plan)
    after = completed_looks([1,-1,1,0,-1]+[100]*30, plan=plan)
    assert before['looks'][0] == after['looks'][0]
    assert len(after['looks']) == 2 and after['maximum_looks_reached']


def test_reference_is_deterministic_and_discloses_tail_precision():
    plan = MonitoringPlan()
    first = reference_distribution([1,-1,.5,-.3]*100, plan=plan)
    assert first == reference_distribution([1,-1,.5,-.3]*100, plan=plan)
    assert not first['coverage_guaranteed']
    assert not first['bands']['expectancy_r']['tail_resolution_sufficient']
    assert first['nominal_per_metric_look_alpha'] == .05/(5*10*4)
    assert drawdown([-1,2,-3]) == -3


def test_invalid_data_and_plan_rejected():
    with pytest.raises(ValueError): completed_looks([float('nan')])
    with pytest.raises(ValueError): MonitoringPlan(trades_per_look=0).identity()
