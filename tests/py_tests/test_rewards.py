import pytest

from neucode.rewards import WeightedMetricsReward, ISEOvershootReward 
from neucode.results import ExperimentResult

@pytest.fixture
def weighted_metrics_strategy() -> WeightedMetricsReward:
    """
    A fixture for our weighted metrics reward strategy.
    """
    return WeightedMetricsReward(weight_tracking=1.0, weight_effort=0.1, weight_overshoot=20.0, weight_speed=2.0, overshoot_threshold=5.0)

@pytest.fixture
def ise_overshoot_strategy():
    """
    A fixture for our ISE + Overshoot reward strategy.
    """
    return ISEOvershootReward(overshoot_penalty_multiplier=10.0, overshoot_threshold=5.0)

@pytest.fixture
def mock_success_result():
    """
    A mock result for a *successful* run.
    """
    return ExperimentResult(
        success=True, final_value=1.0, samples_written=1000,
        ise=10.0,
        isu=5.0,
        overshoot_percent=15.0,
        rise_time=1.2,
        iae=20.0,
        itae=30.0,
        total_time=20.0
    )

def test_ise_overshoot_reward_calculate_cost(ise_overshoot_strategy, mock_success_result):
    """
    Tests that the ISE + Overshoot cost calculation math is correct.
    """
    cost = ise_overshoot_strategy.calculate_cost(mock_success_result)
    
    # Cost = ISE + OvershootPenalty
    # Cost = 10.0 + 10.0 * (15.0-5.0)**2 = 10.0 + 10.0 * 100 = 1010.0
    assert cost == pytest.approx(1010.0)

def test_ise_overshoot_reward_calculate_reward(ise_overshoot_strategy, mock_success_result):
    """
    Tests that the main wrapper returns the correct (reward, loss) tuple.
    """
    reward, loss = ise_overshoot_strategy.calculate_reward(mock_success_result)
    
    assert loss == pytest.approx(1010.0)
    assert reward == pytest.approx(1.0 / (1010.0 + 1e-6))

def test_weighted_metrics_reward_calculate_cost(weighted_metrics_strategy, mock_success_result):
    """
    Tests that the weighted metrics cost calculation math is correct.
    """
    cost = weighted_metrics_strategy.calculate_cost(mock_success_result)

    # Cost = (weight_tracking * ITAE) + (weight_effort * ISU) + (weight_overshoot * OvershootPenalty) + (weight_speed * NormalizedRiseTime)
    # NormalizedRiseTime = 1.2 / 20.0 = 0.06
    # Cost = (1.0 * 30.0) + (0.1 * 5.0) + (20.0 * (15.0-5.0)**2) + (2.0 * 0.06)
    # Cost = 30.0 + 0.5 + (20.0 * 100) + 0.12 = 30.5 + 2000 + 0.12 = 2030.62
    assert cost == pytest.approx(2030.62)

def test_weighted_metrics_reward_calculate_reward(weighted_metrics_strategy, mock_success_result):
    """
    Tests that the main wrapper returns the correct (reward, loss) tuple.
    """
    reward, loss = weighted_metrics_strategy.calculate_reward(mock_success_result)
    
    assert loss == pytest.approx(2030.62)
    assert reward == pytest.approx(1.0 / (2030.62 + 1e-6))

def test_weighted_metrics_reward_calculate_cost(weighted_metrics_strategy, mock_success_result):
    cost = weighted_metrics_strategy.calculate_cost(mock_success_result)
    # Cost = (1.0 * 30.0) + (0.1 * 5.0) + (20.0 * (15.0-5.0)**2) + (2.0 * 0.06) = 2030.62
    
    assert cost == pytest.approx(2030.62)

def test_weighted_metrics_reward_calculate_reward_with_failure_penalty(weighted_metrics_strategy):
    """
    Tests that the failure penalty is correctly applied and reflected in the reward.
    """
    failed_result = ExperimentResult(success=False, final_value=0.0, samples_written=1000,
                                     steady_state_error_percent=15.0, itae=10.0, isu=1.0, overshoot_percent=0.0, total_time=10.0)
    reward, cost = weighted_metrics_strategy.calculate_reward(failed_result)
    # Cost = (weight_tracking * ITAE) + (weight_effort * ISU) + failure_penalty
    # Cost = (1.0 * 10.0) + (0.1 * 1.0) + 1000.0 = 10.0 + 0.1 + 1000.0 = 1010.1
    assert cost == pytest.approx(1010.1)