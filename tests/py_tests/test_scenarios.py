import pytest
from neucode import (
    SimulationHarness,
    PIDController,
    FOPDTPlant,
    ExperimentResult
)

from neucode.scenarios import StepBenchmark

@pytest.fixture
def configured_harness():
    """
    Returns pre-configured SimulationHarness for testing.
    """
    plant = FOPDTPlant(K=0.5, tau=4.284, theta=0.276)
    controller = PIDController(kp=2.0, ki=1.0, kd=0.0)

    harness = SimulationHarness(
        plant=plant,
        controller=controller,
        setpoint=None,
        actuator_limits={'u_min': -10.0, 'u_max': 10.0}
    )
    return harness

def test_base_class_init_raises_error():
    """
    Test that validation in BaseScenario directly raises NotImplementedError.
    """
    with pytest.raises(ValueError, match="must be a positive number"):
        StepBenchmark(dt=0.0, total_time=10.0)
    
    with pytest.raises(ValueError, match="must be a positive number"):
        StepBenchmark(dt=0.01, total_time=0.0)
    
def test_large_step_benchmark_init_raises_error():
    """
    Tests the validation logic in the LargeStepBenchmark constructor.
    """
    # Test step_time > total_time
    with pytest.raises(ValueError, match="step_time must be within the simulation duration"):
        StepBenchmark(dt=0.01, total_time=10.0, step_time=11.0)
    
    # Test step_time < 0
    with pytest.raises(ValueError, match="step_time must be within the simulation duration"):
        StepBenchmark(dt=0.01, total_time=10.0, step_time=-1.0)

def test_large_step_benchmark_run_success(configured_harness):
    """
    Test running the LargeStepBenchmark scenario successfully.
    """
    harness = configured_harness

    benchmark = StepBenchmark(dt=0.01, total_time=20.0, step_value=5.0)
    result = benchmark.run(harness=harness)

    assert isinstance(result, ExperimentResult), "The result should be an instance of SimulationResult."
    assert result.success is True, "The simulation should be marked as successful."

    assert result.final_value == pytest.approx(5.0, rel=0.25), "The final value should match the expected step value."
    assert result.overshoot_percent is not None, "Overshoot percentage should be calculated."