import pytest
import numpy as np

# Make torch an optional import for testing
try:
    import torch
    from neucode.architectures import NoContextMLPArchitecture
    torch_available = True
except ImportError:
    torch_available = False

from neucode import (
    SimulationHarness,
    PIDController,
    ANNController,
    StepSetpoint,
    FOPDTPlant,
    ExperimentResult
)

@pytest.fixture
def baseline_components():
    """
    Returns baseline components for testing the SimulationHarness.
    """
    return {
        'plant': FOPDTPlant(K=0.5, tau=4.284, theta=0.276),
        'controller': PIDController(kp=2.0, ki=1.0, kd=0.0),
        'setpoint': StepSetpoint(time=1.0, value=1.0),
        'actuator_limits': {'u_min': -10.0, 'u_max': 10.0}
    }

def test_harness_init_success(baseline_components):
    """
    Test successful initialization of SimulationHarness with valid components.
    """
    harness = SimulationHarness(
        plant=baseline_components['plant'],
        controller=baseline_components['controller'],
        setpoint=baseline_components['setpoint'],
        actuator_limits=baseline_components['actuator_limits']
    )
    assert isinstance(harness, SimulationHarness)
    assert harness.plant is baseline_components['plant']
    assert harness.controller is baseline_components['controller']
    assert harness.setpoint is baseline_components['setpoint']
    assert harness.disturbance is None
    assert harness.actuator_limits == baseline_components['actuator_limits']

def test_harness_init_invalid_components_raises_error():
    """
    Test that initializing SimulationHarness with invalid components raises an error.
    """
    class InvalidPlant:
        pass

    class InvalidController:
        pass

    class InvalidSetpoint:
        pass

    with pytest.raises(TypeError):
        SimulationHarness(
            plant=InvalidPlant(),
            controller=PIDController(kp=1.0, ki=0.5, kd=0.0),
            setpoint=StepSetpoint(time=1.0, value=1.0)
        )
    with pytest.raises(TypeError):
        SimulationHarness(
            plant=FOPDTPlant(K=1.0, tau=2.0, theta=0.5),
            controller=InvalidController(),
            setpoint=StepSetpoint(time=1.0, value=1.0)
        )
    with pytest.raises(TypeError):
        SimulationHarness(
            plant=FOPDTPlant(K=1.0, tau=2.0, theta=0.5),
            controller=PIDController(kp=1.0, ki=0.5, kd=0.0),
            setpoint=InvalidSetpoint()
        )

def test_harness_e2e_run_simulation_success(baseline_components):
    """
    Test end-to-end simulation run using SimulationHarness.
    -> For now just asserting a few key results.
    """
    harness = SimulationHarness(
        plant=baseline_components['plant'],
        controller=baseline_components['controller'],
        setpoint=baseline_components['setpoint'],
    )
    results = harness.evaluate(dt=0.01, total_time=60.0)
    assert isinstance(results, ExperimentResult), "The harness should return a SimulationResult instance."
    assert results.success is True, "The simulation should be marked as successful."
    assert results.final_value == pytest.approx(1.0, abs=0.02), "The final value {} should be close to the setpoint value of 1.0.".format(results.final_value)
    assert results.overshoot_percent < 15.0, "Overshoot should be less than 15.0%."
    assert results.overshoot_percent > 0, "Overshoot should be greater than 0%."

    assert results.ise > 0.1, "ISE metric was not populated correctly."
    assert results.iae > 0.1, "IAE metric was not populated correctly."
    assert results.itae > 0.1, "ITAE metric was not populated correctly."
    assert results.isu > 0.1, "ISU metric was not populated correctly."

def test_harness_e2e_run_simulation_timeseries_success(baseline_components):
    """
    Test end-to-end simulation run using SimulationHarness with time-series output.
    """
    harness = SimulationHarness(
        plant=baseline_components['plant'],
        controller=baseline_components['controller'],
        setpoint=baseline_components['setpoint'],
    )
    total_time = 10.0
    dt = 0.01
    results = harness.run(dt=dt, total_time=total_time, get_time_series=True)
    assert isinstance(results, ExperimentResult), "The harness should return a SimulationResult instance."
    assert results.success is True, "The simulation should be marked as successful."
    
    assert results.time_series is not None
    assert isinstance(results.time_series, dict)
    
    expected_keys = ['time', 'setpoint', 'measurement', 'control_effort']
    for key in expected_keys:
        assert key in results.time_series, f"Time series dictionary should contain the key '{key}'."

    for key in expected_keys:
        assert isinstance(results.time_series[key], np.ndarray), f"'{key}' should be a NumPy array."
    
    # Check that all arrays have the correct length
    expected_samples = int(total_time / dt) + 1
    for key in expected_keys:
        assert len(results.time_series[key]) == expected_samples, f"'{key}' array should have {expected_samples} samples."
    
    # 4. Sanity check a final value from the time series
    assert results.time_series['measurement'][-1] == pytest.approx(results.final_value, abs=1e-6), "Final measurement from time series should match final_value."

@pytest.fixture
def dummy_ann_files(tmp_path):
    """
    Pytest fixture to create a dummy model and scaler file for testing ANNController.
    This fixture will be skipped if PyTorch is not installed.
    """
    if not torch_available:
        pytest.skip("PyTorch not installed, skipping ANN controller test.")

    model_path = tmp_path / "dummy_model.pth"
    scaler_path = tmp_path / "dummy_scaler.npz"

    dummy_model = NoContextMLPArchitecture(input_size=5)
    torch.save(dummy_model.state_dict(), model_path)

    dummy_mean = np.zeros((5,), dtype=np.float32)
    dummy_scale = np.ones((5,), dtype=np.float32)
    np.savez(scaler_path, mean=dummy_mean, scale=dummy_scale)

    return model_path, scaler_path

def test_harness_with_ann_controller(dummy_ann_files, baseline_components):
    """
    Verifies that the SimulationHarness can run a simulation using an ANNController
    in the hybrid Python/C step-by-step mode.
    """
    model_path, scaler_path = dummy_ann_files

    harness = SimulationHarness(
        plant=baseline_components['plant'],
        controller=ANNController(model_path=model_path, scaler_path=scaler_path),
        setpoint=baseline_components['setpoint']
    )
    result = harness.run(dt=0.01, total_time=5.0)

    assert result.success
    assert result.ise > 0  # Check that some metrics were calculated