import pytest
from neucode.signals import StepSetpoint, RampSetpoint, SinSetpoint, Disturbance

def test_step_setpoint_init_success():
    """
    Test successful initialization of StepSetpoint with valid parameters.
    """
    setpoint = StepSetpoint(time=2.0, value=10.0)
    assert setpoint.config == {'type': 'step', 'step_time': 2.0, 'v': 10.0}

def test_ramp_setpoint_init_success():
    """
    Test successful initialization of RampSetpoint with valid parameters.
    """
    setpoint = RampSetpoint(start_time=1.0, duration=3.0, start_value=5.0, end_value=15.0)
    assert setpoint.config == {'type': 'ramp', 'step_time': 1.0, 'time': 3.0, 'a': 5.0, 'b': 15.0}

def test_sin_setpoint_init_success():
    """
    Test successful initialization of SinSetpoint with valid parameters.
    """
    setpoint = SinSetpoint(start_time=0.0, amplitude=2.0, frequency=0.5)
    assert setpoint.config == {'type': 'sin', 'step_time': 0.0, 'amp': 2.0, 'freq': 0.5}

def test_disturbance_init_defaults_success():
    """
    Test successful initialization of Disturbance with default parameters.
    - We don't check untouched parameters here.
    """
    disturbance = Disturbance()
    assert disturbance.config['noise_type'] == 'none'
    assert disturbance.config['seed'] == 0
    assert not disturbance.config['enable_input_step']
    assert not disturbance.config['enable_output_step']

def test_disturbance_init_input_step_success():
    """
    Test successful initialization of Disturbance that configures an input step.
    - We don't check untouched parameters here.
    """
    disturbance = Disturbance(input_step_time=5.0, input_step_value=0.5)
    assert disturbance.config['enable_input_step'] is True
    assert disturbance.config['input_step_at_s'] == 5.0
    assert disturbance.config['input_step_value'] == 0.5
    assert disturbance.config['enable_output_step'] is False

def test_disturbance_init_noise_success():
    """
    Test successful initialization of Disturbance with noise configuration.
    - We don't check untouched parameters here.
    """
    disturbance = Disturbance(noise_type='gaussian', noise_std=0.1, seed=42)
    assert disturbance.config['noise_type'] == 'gaussian'
    assert disturbance.config['noise_std'] == 0.1
    assert disturbance.config['seed'] == 42
    assert disturbance.config['enable_input_step'] is False

def test_step_setpoint_invalid_time_raises_error():
    with pytest.raises(ValueError, match="must be a non-negative number"):
        StepSetpoint(time=-1.0, value=5.0)

def test_step_setpoint_invalid_value_raises_error():
    with pytest.raises(TypeError, match="must be numeric"):
        StepSetpoint(time=1.0, value="not a number")

def test_ramp_setpoint_invalid_duration_raises_error():
    with pytest.raises(ValueError, match="must be a positive number"):
        RampSetpoint(start_time=1.0, duration=0.0, start_value=0.0, end_value=1.0)

def test_sin_setpoint_invalid_frequency_raises_error():
    with pytest.raises(ValueError, match="must be a positive number"):
        SinSetpoint(start_time=1.0, amplitude=1.0, frequency=0.0)

def test_disturbance_invalid_noise_type_raises_error():
    with pytest.raises(ValueError, match="Invalid noise type"):
        Disturbance(noise_type='some_bad_type')

def test_disturbance_negative_step_time_raises_error():
    with pytest.raises(ValueError, match="Input step time must be a non-negative number"):
        Disturbance(input_step_time=-1.0)

def test_disturbance_negative_noise_std_raises_error():
    with pytest.raises(ValueError, match="Noise standard deviation must be a non-negative number"):
        Disturbance(noise_type='gaussian', noise_std=-0.1)

def test_disturbance_negative_seed_raises_error():
    with pytest.raises(ValueError, match="Seed must be a non-negative integer"):
        Disturbance(seed=-1)