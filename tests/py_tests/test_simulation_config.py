from neucode.simcore import Simulation
import pytest

@pytest.fixture
def sim_instance():
    """
    Fixture to create and destroy a Simulation object for testing.
    """
    try:
        sim = Simulation()
        yield sim
    except Exception as e:
        pytest.fail(f"Failed to create Simulation instance: {e}")

def test_config_set_time_step_success(sim_instance):
    """
    Test that setting a valid time step and total simulation time works without exceptions.
    """
    try:
        sim_instance.set_time_step(0.1, 10.0)
    except Exception as e:
        pytest.fail(f"set_time_step raised an exception unexpectedly: {e}")

def test_config_set_time_step_invalid_arguments(sim_instance):
    """
    Test that setting invalid time step or total simulation time raises ValueError.
    """
    with pytest.raises(ValueError, match="Invalid arguments for setting time step"):
        sim_instance.set_time_step(-0.1, 10.0)  # Invalid dt
    with pytest.raises(ValueError, match="Invalid arguments for setting time step"):
        sim_instance.set_time_step(0.1, -10.0)  # Invalid total_time
    with pytest.raises(ValueError, match="Invalid arguments for setting time step"):
        sim_instance.set_time_step(0.0, 0.0)    # Both invalid
    with pytest.raises(ValueError, match="Invalid arguments for setting time step"):
        sim_instance.set_time_step(0.1, float('nan'))  # total_time is NaN
    with pytest.raises(ValueError, match="Invalid arguments for setting time step"):
        sim_instance.set_time_step(float('nan'), 10.0)  # dt is NaN

def test_set_pid_gains_only_success(sim_instance):
    """
    Test setting PID gains without optional limits.
    """
    gains = {'kp': 2.0, 'ki': 1.0, 'kp': 0.05}
    sim_instance.set_pid(gains=gains)

def test_set_pid_gains_with_limits_success(sim_instance):
    """
    Test setting PID gains with limits.
    """
    gains = {'kp': 5.0, 'ki': 2.5}
    limits = {'u_min': -10.0, 'u_max':10.0}
    sim_instance.set_pid(gains=gains, limits=limits)

def test_set_pid_invalid_gains_error(sim_instance):
    """
    Test that setting PID controller gains with invalid values raises an appropriate error.
    """
    with pytest.raises((AttributeError, ValueError)):
        sim_instance.set_pid(gains=None)


def test_set_fopdt_success(sim_instance):
    """
    Test that valid FOPDT parameters can be set.
    """
    sim_instance.set_time_step(dt=0.01, total_time=10.0)
    params = {'K': 0.5, 'tau': 4.284, 'theta': 0.276}
    sim_instance.set_fopdt(params=params);

def test_set_fopdt_invalid_params_error(sim_instance):
    """
    Test that setting FOPDT parameters with invalid values raises an appropriate error.
    """
    sim_instance.set_time_step(dt=0.01, total_time=10.0)
    with pytest.raises((AttributeError, ValueError)):
        sim_instance.set_fopdt(params=None)
    with pytest.raises((AttributeError, ValueError)):
        sim_instance.set_fopdt(params={'K': 1.0, 'tau': -4.0, 'theta': 0.5})  # Invalid tau
    with pytest.raises((AttributeError, ValueError)):
        sim_instance.set_fopdt(params={'K': 1.0, 'tau': 4.0, 'theta': -0.5})  # Invalid theta

def test_set_setpoint_step_success(sim_instance):
    """
    Test setting a step setpoint.
    """
    sim_instance.set_time_step(dt=0.1, total_time=10.0)
    step_sp_def = {'type': 'step', 'step_time': 1.0, 'v': 5.0}
    sim_instance.set_setpoint(setpoint_config=step_sp_def)

def test_set_setpoint_ramp_success(sim_instance):
    """
    Test setting a ramp setpoint.
    """
    sim_instance.set_time_step(dt=0.1, total_time=10.0)
    ramp_sp_def = {'type': 'ramp', 'step_time': 2.0, 'a': 0.0, 'b': 10.0, 'time': 5.0}
    sim_instance.set_setpoint(setpoint_config=ramp_sp_def)

def test_set_setpoint_sin_success(sim_instance):
    """
    Test setting a sine setpoint.
    """
    sim_instance.set_time_step(dt=0.1, total_time=10.0)
    sine_sp_def = {'type': 'sin', 'step_time': 0.5, 'amp': 2.0, 'freq': 0.5}
    sim_instance.set_setpoint(setpoint_config=sine_sp_def)

def test_set_setpoint_invalid_type_error(sim_instance):
    """
    Test that setting a setpoint with an invalid type raises ValueError.
    """
    inv_sp_def = {'type': 'invalid_type', 'step_time': 1.0, 'v': 5.0}
    with pytest.raises(ValueError, match="Invalid setpoint type"):
        sim_instance.set_setpoint(setpoint_config=inv_sp_def)

def test_set_disturbance_with_noise_and_seed_success(sim_instance):
    """
    Test setting disturbance with gaussian noise and seed.
    """
    disturbance_config = {
        'noise_type': 'gaussian',
        'noise_std': 0.05,
        'seed': 42
    }
    # The test passes if this executes without raising an exception.
    sim_instance.set_disturbance(disturbance_config=disturbance_config)

def test_set_disturbance_with_input_step_success(sim_instance):
    """
    Test setting disturbance with an input step change.
    """
    disturbance_config = {
        'enable_input_step': True,
        'input_step_at_s': 5.0,
        'input_step_value': 0.5
    }
    # The test passes if this executes without raising an exception.
    sim_instance.set_disturbance(disturbance_config=disturbance_config)

def test_set_disturbance_none_value_success(sim_instance):
    """
    Test setting disturbance with 'none' type.
    """
    disturbance_config = {
        'noise_type': 'none'
    }
    # The test passes if this executes without raising an exception.
    sim_instance.set_disturbance(disturbance_config=disturbance_config)

def test_set_disturbance_config_none_type_success(sim_instance):
    """
    Test setting disturbance with config set to None type.
    """
    # The test passes if this executes without raising an exception.
    sim_instance.set_disturbance(disturbance_config=None)

def test_set_disturbance_invalid_type_error(sim_instance):
    """
    Test that setting disturbance with an invalid noise type raises ValueError.
    """
    disturbance_config = {
        'noise_type': 'invalid_type'
    }
    with pytest.raises(ValueError, match="Invalid noise type"):
        sim_instance.set_disturbance(disturbance_config=disturbance_config)

def test_set_metrics_step_mode_true_success(sim_instance):
    """
    Test setting metrics configuration in step mode.
    """
    metrics_config = {
        'step_mode': True,
        'step_time': 1.0,
        'r_final': 5.0,
        'tail_window_s': 2.0
    }
    sim_instance.set_metrics(metrics_config=metrics_config)

def test_set_metrics_step_mode_false_success(sim_instance):
    """
    Test setting metrics configuration with step_mode as False.
    """
    metrics_config = {
        'step_mode': False
    }
    sim_instance.set_metrics(metrics_config=metrics_config)

def test_set_metrics_as_none_success(sim_instance):
    """
    Test setting metrics configuration as None.
    """
    sim_instance.set_metrics(metrics_config=None)

def test_set_metrics_huge_rate_raises_memory_error(sim_instance):
    """
    Tests that requesting a huge buffer (by setting a massive
    sampling rate) correctly propagates a MemoryError.
    """
    metrics_config = {
        'step_mode': True,
        'tail_window_s': 100.0,
        'max_rate_hz': 100e9 # 100GHz so it should blow up (Calc: 100s * 100e9Hz = 10 trillion samples)
    }
    with pytest.raises(MemoryError):
        sim_instance.set_metrics(metrics_config=metrics_config)

