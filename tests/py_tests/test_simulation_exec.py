import pytest
from neucode.simcore import Simulation
import math

@pytest.fixture
def configured_sim_instance():
    """
    Fixture to create and destroy a configured Simulation object for testing.
    """
    sim = Simulation()
    sim.set_time_step(dt=0.01, total_time=60.0)
    sim.set_pid(
        gains={'kp': 2.0, 'ki': 1.0},
        limits={'u_min': -10.0, 'u_max': 10.0, 'i_min': -10.0, 'i_max': 10.0, 'kaw': 0.1}
    )
    sim.set_fopdt(params={'K': 0.5, 'tau': 4.284, 'theta': 0.276})
    sim.set_setpoint(setpoint_config={'type': 'step', 'step_time': 1.0, 'v': 1.0})
    sim.set_metrics(metrics_config={
        'step_mode': True,
        'step_time': 1.0,
        'r_final': 1.0,
        'tail_window_s': 2.0,
        'max_rate_hz': 100.0,
        'min_tail_samples': 20
    })
    return sim

def test_sim_run_success(configured_sim_instance):
    """
    Test that the simulation runs successfully and doesn't raise exceptions.
    """
    results = configured_sim_instance.run()
    assert isinstance(results, dict), "Simulation results should be returned as a dictionary."
    assert 'y_final' in results, "Results should contain 'y_final'."
    assert 'overshoot_percent' in results, "Results should contain 'overshoot_percent'."

def test_sim_run_results_are_reasonable(configured_sim_instance):
    """
    Test that the simulation results are within expected ranges.
    """
    results = configured_sim_instance.run()
    assert results['y_final'] == pytest.approx(1.0, abs=0.02), "Final output should be close to setpoint."
    assert 'overshoot_percent' in results, "Results should contain 'overshoot_percent'."
    assert results['overshoot_percent'] > 0, "Overshoot should be greater than 0."
    assert results['overshoot_percent'] < 15.0, "Overshoot should be less than 15.0."
    assert 'rise_time' in results, "Results should contain 'rise_time'."
    assert not math.isnan(results['rise_time']), "Rise time should be a valid number for a step response."

def test_unconfigured_sim_run_error():
    """
    Test that running a simulation without configuration raises an error.
    """
    sim = Simulation()
    sim.set_time_step(dt=0.01, total_time=10.0)
    sim.set_pid(gains={'kp': 1})
    sim.set_fopdt(params={'K':1, 'tau':1})

    with pytest.raises(RuntimeError):
        sim.run()