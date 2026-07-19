import pytest

from neucode import FOPDTPlant, PIDController
from neucode.tuners import ZieglerNicholsReactionCurveTuner

@pytest.fixture
def fopdt_plant():
    return FOPDTPlant(K=0.5, tau=4.284, theta=0.276)

@pytest.fixture
def dt_positive():
    return 0.01

def test_ziegler_nichols_reaction_curve_tuner_success(fopdt_plant, dt_positive):
    """
    Test successful tuning of PID parameters using Ziegler-Nichols Reaction Curve method.
    """
    tuner = ZieglerNicholsReactionCurveTuner(dt=dt_positive)
    pid_controller = tuner.tune(plant=fopdt_plant)
    
    assert isinstance(pid_controller, PIDController)
    # Check that the PID parameters are within expected ranges
    assert pid_controller.params['kp'] > 0
    assert pid_controller.params['ki'] > 0
    assert pid_controller.params['kd'] > 0

def test_ziegler_nichols_tuner_invalid_params_raises_error(dt_positive):
    """
    Tests that the tuner handles divide-by-zero cases by correctly raising a ValueError.
    """
    # theta = 0 should raise an error
    plant_no_theta = FOPDTPlant(K=1.0, tau=5.0, theta=0.0)
    tuner = ZieglerNicholsReactionCurveTuner(dt=dt_positive)
    with pytest.raises(ValueError, match="Invalid plant parameters"):
        tuner.tune(plant=plant_no_theta)

    # K = 0 should raise an error
    plant_no_gain = FOPDTPlant(K=0.0, tau=5.0, theta=1.0)
    tuner = ZieglerNicholsReactionCurveTuner(dt=dt_positive)
    with pytest.raises(ValueError, match="Invalid plant parameters"):
        tuner.tune(plant=plant_no_gain)
