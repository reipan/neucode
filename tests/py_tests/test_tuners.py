import pytest
import numpy as np
from gymnasium import spaces

pytestmark = pytest.mark.rl

from neucode import (
    PIDController,
    RLTuner,
    Plant,
    FOPDTPlantGenerator,
    FOPDTPlant
)

try:
    import stable_baselines3
    rl_libs_available = True
except ImportError:
    rl_libs_available = False

@pytest.fixture
def fopdt_plant_gen():
    plant = FOPDTPlant(K=0.5, tau=4.284, theta=0.276)
    return FOPDTPlantGenerator(plant=plant)

@pytest.fixture
def gains_space():
    low_bounds = np.array([0, 0, 0], dtype=np.float32)
    high_bounds = np.array([50, 70, 600], dtype=np.float32)
    return spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

@pytest.mark.skipif(not rl_libs_available, reason="RL libs not installed.")
def test_rl_tuner_init_success(fopdt_plant_gen, gains_space):
    """
    Test successful initialization of RLTuner with FOPDT plant generator.
    """
    tuner = RLTuner(
        plant_generator=fopdt_plant_gen,
        gains_space=gains_space,
        algorithm="PPO",
        device="cpu",
    )
    assert isinstance(tuner, RLTuner)
    assert tuner.algorithm_name == "PPO"

@pytest.mark.skipif(not rl_libs_available, reason="RL libs not installed.")
def test_rl_tuner_train_and_tune(fopdt_plant_gen, gains_space):
    """
    Test training and tuning process of RLTuner with FOPDT plant generator.
    """
    tuner = RLTuner(
        plant_generator=fopdt_plant_gen,
        gains_space=gains_space,
        algorithm="PPO",
        device="cpu",
        seed=42
    )
    # Train the tuner for a small number of timesteps for testing
    tuner.train(total_timesteps=100)

    # Tune a specific plant
    plant_to_tune = FOPDTPlant(K=0.5, tau=4.284, theta=0.276)
    tuned_pid = tuner.tune(plant_to_tune)

    assert isinstance(tuned_pid, PIDController)
    assert hasattr(tuned_pid, 'params')
    assert 'kp' in tuned_pid.params

    # Check that the gains are within the specified 'gains_space'
    assert gains_space.contains(np.array([
        tuned_pid.params['kp'],
        tuned_pid.params['ki'],
        tuned_pid.params['kd']
    ], dtype=np.float32))

@pytest.mark.skipif(not rl_libs_available, reason="RL libs not installed.")
def test_rl_tuner_invalid_algorithm_raises_error(fopdt_plant_gen, gains_space):
    """
    Test that RLTuner raises ValueError when initialized with an unknown algorithm.
    """
    with pytest.raises(ValueError, match="Unknown algorithm: UNKNOWN_ALGO. Use 'PPO', 'TD3', or 'SAC'."):
        RLTuner(
            plant_generator=fopdt_plant_gen,
            gains_space=gains_space,
            algorithm="UNKNOWN_ALGO",
            device="cpu",
            seed=42
        )

@pytest.mark.skipif(not rl_libs_available, reason="RL libs not installed.")
def test_rl_tuner_invalid_plant_type_raises_error(fopdt_plant_gen, gains_space):
    """
    Test that RLTuner raises TypeError when tuning a non-FOPDTPlant.
    """
    tuner = RLTuner(
        plant_generator=fopdt_plant_gen,
        gains_space=gains_space,
        algorithm="PPO",
        device="cpu",
        seed=42
    )
    tuner.train(total_timesteps=100)

    class DummyPlant(Plant):
        params = {}

        def get_observation_vector(self):
            pass

    dummy_plant = DummyPlant()

    with pytest.raises(TypeError, match="This tuner currently only supports FOPDTPlant."):
        tuner.tune(dummy_plant)

@pytest.mark.skipif(not rl_libs_available, reason="RL libs not installed.")
def test_rl_tuner_tune_without_training_raises_error(fopdt_plant_gen, gains_space):
    """
    Test that RLTuner raises RuntimeError when tuning before training.
    """
    tuner = RLTuner(
        plant_generator=fopdt_plant_gen,
        gains_space=gains_space,
        algorithm="PPO",
        device="cpu",
        seed=42
    )

    plant_to_tune = FOPDTPlant(K=0.5, tau=4.284, theta=0.276)

    with pytest.raises(RuntimeError, match="The tuner must be trained first."):
        tuner.tune(plant_to_tune)