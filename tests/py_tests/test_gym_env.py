import pytest
import gymnasium as gym
from gymnasium import spaces
import numpy as np

pytestmark = pytest.mark.rl

from neucode import (
    SimulationHarness,
    FOPDTPlant,
)

from neucode.scenarios import StepBenchmark
from neucode.tuners.rl import _PidTuningEnv 
from neucode.generators import FOPDTPlantGenerator
from neucode.rewards import ISEOvershootReward

try:
    import stable_baselines3
    rl_libs_available = True
except ImportError:
    rl_libs_available = False

@pytest.fixture
def blank_harness():
    return SimulationHarness(plant=None, controller=None, setpoint=None)

@pytest.fixture
def gains_space():
    """
    Important: this need to be set as np.float32 to match the action space dtype!
    """
    low_bounds = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    high_bounds = np.array([10.0, 5.0, 2.0], dtype=np.float32)
    return spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

@pytest.fixture
def benchmark_scenario():
    return StepBenchmark(dt=0.01, total_time=30.0)

@pytest.fixture
def plant_fopdt_generator():
    plant = FOPDTPlant(K=0.5, tau=4.284, theta=0.276)
    return FOPDTPlantGenerator(plant=plant)

@pytest.fixture
def reward_strategy():
    return ISEOvershootReward()

def test_env_fopdt_plant_init_success(blank_harness, gains_space, benchmark_scenario, plant_fopdt_generator, reward_strategy):
    """
    Test successful initialization of PidTuningEnv with FOPDT plant generator.
    """
    env = _PidTuningEnv(
        harness=blank_harness,
        gains_space=gains_space,
        benchmark=benchmark_scenario,
        plant_generator=plant_fopdt_generator,
        reward_strategy=reward_strategy
    )
    assert isinstance(env, gym.Env)
    assert env.action_space == gains_space
    
    expected_obs = np.array([0.5, 4.284, 0.276], dtype=np.float32)
    assert isinstance(env.observation_space, spaces.Box)
    assert np.all(env.observation_space.low == expected_obs)
    assert np.all(env.observation_space.high == expected_obs)

@pytest.mark.skipif(not rl_libs_available, reason="RL libs not installed.")
def test_env_fopdt_plant_workflow(blank_harness, gains_space, benchmark_scenario, plant_fopdt_generator, reward_strategy):
    """
    Test the reset and step workflow of PidTuningEnv with FOPDT plant generator.
    """
    env = _PidTuningEnv(
        harness=blank_harness,
        gains_space=gains_space,
        benchmark=benchmark_scenario,
        plant_generator=plant_fopdt_generator,
        reward_strategy=reward_strategy
    )

    expected_obs = np.array([0.5, 4.284, 0.276], dtype=np.float32)
    observation, info = env.reset()

    assert np.allclose(observation, expected_obs)
    assert isinstance(info, dict)
    assert env.harness.plant is not None, "Plant should be set in the harness after reset."

    action = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    next_observation, reward, terminated, truncated, info = env.step(action)

    assert reward > 0, "A stable controller should produce a positive reward."
    assert terminated is True, "The episode should terminate after one step."
    assert truncated is False, "The episode should not be truncated."
    assert isinstance(info, dict)
    assert 'ise' in info
    assert 'overshoot_percent' in info

    assert np.allclose(next_observation, expected_obs)