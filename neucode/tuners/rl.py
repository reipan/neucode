"""
Reinforcement-learning-based PID tuner for the NeuCoDe toolkit.

Provides RLTuner which wraps stable-baselines3 (PPO, TD3, SAC) with a
Gymnasium environment that rewards closed-loop performance metrics.
"""
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RescaleAction
from typing import Optional
import os

from ..harness import SimulationHarness
from ..controllers import PIDController
from ..plants import Plant, FOPDTPlant
from ..scenarios import BenchmarkScenario, StepBenchmark
from ..generators import BasePlantGenerator
from ..rewards import BaseReward, WeightedMetricsReward

from .base import BaseTuner
from .utils import safe_check_env

try:
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    rl_libs_available = True
except ImportError:
    print("Warning: PyTorch or stable-baselines3 not found. RLTuner will not be available.")
    rl_libs_available = False
    # Define dummy classes for type hinting and to allow file import
    class nn:
        """
        Dummy replacement for torch.nn when PyTorch is unavailable.
        """
        Module = object
    class PPO:
        """
        Dummy replacement for stable_baselines3.PPO when the library is unavailable.
        """
        pass
    class SAC:
        """
        Dummy replacement for stable_baselines3.SAC when the library is unavailable.
        """
        pass
    class TD3:
        """
        Dummy replacement for stable_baselines3.TD3 when the library is unavailable.
        """
        pass
    def check_env(env):
        """
        No-op replacement for stable_baselines3 check_env when the library is unavailable.
        """
        pass
    class NormalActionNoise:
        """
        Dummy replacement for stable_baselines3 NormalActionNoise when the library is unavailable.
        """
        pass
    class DummyVecEnv:
        """
        Dummy replacement for stable_baselines3 DummyVecEnv when the library is unavailable.
        """
        pass
    class SubprocVecEnv:
        """
        Dummy replacement for stable_baselines3 SubprocVecEnv when the library is unavailable.
        """
        pass

class _PidTuningEnv(gym.Env):
    """
    (INTERNAL USE ONLY)
    A Gymnasium environment for PID tuning tasks.
    * https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#subclassing-gymnasium-env
    * https://gymnasium.farama.org/api/spaces/fundamental/
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 harness: SimulationHarness,
                 benchmark: BenchmarkScenario,
                 gains_space: spaces.Box,
                 plant_generator: BasePlantGenerator,
                 reward_strategy: BaseReward):
        """
        Initialize the PID tuning environment.

        :param harness: SimulationHarness used to run closed-loop episodes.
        :param benchmark: BenchmarkScenario that configures the setpoint profile for each episode.
        :param gains_space: Gymnasium Box defining the PID gain action space.
        :param plant_generator: BasePlantGenerator that samples a new plant each episode.
        :param reward_strategy: BaseReward implementation used to score each episode.
        """
        super().__init__()
        self.harness = harness
        self.benchmark = benchmark
        self.action_space = gains_space
        self.plant_generator = plant_generator
        self.observation_space = self.plant_generator.get_observation_space()
        self.reward_strategy = reward_strategy
        self._current_plant_params = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment and sample a new plant from the generator.

        :param seed: Optional RNG seed forwarded to the parent reset.
        :param options: Optional environment-specific options (unused).
        :returns: Tuple of (observation, info_dict).
        """
        super().reset(seed=seed, options=options)
        self._current_plant_params, observation = self.plant_generator.generate()
        self.harness.set_plant(self._current_plant_params)
        return observation, {}

    def step(self, action):
        """
        Apply a set of PID gains and return the resulting transition.

        :param action: 1-D array [kp, ki, kd] sampled from the gains action space.
        :returns: Tuple (observation, reward, terminated, truncated, info).
        :raises RuntimeError: If the environment has not been reset first.
        """
        if self._current_plant_params is None:
            raise RuntimeError("Environment must be reset before stepping.")
        
        #kp, ki, kd = action
        # action values are in np.float32, convert to float
        kp = float(action[0])
        ki = float(action[1])
        kd = float(action[2])

        controller = PIDController(kp=kp, ki=ki, kd=kd)
        self.harness.set_controller(controller)

        result = self.benchmark.run(harness=self.harness)

        # Use the strategy set to calculate the reward
        reward, loss = self.reward_strategy.calculate_reward(result)

        terminated = True
        truncated = False
        info = {
            'final_value': result.final_value,
            'overshoot_percent': result.overshoot_percent,
            'ise': result.ise,
            'reward': reward,
            'loss': loss
        }

        _, next_observation = self.plant_generator.generate()

        return next_observation, reward, terminated, truncated, info

    def render(self):
        """
        Print the current plant parameters when render_mode is 'human'.
        """
        if self.metadata.get("render_mode") == "human":
            if self.harness.plant is not None:
                print(f"Current Plant -> {self.harness.plant}")

class _EnvFactory:
    """
    (INTERNAL USE ONLY)
    A factory class that holds the configuration for creating _PidTuningEnv instances.
    This is necessary for parallelization, as each subprocess needs to be able to create its own env.
    * https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    """
    def __init__(self,
                 plant_generator: BasePlantGenerator,
                 gains_space: spaces.Box,
                 reward_strategy: BaseReward,
                 benchmark: BenchmarkScenario,
                 actuator_limits: Optional[dict],
                 algorithm_name: str):
        """
        Store the configuration needed to construct _PidTuningEnv instances.

        :param plant_generator: Generator that samples a plant each episode.
        :param gains_space: Gymnasium Box defining the PID gain action space.
        :param reward_strategy: Reward strategy used to score each episode.
        :param benchmark: Scenario that configures the setpoint profile.
        :param actuator_limits: Optional dict with 'u_min'/'u_max' keys passed to SimulationHarness.
        :param algorithm_name: SB3 algorithm name ('PPO', 'TD3', or 'SAC'); PPO receives a RescaleAction wrapper.
        """
        self.plant_generator = plant_generator
        self.gains_space = gains_space
        self.reward_strategy = reward_strategy
        self.benchmark = benchmark
        self.actuator_limits = actuator_limits
        self.algorithm_name = algorithm_name

    def __call__(self) -> gym.Env:
        """
        Instantiate and return a configured _PidTuningEnv.

        :returns: A ready-to-use Gymnasium environment.
        """
        # Create the base harness
        harness = SimulationHarness(
            plant=None, controller=None, setpoint=None, 
            actuator_limits=self.actuator_limits
        )
        
        # Create the "pure" environment
        env = _PidTuningEnv(
            harness=harness,
            benchmark=self.benchmark,
            gains_space=self.gains_space,
            plant_generator=self.plant_generator,
            reward_strategy=self.reward_strategy
        )
        
        # Apply the wrapper ONLY for PPO
        if self.algorithm_name.upper() in ["PPO"]:
            env = RescaleAction(
                env, 
                min_action=np.float32(-1.0), 
                max_action=np.float32(1.0)
            )
        return env

class RLTuner(BaseTuner):
    """
    A high-level class that encapsulates the entire RL training
    and tuning process for a PID controller.
    This is the main user-facing tool.
    """
    def __init__(self,
                 plant_generator: BasePlantGenerator,
                 gains_space: spaces.Box,
                 reward_strategy: Optional[BaseReward] = None,
                 benchmark: Optional[BenchmarkScenario] = None,
                 algorithm: str = "PPO",
                 actuator_limits: Optional[dict] = None,
                 seed: Optional[int] = None,
                 verbose: int = 0,
                 device: str = "auto",
                 num_parallel_envs: int = 0,
                 **algorithm_kwargs):
        """
        Initialize the RL-based PID tuner.

        :param plant_generator: Generator that samples a new plant each episode.
        :param gains_space: Gymnasium Box defining the PID gain action space.
        :param reward_strategy: Reward strategy used to score episodes. Defaults to WeightedMetricsReward.
        :param benchmark: Scenario used to configure the setpoint. Defaults to a 15 s StepBenchmark.
        :param algorithm: SB3 algorithm to use: 'PPO', 'TD3', or 'SAC'.
        :param actuator_limits: Optional dict with 'u_min'/'u_max' passed to SimulationHarness.
        :param seed: Random seed for reproducibility.
        :param verbose: SB3 verbosity level (0 = silent).
        :param device: Torch device ('auto', 'cpu', or 'cuda').
        :param num_parallel_envs: Number of parallel environments. 0 = auto-detect CPU count.
        :param algorithm_kwargs: Additional keyword arguments forwarded to the SB3 algorithm constructor.
        :raises ImportError: If PyTorch or stable-baselines3 is not installed.
        """
        if not rl_libs_available:
            raise ImportError(f"PyTorch and stable-baselines3 are required to use {self.__class__.__name__}.")

        if benchmark is None:
            benchmark = StepBenchmark(dt=0.01, total_time=15.0)
        
        if reward_strategy is None:
            reward_strategy = WeightedMetricsReward()

        self.real_gains_space = gains_space
        self.plant_generator = plant_generator
        self.algorithm_name = algorithm.upper()

        env_factory = _EnvFactory(
            plant_generator=plant_generator,
            gains_space=gains_space,
            reward_strategy=reward_strategy,
            benchmark=benchmark,
            actuator_limits=actuator_limits,
            algorithm_name=self.algorithm_name
        )

        if num_parallel_envs == 0:
            num_parallel_envs = os.cpu_count() or 1
            print(f"Auto-detecting CPUs: Using {num_parallel_envs} parallel environments.")
        elif num_parallel_envs == 1:
            print("Running in single-environment mode (no parallelization).")

        if num_parallel_envs > 1:
            self.env = SubprocVecEnv([env_factory for i in range(num_parallel_envs)])
        else:
            self.env = DummyVecEnv([env_factory])

        self.seed = seed
        self.device = device
        self.model = self._create_model(
            verbose=verbose,
            algorithm_kwargs=algorithm_kwargs
        )
        self._is_trained = False

    def _create_model(self, verbose: int, algorithm_kwargs: dict):
        """
        Instantiate the SB3 model for the configured algorithm.

        :param verbose: Verbosity level forwarded to the SB3 constructor.
        :param algorithm_kwargs: Extra keyword arguments forwarded to the SB3 constructor.
        :returns: A configured SB3 model (PPO, TD3, or SAC).
        :raises ValueError: If algorithm_name is not one of 'PPO', 'TD3', or 'SAC'.
        """
        if self.algorithm_name == "PPO":
            return PPO("MlpPolicy", self.env, verbose=verbose, seed=self.seed, device=self.device, **algorithm_kwargs)
        elif self.algorithm_name == "TD3":
            n_actions = self.env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            return TD3("MlpPolicy", self.env, action_noise=action_noise, verbose=verbose, seed=self.seed, device=self.device, **algorithm_kwargs)
        elif self.algorithm_name == "SAC":
            return SAC("MlpPolicy", self.env, verbose=verbose, seed=self.seed, device=self.device, **algorithm_kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}. Use 'PPO', 'TD3', or 'SAC'.")

    def train(self, total_timesteps: int, save_path: str = None):
        """
        Train the RL agent.

        :param total_timesteps: Total number of environment steps to train for.
        :param save_path: Optional path to save the trained model after training.
        """
        if hasattr(self.env, "seed"):
            self.env.seed(self.seed)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.env.reset()
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
        self._is_trained = True
        if save_path:
            self.model.save(save_path)

    def tune(self, plant: Plant) -> PIDController:
        """
        Predict optimal PID gains for the given plant using the trained agent.

        :param plant: An FOPDTPlant instance to tune for.
        :returns: A PIDController with gains predicted by the policy.
        :raises RuntimeError: If the tuner has not been trained or loaded yet.
        :raises TypeError: If plant is not an FOPDTPlant.
        """
        if not self._is_trained:
            raise RuntimeError("The tuner must be trained first.")
        
        if not isinstance(plant, FOPDTPlant):
            raise TypeError("This tuner currently only supports FOPDTPlant.")
        
        obs = np.array([
            plant.params['K'], plant.params['tau'], plant.params['theta']
        ], dtype=np.float32)
        
        action, _ = self.model.predict(obs, deterministic=True)

        if self.algorithm_name in ["PPO"]:
            # Get the real bounds we saved during init
            low = self.real_gains_space.low
            high = self.real_gains_space.high
            # Rescale action from [-1, 1] to [low, high]
            real_action = low + (action + 1.0) * 0.5 * (high - low)
        else:
            # For TD3 and SAC, action is already in real space
            real_action = action

        kp, ki, kd = real_action
        return PIDController(kp=float(kp), ki=float(ki), kd=float(kd))

    def load(self, path: str):
        """
        Load a previously saved SB3 model and mark the tuner as trained.

        :param path: Path to the saved SB3 model file.
        """
        if self.algorithm_name == "PPO": self.model = PPO.load(path, env=self.env)
        elif self.algorithm_name == "TD3": self.model = TD3.load(path, env=self.env)
        elif self.algorithm_name == "SAC": self.model = SAC.load(path, env=self.env)
        self._is_trained = True