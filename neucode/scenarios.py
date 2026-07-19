"""
Simulation scenario definitions for the NeuCoDe toolkit.

Provides an abstract BaseScenario interface, the BenchmarkScenario family (step, ramp,
noisy, disturbed), and factory functions for generating randomised training episodes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import math
import numpy as np

from .harness import SimulationHarness
from .results import ExperimentResult
from .signals import StepSetpoint, RampSetpoint, Disturbance
from .rewards import BaseReward

class BaseScenario(ABC):
    """
    Abstract base class for all simulation scenarios.

    Enforces a common dt and total_time contract and the run() interface.
    """
    def __init__(self, dt: float, total_time: float):
        """
        Initialize the base scenario with timing parameters.

        :param dt: Sampling interval in seconds (must be positive).
        :param total_time: Total simulation duration in seconds (must be positive).
        :raises ValueError: If dt or total_time are not positive numbers.
        """
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError("dt must be a positive number.")
        if not isinstance(total_time, (int, float)) or total_time <= 0:
            raise ValueError("total_time must be a positive number.")
        self.dt = dt
        self.total_time = total_time

    @abstractmethod
    def run(self, harness: SimulationHarness, **kwargs):
        """
        Execute the scenario on the given harness.

        :param harness: The SimulationHarness (or compatible harness) to run on.
        :returns: Scenario-specific result object.
        """
        pass
        
class BenchmarkScenario(BaseScenario):
    """
    Base class for deterministic benchmark scenarios.

    Subclasses implement _configure() to attach a setpoint and optional disturbance,
    then delegate execution to run(), which returns an ExperimentResult.
    """
    @abstractmethod
    def _configure(self, harness: SimulationHarness):
        """
        Attach setpoint and disturbance signals to the harness before the run.

        :param harness: The SimulationHarness instance to configure.
        """
        pass

    def run(self,
            harness: SimulationHarness,
            get_time_series: bool = False,
            show_summary: bool = False) -> ExperimentResult:
        """
        Configure the harness and execute the simulation.

        :param harness: The SimulationHarness to run the scenario on.
        :param get_time_series: If True, include full time-series data in the result.
        :param show_summary: If True, print a brief metric summary to stdout.
        :returns: ExperimentResult containing performance metrics.
        """
        self._configure(harness)

        if get_time_series:
            result = harness.run(dt=self.dt, total_time=self.total_time, get_time_series=True)
        else:
            result = harness.evaluate(dt=self.dt, total_time=self.total_time)

        if show_summary:
            print(self._quick_summary(result))

        return result

    def _quick_summary(self, result: ExperimentResult) -> str:
        """
        Build a human-readable metric summary string from a result.

        :param result: ExperimentResult to summarise.
        :returns: Multi-line string with key metrics formatted for console output.
        """
        summary_metrics = [
            # (attribute_name, display_name, format_specifier)
            # This list can be extended with more metrics as needed
            ('final_value', 'Final Value', '.4f'),
            ('overshoot_percent', 'Overshoot', '.2f'),
            ('ise', 'ISE', '.4f'),
            ('isu', 'ISU', '.4f'),
            ('rise_time', 'Rise Time', '.4f')
        ]

        summary_lines = []
        for attr, name, fmt in summary_metrics:
            value = getattr(result, attr, None)
            
            # For uniformity, handle None or NaN values as "N/A"
            if value is None or math.isnan(value):
                value_str = "N/A"
            else:
                # Some value unit cosmetics (add % or s where appropriate)
                unit = '%' if 'percent' in attr else ('s' if 'time' in attr else '')
                value_str = f"{value:{fmt}}{unit}"
                
            summary_lines.append(f"  - {name}: {value_str}")
            
        return "\n".join(summary_lines)

class StepBenchmark(BenchmarkScenario):
    """
    A standard benchmark scenario using a step input.
    """
    def __init__(self, dt: float, total_time: float, step_time: float = 1.0,
                 step_value: float = 5.0, disturbance: Disturbance = None):
        """
        Initialize a clean step benchmark scenario.

        :param dt: Sampling interval in seconds.
        :param total_time: Total simulation duration in seconds.
        :param step_time: Time at which the step setpoint occurs (must be within [0, total_time)).
        :param step_value: Magnitude of the step setpoint change.
        :param disturbance: Optional Disturbance to apply (e.g. cogging). Default None.
        :raises ValueError: If step_time falls outside the simulation duration.
        """
        super().__init__(dt, total_time)
        if step_time < 0 or step_time >= total_time:
            raise ValueError("step_time must be within the simulation duration.")
        self.step_time = step_time
        self.step_value = step_value
        self.disturbance = disturbance

    def _configure(self, harness):
        """
        Configure the harness with a step setpoint and optional disturbance.

        :param harness: The SimulationHarness instance to configure.
        """
        harness.set_setpoint(StepSetpoint(time=self.step_time, value=self.step_value))
        harness.set_disturbance(self.disturbance)

class RobustStepBenchmark(BenchmarkScenario):
    """
    A robust benchmark scenario using a step input with added noise + load disturbance.
    """
    def __init__(self,
                 dt: float,
                 total_time: float,
                 step_time: float = 1.0,
                 step_value: float = 5.0,
                 noise_std: float = 0.01,
                 disturbance_time: float = 2.0,
                 disturbance_value: float = 2.0,
                 ):
        """
        Initialize a robust step benchmark with Gaussian noise and a load disturbance.

        :param dt: Sampling interval in seconds.
        :param total_time: Total simulation duration in seconds.
        :param step_time: Time at which the step setpoint occurs.
        :param step_value: Magnitude of the step setpoint change.
        :param noise_std: Standard deviation of Gaussian sensor noise.
        :param disturbance_time: Time at which the load disturbance is applied.
        :param disturbance_value: Magnitude of the load disturbance input step.
        """
        super().__init__(dt, total_time)
        self.step_time = step_time
        self.step_value = step_value

        # Generate a non-zero seed to ensure noise RNG is properly initialized
        # (seed=0 would trigger default in C layer, potentially causing issues)
        seed = np.random.randint(1, 100001)  # Range [1, 100001)
        
        self.disturbance = Disturbance(
            input_step_time=disturbance_time,
            input_step_value=disturbance_value,
            noise_type='gaussian',
            noise_std=noise_std,
            seed=seed
        )

    def _configure(self, harness):
        """
        Configure the harness with a step setpoint and the pre-built noise+disturbance signal.

        :param harness: The SimulationHarness instance to configure.
        """
        harness.set_setpoint(StepSetpoint(time=self.step_time, value=self.step_value))
        harness.set_disturbance(self.disturbance)


class NoisyBenchmark(BenchmarkScenario):
    """
    A benchmark scenario with step input and sensor noise only.
    """
    def __init__(self,
                 dt: float,
                 total_time: float,
                 step_time: float = 1.0,
                 step_value: float = 5.0,
                 noise_std: float = 0.01,
                 seed: int | None = None):
        """
        Initialize a noisy benchmark scenario.
        
        :param dt: Time step for simulation
        :param total_time: Total simulation duration
        :param step_time: Time at which the step input occurs
        :param step_value: Magnitude of the step input
        :param noise_std: Standard deviation of Gaussian sensor noise
        :param seed: Random seed for noise generation
        """
        super().__init__(dt, total_time)
        if step_time < 0 or step_time >= total_time:
            raise ValueError("step_time must be within the simulation duration.")
        
        self.step_time = step_time
        self.step_value = step_value

        if seed is None:
            seed = np.random.randint(1, 100001)
        self.disturbance = Disturbance(
            input_step_time=999.0, # Disabled
            input_step_value=0.0,
            noise_type='gaussian',
            noise_std=noise_std,
            seed=seed
        )

    def _configure(self, harness):
        """
        Configure the harness with a step setpoint and sensor noise only (no load disturbance).

        :param harness: The SimulationHarness instance to configure.
        """
        harness.set_setpoint(StepSetpoint(time=self.step_time, value=self.step_value))
        harness.set_disturbance(self.disturbance)


class DisturbedBenchmark(BenchmarkScenario):
    """
    A benchmark scenario with step input, sensor noise, and a single load disturbance.
    """
    def __init__(self,
                 dt: float,
                 total_time: float,
                 step_time: float = 1.0,
                 step_value: float = 5.0,
                 noise_std: float = 0.01,
                 disturbance_time: float = 2.0,
                 disturbance_value: float = 2.0,
                 seed: int | None = None):
        """
        Initialize a disturbed benchmark scenario.
        
        :param dt: Time step for simulation
        :param total_time: Total simulation duration
        :param step_time: Time at which the step input occurs
        :param step_value: Magnitude of the step input
        :param noise_std: Standard deviation of Gaussian sensor noise
        :param disturbance_time: Time at which the load disturbance occurs
        :param disturbance_value: Magnitude of the load disturbance
        :param seed: Random seed for noise generation
        """
        super().__init__(dt, total_time)
        if step_time < 0 or step_time >= total_time:
            raise ValueError("step_time must be within the simulation duration.")
        if disturbance_time < 0 or disturbance_time >= total_time:
            raise ValueError("disturbance_time must be within the simulation duration.")
        
        self.step_time = step_time
        self.step_value = step_value
        self.disturbance_time = disturbance_time
        self.disturbance_value = disturbance_value

        if seed is None:
            seed = np.random.randint(1, 100001)
        self.disturbance = Disturbance(
            input_step_time=disturbance_time,
            input_step_value=disturbance_value,
            noise_type='gaussian',
            noise_std=noise_std,
            seed=seed
        )

    def _configure(self, harness):
        """
        Configure the harness with a step setpoint and a combined noise+load disturbance.

        :param harness: The SimulationHarness instance to configure.
        """
        harness.set_setpoint(StepSetpoint(time=self.step_time, value=self.step_value))
        harness.set_disturbance(self.disturbance)


class RampBenchmark(BenchmarkScenario):
    """
    A benchmark scenario using a ramp input.
    """
    def __init__(self,
                 dt: float,
                 total_time: float,
                 start_time: float,
                 duration: float,
                 start_value: float,
                 end_value: float):
        """
        Initialize a ramp benchmark scenario.

        :param dt: Sampling interval in seconds.
        :param total_time: Total simulation duration in seconds.
        :param start_time: Time at which the ramp begins (must be non-negative).
        :param duration: Duration of the ramp in seconds (must not extend beyond total_time).
        :param start_value: Setpoint value at the start of the ramp.
        :param end_value: Setpoint value at the end of the ramp.
        :raises ValueError: If start_time or duration place the ramp outside the simulation window.
        """
        super().__init__(dt, total_time)
        if start_time < 0 or duration > total_time:
            raise ValueError("start_time and duration must be within the simulation duration")
        if start_time + duration > total_time:
            raise ValueError("Ramp duration extends beyond total simulation time")
        self.start_time = start_time
        self.duration = duration
        self.start_value = start_value
        self.end_value = end_value

    def _configure(self, harness):
        """
        Configure the harness with a ramp setpoint and no disturbance.

        :param harness: The SimulationHarness instance to configure.
        """
        harness.set_setpoint(RampSetpoint(
            start_time=self.start_time,
            duration=self.duration,
            start_value=self.start_value,
            end_value=self.end_value
        ))
        harness.set_disturbance(None)

# Factory functions for generating benchmark scenarios with specific characteristics
def resilience_training_factory(
    dt: float = 0.01,
    noise_std: float = 0.05,
    clean_prob: float = 0.1,
    noisy_prob: float = 0.1,
    disturbed_prob: float = 0.8,
    setpoint_range: tuple = (0.1, 5.0),
    step_time_range: tuple = (0.1, 3.0),
    disturbance_range: tuple = (-2.5, 2.5),
    disturbance_window: tuple = (1.5, 2.0),
    episode_duration: float = 20.0
) -> BenchmarkScenario:
    """
    Factory for resilience training scenarios.
    Generates a randomized split of Clean, Noisy, and Disturbed episodes.
    
    :param dt: Time step for the benchmark
    :param noise_std: Standard deviation of sensor noise
    :param clean_prob: Probability of clean scenario (no noise, no disturbance)
    :param noisy_prob: Probability of noisy-only scenario (noise, no disturbance)
    :param disturbed_prob: Probability of disturbed scenario (noise + disturbance)
    :param setpoint_range: Range for random setpoint magnitude (min, max)
    :param step_time_range: Range for random step time (min, max) relative to episode start
    :param disturbance_range: Range of random disturbance magnitudes (min, max)
    :param disturbance_window: Window of when disturbance can occur post-step (min, max)
    :param episode_duration: Total duration of each episode in seconds
    :returns: A BenchmarkScenario instance (StepBenchmark, NoisyBenchmark, or DisturbedBenchmark).
    """
    random_setpoint = float(np.random.uniform(*setpoint_range))
    random_step_time = float(np.random.uniform(*step_time_range))
    
    scenario_type = np.random.choice(
        ['clean', 'noisy', 'disturbed'],
        p=[clean_prob, noisy_prob, disturbed_prob]
    )
    
    if scenario_type == 'clean':
        return StepBenchmark(
            dt=dt,
            total_time=episode_duration,
            step_time=random_step_time,
            step_value=random_setpoint
        )
    elif scenario_type == 'noisy':
        return NoisyBenchmark(
            dt=dt,
            total_time=episode_duration,
            step_time=random_step_time,
            step_value=random_setpoint,
            noise_std=noise_std
        )
    else:
        rand_dist_time = float(np.random.uniform(
            random_step_time + disturbance_window[0],
            random_step_time + disturbance_window[1]
        ))
        rand_dist_val = float(np.random.uniform(*disturbance_range))
        
        return DisturbedBenchmark(
            dt=dt,
            total_time=episode_duration,
            step_time=random_step_time, 
            step_value=random_setpoint,
            noise_std=noise_std,
            disturbance_time=rand_dist_time,
            disturbance_value=rand_dist_val
        )


def noise_focused_factory(
    dt: float = 0.01,
    noise_std: float = 0.01,
    clean_prob: float = 0.5,
    noisy_prob: float = 0.5,
    setpoint_range: tuple = (0.1, 5.0),
    step_time_range: tuple = (0.1, 3.0),
    episode_duration: float = 40.0,
    disturbance: Disturbance = None,
) -> BenchmarkScenario:
    """
    Factory for noise-focused training scenarios.
    Generates clean and noisy episodes with no disturbances.
    Longer episodes (30-40 s) provide natural steady-state observation windows.

    :param dt: Sampling interval in seconds.
    :param noise_std: Standard deviation of sensor noise for noisy scenarios.
    :param clean_prob: Probability of a clean (no-noise) episode.
    :param noisy_prob: Probability of a noisy episode.
    :param setpoint_range: Range (min, max) for the random setpoint magnitude.
    :param step_time_range: Range (min, max) for the random step time in seconds.
    :param episode_duration: Total duration of each episode in seconds.
    :param disturbance: Optional Disturbance applied to every episode (e.g. cogging for v5+).
    :returns: A BenchmarkScenario instance (StepBenchmark or NoisyBenchmark).
    """
    random_setpoint = float(np.random.uniform(*setpoint_range))
    random_step_time = float(np.random.uniform(*step_time_range))
    
    scenario_type = np.random.choice(['clean', 'noisy'], p=[clean_prob, noisy_prob])
    
    if scenario_type == 'clean':
        return StepBenchmark(
            dt=dt,
            total_time=episode_duration,
            step_time=random_step_time,
            step_value=random_setpoint,
            disturbance=disturbance,
        )
    else:
        bench = NoisyBenchmark(
            dt=dt,
            total_time=episode_duration,
            step_time=random_step_time,
            step_value=random_setpoint,
            noise_std=noise_std,
        )
        # Merge cogging on top of the noise disturbance if both are requested.
        # Simple approach: cogging takes precedence; noise is retained via NoisyBenchmark's own
        # disturbance field.  For independent combination, extend Disturbance to chain signals.
        if disturbance is not None:
            bench.disturbance = disturbance
        return bench