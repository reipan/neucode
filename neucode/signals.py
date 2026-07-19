"""
Setpoint and disturbance signal definitions for the NeuCoDe simulation pipeline.

Provides concrete setpoint types (step, ramp, sinusoidal) and the Disturbance class
that configures input/output load steps and additive sensor noise for the simulation core.
"""
from abc import ABC

class Setpoint(ABC):
    """
    Abstract base class for setpoint signal configurations.

    Subclasses populate the ``config`` dict with type-specific fields
    consumed by the simulation core.
    """
    config: dict

class StepSetpoint(Setpoint):
    """
    Define a step change in the setpoint.
    """
    def __init__(self, time: float, value: float):
        """
        Initialize a step setpoint.

        :param time: Time in seconds at which the step occurs (must be non-negative).
        :param value: Setpoint value after the step.
        :raises ValueError: If time is negative or not a number.
        :raises TypeError: If value is not numeric.
        """
        if not isinstance(time, (int, float)) or time < 0:
            raise ValueError("Step time must be a non-negative number.")
        if not isinstance(value, (int, float)):
            raise TypeError("Step value must be numeric.")

        self.config = {
            'type': 'step',
            'step_time': float(time),
            'v': float(value)
        }

    def __repr__(self):
        """
        Return a string of the form '(SETPOINT: type=step, time=..., value=...)'.
        """
        return f" (SETPOINT: type=step, time={self.config['step_time']}, value={self.config['v']})"

class RampSetpoint(Setpoint):
    """
    Define a ramp change in the setpoint.
    """
    def __init__(self, start_time: float, duration: float, start_value: float, end_value: float):
        """
        Initialize a ramp setpoint.

        :param start_time: Time in seconds at which the ramp begins (must be non-negative).
        :param duration: Duration of the ramp in seconds (must be positive).
        :param start_value: Setpoint value at the beginning of the ramp.
        :param end_value: Setpoint value at the end of the ramp.
        :raises ValueError: If any parameter fails its validity check.
        """
        if not isinstance(start_time, (int, float)) or start_time < 0:
            raise ValueError("Ramp start time must be a non-negative number.")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError("Ramp duration must be a positive number.")
        if not all(isinstance(v, (int, float)) for v in [start_value, end_value]):
            raise ValueError("Ramp start and end values must be numeric.")

        self.config = {
            'type': 'ramp',
            'step_time': float(start_time),
            'time': float(duration),
            'a': float(start_value),
            'b': float(end_value)
        }

    def __repr__(self):
        """
        Return a string of the form '(SETPOINT: type=ramp, start_time=..., duration=..., start_value=..., end_value=...)'.
        """
        return f" (SETPOINT: type=ramp, start_time={self.config['step_time']}, duration={self.config['time']}, start_value={self.config['a']}, end_value={self.config['b']})"

class SinSetpoint(Setpoint):
    """
    Define a sinusoidal variation in the setpoint.
    """
    def __init__(self, start_time: float, amplitude: float, frequency: float):
        """
        Initialize a sinusoidal setpoint.

        :param start_time: Time in seconds at which the sinusoid begins (must be non-negative).
        :param amplitude: Peak amplitude of the sinusoidal variation.
        :param frequency: Frequency of the sinusoid in Hz (must be positive).
        :raises ValueError: If any parameter fails its validity check.
        """
        if not isinstance(start_time, (int, float)) or start_time < 0:
            raise ValueError("Sine start time must be a non-negative number.")
        if not isinstance(amplitude, (int, float)):
            raise ValueError("Sine amplitude must be a numeric.")
        if not isinstance(frequency, (int, float)) or frequency <= 0:
            raise ValueError("Sine frequency must be a positive number.")

        self.config = {
            'type': 'sin',
            'step_time': float(start_time),
            'amp': float(amplitude),
            'freq': float(frequency)
        }

    def __repr__(self):  
        """
        Return a string of the form '(Sin: start_time=..., amplitude=..., frequency=...)'.
        """
        return f" (Sin: start_time={self.config['step_time']}, amplitude={self.config['amp']}, frequency={self.config['freq']})"

class MultiStepSetpoint(Setpoint):
    """
    A sequence of step setpoints for multi-episode calibration or sweep runs.

    Unlike single-episode setpoints (StepSetpoint, RampSetpoint, etc.), this
    class is iterable and yields one StepSetpoint per episode. It is intended
    for use with the Akida export pipeline's export(calibration_harness=...) to
    generate calibration data across multiple operating points.

    It cannot be passed to SimulationHarness.run() directly - doing so raises
    a RuntimeError.
    """

    def __init__(self, values: list, step_time: float = 0.1):
        """
        Initialize a multi-step setpoint sequence.

        :param values: List of step values to iterate over (one episode each).
        :param step_time: Time in seconds at which each step occurs (default 0.1).
        :raises ValueError: If values is empty or step_time is negative.
        """
        if not values:
            raise ValueError("MultiStepSetpoint requires at least one value.")
        if not isinstance(step_time, (int, float)) or step_time < 0:
            raise ValueError("step_time must be a non-negative number.")
        self._steps = [StepSetpoint(time=step_time, value=v) for v in values]

    @property
    def config(self):
        raise RuntimeError(
            "MultiStepSetpoint cannot be used directly with SimulationHarness.run(). "
            "Pass the harness to the Akida exporter's export(calibration_harness=...) instead."
        )

    def __iter__(self):
        return iter(self._steps)

    def __len__(self):
        return len(self._steps)

    def __repr__(self):
        values = [s.config['v'] for s in self._steps]
        return f"(MultiStepSetpoint: {len(self._steps)} episodes, values={values})"


class Disturbance:
    """
    Class for disturbance and noise configuration.
    """
    def __init__(self,
            input_step_time: float = 0.0, input_step_value: float = 0.0,
            output_step_time: float = 0.0, output_step_value: float = 0.0,
            noise_type: str = 'none', noise_std: float = 0.0, noise_amp: float = 0.0,
            seed: int = 0,
            cogging_sine_amp: float = 0.0, cogging_freq_mult: int = 1):
        """
        Initialize a disturbance configuration.

        :param input_step_time: Time in seconds at which the input load step is applied.
        :param input_step_value: Magnitude of the input load step (0 disables it).
        :param output_step_time: Time in seconds at which the output load step is applied.
        :param output_step_value: Magnitude of the output load step (0 disables it).
        :param noise_type: Additive sensor noise type: 'none', 'gaussian', or 'uniform'.
        :param noise_std: Standard deviation for Gaussian noise (must be non-negative).
        :param noise_amp: Half-amplitude for uniform noise (must be non-negative).
        :param seed: RNG seed for noise generation (must be a non-negative integer).
        :param cogging_sine_amp: Peak amplitude of the cogging torque disturbance
            [same units as u]. Set to 0 (default) to disable.
        :param cogging_freq_mult: Cogging periods per revolution. Use pole-pair count
            (quick) or ``LCM(N_slots, 2*n_pp)`` (exact). Default 1.
        :raises ValueError: If any numeric parameter fails its validity check or
            noise_type is not one of the accepted values.
        """
        if not isinstance(input_step_time, (int, float)) or input_step_time < 0:
            raise ValueError("Input step time must be a non-negative number.")
        if not isinstance(output_step_time, (int, float)) or output_step_time < 0:
            raise ValueError("Output step time must be a non-negative number.")
        if not isinstance(noise_std, (int, float)) or noise_std < 0:
            raise ValueError("Noise standard deviation must be a non-negative number.")
        if not isinstance(noise_amp, (int, float)) or noise_amp < 0:
            raise ValueError("Noise amplitude must be a non-negative number.")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("Seed must be a non-negative integer.")
        noise_type_lower = noise_type.lower()
        if noise_type_lower not in ['none', 'gaussian', 'uniform']:
            raise ValueError("Invalid noise type. Must be 'none', 'gaussian', or 'uniform'.")
        
        self.config = {
            'enable_input_step': input_step_value != 0.0,
            'input_step_at_s': float(input_step_time),
            'input_step_value': float(input_step_value),
            'enable_output_step': output_step_value != 0.0,
            'output_step_at_s': float(output_step_time),
            'output_step_value': float(output_step_value),
            'noise_type': noise_type_lower,
            'noise_std': float(noise_std),
            'noise_amp': float(noise_amp),
            'seed': int(seed),
            'enable_cogging_sine': cogging_sine_amp > 0.0,
            'cogging_sine_amp': float(cogging_sine_amp),
            'cogging_freq_mult': int(cogging_freq_mult),
        }

    def __repr__(self):
        """
        Return a human-readable summary of active disturbance and noise settings.
        """
        desc = " (DISTURBANCE:"
        if self.config['enable_input_step']:
            desc += f" input_step_at_s={self.config['input_step_at_s']}s, input_step_value={self.config['input_step_value']})"
        if self.config['enable_output_step']:
            desc += f" output_step_at_s={self.config['output_step_at_s']}s, output_step_value={self.config['output_step_value']})"
        desc += f" noise_type={self.config['noise_type']}"
        if self.config['noise_type'] != 'none':
            desc += f", std={self.config['noise_std']}, amp={self.config['noise_amp']}, seed={self.config['seed']})"
        return desc
