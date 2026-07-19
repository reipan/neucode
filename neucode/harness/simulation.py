"""
Simulation harness for the NeuCoDe toolkit.

Wraps the Cython simulation core (Simulation) and provides a Python-level
harness that accepts any Controller/Plant/Setpoint combination and returns
a standardised ExperimentResult.
"""
from __future__ import annotations
import numpy as np

from .base import BaseHarness
from neucode.plants import Plant, FOPDTPlant, FOIPDTPlant
from neucode.controllers import Controller, PIDController
from neucode.signals import Setpoint, Disturbance
from neucode.results import ExperimentResult
from neucode.simcore import Simulation

class SimulationHarness(BaseHarness):
    """
    Harness that runs closed-loop control experiments inside the Cython simulation core.

    For PID controllers the full simulation runs natively in C (fast path).
    For ANN/SNN controllers a Python step-loop calls controller.predict() each tick
    and drives the C core one step at a time (hybrid path).
    """
    def __init__(self,
                 plant: Plant | None,
                 controller: Controller,
                 setpoint: Setpoint | None,
                 disturbance: Disturbance = None,
                 actuator_limits: dict = None,
                 sensor_iir_alpha: float | None = None,
                 ):
        """
        Initialize the simulation harness.

        :param plant: A Plant instance (FOPDTPlant or FOIPDTPlant), or None for deferred assignment.
        :param controller: A Controller instance (PIDController, ANNController, or SNNController).
        :param setpoint: A Setpoint instance defining the reference trajectory, or None.
        :param disturbance: An optional Disturbance for load/noise injection.
        :param actuator_limits: Dict with keys 'u_min'/'u_max' (and optionally 'i_min'/'i_max',
            'd_alpha', 'kaw'). Defaults to ``{'u_min': -10.0, 'u_max': 10.0}``.
        :param sensor_iir_alpha: IIR low-pass filter coefficient applied to the
            measurement in the hybrid step loop.  Matches the firmware sensor
            filter (``alpha * raw + (1-alpha) * prev``).  ``None`` disables
            filtering (default).  Firmware uses 0.04.
        :raises TypeError: If plant, controller, or actuator_limits are the wrong types.
        :raises ValueError: If actuator_limits is missing required keys.
        """
        super().__init__(controller=controller, setpoint=setpoint, disturbance=disturbance)
        self.sensor_iir_alpha = sensor_iir_alpha
        if plant is not None and not isinstance(plant, Plant):
            raise TypeError("plant must be an instance of a Plant subclass or None.")
        self.plant = plant

        if actuator_limits is None:
            self.actuator_limits = {'u_min': -10.0, 'u_max': 10.0}
        else:
            # check if actuator_limits is a dict with 'u_min' and 'u_max' keys
            if not isinstance(actuator_limits, dict):
                raise TypeError("actuator_limits must be a dict with 'u_min' and 'u_max' keys.")
            if 'u_min' not in actuator_limits or 'u_max' not in actuator_limits:
                raise ValueError("actuator_limits must contain 'u_min' and 'u_max' keys.")
            self.actuator_limits = actuator_limits

    def set_plant(self, plant: Plant):
        """
        Swap the plant used by this simulation harness.

        :param plant: A Plant instance to use for subsequent runs.
        :raises TypeError: If plant is not a Plant instance.
        """
        if not isinstance(plant, Plant):
            raise TypeError("plant must be an instance of Plant.")
        self.plant = plant

    def run(self, dt: float, total_time: float, get_time_series: bool = False) -> ExperimentResult:
        """
        Run the closed-loop simulation and return performance metrics.

        :param dt: Sampling interval in seconds.
        :param total_time: Total simulation duration in seconds.
        :param get_time_series: If True, populate time-series arrays in the returned result.
        :returns: ExperimentResult with metrics and optional time-series data.
        :raises RuntimeError: If no setpoint has been configured.
        :raises ValueError: If dt and total_time produce fewer than two samples.
        :raises TypeError: If the controller or plant type is not supported.
        """
        # validate setpoint
        if self.setpoint is None:
            raise RuntimeError("Cannot run simulation: Setpoint is not configured.")

        # Setup simulation
        simulation = Simulation()
        simulation.set_time_step(dt, total_time)
        num_samples = int(total_time / dt) + 1
        if num_samples <= 1:
            raise ValueError("total_time and dt must result in at least two samples.")

        if isinstance(self.controller, PIDController):
            # This configures the internal PID controller in the C core
            limits = {
                'u_min': self.actuator_limits.get('u_min', -10.0),
                'u_max': self.actuator_limits.get('u_max', 10.0),
                'i_min': self.actuator_limits.get('i_min', self.actuator_limits.get('u_min', -10.0)),
                'i_max': self.actuator_limits.get('i_max', self.actuator_limits.get('u_max', 10.0)),
                'd_alpha': self.actuator_limits.get('d_alpha', 0.1),
                'kaw': self.actuator_limits.get('kaw', 0.0)
            }

            simulation.set_pid(gains=self.controller.params, limits=limits)
        elif hasattr(self.controller, 'predict'):
            # As this is a custom controller, we will use the hybrid path.
            # The harness will call the controller's `predict` method in a loop.
            pass
        else:
            raise TypeError("'{}' is not a supported controller type.".format(type(self.controller).__name__))

        # Plant
        if isinstance(self.plant, FOIPDTPlant):
            simulation.set_foipdt(params=self.plant.params)
        elif isinstance(self.plant, FOPDTPlant):
            simulation.set_fopdt(params=self.plant.params)
        else:
            raise TypeError("'{}' is not a supported plant type.".format(type(self.plant).__name__))

        # Sensor filter
        if self.sensor_iir_alpha is not None:
            simulation.set_sensor_filter(self.sensor_iir_alpha)

        # Setpoint & Disturbance
        simulation.set_setpoint(setpoint_config=self.setpoint.config)
        simulation.set_disturbance(disturbance_config=self.disturbance.config if self.disturbance else None)

        # Metrics
        if self.setpoint.config.get('type') == 'step':
            metrics_config = {
                'step_mode': True,
                'step_time': self.setpoint.config.get('step_time', 0.0),
                'r_final': self.setpoint.config.get('v', 1.0),
                'max_rate_hz': (1.0 / dt) if dt > 0 else 1000.0
            }
        else:
            metrics_config = {'step_mode': False}
        simulation.set_metrics(metrics_config=metrics_config)

        time_series_data = None

        # Run simulation
        if isinstance(self.controller, PIDController):
            # Fast path: run full simulation entirely in C
            time_out = np.empty(num_samples, dtype=np.float32)
            sp_out = np.empty(num_samples, dtype=np.float32)
            y_out = np.empty(num_samples, dtype=np.float32)
            u_out = np.empty(num_samples, dtype=np.float32)
            results_dict = simulation.run(time_out=time_out, sp_out=sp_out, y_out=y_out, u_out=u_out)
            if get_time_series:
                # Trim to samples_written: Python allocates num_samples using float64 arithmetic
                # while the C core uses float32, so N may differ by 1 (e.g. floorf(499.999) = 499).
                # Without trimming, the last element stays zero-initialised, causing a spurious
                # line back to (t=0, y=0) in plots.
                n = results_dict.get('samples_written', num_samples)
                time_series_data = {
                    'time': time_out[:n],
                    'setpoint': sp_out[:n],
                    'measurement': y_out[:n],
                    'control_effort': u_out[:n]
                }

        elif hasattr(self.controller, 'predict'):
            # Hybrid path: We run the simulation step-by-step, calling the controller's predict method
            simulation.reset()
            
            # This ensures SNN membrane potentials and internal state are cleared
            if hasattr(self.controller, 'reset'):
                self.controller.reset()
            
            state_vector = np.zeros(3, dtype=np.float32)

            # Allocate buffers if requested
            if get_time_series:
                time_out = np.empty(num_samples, dtype=np.float32)
                sp_out = np.empty(num_samples, dtype=np.float32)
                y_out = np.empty(num_samples, dtype=np.float32)
                u_out = np.empty(num_samples, dtype=np.float32)

            for k in range(num_samples):
                simulation.get_state_vector(state_vector)
                error, measurement, setpoint = state_vector

                control_signal = self.controller.predict(setpoint, measurement, error)
                simulation.step(control_signal)

                if get_time_series:
                    time_out[k] = k * dt
                    sp_out[k] = setpoint
                    y_out[k] = measurement
                    u_out[k] = control_signal
            
            if get_time_series:
                time_series_data = {
                    'time': time_out,
                    'setpoint': sp_out,
                    'measurement': y_out,
                    'control_effort': u_out
                }

            # Manually add the final value and samples written for the hybrid run.
            # Use num_samples-1 to match the C fast path: the C core uses float32
            # arithmetic and writes one fewer sample than Python's int(t/dt)+1.
            final_measurement = y_out[-1] if get_time_series else measurement
            results_dict = simulation.get_metrics_results()
            # Use the mean of the last 2 seconds rather than the last sample so that
            # controllers with quantization-induced limit cycles (e.g. 4-bit Akida)
            # report their true steady-state value instead of the oscillation phase
            # at the exact end time.
            if get_time_series and len(y_out) >= 200:
                final_measurement = float(np.mean(y_out[-200:]))
            results_dict['y_final'] = final_measurement
            results_dict['samples_written'] = num_samples - 1
            if get_time_series:
                time_series_data = {k: v[:num_samples - 1] for k, v in time_series_data.items()}

        else:
            results_dict = simulation.get_metrics_results()

        # Results
        result = ExperimentResult(
            success=True,
            final_value=results_dict.get('y_final', 0.0),
            samples_written=results_dict.get('samples_written', 0),
            overshoot_percent=results_dict.get('overshoot_percent', 0.0),
            rise_time=results_dict.get('rise_time', 0.0),
            ise=results_dict.get('ise', 0.0),
            iae=results_dict.get('iae', 0.0),
            itae=results_dict.get('itae', 0.0),
            isu=results_dict.get('isu', 0.0),
            peak_value=results_dict.get('peak_value', 0.0),
            peak_time=results_dict.get('peak_time', 0.0),
            steady_state_error=results_dict.get('steady_state_error', 0.0),
            steady_state_error_percent=results_dict.get('steady_state_error_percent', 0.0),
            total_time=total_time,
            time_series=time_series_data
        )
        return result
