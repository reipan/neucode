"""
Abstract harness base class for the NeuCoDe toolkit.

Defines the common controller/setpoint/disturbance contract shared by
SimulationHarness and HardwareHarness.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

from neucode.controllers import Controller
from neucode.signals import Setpoint, StepSetpoint, Disturbance
from neucode.plants import Plant
from neucode.results import ExperimentResult

class BaseHarness(ABC):
    """
    Abstract base class defining the interface for all harnesses.
    Used so that simulation and hardware harnesses can be used interchangeably.
    """
    def __init__(self, controller: Controller, setpoint: Setpoint | None, disturbance: Disturbance = None):
        """
        Initialize the harness with a controller, setpoint, and optional disturbance.

        :param controller: A Controller instance (PID, ANN, or SNN), or None for deferred assignment.
        :param setpoint: A Setpoint instance defining the reference trajectory, or None.
        :param disturbance: An optional Disturbance instance for load/noise injection.
        :raises TypeError: If controller or setpoint are provided but are not the correct types.
        """
        # Note to myself: We dont add a plant here, because we only need it on the sim harness.
        # The hardware harness will use the plant that is physically connected to the hardware.
        if controller is not None and not isinstance(controller, Controller):
            raise TypeError("controller must be an instance of Controller.")
        if setpoint is not None and not isinstance(setpoint, Setpoint):
            raise TypeError("setpoint must be an instance of Setpoint or None.")
        
        self.controller = controller
        self.setpoint = setpoint
        self.disturbance = disturbance

    def set_controller(self, controller: Controller):
        """
        Swap the active controller.

        :param controller: The new Controller instance to use.
        :raises TypeError: If controller is not a Controller instance.
        """
        if not isinstance(controller, Controller):
            raise TypeError("controller must be an instance of Controller.")
        self.controller = controller

    def set_setpoint(self, setpoint: Setpoint):
        """
        Swap the active setpoint signal.

        :param setpoint: The new Setpoint instance to use.
        :raises TypeError: If setpoint is not a Setpoint instance.
        """
        if not isinstance(setpoint, Setpoint):
            raise TypeError("setpoint must be an instance of Setpoint.")
        self.setpoint = setpoint

    def set_disturbance(self, disturbance: Disturbance):
        """
        Swap the active disturbance configuration.

        :param disturbance: A Disturbance instance, or None to disable disturbances.
        :raises TypeError: If disturbance is neither a Disturbance instance nor None.
        """
        if disturbance is not None and not isinstance(disturbance, Disturbance):
            raise TypeError("disturbance must be an instance of Disturbance or None.")
        self.disturbance = disturbance

    @abstractmethod
    def set_plant(self, plant: Plant):
        """
        Swap the plant used by the harness.

        :param plant: A Plant instance (simulation) or ignored (hardware).
        """
        pass

    @abstractmethod
    def run(self, dt: float, total_time: float, get_time_series: bool = False) -> ExperimentResult:
        """
        Execute the experiment for the specified duration.

        :param dt: Sampling interval in seconds.
        :param total_time: Total experiment duration in seconds.
        :param get_time_series: If True, populate time-series arrays in the returned result.
        :returns: An ExperimentResult with performance metrics and optional time-series data.
        """
        pass

    def evaluate(self, dt: float, total_time: float) -> ExperimentResult:
        """
        Run the experiment without collecting time-series data.

        :param dt: Sampling interval in seconds.
        :param total_time: Total experiment duration in seconds.
        :returns: An ExperimentResult with performance metrics only.
        """
        return self.run(dt, total_time, get_time_series=False)

    def load_weights(self, model_path: str | Path, scaler_path: str | Path | None = None):
        """
        Reload controller weights in-place (no reconstruction needed).

        :param model_path: Path to model checkpoint (.pth).
        :param scaler_path: Optional path to scaler (.npz). When provided the
            controller also reloads its input normalization parameters.
        """
        self.controller.load_weights(model_path, scaler_path)

    def rollout(self, setpoint_value: float, episode_time: float, dt: float,
                step_delay: float = 0.0) -> list[dict]:
        """
        Run one episode and return visited states as feature dicts.

        Each dict contains the keys expected by the replacement trainer CSV:
        ``setpoint, measurement, error, integral_error, derivative_error``.

        Works for ANN/SNN controllers with ``record_state=True``.
        Subclasses may override for hardware-specific state reconstruction.

        :param setpoint_value: Step setpoint magnitude.
        :param episode_time: Episode duration in seconds.
        :param dt: Control loop timestep in seconds.
        :param step_delay: Seconds before the step fires (default 0.0).
            Set to ~1.0 for SNN controllers so LIF membranes can warm up.
        :returns: List of state dicts, one per control tick.
        """
        if hasattr(self.controller, 'reset'):
            self.controller.reset()
        if hasattr(self.controller, 'reset_state_history'):
            self.controller.reset_state_history()

        self.set_setpoint(StepSetpoint(time=step_delay, value=setpoint_value))
        self.run(dt=dt, total_time=step_delay + episode_time)

        states = self.controller.get_state_history()
        return [
            {
                'setpoint': s['setpoint'],
                'measurement': s['measurement'],
                'error': s['error'],
                'integral_error': s['integral'],
                'derivative_error': s['derivative'],
            }
            for s in states
        ]
