"""
Controller comparison evaluation for the NeuCoDe toolkit.

Runs multiple controllers through the same harness (sim or hardware),
collects metrics and plots, and saves artifacts.  Works with any
BaseHarness -- SimulationHarness and HardwareHarness are both supported
without special-casing.
"""
from __future__ import annotations

import datetime

from neucode.harness.base import BaseHarness
from neucode.harness import HardwareHarness
from neucode.reporting import MetricsTable, GraphPlotter
from neucode.signals import StepSetpoint


class ControllerComparison:
    """
    Run N controllers through the same harness and compare results.

    :param harness: A configured BaseHarness (sim or hardware).
    :param controllers: Dict mapping display names to controller instances.
        Entries with a None value are silently skipped (convenient for
        ``experiment.load_controller()`` which returns None on missing models).
    :param setpoints: List of step setpoint values to test.
    :param dt: Sampling interval passed to ``harness.run()``.
    :param total_time: Experiment duration in seconds.
    :param step_delay: Delay before the setpoint step (SNN warm-up).
    :param reset_controller: If provided, run this controller at setpoint=0
        between experiments to return the plant to its origin.  Typical for
        hardware runs; not needed for simulation.
    :param reset_time: Duration of the reset episode in seconds.
    :param reset_dt: Sampling interval for the reset episode.  Defaults
        to ``dt`` if not specified.
    :param title: Title for the metrics table and plot.
    :param color_map: Optional dict mapping controller names to hex colors.
    """

    def __init__(
        self,
        harness: BaseHarness,
        controllers: dict,
        setpoints: list[float],
        dt: float,
        total_time: float,
        step_delay: float = 0.0,
        reset_controller=None,
        reset_time: float = 4.0,
        reset_dt: float = None,
        title: str = None,
        color_map: dict = None,
    ):
        self.harness = harness
        self.controllers = {k: v for k, v in controllers.items() if v is not None}
        self.setpoints = setpoints
        self.dt = dt
        self.total_time = total_time
        self.step_delay = step_delay
        self.reset_controller = reset_controller
        self.reset_time = reset_time
        self.reset_dt = reset_dt or dt
        self.color_map = color_map or {}

        sp_label = "|".join(str(sp) for sp in setpoints)
        self.title = title or f"Comparison (sp={sp_label})"

        self.table = MetricsTable(title=self.title)
        self.plotter = GraphPlotter(title=self.title, color_map=self.color_map)
        self.results = {}

    def _reset_to_zero(self) -> None:
        self.harness.set_controller(self.reset_controller)
        self.harness.set_setpoint(StepSetpoint(time=0.0, value=0.0))
        kwargs = dict(dt=self.reset_dt, total_time=self.reset_time,
                      get_time_series=False)
        if isinstance(self.harness, HardwareHarness):
            kwargs['nozero'] = True
        res = self.harness.run(**kwargs)
        print(f"    [reset] final_y={res.final_value:.2f}")

    def run(self) -> dict:
        """
        Run all controllers across all setpoints.

        :returns: Dict mapping labels to ExperimentResult objects.
        """
        run_time = self.total_time + self.step_delay
        multi_sp = len(self.setpoints) > 1
        need_reset = self.reset_controller is not None
        first_run = True

        for sp in self.setpoints:
            print(f"\n  --- Setpoint {sp} ---")
            setpoint = StepSetpoint(time=self.step_delay, value=sp)
            self.harness.set_setpoint(setpoint)

            for name, ctrl in self.controllers.items():
                if not first_run and need_reset:
                    self._reset_to_zero()
                    self.harness.set_setpoint(setpoint)
                first_run = False

                if hasattr(ctrl, 'reset'):
                    ctrl.reset()

                self.harness.set_controller(ctrl)
                print(f"  Running {name} @ sp={sp}...")
                result = self.harness.run(
                    dt=self.dt, total_time=run_time, get_time_series=True,
                )

                label = f"{name} sp={sp}" if multi_sp else name
                self.table.add_run(result, label=label)
                self.plotter.add_run(result, label=label)
                self.results[label] = result
                print(f"    final={result.final_value:.2f}  "
                      f"rise={result.rise_time:.3f}s  "
                      f"sse={result.steady_state_error:.2f}")

        self.table.report()
        return self.results

    def save(self, prefix: str = None) -> str:
        """Save metrics table and plots to timestamped files.

        :param prefix: File path prefix (e.g. 'artifacts/v10_hw').
            Timestamp is appended automatically.
        :returns: The full prefix with timestamp.
        """
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{prefix}_{stamp}" if prefix else f"artifacts/comparison_{stamp}"
        self.plotter.save_pdf(path)
        self.plotter.save_png(path)
        self.table.save_json(path)
        print(f"  Saved -> {path}")
        return path
