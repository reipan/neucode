"""
Hardware harness for the NeuCoDe toolkit.

Provides HardwareHarness which drives real embedded targets via CommunicationClient,
collects telemetry, and evaluates performance metrics using StandaloneMetrics.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import time
import numpy as np

from neucode.harness.base import BaseHarness
from neucode.results import ExperimentResult
from neucode.communication.client import CommunicationClient, LogLine, TelemetryLine
from neucode.communication.interface import BaseInterface
from neucode.controllers import Controller, ANNController, PIDController, SNNController
from neucode.signals import Setpoint, StepSetpoint, Disturbance
from neucode.simcore import StandaloneMetrics

@dataclass
class HardwareRunConfig:
    """
    Timing and behaviour options for a HardwareHarness run.

    :param empty_buffer_seconds: Duration in seconds to drain the receive buffer before each command.
    :param wait_pong_timeout_seconds: Maximum time in seconds to wait for a pong acknowledgement.
    :param send_stop_cmd_on_close: If True, send an exp_stop command when closing the connection.
    :param max_log_lines: Maximum number of firmware log lines to retain (None = unlimited).
    """
    empty_buffer_seconds: float = 0.2
    wait_pong_timeout_seconds: float = 2.0
    send_stop_cmd_on_close: bool = True
    max_log_lines: Optional[int] = 1000

class HardwareHarness(BaseHarness):
    """
    Harness for running experiments on real hardware via CommunicationClient.
    """
    def __init__(self,
                 controller: Controller,
                 setpoint: Optional[Setpoint] = None,
                 disturbance: Optional[Disturbance] = None,
                 interface: Optional[BaseInterface] = None,
                 run_config: Optional[HardwareRunConfig] = None):
        """
        Initialize the hardware harness.

        :param controller: A Controller instance (PIDController, ANNController, or SNNController).
        :param setpoint: An optional Setpoint defining the reference trajectory.
        :param disturbance: An optional Disturbance configuration (informational; not sent to hardware).
        :param interface: A BaseInterface implementation (e.g. SerialInterface) for byte transport.
        :param run_config: Optional HardwareRunConfig; defaults are used if None.
        """
        super().__init__(controller=controller, setpoint=setpoint, disturbance=disturbance)
        self.client = CommunicationClient(interface=interface)
        self.run_config = run_config or HardwareRunConfig()
        self.export_fn = None
        self._needs_reset = False

    def set_plant(self, plant):
        """
        HardwareHarness does not support setting a plant, as it uses real hardware.
        This is just here to satisfy the BaseHarness interface.
        """
        raise NotImplementedError("HardwareHarness uses real hardware; plant cannot be set.")
    
    def _wait_for_pong(self) -> bool:
        """
        Drain the buffer, send a ping, and wait for the pong acknowledgement.

        :returns: True if a pong was received within the configured timeout, False otherwise.
        """
        self.client.empty_buffer(seconds=self.run_config.empty_buffer_seconds)
        self.client.ping()

        start_time = time.time()
        while (time.time() - start_time) < self.run_config.wait_pong_timeout_seconds:
            msg = self.client.read(timeout=0.2)
            if msg is None:
                continue
            if isinstance(msg, LogLine) and msg.raw.strip() == "L,pong":
                return True
        return False
    
    def _set_controller_params(self):
        """
        Set the controller parameters on the hardware.
        Supports all three controller types: PID, ANN, and SNN.

        Notes:
        - SNN controller also needs the PID gains to be set for the hardware to set the correct context.
        - ANN controller does not need it yet, but we will set it anyway for consistency and future compatibility (we will switch to a context-based approach later).
        """
        required_keys = ['kp', 'ki', 'kd']
        if isinstance(self.controller, (ANNController, PIDController, SNNController)):
            # Check if params attribute exists and contains required keys
            if not hasattr(self.controller, 'params') or not all(k in self.controller.params for k in required_keys):
                raise AttributeError(f"Controller {type(self.controller).__name__} must have 'params' dict with keys {required_keys}")
            if isinstance(self.controller, ANNController):
                self.client.mode("ann")
            elif isinstance(self.controller, PIDController):
                self.client.mode("pid")
            elif isinstance(self.controller, SNNController):
                self.client.mode("snn")
            self.client.pid(
                float(self.controller.params['kp']),
                float(self.controller.params['ki']),
                float(self.controller.params['kd'])
            )
            # Set anti-windup gain if provided (prevents integral drift after settling)
            if hasattr(self.controller, 'params') and 'kaw' in self.controller.params:
                self.client.send(f"set kaw {float(self.controller.params['kaw'])}")
            if hasattr(self.controller, 'params') and 'd_alpha' in self.controller.params:
                self.client.send(f"set d_alpha {float(self.controller.params['d_alpha'])}")
        else:
            raise NotImplementedError("Only ANNController, PIDController, and SNNController are implemented in HardwareHarness.")
            
    def run(self, dt: float, total_time: float, get_time_series: bool = False,
            nozero: bool = False) -> ExperimentResult:
        """
        Run the experiment on connected hardware and return performance metrics.

        Always fetches the firmware dump buffer after the experiment for
        high-resolution (100 Hz) metrics and plots. Falls back to the 10 Hz
        real-time telemetry stream if the dump fails.

        :param dt: Expected sampling interval in seconds (fallback rate).
        :param total_time: Experiment duration in seconds.
        :param get_time_series: If True, populate time-series arrays in the returned result.
        :param nozero: If True, sends ``exp start nozero`` to skip sensor zeroing.
        :returns: ExperimentResult with performance metrics and optional time-series data.
        :raises RuntimeError: If no pong is received from the hardware within the timeout.
        :raises NotImplementedError: If the setpoint type is not StepSetpoint.
        """
        self.client.open()

        # check if we get pong
        if not self._wait_for_pong():
            self.client.close()
            raise RuntimeError("No pong response from hardware within timeout.")
        
        # configure controller
        self._set_controller_params()

        # set setpoint
        if self.setpoint is not None:
            if self.setpoint.config['type'] == 'step':
                self.client.setpoint_step(
                    float(self.setpoint.config['step_time']),
                    float(self.setpoint.config['v'])
                )
            else:
                raise NotImplementedError("Only StepSetpoint is implemented in HardwareHarness.")

        # start
        self.client.empty_buffer(seconds=self.run_config.empty_buffer_seconds)
        self.client.exp_start(nozero=nozero)

        # collect data
        time_series = {'t': [], 'sp': [], 'y': [], 'u': []} if get_time_series else None
        end_time = time.time() + total_time

        buffer_frames = None
        try:
            while True:
                if time.time() >= end_time:
                    break

                msg = self.client.read(timeout=dt + 0.1)
                if msg is None:
                    continue

                if isinstance(msg, TelemetryLine):
                    if get_time_series:
                        time_series['t'].append(msg.t)
                        time_series['sp'].append(msg.sp)
                        time_series['y'].append(msg.y)
                        time_series['u'].append(msg.u)
                elif isinstance(msg, LogLine):
                    print(f"    [FW] {msg.msg.strip()}")
        finally:
            if self.run_config.send_stop_cmd_on_close:
                try:
                    self.client.exp_stop()
                except Exception:
                    pass
            try:
                buffer_frames = self.client.exp_dump()
                print(f"    [dump] {len(buffer_frames)} frames at firmware rate")
            except Exception as exc:
                print(f"    [dump] failed: {exc}")
                buffer_frames = []
            self.client.close()

        if buffer_frames:
            t_arr = np.array([f.t for f in buffer_frames], dtype=np.float32)
            sp_arr = np.array([f.sp for f in buffer_frames], dtype=np.float32)
            y_arr = np.array([f.y for f in buffer_frames], dtype=np.float32)
            u_arr = np.array([f.u for f in buffer_frames], dtype=np.float32)
            dt_med = float(np.median(np.diff(t_arr))) if len(t_arr) > 1 else dt
            max_rate_hz = 1.0 / dt_med
        else:
            t_arr = np.array(time_series['t'], dtype=np.float32)
            sp_arr = np.array(time_series['sp'], dtype=np.float32)
            y_arr = np.array(time_series['y'], dtype=np.float32)
            u_arr = np.array(time_series['u'], dtype=np.float32)
            max_rate_hz = (1.0 / dt) if dt > 0 else 1000.0

        # Configure C metrics core for step-response metrics on hardware.
        if self.setpoint is not None and self.setpoint.config.get('type') == 'step':
            # StandaloneMetrics uses an internal time base starting at ~0 and advances by dt deltas
            # (i.e., effectively t - t0 when calc_batch integrates diff(t)).
            # Our telemetry timestamps start at the device time since experiment start and may not
            # begin at exactly 0.0 (often the first sample is ~0.01s). To keep step detection aligned,
            # express step_time relative to the first telemetry timestamp.
            t0 = float(t_arr[0]) if len(t_arr) > 0 else 0.0
            step_time_abs = float(self.setpoint.config.get('step_time', 0.0))
            metrics_config = {
                'step_mode': True,
                'step_time': max(0.0, step_time_abs - t0),
                'r_final': float(self.setpoint.config.get('v', 0.0)),
                'max_rate_hz': float(max_rate_hz),
            }
        else:
            metrics_config = {
                'step_mode': False,
                'max_rate_hz': float(max_rate_hz),
            }

        metrics = StandaloneMetrics(metrics_config=metrics_config)

        # Run metrics batch calculation
        metrics.process_telemetry(t_arr, sp_arr, y_arr, u_arr)
        results_dict = metrics.get_results()

        time_series_data = {
            'time': t_arr.tolist(),
            'setpoint': sp_arr.tolist(),
            'measurement': y_arr.tolist(),
            'control_effort': u_arr.tolist(),
            'buffer_frames': buffer_frames,
        }

        # We need to manually add y_final and samples_written 
        results_dict['y_final'] = y_arr[-1] if len(y_arr) > 0 else 0.0
        results_dict['samples_written'] = len(t_arr)

        # Some result do not work or have wrong scaling on hardware runs yet
        # But it works :D
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

    # DAgger support
    def load_weights(self, model_path: str | Path, scaler_path: str | Path | None = None):
        """
        Reload controller weights, export firmware headers, and prompt for flash.

        :param model_path: Path to model checkpoint (.pth).
        :param scaler_path: Optional path to scaler (.npz).
        """
        print(f"  [HW] Loading weights: {model_path}")
        self.controller.load_weights(model_path, scaler_path)
        if self.export_fn is not None:
            print(f"  [HW] Exporting model for firmware...")
            self.export_fn(model_path)
        input("  [HW] Flash the MCU with the exported model, then press Enter...")
        self._needs_reset = False

    def reset_to_zero(self, dt: float = 0.1, reset_time: float = 8.0,
                      pid_gains: dict | None = None):
        """
        PID-drive the motor back to y~0 between DAgger episodes.

        Uses ``nozero=True`` so the original sensor zero-reference is preserved
        across the entire DAgger session.

        :param dt: Sampling interval passed to ``run()`` (default 0.1 = telemetry rate).
        :param reset_time: Duration of the PID reset episode in seconds (default 8.0).
        :param pid_gains: PID gains dict with ``kp, ki, kd`` keys.
            Defaults to expert PID gains for reliable return through friction.
        """
        if pid_gains is None:
            pid_gains = {'kp': 0.020, 'ki': 0.050, 'kd': 0.015}

        saved_controller = self.controller
        saved_setpoint = self.setpoint

        self.controller = PIDController(
            kp=pid_gains['kp'], ki=pid_gains['ki'], kd=pid_gains['kd'],
        )
        self.set_setpoint(StepSetpoint(time=0.0, value=0.0))
        result = self.run(dt=dt, total_time=reset_time, get_time_series=True,
                          nozero=True)

        self.controller = saved_controller
        self.setpoint = saved_setpoint

        print(f"  [reset] PID->0  final_y={result.final_value:.2f} deg")

    def rollout(self, setpoint_value: float, episode_time: float, dt: float,
                step_delay: float = 0.0,
                reset_time: float = 8.0, pid_gains: dict | None = None,
                ) -> list[dict]:
        """
        Run one hardware episode and return reconstructed states.

        Sequence: PID-reset to zero -> step experiment -> dump 100 Hz buffer ->
        reconstruct ``(sp, y)`` into feature dicts for DAgger labeling.

        :param setpoint_value: Step setpoint magnitude.
        :param episode_time: Episode duration in seconds.
        :param dt: Sampling interval for ``run()``.
        :param step_delay: Seconds before the step fires (default 0.0).
            Set to ~1.0 for SNN controllers so LIF membranes can warm up.
        :param reset_time: PID reset duration before the episode.
        :param pid_gains: PID gains for the reset phase.
        :returns: List of state dicts with keys matching trainer CSV columns.
        """
        if self._needs_reset:
            self.reset_to_zero(reset_time=reset_time, pid_gains=pid_gains)

        self.set_setpoint(StepSetpoint(time=step_delay, value=setpoint_value))
        result = self.run(dt=dt, total_time=step_delay + episode_time,
                          get_time_series=True)
        self._needs_reset = True

        return self._reconstruct_states(result, firmware_dt=dt,
                                         target_dt=getattr(self, '_target_dt', None))

    @staticmethod
    def _reconstruct_states(result: ExperimentResult,
                            firmware_dt: float = 0.01,
                            target_dt: float | None = None) -> list[dict]:
        """
        Reconstruct controller input features from hardware telemetry.

        Uses the dump buffer if available, otherwise falls back to the
        telemetry stream. When *target_dt* is set and differs from the
        source sample period, (sp, y) pairs are linearly interpolated
        so that derivative and integral scales match the training data.

        :param result: ExperimentResult with time-series data.
        :param firmware_dt: Firmware control loop period (default 0.01 s).
        :param target_dt: Desired output sample period.  If ``None`` or equal
            to the source sample period, no interpolation is performed.
        :returns: List of state dicts.
        """
        ts = result.time_series or {}
        buffer_frames = ts.get('buffer_frames')

        if buffer_frames:
            t_src = np.array([float(f.t) for f in buffer_frames], dtype=np.float64)
            sp_src = np.array([float(f.sp) for f in buffer_frames], dtype=np.float64)
            y_src = np.array([float(f.y) for f in buffer_frames], dtype=np.float64)
            sample_dt = firmware_dt
        else:
            sp_arr = ts.get('setpoint', [])
            y_arr = ts.get('measurement', [])
            t_arr = ts.get('time', [])
            t_src = np.array([float(t) for t in t_arr], dtype=np.float64)
            sp_src = np.array([float(s) for s in sp_arr], dtype=np.float64)
            y_src = np.array([float(y) for y in y_arr], dtype=np.float64)
            if len(t_src) >= 2:
                sample_dt = float(t_src[1] - t_src[0])
            else:
                sample_dt = firmware_dt

        if target_dt is not None and target_dt < sample_dt - 1e-9 and len(t_src) >= 2:
            t_interp = np.arange(t_src[0], t_src[-1], target_dt)
            sp_interp = np.interp(t_interp, t_src, sp_src)
            y_interp = np.interp(t_interp, t_src, y_src)
            pairs = list(zip(sp_interp.tolist(), y_interp.tolist()))
            sample_dt = target_dt
        else:
            pairs = list(zip(sp_src.tolist(), y_src.tolist()))

        rows: list[dict] = []
        integral = 0.0
        prev_error = None
        for sp, y in pairs:
            error = sp - y
            integral += error * sample_dt
            if prev_error is None:
                derivative = 0.0
            else:
                derivative = (error - prev_error) / sample_dt
            rows.append({
                'setpoint':         sp,
                'measurement':      y,
                'error':            error,
                'integral_error':   integral,
                'derivative_error': derivative,
            })
            prev_error = error
        return rows