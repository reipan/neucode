"""
Hardware telemetry recording for the NeuCoDe toolkit.

Provides HardwareRecorder which runs a HardwareHarness, captures the resulting
time series, and persists it alongside a JSON metadata file for later analysis or replay.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import json

from neucode.harness.hardware import HardwareHarness
from neucode.results import ExperimentResult

@dataclass
class RecordingMetadata:
    """
    Metadata captured alongside a hardware telemetry recording.

    :param created_timestamp: Unix timestamp (seconds) at which the recording was initiated.
    :param dt: Sampling interval in seconds used during the run.
    :param total_time: Total run duration in seconds.
    :param controller: Serialised controller configuration dictionary.
    :param setpoint: Serialised setpoint configuration, or ``None`` if not set.
    :param disturbance: Serialised disturbance configuration, or ``None`` if not set.
    :param actuator_limits: Actuator saturation limits dict, or ``None`` if unrestricted.
    :param metrics_config: Metrics engine configuration captured for reproducibility.
    :param notes: Free-text annotation attached to this recording.
    """
    created_timestamp: float
    dt: float
    total_time: float
    controller: dict
    setpoint: dict | None
    disturbance: dict | None
    actuator_limits: dict | None
    metrics_config: dict | None = None
    notes: str = ""


@dataclass(frozen=True)
class HardwareRecording:
    """
    Immutable bundle returned by HardwareRecorder.record when
    ``return_recording=True``.

    :param output_dir: Directory where telemetry CSV and metadata JSON were written.
    :param result: Full ExperimentResult including time series.
    :param metadata: Metadata snapshot captured at record time.
    """
    output_dir: Path
    result: ExperimentResult
    metadata: RecordingMetadata

class HardwareRecorder:
    """
    Simple hardware recorder that stores telemetry and metadata from a hardware harness run.
    @todo: later make sure we can run a scenario and record the data automatically (scenarios right now only run a simulation).
    """
    def __init__(self, harness: HardwareHarness):
        """
        Initialise the recorder with a configured hardware harness.

        :param harness: The HardwareHarness instance to record from.
            It must already have a controller and setpoint attached.
        """
        self.harness = harness

    def record(
        self,
        dt: float,
        total_time: float,
        output_dir: str | Path = None,
        notes: str = "",
        return_recording: bool = False,
    ) -> Path | HardwareRecording:
        """
        Run the harness, save telemetry and metadata, and return the output path.

        :param dt: Sampling interval in seconds.
        :param total_time: Total run duration in seconds.
        :param output_dir: Directory to write output files into.  Defaults to
            ``artifacts/recordings/<unix_timestamp>``.
        :param notes: Optional free-text annotation stored in the metadata file.
        :param return_recording: If ``True``, return a HardwareRecording
            bundle instead of just the output path.
        :returns: The output directory path, or a HardwareRecording when
            ``return_recording=True``.
        :rtype: pathlib.Path or HardwareRecording
        :raises RuntimeError: If the harness run produces no time series data.
        """
        record_time = time.time()

        if output_dir is None:
            output_dir = Path("artifacts/recordings") / str(int(record_time))
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.harness.run(dt=dt, total_time=total_time, get_time_series=True)
        if result.time_series is None:
            raise RuntimeError("No time series data available from the hardware harness run.")
        
        # Calc error
        result.time_series['error'] = np.array(result.time_series['setpoint']) - np.array(result.time_series['measurement'])

        # Convert time series data to numpy arrays
        t = np.asarray(result.time_series['time'], dtype=float)
        sp = np.asarray(result.time_series['setpoint'], dtype=float)
        y = np.asarray(result.time_series['measurement'], dtype=float)
        u = np.asarray(result.time_series['control_effort'], dtype=float)
        e = np.asarray(result.time_series['error'], dtype=float)

        # Capture the effective metrics configuration for reproducibility.
        # StandaloneMetrics uses an internal time base starting near 0 and advances by dt deltas.
        # HardwareHarness therefore expresses step_time relative to the first telemetry timestamp.
        metrics_config: dict | None = None
        try:
            max_rate_hz = (1.0 / dt) if dt > 0 else 1000.0
            sp_cfg = getattr(self.harness.setpoint, "config", None) if getattr(self.harness, "setpoint", None) is not None else None
            if isinstance(sp_cfg, dict) and sp_cfg.get("type") == "step":
                t0 = float(t[0]) if t.size else 0.0
                step_time_abs = float(sp_cfg.get("step_time", 0.0))
                metrics_config = {
                    "step_mode": True,
                    "step_time": max(0.0, step_time_abs - t0),
                    "r_final": float(sp_cfg.get("v", 0.0)),
                    "max_rate_hz": float(max_rate_hz),
                    "telemetry_t0": t0,
                    "step_time_abs": step_time_abs,
                }
            else:
                metrics_config = {
                    "step_mode": False,
                    "max_rate_hz": float(max_rate_hz),
                }
        except Exception:
            metrics_config = None
        
        # Plant Observation Space Padding Array
        obs_0 = np.zeros_like(t, dtype=float)
        obs_1 = np.zeros_like(t, dtype=float)
        obs_2 = np.zeros_like(t, dtype=float)

        # Fake Episode ID
        episode_id = np.zeros_like(t, dtype=float)

        # Save telemetry data
        telemetry_data = np.vstack((t, sp, y, u, e, obs_0, obs_1, obs_2, episode_id)).T
        telemetry_file = output_dir / "telemetry.csv"
        np.savetxt(
            telemetry_file,
            telemetry_data,
            delimiter=",",
            header="time,setpoint,measurement,control_effort,error,obs_0,obs_1,obs_2,episode_id",
            comments='',
            fmt="%.4f",
        )

        # Save metadata
        metadata = RecordingMetadata(
            created_timestamp=record_time,
            dt=dt,
            total_time=total_time,
            controller=getattr(self.harness.controller, "to_dict", lambda: vars(self.harness.controller))(),
            setpoint=getattr(self.harness.setpoint, "to_dict", lambda: getattr(self.harness.setpoint, "config", None))(),
            disturbance=getattr(self.harness.disturbance, "to_dict", lambda: getattr(self.harness.disturbance, "config", None))(),
            actuator_limits=getattr(self.harness, "actuator_limits", None),
            metrics_config=metrics_config,
            notes=notes
        )
        meta_path = output_dir / "meta.json"
        meta_path.write_text(json.dumps(metadata.__dict__, indent=4))

        if return_recording:
            return HardwareRecording(output_dir=output_dir, result=result, metadata=metadata)

        return output_dir