"""
Runtime controller implementations for the NeuCoDe simulation and hardware harnesses.

Provides PIDController (reference teacher), ANNController (MLP-based replacement),
SNNController (spiking neural network replacement), KerasController (quantized Keras
model inference), and AkidaController (neuromorphic hardware inference via .fbz model).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

from ._torch_optional import (
    torch, nn,
    torch_available,
    snntorch_available
)

# Only import architecture classes when PyTorch is available.
# Deployed controllers don't need these.
if torch_available:
    from .architectures import NoContextMLPArchitecture, HybridControlSNN

class Controller(ABC):
    """
    Abstract base class for controllers.
    """

class PIDController(Controller):
    """
    Proportional-Integral-Derivative (PID) controller model.
    """
    def __init__(self, kp: float, ki: float, kd: float):
        """
        Initialize the PID controller with the given gains.

        :param kp: Proportional gain.
        :param ki: Integral gain.
        :param kd: Derivative gain in the pre-scaled industrial convention
            (kd_industrial = kd_textbook / dt), consistent with the C simulation core.
        """
        if not all(isinstance(g, (int, float)) for g in [kp, ki, kd]):
            raise TypeError("PID gains (kp, ki, kd) must be numeric values.")

        self.params = {
            'kp': kp,
            'ki': ki,
            'kd': kd
        }

    def __repr__(self):
        """
        Return a string representation of the PID controller with its gains.
        """
        return f" (PID: kp={self.params['kp']}, ki={self.params['ki']}, kd={self.params['kd']})"

class ANNController(Controller):
    """
    An Artificial Neural Network (ANN) based controller that uses a trained
    PyTorch model to compute the control variable.
    """
    def __init__(self,
                 model_path: str,
                 scaler_path: str,
                 model_architecture: nn.Module = None,
                 dt: float = 0.01,
                 device: str = 'cpu',
                 actuator_limits: dict | None = None,
                 record_state: bool = False):
        """
        Initialize the ANN controller with a trained model and scaler.

        :param model_path: Path to the saved PyTorch model state dict (.pth).
        :param scaler_path: Path to the saved scaler parameters (.npz); must contain
            'mean', 'scale', and optionally 'deriv_clip'.
        :param model_architecture: Optional pre-instantiated nn.Module to use as the
            inference model. If None, a NoContextMLPArchitecture is inferred from the
            state dict shape.
        :param dt: Control loop time step in seconds (default 0.01).
        :param device: Torch device string, e.g. 'cpu' or 'cuda' (default 'cpu').
        :param actuator_limits: Optional dict with 'u_min' and 'u_max' keys defining
            the output clamp range (default +/-10.0).
        :param record_state: If True, the controller will record visited states (setpoint, measurement, error, integral, derivative)
        """
        if not torch_available:
            raise ImportError("Cannot initialize ANNController: PyTorch is not available.")

        self.device = torch.device(device)
        self.model_path = model_path
        self.dt = dt
        self.integral_error = 0.0
        self.prev_error = 0.0
        self._record_state = record_state
        self._state_history: list[dict] = []

        # Extract actuator limits
        if actuator_limits is not None:
            self.u_max = float(actuator_limits.get('u_max', 10.0))
            self.u_min = float(actuator_limits.get('u_min', -10.0))
        else:
            self.u_max = 10.0
            self.u_min = -10.0

        with np.load(scaler_path) as scaler_data:
            self.scaler_mean  = torch.tensor(scaler_data['mean'],  dtype=torch.float32).to(self.device)
            self.scaler_scale = torch.tensor(scaler_data['scale'], dtype=torch.float32).to(self.device)
            self.deriv_clip   = float(scaler_data['deriv_clip'][0]) if 'deriv_clip' in scaler_data.files else float('inf')
            _int_mean = float(scaler_data['mean'][3])
            _int_std  = float(scaler_data['scale'][3])
            # Safety rail: keeps integral within training distribution if error persists.
            # 5 sigma covers 99.99994% of training samples so it never fires in normal operation,
            # but catches runaway accumulation (e.g. sensor loss, persistent large error).
            self.integral_clip_max = _int_mean + 5.0 * _int_std
            self.integral_clip_min = _int_mean - 5.0 * _int_std

        state_dict = torch.load(model_path, map_location=self.device)
        if model_architecture is None:
            input_size  = state_dict['layer1.weight'].shape[1]
            hidden_size = state_dict['layer1.weight'].shape[0]
            self.model = NoContextMLPArchitecture(input_size=input_size, hidden_size=hidden_size).to(self.device)
        else:
            if not isinstance(model_architecture, nn.Module):
                raise TypeError("model_architecture must be an instance of torch.nn.Module.")
            self.model = model_architecture.to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def reset(self):
        """
        Reset integral and derivative state between episodes.
        """
        self.integral_error = 0.0
        self.prev_error = 0.0

    def reset_state_history(self) -> None:
        """
        Clear recorded state history.
        """
        self._state_history = []

    def get_state_history(self) -> list[dict]:
        """
        Return states visited during the last rollout.

        Each dict contains: setpoint, measurement, error, integral, derivative.
        Only populated when record_state=True was passed at construction time.
        """
        return list(self._state_history)

    def load_weights(self, model_path, scaler_path=None):
        """
        Load new model weights and optionally a new scaler.

        :param model_path: Path to the saved PyTorch model state dict (.pth).
        :param scaler_path: Optional path to the saved scaler parameters (.npz).
        """
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model_path = model_path

        if scaler_path is not None:
            with np.load(scaler_path) as scaler_data:
                self.scaler_mean  = torch.tensor(scaler_data['mean'],  dtype=torch.float32).to(self.device)
                self.scaler_scale = torch.tensor(scaler_data['scale'], dtype=torch.float32).to(self.device)
                self.deriv_clip   = float(scaler_data['deriv_clip'][0]) if 'deriv_clip' in scaler_data.files else float('inf')
                _int_mean = float(scaler_data['mean'][3])
                _int_std  = float(scaler_data['scale'][3])
                self.integral_clip_max = _int_mean + 5.0 * _int_std
                self.integral_clip_min = _int_mean - 5.0 * _int_std

        self.reset()

    def predict(self, setpoint: float, measurement: float, error: float) -> float:
        """
        Compute control output using features prepared identically to the training data generator.

        The derivative is finite-differenced from the previous error and clipped to the
        99th-percentile training bound. The integral is a plain cumulative sum of error*dt,
        clipped to +/-5 sigma of the training distribution to prevent runaway accumulation.

        :param setpoint: Current reference value.
        :param measurement: Current process variable.
        :param error: Pre-computed tracking error (setpoint - measurement).
        :return: Control effort clamped to [u_min, u_max].
        """
        derivative_error = (error - self.prev_error) / self.dt
        if self.deriv_clip != float('inf'):
            derivative_error = max(-self.deriv_clip, min(self.deriv_clip, derivative_error))

        # Plain cumsum integral - matches datagen; clipped to 5 sigma safety bound.
        self.integral_error = float(np.clip(
            self.integral_error + error * self.dt,
            self.integral_clip_min, self.integral_clip_max
        ))

        # Record visited state for DAgger re-labelling.
        if self._record_state:
            self._state_history.append({
                'setpoint':   setpoint,
                'measurement': measurement,
                'error':      error,
                'integral':   self.integral_error,
                'derivative': derivative_error,
            })

        features = torch.tensor(
            [[setpoint, measurement, error, self.integral_error, derivative_error]],
            dtype=torch.float32, device=self.device
        )
        scaled = (features - self.scaler_mean) / self.scaler_scale

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                u_raw = self.model(scaled).item()

        self.prev_error = error
        return float(np.clip(u_raw, self.u_min, self.u_max))

    def __repr__(self):
        """
        Return a string representation of the ANN controller with its model path.
        """
        return f" (ANN: model_path='{self.model_path}')"


class BaseSNNController(Controller):
    """
    Abstract base class for SNN-family controllers.

    Owns all signal preprocessing shared by SNNController, KerasController and
    AkidaController:

    - Integral and derivative accumulation with safety clipping
    - MinMax scaling of the 5-dim context vector
      ``[setpoint, measurement, error, integral_error, derivative_error]``
    - Zero-setpoint deadband (drains state when setpoint ~= 0 and error ~= 0)
    - EMA output filter and actuator clamping

    Encoding (delta spikes, rate spikes, etc.) is the model's responsibility.
    Subclasses whose models were trained on the legacy 7-dim fused format can
    call ``_compute_delta_spikes(error)`` in their ``_forward``.

    Subclasses must implement :meth:`_forward` which receives the 5-dim scaled
    context vector and the raw error, and returns the raw scalar output in
    ``[0, 1]``.
    """

    def __init__(
        self,
        scaler_path: str,
        dt: float = 0.01,
        filter_alpha: float = 0.5,
        delta_threshold: float = 0.005,
        integral_deadzone: float = 0.0,
        output_deadzone: float = 0.0,
        err_deadband: float = 0.15,
        actuator_limits: dict | None = None,
        record_state: bool = False,
        error_encoding: str = 'delta',
        max_rate_input: float = 1.0,
    ):
        """
        Initialise shared preprocessing state.

        :param scaler_path: Path to the MinMax scaler ``.npz`` file; must contain
            ``data_min``, ``data_scale``, ``target_scale``, and optionally
            ``deriv_clip``.
        :param dt: Control loop time step in seconds (default 0.01).
        :param filter_alpha: EMA smoothing coefficient for the output signal.
            ``1.0`` is unfiltered; lower values suppress spike-induced noise at
            the cost of added latency (default 0.5).
        :param delta_threshold: Minimum error change required to emit a spike in
            the delta encoder (default 0.005, matches ``HybridControlSNN``).
        :param integral_deadzone: Absolute error threshold below which the
            integral is frozen.  Prevents limit-cycle windup when a quantized
            output (e.g. 4-bit Akida) alternates between discrete levels near
            the setpoint (default 0.0 = no deadzone).
        :param err_deadband: Absolute error below which output is zeroed.
            Matches firmware ``NC_SNN_ERR_DEADBAND_DEG`` (default 0.15).
            Set to 0.0 to disable.
        :param actuator_limits: Optional dict with ``u_min`` / ``u_max`` keys
            (default +/-10.0).
        :param record_state: If True, the controller will record visited states
            (setpoint, measurement, error, integral, derivative).
        """
        self.dt                = dt
        self.filter_alpha      = filter_alpha
        self.delta_threshold   = delta_threshold
        self.integral_deadzone = integral_deadzone
        self.output_deadzone   = output_deadzone
        self.err_deadband      = err_deadband
        self.error_encoding    = error_encoding
        self.max_rate_input    = max_rate_input

        self._record_state = record_state
        self._state_history: list[dict] = []

        if actuator_limits is not None:
            self.u_max = float(actuator_limits.get('u_max',  10.0))
            self.u_min = float(actuator_limits.get('u_min', -10.0))
        else:
            self.u_max =  10.0
            self.u_min = -10.0

        self._load_scaler(scaler_path)
        self._reset_state()

    def _load_scaler(self, scaler_path):
        """
        Load scaler parameters and per-feature clip bounds from an .npz file.

        :param scaler_path: Path to the saved MinMax scaler parameters (.npz); must contain
            ``data_min``, ``data_scale``, ``target_scale``, and optionally
            ``deriv_clip``.
        """
        with np.load(scaler_path) as data:
            self.scaler_min   = data['data_min'].astype(np.float32)
            self.scaler_scale = data['data_scale'].astype(np.float32)
            self.output_scale = float(data['target_scale']) if 'target_scale' in data else 10.0

            self.clip_min = data['clip_min'].astype(np.float32)
            self.clip_max = data['clip_max'].astype(np.float32)
            self.deriv_clip = float(self.clip_max[4])

            if 'max_rate_input' in data:
                self.max_rate_input = float(data['max_rate_input'])

    def _reset_state(self):
        """Zero all incremental state (integral, derivative, EMA, delta encoder)."""
        self.integral_error    = 0.0
        self.prev_error        = 0.0
        self.prev_output       = 0.0
        self._delta_prev       = 0.0   # delta encoder memory
        self._delta_initialized = False
        self._last_error       = 0.0   # tracked for adaptive filter

    def reset(self):
        """
        Reset all internal state between simulation episodes.

        Subclasses that hold additional state (e.g. membrane potentials) should
        call ``super().reset()`` and then reset their own state.
        """
        self._reset_state()

    def reset_state_history(self) -> None:
        """
        Clear recorded state history (call between DAgger episodes).
        """
        self._state_history = []

    def get_state_history(self) -> list[dict]:
        """
        Return states visited during the last rollout.

        Each dict contains: setpoint, measurement, error, integral, derivative.
        Only populated when record_state=True was passed at construction time.
        """
        return list(self._state_history)

    def _preprocess(self, setpoint: float, measurement: float, error: float):
        """
        Build the scaled 5-dim context vector.

        Handles derivative/integral accumulation, feature clipping, and MinMax
        scaling.  Encoding (delta spikes, rate spikes, etc.) is left to the
        model or the subclass ``_forward`` implementation.

        Returns ``None`` when the zero-setpoint deadband fires (caller should
        return 0.0 immediately).

        :return: float32 ndarray of shape ``(5,)`` or ``None``.
        """
        # Zero-setpoint deadband
        if abs(setpoint) < 0.001 and abs(error) < 0.05:
            return None

        # Derivative (finite-difference)
        derivative_error = (error - self.prev_error) / self.dt
        self.prev_error  = error

        # Integral (cumsum, clipped to feature clip bounds).
        # Deadzone: freeze integral when |error| is below threshold to prevent
        # limit-cycle windup caused by quantized output levels (e.g. 4-bit Akida).
        if abs(error) > self.integral_deadzone:
            self.integral_error = float(np.clip(
                self.integral_error + error * self.dt,
                self.clip_min[3], self.clip_max[3],
            ))

        # Record visited state for DAgger re-labelling (before model call).
        if self._record_state:
            self._state_history.append({
                'setpoint':   setpoint,
                'measurement': measurement,
                'error':      error,
                'integral':   self.integral_error,
                'derivative': derivative_error,
            })

        # Clip and scale context
        raw_context = np.array(
            [setpoint, measurement, error, self.integral_error, derivative_error],
            dtype=np.float32,
        )
        clipped_context = np.clip(raw_context, self.clip_min, self.clip_max)
        scaled_context = (clipped_context - self.scaler_min) / np.maximum(self.scaler_scale, 1e-12)

        return scaled_context

    def _compute_delta_spikes(self, error: float) -> np.ndarray:
        """
        Stateful delta spike encoding of the error signal.

        Returns a 2-element float32 array ``[pos_spike, neg_spike]``.
        Used by subclasses whose models were trained on the fused 7-dim
        input format (e.g. KerasController, AkidaController).
        """
        if not self._delta_initialized:
            self._delta_prev        = error
            self._delta_initialized = True
        diff   = error - self._delta_prev
        spikes = np.zeros(2, dtype=np.float32)
        if diff > self.delta_threshold:
            spikes[0]        = 1.0
            self._delta_prev += self.delta_threshold
        elif diff < -self.delta_threshold:
            spikes[1]        = 1.0
            self._delta_prev -= self.delta_threshold
        return spikes

    def _compute_rate_spikes(self, error: float) -> np.ndarray:
        """
        Stateless rate encoding of the error signal.

        Returns a 2-element float32 array ``[pos_rate, neg_rate]`` with
        continuous values in [0, 1] proportional to |error|/max_rate_input.
        """
        rate = min(abs(error) / self.max_rate_input, 1.0)
        spikes = np.zeros(2, dtype=np.float32)
        if error > 0:
            spikes[0] = rate
        elif error < 0:
            spikes[1] = rate
        return spikes

    def _compute_error_encoding(self, error: float) -> np.ndarray:
        """Dispatch to delta or rate encoding based on ``self.error_encoding``."""
        if self.error_encoding == 'rate':
            return self._compute_rate_spikes(error)
        return self._compute_delta_spikes(error)

    @property
    def _error_for_filter(self) -> float:
        """Most recent error magnitude, used by adaptive-alpha subclasses."""
        return self._last_error

    def _postprocess(self, raw_output: float) -> float:
        """
        Apply EMA filter, output scaling and actuator clamping.

        :param raw_output: Raw scalar from ``_forward()`` in ``[0, 1]``.
        :return: Control effort clamped to ``[u_min, u_max]``.
        """
        u = (raw_output - 0.5) * self.output_scale
        u = self.filter_alpha * u + (1.0 - self.filter_alpha) * self.prev_output
        self.prev_output = u
        return float(np.clip(u, self.u_min, self.u_max))

    @abstractmethod
    def _forward(self, scaled_context: np.ndarray, error: float) -> float:
        """
        Run one inference step.

        :param scaled_context: float32 ndarray of shape ``(5,)`` -- MinMax-scaled
            ``[setpoint, measurement, error, integral, derivative]``.
        :param error: Raw (unscaled) tracking error, available for subclasses
            that need it for encoding (e.g. delta spikes, model encoder).
        :return: Raw scalar output in ``[0, 1]``.
        """

    def predict(self, setpoint: float, measurement: float, error: float) -> float:
        """
        Compute one control step.

        :param setpoint: Current reference value.
        :param measurement: Current process variable.
        :param error: Pre-computed tracking error (setpoint - measurement).
        :return: Control effort clamped to ``[u_min, u_max]``.
        """
        # Error deadband: zero output near setpoint (matches firmware NC_SNN_ERR_DEADBAND_DEG).
        if self.err_deadband > 0.0 and abs(error) < self.err_deadband:
            self.prev_output = 0.0
            return 0.0

        # Output deadzone: hold last output when near setpoint to prevent
        # limit-cycle oscillation from quantized (e.g. 4-bit Akida) outputs.
        if self.output_deadzone > 0.0 and abs(error) < self.output_deadzone:
            return float(np.clip(self.prev_output, self.u_min, self.u_max))

        scaled = self._preprocess(setpoint, measurement, error)
        if scaled is None:
            self.prev_output = 0.0
            return 0.0
        self._last_error = error
        return self._postprocess(self._forward(scaled, error))

class SNNController(BaseSNNController):
    """
    Spiking Neural Network controller. Maintains membrane state between steps.
    """
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        device: str = 'cpu',
        dt: float = 0.01,
        filter_alpha: float = 0.5,
        hidden_size: int = 256,
        architecture: str = 'hybrid',
        population_size: int = 64,
        actuator_limits: dict | None = None,
        record_inputs: bool = False,
        record_state: bool = False
    ):
        """
        Initialize the SNN controller with a trained model and scaler.

        :param model_path: Path to the saved PyTorch model state dict (.pth).
        :param scaler_path: Path to the saved MinMax scaler parameters (.npz); must contain
            ``data_min``, ``data_scale``, ``target_scale``, and optionally ``deriv_clip``.
        :param device: Torch device string, e.g. ``'cpu'`` or ``'cuda'`` (default ``'cpu'``).
        :param dt: Control loop time step in seconds (default 0.01).
        :param filter_alpha: EMA smoothing coefficient applied to the raw SNN output.
            ``1.0`` is unfiltered; lower values suppress spike-induced output noise
            at the cost of added latency (default 0.5).
        :param hidden_size: Number of hidden neurons in the HybridControlSNN (default 256).
        :param actuator_limits: Optional dict with ``u_min`` / ``u_max`` keys defining
            the output clamp range (default +/-10.0).
        :param record_inputs: If True, record the fused 7-dim input vector
            ``[spike_pos, spike_neg, ctx_0..4]`` and raw SNN output at each
            predict() call for Akida QAT calibration. Retrieve with
            ``get_recorded_dataset()``. Resets on each ``reset()`` call
            (default False).
        :param record_state: If True, the controller will record visited states
            (setpoint, measurement, error, integral, derivative) for DAgger re-l
        """
        if not torch_available and snntorch_available:
            raise ImportError("Cannot initialize SNNController: PyTorch or snnTorch is not available.")

        super().__init__(
            scaler_path=scaler_path,
            dt=dt,
            filter_alpha=filter_alpha,
            actuator_limits=actuator_limits,
            record_state=record_state
        )

        self.device      = torch.device(device)
        self.model_path  = model_path
        self.hidden_size = hidden_size

        # Load snnTorch model
        if architecture == 'population':
            from neucode.architectures import PopulationControlSNN
            self.model = PopulationControlSNN(hidden_size=self.hidden_size, population_size=population_size).to(self.device)
        elif architecture == 'spike':
            from neucode.architectures import SpikeControlSNN
            rate_zp = float((0.0 - self.scaler_min[2]) / self.scaler_scale[2]) if self.scaler_scale[2] != 0 else 0.0
            self.model = SpikeControlSNN(hidden_size=self.hidden_size, population_size=population_size, rate_zero_point=rate_zp).to(self.device)
        else:
            self.model = HybridControlSNN(hidden_size=self.hidden_size).to(self.device)
            
        state_dict = torch.load(model_path, map_location=self.device)
        if "encoder.prev_spike_value" in state_dict:
            del state_dict["encoder.prev_spike_value"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.mem = None

        # Input recording via encoder forward hook - captures exact spikes without
        # double-stepping the stateful delta encoder.
        self._record_inputs  = False
        self._recorded_spike = None
        self._recorded_inputs:  list = []
        self._recorded_outputs: list = []
        if record_inputs:
            self.enable_recording()

    def enable_recording(self):
        if self._record_inputs:
            return
        self._record_inputs = True
        self.model.encoder.register_forward_hook(
            lambda _m, _i, out: setattr(
                self, '_recorded_spike', out.detach().cpu().numpy()[0, 0, :]
            )
        )

    def load_weights(self, model_path, scaler_path=None):
        state_dict = torch.load(model_path, map_location=self.device)
        if "encoder.prev_spike_value" in state_dict:
            del state_dict["encoder.prev_spike_value"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model_path = model_path

        if scaler_path is not None:
            self._load_scaler(scaler_path)

        self.reset()

    def reset(self):
        """Reset membrane state, encoder state and recording buffers."""
        super().reset()
        self.model.reset_states()
        self.mem             = None
        self._recorded_spike = None
        self._recorded_inputs.clear()
        self._recorded_outputs.clear()

    def _forward(self, scaled_context: np.ndarray, error: float) -> float:
        """Run one snnTorch forward step and return raw output in [0, 1]."""
        # Clip scaled features to [0,1] -- MinMaxScaler guarantees this range during
        # training, but OOD inputs can exceed it at inference.
        scaled_context = np.clip(scaled_context, 0.0, 1.0)

        context_tensor = torch.tensor(
            scaled_context, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)
        error_tensor = torch.tensor(
            [[[error]]], dtype=torch.float32, device=self.device
        )

        if self.mem is None:
            self.mem = torch.zeros(1, self.hidden_size, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                output_tensor, self.mem = self.model(
                    error_tensor, context_tensor, self.mem
                )
            self.mem = torch.clamp(self.mem, -1.0, 1.5)

        raw = output_tensor.item()

        # Record fused 7-dim input for Akida QAT calibration: the Akida model
        # expects [encoded_0, encoded_1, ctx_0..4].  Use the encoder's actual
        # output (captured by forward hook) so the recording matches the model's
        # encoding -- works for both DeltaEncoder and ErrorRateEncoder.
        if self._record_inputs and self._recorded_spike is not None:
            fused_input = np.concatenate([self._recorded_spike, scaled_context])
            self._recorded_inputs.append(fused_input)
            self._recorded_outputs.append(raw)

        return raw

    def predict(self, setpoint: float, measurement: float, error: float) -> float:
        """
        Compute one control step.

        Overrides the base to drain membrane state on startup deadband.
        """
        # Startup deadband - drain membrane when idle
        if abs(setpoint) < 0.001 and abs(error) < 0.05:
            self.mem         = None
            self.prev_output = 0.0
            return 0.0
        # Error deadband - matches firmware NC_SNN_ERR_DEADBAND_DEG
        if self.err_deadband > 0.0 and abs(error) < self.err_deadband:
            self.prev_output = 0.0
            return 0.0

        scaled = self._preprocess(setpoint, measurement, error)
        if scaled is None:
            return 0.0
        return self._postprocess(self._forward(scaled, error))

    def get_recorded_dataset(self):
        """
        Return the fused inputs and raw SNN outputs recorded since the last
        ``reset()``.

        Only meaningful when ``record_inputs=True`` was passed to the constructor.

        :return: Tuple ``(X, y)`` where

            - ``X`` is a float32 ndarray of shape ``(N, 7)`` --
              ``[spike_pos, spike_neg, ctx_0, ..., ctx_4]``;
            - ``y`` is a float32 ndarray of shape ``(N,)`` -- the raw SNN output in
              ``[0, 1]`` before EMA and ``target_scale``.

        :raises RuntimeError: If ``record_inputs`` was not enabled.
        """
        if not self._record_inputs:
            raise RuntimeError("record_inputs was not enabled at construction time.")
        X = np.array(self._recorded_inputs,  dtype=np.float32)
        y = np.array(self._recorded_outputs, dtype=np.float32)
        return X, y

    def __repr__(self):
        return f" (SNNController: model_path='{self.model_path}')"

class KerasController(BaseSNNController):
    """
    Controller that runs inference on a quantized ``tf_keras`` model.

    The model is expected to have been produced by the ``AKD1000Exporter`` pipeline
    (float Keras Sequential trained on SNN trajectories). It shares the full
    preprocessing stack with ``SNNController`` so metrics are directly comparable.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        dt: float = 0.01,
        filter_alpha: float = 0.5,
        integral_deadzone: float = 0.0,
        output_deadzone: float = 0.0,
        actuator_limits: dict | None = None,
        error_encoding: str = 'delta',
        max_rate_input: float = 1.0,
    ):
        """
        :param model_path: Path to a saved ``tf_keras`` model (SavedModel dir or ``.h5``).
        :param scaler_path: Path to the MinMax scaler ``.npz`` (same file used for SNN training).
        :param dt: Control loop time step in seconds (default 0.01).
        :param filter_alpha: EMA smoothing coefficient (default 0.5).
        :param actuator_limits: Optional dict with ``u_min`` / ``u_max`` (default +/-10.0).
        :param error_encoding: ``'delta'`` for DeltaEncoder or ``'rate'`` for ErrorRateEncoder (default ``'delta'``).
        :param max_rate_input: Error magnitude that saturates the rate encoder (default 1.0).
        """
        try:
            import tf_keras as keras
        except ImportError:
            raise ImportError("KerasController requires tf_keras. Install it with: pip install tf_keras")

        super().__init__(
            scaler_path=scaler_path,
            dt=dt,
            filter_alpha=filter_alpha,
            integral_deadzone=integral_deadzone,
            output_deadzone=output_deadzone,
            actuator_limits=actuator_limits,
            error_encoding=error_encoding,
            max_rate_input=max_rate_input,
        )
        self.model_path = model_path
        self._model = keras.models.load_model(model_path)

        # The Keras model is fine-tuned on per-feature normalized QAT data
        # (same normalization as AkidaController). Apply it at inference too.
        self._feat_min   = None
        self._feat_range = None
        _scaler_path = Path(model_path).parent / "akida_input_scaler.npz"
        if _scaler_path.exists():
            with np.load(str(_scaler_path)) as _d:
                self._feat_min   = _d['feat_min'].astype(np.float32)
                self._feat_range = _d['feat_range'].astype(np.float32)

    def _forward(self, scaled_context: np.ndarray, error: float) -> float:
        """Run one Keras inference step."""
        encoded = self._compute_error_encoding(error)
        fused_input = np.concatenate([encoded, scaled_context])
        if self._feat_min is not None:
            x = np.clip((fused_input - self._feat_min) / self._feat_range, 0.0, 1.0)
        else:
            x = fused_input
        x = x.reshape(1, 1, 1, -1)
        return float(self._model.predict(x, verbose=0).ravel().mean())

    def __repr__(self):
        return f" (KerasController: model_path='{self.model_path}')"

class AkidaController(BaseSNNController):
    """
    Controller that runs inference on an Akida ``.fbz`` model.

    Intended for hardware-in-the-loop validation or pre-deployment simulation
    benchmarking. Shares the full preprocessing stack with ``SNNController`` and
    ``KerasController`` so all three can be evaluated in the same harness loop.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        dt: float = 0.01,
        filter_alpha: float = 0.5,
        integral_deadzone: float = 0.0,
        output_deadzone: float = 0.0,
        map_to_hardware: bool = True,
        actuator_limits: dict | None = None,
        error_encoding: str = 'delta',
        max_rate_input: float = 1.0,
    ):
        """
        :param model_path: Path to the Akida ``.fbz`` model file.
        :param scaler_path: Path to the MinMax scaler ``.npz`` (same file used for SNN training).
        :param dt: Control loop time step in seconds (default 0.01).
        :param filter_alpha: EMA smoothing coefficient (default 0.5).
        :param integral_deadzone: Error threshold below which the integral is frozen
            (default 0.0 = disabled). Can be set to reduce limit-cycle amplitude on plants
            with aggressive integral action.
        :param map_to_hardware: If True (default), automatically map the model to the first
            available Akida hardware device so inference runs on-chip rather than
            in the CPU software simulator. Set to False to force software simulation.
        :param actuator_limits: Optional dict with ``u_min`` / ``u_max`` (default +/-10.0).
        :param error_encoding: ``'delta'`` for DeltaEncoder or ``'rate'`` for ErrorRateEncoder (default ``'delta'``).
        :param max_rate_input: Error magnitude that saturates the rate encoder (default 1.0).
        """
        try:
            import akida
        except ImportError:
            raise ImportError("AkidaController requires the akida package. Install it with: pip install akida")

        super().__init__(
            scaler_path=scaler_path,
            dt=dt,
            filter_alpha=filter_alpha,
            integral_deadzone=integral_deadzone,
            output_deadzone=output_deadzone,
            actuator_limits=actuator_limits,
            error_encoding=error_encoding,
            max_rate_input=max_rate_input,
        )
        self.model_path = model_path
        self._model = akida.Model(model_path)
        self._device = None
        if map_to_hardware:
            devices = akida.devices()
            if devices:
                self._model.map(devices[0])
                # SW/-prefixed sequences run on the host CPU even after mapping,
                # routing data through PCIe adds ~500 us overhead with no chip benefit.
                # Only keep the mapping if at least one sequence is hardware-executed.
                hw_seqs = [s for s in self._model.sequences if not s.name.startswith("SW/")]
                if hw_seqs:
                    self._device = devices[0]
                    print(f"  Mapped to Akida hardware: {devices[0].desc}")
                else:
                    self._model = akida.Model(model_path)  # reload unmapped
                    print("  SW/-only model - skipping hardware map (avoids PCIe overhead penalty)")
            else:
                print("  No Akida hardware detected - running in software simulation mode")


        # Quantization output bias: 4-bit weights introduce a systematic offset
        # that integrating plants amplify into steady-state error.
        self._output_bias = 0.0
        _bias_path = Path(model_path).parent / "akida_output_bias.npz"
        if _bias_path.exists():
            self._output_bias = float(np.load(str(_bias_path))['bias'])

        # Per-feature scaler computed from QAT data in prototype_akida_export.py.
        # Covers the actual SNN inference distribution (integral_error can reach 12x
        # the PID training range). Without this, large integral values all map to
        # uint8 level 15 and the controller loses the correction signal.
        self._feat_min   = None
        self._feat_range = None
        _scaler_path = Path(model_path).parent / "akida_input_scaler.npz"
        if _scaler_path.exists():
            with np.load(str(_scaler_path)) as _d:
                self._feat_min   = _d['feat_min'].astype(np.float32)
                self._feat_range = _d['feat_range'].astype(np.float32)

            # Tighten integral clip to what the Akida uint8 quantization can represent.
            _snn_m  = float(self.scaler_min[3])
            _snn_s  = float(self.scaler_scale[3])
            _f_lo   = float(self._feat_min[5])   - 0.10 * float(self._feat_range[5])
            _f_hi   = float(self._feat_min[5])   + 1.10 * float(self._feat_range[5])
            self.clip_min[3] = _snn_m + _snn_s * _f_lo
            self.clip_max[3] = _snn_m + _snn_s * _f_hi

    def _forward(self, scaled_context: np.ndarray, error: float) -> float:
        """Run one Akida inference step."""
        encoded = self._compute_error_encoding(error)
        fused_input = np.concatenate([encoded, scaled_context])
        if self._feat_min is not None:
            x = np.clip((fused_input - self._feat_min) / self._feat_range, 0.0, 1.0)
        else:
            x = np.clip(fused_input, 0.0, 1.0)
        x = (x * 15).astype(np.uint8).reshape(1, 1, 1, -1)
        result = self._model.predict(x)
        mean_pred = result.ravel().mean()
        return float(mean_pred) - self._output_bias

    def __repr__(self):
        return f" (AkidaController: model_path='{self.model_path}')"