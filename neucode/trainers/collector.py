"""
Calibration statistics collectors for quantisation-aware export in NeuCoDe.

Provides LinearRangeStats / collect_linear_ranges for ANN layers and
SNNLayerStats / collect_snn_layer_stats for snnTorch Leaky layers.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union
from .._torch_optional import torch, nn

@dataclass
class LinearRangeStats:
    """
    Running statistics collected at nn.Linear boundaries (pre-activation) over all seen batches.

    :param in_maxabs: 99.9th-percentile absolute maximum of the layer input across all batches.
    :param out_min: Global minimum of the layer output (pre-activation) across all batches.
    :param out_max: Global maximum of the layer output (pre-activation) across all batches.
    """
    in_maxabs: float = 0.0
    out_min: float = 0.0
    out_max: float = 0.0
    _init: bool = False

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update running statistics with a new batch of layer inputs and outputs.

        :param x: Input tensor to the Linear layer (pre-weight-multiply).
        :param y: Output tensor from the Linear layer (pre-activation).
        """
        # input maxabs (use 99.9th percentile to be robust to outliers, but still capture the typical range well)
        x_ma = torch.quantile(x.detach().abs().float(), 0.999).item()
        if x_ma > self.in_maxabs:
            self.in_maxabs = float(x_ma)

        # output min/max
        y_min = y.detach().min().item()
        y_max = y.detach().max().item()
        if not self._init:
            self.out_min = float(y_min)
            self.out_max = float(y_max)
            self._init = True
        else:
            self.out_min = min(self.out_min, float(y_min))
            self.out_max = max(self.out_max, float(y_max))

    def to_dict(self) -> Dict[str, float]:
        """
        Return a plain dict serialisation of the stats for JSON export.

        :returns: Dict with keys 'in_maxabs', 'out_min', and 'out_max'.
        """
        return {"in_maxabs": float(self.in_maxabs), "out_min": float(self.out_min), "out_max": float(self.out_max)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LinearRangeStats":
        """
        Reconstruct a LinearRangeStats instance from a plain dict.

        :param d: Dict with keys 'in_maxabs', 'out_min', and 'out_max'.
        :returns: A fully initialised LinearRangeStats instance.
        """
        s = LinearRangeStats()
        s.in_maxabs = float(d["in_maxabs"])
        s.out_min = float(d["out_min"])
        s.out_max = float(d["out_max"])
        s._init = True
        return s

def collect_linear_ranges(                
        model: nn.Module,
        calibration_loader: Iterable,
        *,
        max_batches: Optional[int] = 8,
        device: str = "auto",
) -> Dict[str, Dict[str, float]]:
    """
    Register forward hooks on all nn.Linear layers and collect input/output range statistics
    over a calibration dataset, for use when exporting quantised models to embedded hardware.

    Note: PyTorch's built-in observer and quantization utilities (torch.quantization) are not
    used here because they target TFLite/ONNX-style INT8 formats, not the CMSIS-NN Q-format
    arithmetic required for deployment on Arm Cortex-M targets. Forward hooks give us direct
    access to the per-layer min/max and percentile values we need.

    :param model: The PyTorch model to calibrate.
    :param calibration_loader: Iterable of batches (tensor, tuple, list, or dict).
    :param max_batches: Maximum number of batches to process (None = all).
    :param device: Device to run calibration on ('auto', 'cpu', or 'cuda').
    :returns: Dict mapping layer names to LinearRangeStats serialised as plain dicts.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    model.eval()

    stats: Dict[str, LinearRangeStats] = {}
    hooks = []

    def pick_x(batch):
        if isinstance(batch, (tuple, list)):
            return batch[0]
        if isinstance(batch, dict):
            if "x" in batch:
                return batch["x"]
            # fallback to first item
            return next(iter(batch.values()))
        return batch

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            stats_data = LinearRangeStats()
            stats[name] = stats_data

            def hook(module, inputs, output, stats_data=stats_data):
                x = inputs[0]
                y = output
                stats_data.update(x, y)

            hooks.append(m.register_forward_hook(hook))

    with torch.no_grad():
        for bi, batch in enumerate(calibration_loader):
            if max_batches is not None and bi >= max_batches:
                break
            x = pick_x(batch)
            if device is not None:
                x = x.to(device)
            model(x)

    for h in hooks:
        h.remove()

    return {name: s.to_dict() for name, s in stats.items()}

@dataclass
class SNNLayerStats:
    """
    Running statistics for internal state safety and activity estimation in SNN (LIF) layers.

    :param max_abs_membrane_potential: Maximum absolute membrane potential observed;
        used to verify accumulator bit-width is sufficient.
    :param total_spike_count: Total number of spikes emitted across all observed steps.
    :param total_step_count: Total number of time steps observed (denominator for firing rate).
    :param min_input: Minimum input value observed; used to calculate INPUT_FRAC_BITS.
    :param max_input: Maximum input value observed (retained for future use).
    """
    max_abs_membrane_potential: float = 0.0
    total_spike_count: int = 0
    total_step_count: int = 0
    min_input: float = float('inf')
    max_input: float = float('-inf')

    def update(self, x_in: torch.Tensor, spike_out: torch.Tensor, mem_out: torch.Tensor) -> None:
        """
        Update running statistics with one forward-pass observation.

        :param x_in: Input tensor to the LIF layer.
        :param spike_out: Binary spike output tensor from the LIF layer.
        :param mem_out: Membrane potential tensor, or None if not returned.
        """
        # Update input range
        if (x_in.numel() > 0):
            x_min = x_in.detach().min().item()
            x_max = x_in.detach().max().item()
            if x_min < self.min_input:
                self.min_input = float(x_min)
            if x_max > self.max_input:
                self.max_input = float(x_max)
        # Update max absolute membrane potential
        if mem_out is not None and mem_out.numel() > 0:
            mem_max_abs = mem_out.detach().abs().max().item()
            if mem_max_abs > self.max_abs_membrane_potential:
                self.max_abs_membrane_potential = mem_max_abs
        # Update spike activity
        self.total_spike_count += int(spike_out.detach().sum().item())
        self.total_step_count += spike_out.numel()

    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Return a plain dict serialisation of the stats for JSON export.

        :returns: Dict with keys 'max_abs_membrane_potential', 'total_spike_count',
            'total_step_count', 'min_input', and 'max_input'.
        """
        return {
            "max_abs_membrane_potential": float(self.max_abs_membrane_potential),
            "total_spike_count": int(self.total_spike_count),
            "total_step_count": int(self.total_step_count),
            "min_input": float(self.min_input),
            "max_input": float(self.max_input),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SNNLayerStats":
        """Reconstruct an SNNLayerStats instance from a plain dict.

        :param d: Dict with the five stat keys produced by to_dict().
        :returns: A fully initialised SNNLayerStats instance.
        """
        s = SNNLayerStats()
        s.max_abs_membrane_potential = float(d["max_abs_membrane_potential"])
        s.total_spike_count = int(d["total_spike_count"])
        s.total_step_count = int(d["total_step_count"])
        s.min_input = float(d["min_input"])
        s.max_input = float(d["max_input"])
        return s

    @property
    def average_firing_rate(self) -> float:
        """
        Average firing rate of this layer.

        :returns: total_spike_count / total_step_count, or 0.0 if no steps were observed.
        """
        if self.total_step_count == 0:
            return 0.0
        return self.total_spike_count / self.total_step_count
    
def collect_snn_layer_stats(stats: dict, layer_name: str):
    """
    Return a forward hook that accumulates SNNLayerStats for a named LIF layer.

    Register the returned callable with ``layer.register_forward_hook(...)``.
    The hook writes into ``stats[layer_name]`` and creates the entry on first call.

    :param stats: Shared dict to accumulate results into (modified in-place).
    :param layer_name: Key used to store this layer's SNNLayerStats in ``stats``.
    :returns: A forward-hook callable compatible with nn.Module.register_forward_hook.
    """
    def hook(module, input_args, output):
        x_in = input_args[0]

        if isinstance(output, tuple) and len(output) == 2:
            spike_out, mem_out = output
        else:
            spike_out = output
            mem_out = None

        if layer_name not in stats:
            stats[layer_name] = SNNLayerStats()
        stats[layer_name].update(x_in, spike_out, mem_out)

    return hook