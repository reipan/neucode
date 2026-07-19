"""
SNN model exporter for the NeuCoDe toolkit.

Converts a trained HybridControlSNN PyTorch model into a fixed-point C header
file, handling the dual spike/context input branches, LIF neuron parameters,
and MinMaxScaler export for firmware-side normalisation.
"""
from __future__ import annotations
import json
import numpy as np
import math

from pathlib import Path
from typing import Optional

from .._torch_optional import torch, nn
from .base import BaseExporter
import neucode.nn.layers


class SNNExporter(BaseExporter):
    """
    Exporter for HybridControlSNN controllers.

    Quantises spike-input, context-input, and hidden/output layer weights using
    branch-aware Q-format arithmetic, extracts LIF decay and threshold parameters,
    and writes a self-contained C header for embedded deployment.
    """
    NC_CONTEXT, NC_DELTA, NC_RATE = 0, 1, 2

    def __init__(self, total_bits: int = 8):
        """
        Initialise the SNN exporter.

        :param total_bits: Total bit-width for Q-format quantisation (default 8).
        """
        self.total_bits = total_bits

    @staticmethod
    def _load_model(model_path: str, hidden_size: int = 256,
                    architecture: str = None) -> nn.Module:
        """
        Load an SNN model from a state dict file.

        :param model_path: Path to the .pth state dict.
        :param hidden_size: Hidden layer size (default 256).
        :param architecture: One of 'hybrid', 'population', 'spike'.
            Auto-detected from state dict keys when None.
        :returns: Loaded nn.Module in eval mode.
        """
        from ..experiment import Experiment
        from ..architectures import HybridControlSNN, PopulationControlSNN, SpikeControlSNN

        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

        if architecture is None:
            architecture = Experiment._detect_controller_type(model_path)
            if architecture == 'ann':
                raise ValueError("State dict looks like an ANN, not an SNN")

        if architecture == 'population':
            model = PopulationControlSNN(hidden_size=hidden_size)
        elif architecture == 'spike':
            model = SpikeControlSNN(hidden_size=hidden_size)
        else:
            model = HybridControlSNN(hidden_size=hidden_size)

        state_dict.pop('encoder.prev_spike_value', None)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def _calc_safe_input_frac(self, stats: dict, accumulator_bits: int = 32, physical_max: float = None) -> int:
        """
        Calculate the maximum safe fractional bits for input scaling, so that
        the accumulated input does not exceed the range of the accumulator.

        Reference:
            Yates et al., "Fixed-Point Arithmetic: An Introduction", Rev PA11 (16.05.24)
            Section 4.4: "Signed Range"

        :param stats: Calibration statistics containing min/max input values.
        :param accumulator_bits: Bit-width of the accumulator (default is 32).
        :param physical_max: Optional physical maximum input value to consider.
        :returns: Maximum safe number of fractional bits for input scaling.
        """
        try:
            # SNN calibration stats are collected by forward hooks on snn.Leaky modules.
            # Prefer a stable key if present, otherwise fall back to the first recorded layer.
            layer_key = 'lif' if isinstance(stats, dict) and 'lif' in stats else next(iter(stats))
            min_input = stats[layer_key]['min_input']
            max_input = stats[layer_key]['max_input']
            max_input_val = max(abs(min_input), abs(max_input))
        except (KeyError, TypeError, IndexError):
            # Assume normalized input in [-1, 1]
            print("* Warning: Missing calibration stats for SNN layer input, using normalized assumption.")
            max_input_val = 1.0

        if max_input_val < 1e-6:
            max_input_val = 1.0

        # If physical max is provided, use it to adjust max input value
        if physical_max is not None:
            if physical_max > max_input_val:
                print(f"* Warning: Provided physical_max {physical_max} is greater than observed max input {max_input_val}. Using physical_max.")
                max_input_val = physical_max

        # Calc maximum accumulator value based on bit-width
        max_accumulator_val = (1 << (accumulator_bits - 1)) - 1
        # Set maximum input weight (assuming int8 weights)
        max_input_weight = 127
        # Reasonable assumption for maximum number of inputs
        max_num_inputs = 8
        # Calculate safe input integer range (with 10% margin)
        max_safe_input_int = (max_accumulator_val / (max_input_weight * max_num_inputs)) * 0.9

        # Calculate fractional bits: max_input_val * 2^bits < max_safe_input_int
        bits = math.floor(math.log2(max_safe_input_int / max_input_val))

        # Limit to a maximum of 16 bits to leave headroom for weight shifts
        input_frac_bits = max(0, min(bits, 16))

        return input_frac_bits

    def _calc_common_acc_shift(self, model: torch.nn.Module, input_frac_bits: int, has_rate: bool = False) -> int:
        """
        Calculate Common Accumulation Shift for Context Branch
        - In a hybrid spiking neural network, both spike inputs and context (analog) inputs must result in the same scale during accumulation.
        - Uses the first SNNDirectInputLinear layer to determine the weight fractional bits.

        Reference:
            Yates et al., "Fixed-Point Arithmetic: An Introduction", Rev PA11 (16.05.24)
            Section 6.4: "Principle of Scaling Homogeneity"

        :param model: The SNN model containing the layers.
        :param input_frac_bits: Pre-calculated safe fractional bits for input scaling.
        :returns: Common accumulation shift that aligns the context and spike branches.
        """
        direct_input_modules = [m for m in model.modules() if isinstance(m, neucode.nn.layers.SNNDirectInputLinear)]
        if direct_input_modules:
            weight_context_max = direct_input_modules[0].weight.abs().max().item()
            _, context_weight_frac, _ = self._get_q_format(weight_context_max)
            return input_frac_bits + context_weight_frac

        spike_input_modules = [m for m in model.modules() if isinstance(m, neucode.nn.layers.SNNSpikeInputLinear)]
        if spike_input_modules:
            weight_max = spike_input_modules[0].weight.abs().max().item()
            _, weight_frac, _ = self._get_q_format(weight_max)
            if has_rate:
                return input_frac_bits + weight_frac
            return weight_frac

        raise ValueError("Model must have at least one SNNDirectInputLinear or SNNSpikeInputLinear layer.")

    def _get_quantized_weights(self, model: torch.nn.Module, input_frac_bits: int, channels=None) -> dict:
        """
        Branch-Aware weights quantization for HybridControlSNN without Scaler Fusion.
        Firmware must handle scaling prior to passing data to the network.

        :param model: The SNN model to quantize.
        :param input_frac_bits: Fractional bits for input scaling.
        :returns: Dictionary mapping layer names to quantised weight/bias arrays and metadata.
        """
        layers = {}

        # Calculate common accumulation shift to align both branches
        has_rate = channels is not None and any(t == self.NC_RATE for t, _, _ in channels)
        common_acc_shift = self._calc_common_acc_shift(model=model, input_frac_bits=input_frac_bits, has_rate=has_rate)
        context_weight_frac = common_acc_shift - input_frac_bits

        # Quantize each layer
        for name, module in model.named_modules():
            if not isinstance(module, (neucode.nn.layers.SNNDirectInputLinear, neucode.nn.layers.SNNSpikeInputLinear, torch.nn.Linear)):
                continue

            weight_float = module.weight.detach().numpy()
            bias_float = module.bias.detach().numpy() if module.bias is not None else np.zeros(module.out_features)
            output_bias_mean = 0.0

            # Branch: Analog Context
            if isinstance(module, neucode.nn.layers.SNNDirectInputLinear):
                # Firmware passes quantized inputs (already shifted by input_frac_bits).
                # To reach common_acc_shift, weights only need to be shifted by context_weight_frac.
                weight_scale = (2 ** context_weight_frac)

                weight_final = weight_float * weight_scale

                # Final Bias needs to be scaled to match the common accumulation shift
                bias_final = bias_float * (2 ** common_acc_shift)

                # Quantization
                weight_quant = np.clip(np.round(weight_final), -128, 127).astype(np.int8)
                bias_quant = np.round(bias_final).astype(np.int32)

                # Meta info
                sort_order = 0
                layer_type = 'standard'
                layer_shift = common_acc_shift

            # Branch: Spike Input (Forced Alignment)
            elif isinstance(module, neucode.nn.layers.SNNSpikeInputLinear):
                # Identify which columns carry continuous (rate) inputs vs binary (delta).
                # Rate columns need weight / 2^input_frac so that
                # rate_q(Qfrac) * weight_q(Qacc-frac) = Qacc.
                rate_cols = set()
                if channels is not None:
                    spike_ch_idx = 0
                    for ch_type, _, _ in channels:
                        if ch_type in (self.NC_DELTA, self.NC_RATE):
                            if ch_type == self.NC_RATE:
                                rate_cols.add(spike_ch_idx)
                            spike_ch_idx += 1

                if rate_cols:
                    weight_final = np.zeros_like(weight_float)
                    for col in range(module.in_features):
                        if col in rate_cols:
                            weight_final[:, col] = weight_float[:, col] * (2 ** (common_acc_shift - input_frac_bits))
                        else:
                            weight_final[:, col] = weight_float[:, col] * (2 ** common_acc_shift)
                else:
                    weight_final = weight_float * (2 ** common_acc_shift)

                weight_quant = np.round(weight_final).astype(np.int32)
                bias_quant = None
                sort_order = 1
                layer_type = 'input_spikes'
                layer_shift = common_acc_shift

            # Branch: Output layer - keep float32 to avoid sparsity from int8 quantisation.
            elif name in ('output_scale', 'output_population'):
                if module.out_features > 1:
                    weight_quant = weight_float.mean(axis=0, keepdims=True).astype(np.float32)
                    if module.bias is not None:
                        output_bias_mean = float(bias_float.mean())
                else:
                    weight_quant = weight_float.astype(np.float32)
                bias_quant = None

                sort_order = 2
                layer_type = 'output_float'
                layer_shift = 16

            # Branch: Other hidden layers
            else:
                # Standard quantization
                weight_max = module.weight.abs().max().item()
                _, weight_frac, weight_scale = self._get_q_format(weight_max)

                # Uses power-of-two scaling directly from Q-format
                weight_final = weight_float * weight_scale
                weight_quant = np.clip(np.round(weight_final), -128, 127).astype(np.int8)

                # Meta info
                sort_order = 2
                layer_type = 'standard'
                layer_shift = weight_frac

                # Bias must match weight scaling (since input scale=1, shift=0)
                if module.bias is not None:
                    bias_final = bias_float * (2 ** weight_frac)
                    bias_quant = np.round(bias_final).astype(np.int32)
                else:
                    bias_quant = None

            layers[name] = {
                'weights': weight_quant,
                'bias': bias_quant,
                'type': layer_type,
                'total_shift': layer_shift,
                'in_dim': module.in_features,
                'out_dim': module.out_features,
                'sort_order': sort_order,
                'output_bias_mean': output_bias_mean,
            }

        return layers

    def _get_decay_params(self, beta: float, shift: int = 15) -> tuple[int, int]:
        """
        Find the fixed-point representation for the decay factor (beta) given a right shift.
        This is smilar to _get_q_format (from the base class) but specifically tailored for decay factor.

        Reference:
            Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018.
            Section 2.2: "Integer-arithmetic-only matrix multiplication"

        Slight modification:
            We don't normalize the multiplier to be in [0.5, 1) because that keeps hardware implementation simpler (shift stays constant).

        :param beta: Decay factor (0 < beta < 1).
        :param shift: Right shift to be applied during decay multiplication.
        :returns: Tuple of (multiplier, shift) for fixed-point decay computation.
        """
        if beta >= 1.0:
            return (1 << shift), shift # No decay
        multiplier = int(beta * (1 << shift))
        return multiplier, shift

    def _build_channel_descriptors(self, model):
        """Build per-input-channel encoding descriptors from the model.

        Each channel is a tuple (type, feature_idx, polarity) describing how
        the firmware should encode one slot of the input buffer.
        The ordering must match the weight matrix column layout.
        """
        channels = []
        has_context = any(isinstance(m, neucode.nn.layers.SNNDirectInputLinear)
                          for m in model.modules())
        has_rate_encoder = (hasattr(model, 'encoder') and
                            hasattr(model.encoder, 'rate_index'))

        if has_context:
            n_ctx = 5
            for m in model.modules():
                if isinstance(m, neucode.nn.layers.SNNDirectInputLinear):
                    n_ctx = m.in_features
                    break
            for i in range(n_ctx):
                channels.append((self.NC_CONTEXT, i, 0))
            if hasattr(model, 'encoder'):
                channels.append((self.NC_DELTA, 2, 0))
                channels.append((self.NC_DELTA, 2, 1))
        elif has_rate_encoder:
            enc = model.encoder
            ri = enc.rate_index
            n_feat = enc.n_features
            channels.append((self.NC_RATE, ri, 0))
            channels.append((self.NC_RATE, ri, 1))
            rest = [i for i in range(n_feat) if i != ri]
            for i in rest:
                channels.append((self.NC_DELTA, i, 0))
            for i in rest:
                channels.append((self.NC_DELTA, i, 1))
        elif hasattr(model, 'encoder'):
            channels.append((self.NC_DELTA, 2, 0))
            channels.append((self.NC_DELTA, 2, 1))

        return channels

    # ------------------------------------------------------------------
    # Header emission helpers (called by export())
    # ------------------------------------------------------------------

    def _emit_scaler_params(self, scaler_path):
        """Load scaler and emit C arrays + config defines.

        :returns: (scaler_params, c_lines, config_lines)
        """
        if scaler_path is None:
            return None, [], []

        scaler_params = np.load(scaler_path)
        c_lines = []
        config_lines = []

        data_min = scaler_params['data_min'] if 'data_min' in scaler_params.files else np.zeros(5)
        data_range = scaler_params['data_scale'] if 'data_scale' in scaler_params.files else np.ones(5)
        data_range[data_range == 0] = 1.0

        c_lines.append("// Input Scaler Parameters (MinMaxScaler: (x - min) / (max - min))")
        config_lines.append(f"#define SCALER_DIM {len(data_min)}")
        scaler_min_str = ", ".join([f"{x:.8e}" for x in data_min])
        c_lines.append(f"const float SCALER_MIN[] = {{ {scaler_min_str} }};")
        scaler_range_str = ", ".join([f"{x:.8e}" for x in data_range])
        c_lines.append(f"const float SCALER_RANGE[] = {{ {scaler_range_str} }};")

        deriv_clip = float(scaler_params['deriv_clip'][0]) if 'deriv_clip' in scaler_params.files else float('inf')
        c_lines.append(f"// Derivative-error clamp applied before scaling (matches training p99 clip)")
        c_lines.append(f"#define DERIV_CLIP_VALUE {deriv_clip:.8e}f")
        c_lines.append("")

        if 'clip_min' in scaler_params.files and 'clip_max' in scaler_params.files:
            feat_clip_min = scaler_params['clip_min']
            feat_clip_max = scaler_params['clip_max']
        else:
            feat_clip_min = np.full(len(data_min), -3.40282347e+38, dtype=np.float32)
            feat_clip_max = np.full(len(data_min),  3.40282347e+38, dtype=np.float32)
            if deriv_clip != float('inf'):
                feat_clip_min[4] = -deriv_clip
                feat_clip_max[4] =  deriv_clip
            feat_clip_min[3] = data_min[3]
            feat_clip_max[3] = data_min[3] + data_range[3]

        clip_min_str = ", ".join([f"{x:.8e}f" for x in feat_clip_min])
        clip_max_str = ", ".join([f"{x:.8e}f" for x in feat_clip_max])
        c_lines.append(f"// Per-feature clip bounds (applied before scaling)")
        c_lines.append(f"const float FEATURE_CLIP_MIN[] = {{ {clip_min_str} }};")
        c_lines.append(f"const float FEATURE_CLIP_MAX[] = {{ {clip_max_str} }};")
        c_lines.append("")

        return scaler_params, c_lines, config_lines

    def _emit_global_defines(self, model, layers, scaler_params, input_frac_bits):
        """Emit model-level #defines for model_data.h."""
        c_lines = []

        output_weight_shift = 0
        for name, data in layers.items():
            if "output" in name.lower() or "scale" in name.lower():
                output_weight_shift = data['total_shift']
                break
        c_lines.append(f"#define OUTPUT_WEIGHT_SHIFT {output_weight_shift}")

        if hasattr(model, 'encoder'):
            threshold_float = model.encoder.delta
            threshold_int = int(threshold_float * (2 ** input_frac_bits))
            c_lines.append(f"#define ENCODER_DELTA_THRESHOLD {threshold_int}")
        else:
            c_lines.append(f"#define ENCODER_DELTA_THRESHOLD 1")

        if scaler_params is not None and 'target_scale' in scaler_params.files:
            target_scale = float(np.array(scaler_params['target_scale']).reshape(-1)[0])
            c_lines.append(f"#define OUTPUT_SCALE {target_scale}f")
        else:
            c_lines.append(f"#define OUTPUT_SCALE 1.0f")

        return c_lines

    def _emit_lif_params(self, model, layers):
        """Emit LIF neuron parameters for model_data.h.

        :returns: (c_lines, beta_multiplier, beta_shift)
        """
        if not hasattr(model, 'lif'):
            return [], 0, 0

        beta_value = model.lif.beta
        if hasattr(beta_value, 'item'):
            beta_value = float(beta_value.item())
        else:
            beta_value = float(beta_value)
        beta_shift = 15
        beta_multiplier, _ = self._get_decay_params(beta=beta_value, shift=beta_shift)

        c_lines = []
        c_lines.append(f"#define LIF_BETA_MULTIPLIER {beta_multiplier}")
        c_lines.append(f"#define LIF_BETA_SHIFT {beta_shift}")

        accumulator_shift = 0
        for name, mod in model.named_modules():
            if isinstance(mod, (neucode.nn.layers.SNNDirectInputLinear, neucode.nn.layers.SNNSpikeInputLinear)) and name in layers:
                accumulator_shift = layers[name]['total_shift']
                break

        threshold_value = model.lif.threshold
        if hasattr(threshold_value, 'item'):
            threshold_value = float(threshold_value.item())
        else:
            threshold_value = float(threshold_value)
        threshold_int32 = int(threshold_value * (2 ** accumulator_shift))
        c_lines.append(f"#define LIF_THRESHOLD {threshold_int32}")

        return c_lines, beta_multiplier, beta_shift

    def _emit_layer_arrays(self, model, layers, beta_multiplier, beta_shift):
        """Emit per-layer weight/bias C arrays and build layer config structs.

        :returns: (c_lines, layer_configs)
        """
        c_lines = []
        sorted_layers = sorted(layers.items(), key=lambda item: item[1]['sort_order'])
        layer_configs = []

        for layer_name, data in sorted_layers:
            prefix = layer_name.upper().replace('.', '_')
            if data['type'] == 'output_float':
                prefix = 'OUTPUT_SCALE'

            c_lines.append(f"// Layer: {layer_name} ({data['type']})")

            if data['type'] == 'input_spikes':
                weight_dtype = 'int32_t'
            elif data['type'] == 'output_float':
                weight_dtype = 'float'
            else:
                weight_dtype = 'int8_t'

            if data['type'] == 'output_float':
                w_str = ", ".join([f"{w:.8e}f" for w in data['weights'].flatten()])
            else:
                w_str = ", ".join([str(w) for w in data['weights'].flatten()])
            c_lines.append(f"static const {weight_dtype} {prefix}_WEIGHTS[] = {{ {w_str} }};")

            if data['bias'] is not None:
                b_str = ", ".join(map(str, data['bias'].flatten()))
                c_lines.append(f"static const int32_t {prefix}_BIASES[] = {{ {b_str} }};")
                bias_ref = f"{prefix}_BIASES"
            else:
                bias_ref = "0"

            if 'output' in layer_name:
                cfg_threshold = 2147483647
                cfg_beta_multiplier = 0
                cfg_beta_shift = 0
            else:
                cfg_beta_multiplier = beta_multiplier
                cfg_beta_shift = beta_shift
                th = model.lif.threshold
                if hasattr(th, 'item'):
                    th = float(th.item())
                else:
                    th = float(th)
                cfg_threshold = int(th * (2 ** data['total_shift']))

            layer_configs.append({
                'layer_name': prefix,
                'type': data['type'],
                'bias_ref': bias_ref,
                'in_dim': data['in_dim'],
                'out_dim': data['out_dim'],
                'threshold': cfg_threshold,
                'beta_multiplier': cfg_beta_multiplier,
                'beta_shift': cfg_beta_shift
            })

        return c_lines, layer_configs

    def _emit_config(self, model, layers, channels, scaler_params,
                     firmware_filter_alpha):
        """Emit all model_config.h entries (excluding boilerplate and SCALER_DIM)."""
        config_lines = []

        # Actuator limits
        if scaler_params is not None and 'u_min' in scaler_params.files:
            snn_u_min = float(np.array(scaler_params['u_min']).reshape(-1)[0])
            snn_u_max = float(np.array(scaler_params['u_max']).reshape(-1)[0])
        else:
            snn_u_min = -1.0
            snn_u_max =  1.0
        config_lines.append(f"#define NC_SNN_U_MIN {snn_u_min:.8e}f")
        config_lines.append(f"#define NC_SNN_U_MAX {snn_u_max:.8e}f")

        # Output EMA filter
        if firmware_filter_alpha is not None:
            ffa = float(firmware_filter_alpha)
            if not (0.0 <= ffa < 1.0):
                raise ValueError(f"firmware_filter_alpha must be in [0, 1), got {ffa}")
            config_lines.append(f"// Output EMA: alpha={ffa:.4f} (firmware 1kHz rate)")
            config_lines.append(f"#define NC_SNN_OUTPUT_EMA_ALPHA {ffa:.8e}f")

        # Rate encoder parameters
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'max_rate_input'):
            config_lines.append(f"#define NC_SNN_ENCODER_RATE_MAX_INPUT {model.encoder.max_rate_input:.8e}f")
            rate_zp = getattr(model.encoder, 'rate_zero_point', 0.0)
            config_lines.append(f"#define NC_SNN_ENCODER_RATE_ZERO_POINT {rate_zp:.8e}f")

        # Output centering constant
        output_center = 0.5
        for _, data in layers.items():
            if data['type'] == 'output_float':
                output_center = 0.5 - data.get('output_bias_mean', 0.0)
                break
        config_lines.append(f"#define NC_SNN_OUTPUT_CENTER {output_center:.8e}f")

        # Channel encoding descriptors
        config_lines.append("")
        config_lines.append("#define NC_INPUT_CONTEXT 0")
        config_lines.append("#define NC_INPUT_DELTA   1")
        config_lines.append("#define NC_INPUT_RATE    2")
        config_lines.append(f"#define INPUT_CHANNEL_COUNT {len(channels)}")
        types_str = ", ".join(str(t) for t, _, _ in channels)
        features_str = ", ".join(str(f) for _, f, _ in channels)
        polarities_str = ", ".join(str(p) for _, _, p in channels)
        config_lines.append(f"static const uint8_t INPUT_CHANNEL_TYPE[] = {{ {types_str} }};")
        config_lines.append(f"static const uint8_t INPUT_CHANNEL_FEATURE[] = {{ {features_str} }};")
        config_lines.append(f"static const uint8_t INPUT_CHANNEL_POLARITY[] = {{ {polarities_str} }};")

        # Derived sizes for inference kernel
        context_count = sum(1 for t, _, _ in channels if t == self.NC_CONTEXT)
        spike_count = sum(1 for t, _, _ in channels if t != self.NC_CONTEXT)
        sorted_layers = sorted(layers.items(), key=lambda item: item[1]['sort_order'])
        hidden_size_val = next((data['out_dim'] for _, data in sorted_layers if data['sort_order'] < 2), 256)
        config_lines.append(f"#define INPUT_CONTEXT_SIZE {context_count}")
        config_lines.append(f"#define INPUT_SPIKE_SIZE {spike_count}")
        config_lines.append(f"#define HIDDEN_SIZE {hidden_size_val}")

        return config_lines

    def _emit_inference_tables(self, layer_configs):
        """Emit NUM_LAYERS and the per-layer metadata tables for model_data.h."""
        c_lines = [""]
        c_lines.append(f"#define NUM_LAYERS {len(layer_configs)}")

        weight_pointers = []
        for cfg in layer_configs:
            if cfg['type'] == 'input_spikes':
                weight_pointers.append(f"(const int8_t*){cfg['layer_name']}_WEIGHTS")
            elif cfg['type'] == 'output_float':
                weight_pointers.append("0")
            else:
                weight_pointers.append(f"{cfg['layer_name']}_WEIGHTS")

        c_lines.append(f"static const int8_t* const WEIGHT_POINTERS[] = {{ " + ", ".join(weight_pointers) + " };")
        c_lines.append(f"static const int32_t* const BIAS_POINTERS[] = {{ " + ", ".join([str(cfg['bias_ref']) for cfg in layer_configs]) + " };")

        c_lines.append(f"static const int32_t LAYER_THRESHOLDS[] = {{ " + ", ".join([str(cfg['threshold']) for cfg in layer_configs]) + " };")
        c_lines.append(f"static const int32_t LAYER_BETA_MULTIPLIERS[] = {{ " + ", ".join([str(cfg['beta_multiplier']) for cfg in layer_configs]) + " };")
        c_lines.append(f"static const int32_t LAYER_BETA_SHIFTS[] = {{ " + ", ".join([str(cfg['beta_shift']) for cfg in layer_configs]) + " };")

        c_lines.append(f"static const int LAYER_IN_DIMS[] = {{ " + ", ".join([str(cfg['in_dim']) for cfg in layer_configs]) + " };")
        c_lines.append(f"static const int LAYER_OUT_DIMS[] = {{ " + ", ".join([str(cfg['out_dim']) for cfg in layer_configs]) + " };")

        return c_lines

    def _write_headers(self, output_path, c_content, config_content,
                       deployment_target):
        """Write model_data.h and model_config.h to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_path = output_path.parent / "model_config.h"
        config_path.write_text("\n".join(config_content), encoding='utf-8')
        output_path.write_text("\n".join(c_content), encoding='utf-8')

        if deployment_target:
            deployment_path = Path(deployment_target)
            deployment_path.parent.mkdir(parents=True, exist_ok=True)
            deployment_path.write_text("\n".join(c_content), encoding='utf-8')
            config_deploy_path = deployment_path.parent / "model_config.h"
            config_deploy_path.write_text("\n".join(config_content), encoding='utf-8')

        print(f"* Exported SNN model to {output_path}")
        print(f"* Exported SNN config to {config_path}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def export(self,
               model: nn.Module = None,
               stats_path: str = None,
               scaler_path: str = None,
               output_path: str = None,
               physical_max: Optional[float] = None,
               integral_window: Optional[float] = None,
               windup_limit: Optional[float] = None,
               firmware_filter_alpha: Optional[float] = None,
               deployment_target: Optional[str] = None,
               experiment=None,
               tag: str = None,
               deployment_type: str = None,
               hidden_size: int = 256,
               architecture: str = None) -> None:
        """
        Export the SNN model to a C header file with fixed-point representation.

        Pipeline:
            Load Artifacts -> Load Scaler -> Quantize Weights -> Write Header.

        Reference:
            Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", https://arxiv.org/pdf/1712.05877

        Some more info about the quantization approach:
        - Input fractional bits are defined based on a safe range to prevent accumulator overflow.
        - The MinMaxScaler is exported separately and applied by firmware BEFORE quantization.
            Formula: (x - data_min) / data_range, where data_range = (data_max - data_min), resulting in [0, 1] range.
        - Weights are quantized to symmetric int8 and biases to int32 (to avoid overflow during accumulation).
        - Performant requantization requires right shift with a power-of-two scaling factor.
        - Additionally, SNN-specific parameters like spike threshold and decay (beta) are quantized.

        :param model: The trained SNN model to export. If None, loaded automatically
            from the experiment's model path using architecture auto-detection.
        :param stats_path: Path to the calibration statistics JSON file.
        :param scaler_path: Path to the scaler .npz file.
        :param output_path: Path to save the generated C header file.
        :param physical_max: Optional physical maximum input value for fractional bit calculation.
        :param integral_window: Optional integral window size.
        :param windup_limit: Optional windup limit.
        :param firmware_filter_alpha: EMA smoothing coefficient for the firmware output filter.
            Exported as NC_SNN_OUTPUT_EMA_ALPHA in the model header; firmware snn.c picks it
            up via its #ifndef guard. Tuned for the firmware sample rate (1 kHz), not the
            Python simulation rate. If None, firmware falls back to its compiled-in default
            (currently 0.9). Example: use 0.95 for a 20 ms time-constant at 1 kHz.
        :param deployment_target: Optional path to also save the header for deployment.
        :param experiment: Optional Experiment instance for path resolution.
        :param tag: Training tag within the experiment (required when experiment is passed).
        :param deployment_type: Controller type ('ann', 'snn') for firmware deployment path.
        :param hidden_size: Hidden layer size when loading from path (default 256).
        :param architecture: Architecture type ('hybrid', 'population', 'spike').
            Auto-detected from state dict keys when None.
        """
        # Resolve paths from experiment
        model_path = None
        if experiment is not None:
            if tag is None:
                raise ValueError("tag is required when passing experiment")
            model_path = experiment.get_model_path(tag)
            stats_path = stats_path or experiment.get_stats_path(tag)
            scaler_path = scaler_path or experiment.get_scaler_path(tag)
            output_path = output_path or experiment.get_model_data_header_path(tag)
            if deployment_type:
                deployment_target = deployment_target or experiment.get_model_data_deployment_target_path(deployment_type)

        if model is None:
            if model_path is None:
                raise ValueError("Either model or experiment+tag must be provided")
            model = self._load_model(model_path, hidden_size=hidden_size,
                                     architecture=architecture)

        channels = self._build_channel_descriptors(model)

        # model_data.h: weight arrays, scaler arrays, quantization defines
        c_header = [
            "// AUTO-GENERATED BY neucode.exporters.snn.SNNExporter",
            "// Scaler is applied by firmware BEFORE quantization to preserve input resolution",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            '#include "model_config.h"',
            "",
        ]
        # model_config.h: lightweight #define constants shared by model_data.h and snn.c
        config_header = [
            "// AUTO-GENERATED BY neucode.exporters.snn.SNNExporter",
            "// Runtime configuration constants for the SNN firmware controller.",
            "// This file is intentionally free of weight arrays so snn.c can safely include it.",
            "#pragma once",
            "#include <stdint.h>",
            "",
        ]

        with open(stats_path, 'r') as f:
            stats = json.load(f)

        scaler_params, scaler_c, scaler_cfg = self._emit_scaler_params(scaler_path)
        c_header.extend(scaler_c)
        config_header.extend(scaler_cfg)

        input_frac_bits = self._calc_safe_input_frac(stats, physical_max=physical_max)
        c_header.append(f"#define INPUT_FRAC_BITS {input_frac_bits}")

        layers = self._get_quantized_weights(model=model, input_frac_bits=input_frac_bits, channels=channels)

        c_header.extend(self._emit_global_defines(model, layers, scaler_params, input_frac_bits))
        config_header.extend(self._emit_config(model, layers, channels, scaler_params, firmware_filter_alpha))

        lif_lines, beta_multiplier, beta_shift = self._emit_lif_params(model, layers)
        c_header.extend(lif_lines)
        c_header.append("")

        layer_lines, layer_configs = self._emit_layer_arrays(model, layers, beta_multiplier, beta_shift)
        c_header.extend(layer_lines)

        c_header.append("")
        if integral_window is not None:
            c_header.append(f"#define INTEGRAL_WINDOW {float(integral_window):.6f}f")
        if windup_limit is not None:
            c_header.append(f"#define WINDUP_LIMIT {float(windup_limit):.6f}f")

        c_header.extend(self._emit_inference_tables(layer_configs))

        self._write_headers(output_path, c_header, config_header, deployment_target)
