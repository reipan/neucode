"""NIR exporter for the NeuCoDe toolkit.

Exports a trained spike-based SNN to the Neuromorphic Intermediate
Representation (NIR) format.  The output .nir file contains the
Linear -> LIF -> Affine graph; encoder and scaler metadata are stored
in the input node so downstream backends can reconstruct preprocessing.

NIR is a hardware-agnostic interchange format -- the same .nir file
can be imported into Lava-DL, Sinabs, snnTorch, Rockpool, or any
other framework with a NIR reader.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch.nn as nn

from .base import BaseExporter
import neucode.nn.layers


class NIRExporter(BaseExporter):
    """
    Export a spike-based SNN to NIR (.nir).

    The exported graph is:
        Input -> Linear -> LIF -> Affine -> Output

    Encoder parameters and MinMax scaler values are attached as metadata
    on the Input and Output nodes respectively, so a backend can
    reconstruct the full inference pipeline without external files.

    Layers are discovered by type (SNNSpikeInputLinear, LIF, output
    Linear) rather than by name, so any single-hidden-layer spike SNN
    architecture is supported.
    """

    def __init__(self, dt: float = 0.001):
        """
        :param dt: Simulation timestep used to convert snnTorch beta
            to the continuous-time tau_mem that NIR expects.
        """
        self.dt = dt
        self.total_bits = 32  # unused, satisfies BaseExporter._get_q_format

    def export(
        self,
        model,
        output_path: str = None,
        scaler_path: Optional[str] = None,
        experiment=None,
        tag: str = None,
    ) -> Path:
        """
        Export a trained spike SNN to a .nir file.

        :param model: Trained SNN with spike input, LIF hidden layer, and
            linear output.  Must have a ``lif`` attribute (snnTorch Leaky).
        :param output_path: Destination path for the .nir file.
        :param scaler_path: Optional path to the scaler .npz file.
            If provided, scaler parameters are included in the output
            node metadata so the backend can reconstruct normalisation.
        :param experiment: Optional Experiment instance for path resolution.
        :param tag: Training tag within the experiment (required when experiment is passed).
        :returns: Path to the written .nir file.
        """
        if experiment is not None:
            if tag is None:
                raise ValueError("tag is required when passing experiment")
            output_path = output_path or str(experiment._get_train_dir(tag) / f"{tag}.nir")
            scaler_path = scaler_path or experiment.get_scaler_path(tag)
        try:
            import nir as _nir
        except ImportError:
            raise ImportError(
                "The 'nir' package is required for NIR export. "
                "Install it with: pip install nir"
            )

        if not hasattr(model, 'lif'):
            raise TypeError(
                f"NIR export requires a model with a 'lif' attribute, "
                f"got {type(model).__name__}."
            )

        # Discover layers by type
        spike_input_layer = None
        output_layer = None
        for name, mod in model.named_modules():
            if isinstance(mod, neucode.nn.layers.SNNSpikeInputLinear):
                spike_input_layer = (name, mod)
            elif isinstance(mod, nn.Linear) and 'output' in name:
                output_layer = (name, mod)

        if spike_input_layer is None:
            raise TypeError("Model has no SNNSpikeInputLinear layer.")
        if output_layer is None:
            raise TypeError("Model has no output Linear layer.")

        input_name, input_mod = spike_input_layer
        output_name, output_mod = output_layer

        output_path = Path(output_path)
        if output_path.suffix != ".nir":
            output_path = output_path.with_suffix(".nir")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        h = input_mod.out_features
        dt = self.dt

        beta = float(model.lif.beta.detach())
        tau_mem = np.full(h, dt / (1.0 - beta))
        r = tau_mem / dt
        vthr = np.full(h, float(model.lif.threshold.detach()))
        v_leak = np.zeros(h)

        # Encoder metadata -- extract whatever the encoder provides
        NC_CONTEXT, NC_DELTA, NC_RATE = 0, 1, 2
        encoder_meta = {
            "feature_names": ["setpoint", "measurement", "error",
                              "integral_error", "derivative_error"],
            "encoding_types": {"context": NC_CONTEXT, "delta": NC_DELTA,
                               "rate": NC_RATE},
        }
        has_context = any(isinstance(m, neucode.nn.layers.SNNDirectInputLinear)
                          for m in model.modules())
        if hasattr(model, 'encoder'):
            enc = model.encoder
            encoder_meta["encoder"] = type(enc).__name__
            for attr in ("rate_index", "rate_zero_point", "max_rate_input",
                         "delta", "n_features"):
                if hasattr(enc, attr):
                    val = getattr(enc, attr)
                    encoder_meta[attr] = float(val) if isinstance(val, (int, float)) else val

            # Channel descriptors: [type, feature_idx, polarity] per input slot
            channels = []
            if has_context:
                n_ctx = 5
                for m in model.modules():
                    if isinstance(m, neucode.nn.layers.SNNDirectInputLinear):
                        n_ctx = m.in_features
                        break
                for i in range(n_ctx):
                    channels.append([NC_CONTEXT, i, 0])
                channels.append([NC_DELTA, 2, 0])
                channels.append([NC_DELTA, 2, 1])
            elif hasattr(enc, 'rate_index'):
                ri = int(enc.rate_index)
                n_feat = int(enc.n_features)
                channels.append([NC_RATE, ri, 0])
                channels.append([NC_RATE, ri, 1])
                rest = [i for i in range(n_feat) if i != ri]
                for i in rest:
                    channels.append([NC_DELTA, i, 0])
                for i in rest:
                    channels.append([NC_DELTA, i, 1])
            else:
                channels.append([NC_DELTA, 2, 0])
                channels.append([NC_DELTA, 2, 1])
            encoder_meta["channel_descriptors"] = channels

        # Scaler metadata
        scaler_meta = {}
        if scaler_path is not None:
            sc = np.load(scaler_path)
            scaler_meta = {
                "scaler_data_min": sc["data_min"].tolist(),
                "scaler_data_scale": sc["data_scale"].tolist(),
                "scaler_target_scale": float(sc["target_scale"]),
            }
            if "clip_min" in sc:
                scaler_meta["scaler_clip_min"] = sc["clip_min"].tolist()
                scaler_meta["scaler_clip_max"] = sc["clip_max"].tolist()

        has_bias = output_mod.bias is not None
        output_w = output_mod.weight.detach().cpu().numpy()
        output_b = output_mod.bias.detach().cpu().numpy() if has_bias else np.zeros(output_mod.out_features)

        nodes = {
            "input": _nir.Input(
                input_type=[input_mod.in_features],
                metadata=encoder_meta,
            ),
            input_name: _nir.Linear(
                weight=input_mod.weight.detach().cpu().numpy(),
            ),
            "lif": _nir.LIF(
                tau=tau_mem,
                v_threshold=vthr,
                v_leak=v_leak,
                r=r,
            ),
            output_name: _nir.Affine(
                weight=output_w,
                bias=output_b,
            ),
            "output": _nir.Output(
                output_type=[output_mod.out_features],
                metadata=scaler_meta,
            ),
        }
        edges = [
            ("input", input_name),
            (input_name, "lif"),
            ("lif", output_name),
            (output_name, "output"),
        ]

        graph = _nir.NIRGraph(nodes=nodes, edges=edges)
        _nir.write(str(output_path), graph)
        print(f"* NIR graph saved to {output_path}")
        print(f"  Nodes: {' -> '.join(list(edges[0][:1]) + [e[1] for e in edges])}")
        print(f"  LIF: {h} neurons, beta={beta:.4f}, tau={tau_mem[0]:.6f}s")
        print(f"  Output: {output_mod.out_features} neurons")

        return output_path
