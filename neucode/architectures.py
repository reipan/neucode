# We should move that into trainers
from ._torch_optional import (
    torch, nn, 
    snn, surrogate, utils,
    torch_available, snntorch_available
)

import neucode.nn.layers

class DefaultMLPArchitecture(nn.Module):
    """
    WE SHOULD REMOVE IT!
    A default Multi-Layer Perceptron (MLP) architecture for the ANN controller.

    This architecture consists of:
    - An input layer
    - Two hidden layers with ReLU activation
    - An output layer

    The default input size is 6 (for setpoint, measurement, error, obs_0, obs_1, obs_2),
    and the output size is 1 (for control variable).
    """
    def __init__(self, input_size=6, hidden_size=32, output_size=1):
        """
        Initializes the DefaultMLPArchitecture.

        :param input_size: Number of input features (default 6).
        :param hidden_size: Number of hidden neurons per layer (default 32).
        :param output_size: Number of output neurons (default 1).
        """
        if not torch_available:
            raise ImportError("Cannot instantiate DefaultMLPArchitecture: PyTorch is not available.")
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the DefaultMLPArchitecture.

        :param x: Input tensor of shape [Batch, InputSize].
        :return: Output tensor of shape [Batch, OutputSize].
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
class NoContextMLPArchitecture(nn.Module):
    """
    A stateless MLP architecture for the replacement controller.
    Takes the five closed-loop context features (setpoint, measurement, error,
    integral_error, derivative_error) as input and predicts the control effort.
    No plant observation context is used, making the controller plant-agnostic
    at inference time.
    """
    def __init__(self, input_size=5, hidden_size=32, output_size=1):
        """
        Initializes the NoContextMLPArchitecture.

        :param input_size: Number of input features (default 5: setpoint, measurement, error, integral_error, derivative_error).
        :param hidden_size: Number of hidden neurons in the MLP (default 32).
        :param output_size: Number of output neurons (default 1).
        """
        if not torch_available:
            raise ImportError("Cannot instantiate NoContextMLPArchitecture: PyTorch is not available.")
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the NoContextMLPArchitecture.

        :param x: Input tensor of shape [Batch, InputSize].
        :return: Output tensor of shape [Batch, OutputSize].
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class ErrorRateEncoder(nn.Module):
    """
    Rate encoder for the error signal.

    Produces two channels (positive, negative) with continuous values
    proportional to |error| / max_rate_input, split by sign.  A constant
    steady-state error produces a sustained signal the downstream LIF can
    integrate against -- unlike DeltaEncoder which goes silent when the
    error stops changing.

    Drop-in replacement for DeltaEncoder: same input [B, W, 1], same
    output [B, W, 2].
    """
    def __init__(self, max_rate_input=1.0):
        super().__init__()
        self.max_rate_input = max_rate_input

    def reset(self):
        pass

    def forward(self, x):
        """
        :param x: [Batch, Window, 1] -- raw error signal.
        :return:  [Batch, Window, 2] -- (pos_rate, neg_rate).
        """
        rate = (x.abs() / self.max_rate_input).clamp(0.0, 1.0)
        pos = torch.where(x > 0, rate, torch.zeros_like(rate))
        neg = torch.where(x < 0, rate, torch.zeros_like(rate))
        return torch.cat([pos, neg], dim=-1)


class DeltaEncoder(nn.Module):
    """
    Delta Spike Encoder.
    Encodes continuous input signals into spike trains based on changes exceeding a threshold.
    """
    def __init__(self, delta_threshold=0.0001):
        """
        Initializes the DeltaEncoder.

        :param delta_threshold: Minimum change in input to generate a spike (default 0.0001).
        """
        super().__init__()
        self.delta = delta_threshold
        self.register_buffer('prev_spike_value', torch.tensor(0.0))
        self.initialized = False

    def reset(self):
        """
        Resets the encoder state.
        """
        self.initialized = False
        self.prev_spike_value = torch.tensor(0.0)

    def forward(self, x):
        """
        Forward encoding pass to generate delta spikes.
        It produces two channels: positive and negative spikes, those are needed
        for representing increases and decreases in the input signal.

        :param x: Input tensor of shape [Batch, Window, 1].
        :return: Spike tensor of shape [Batch, Window, 2] (positive and negative spikes).
        """
        # x should be of shape: [Batch, Window, 1]
        batch_size, window, _ = x.shape
        spikes = torch.zeros(batch_size, window, 2, device=x.device)

        if not self.initialized:
            self.prev_spike_value = x[:, 0, :].detach().clone()
            self.initialized = True
        
        for t in range(window):
            curr = x[:, t, :]
            diff = curr - self.prev_spike_value
            
            # View(-1) ensures safe 1D masking even if Batch=1 (Inference)
            pos_mask = (diff > self.delta).view(-1)
            neg_mask = (diff < -self.delta).view(-1)

            if pos_mask.any():
                spikes[pos_mask, t, 0] = 1.0
                self.prev_spike_value[pos_mask] += self.delta

            if neg_mask.any():
                spikes[neg_mask, t, 1] = 1.0
                self.prev_spike_value[neg_mask] -= self.delta

        return spikes



class HybridSpikeEncoder(nn.Module):
    """
    Rate-encode the error channel, delta-encode the remaining context features.

    Channel 0 (error) is rate-encoded: the spike probability each timestep is
    proportional to |error| / max_rate_input, split into positive (error > 0)
    and negative (error < 0) channels.  A constant 8 deg error therefore produces
    a steady spike stream the downstream LIF can integrate against.

    Channels 1-4 (derivative, integral, prev_output, dt) keep delta encoding
    because they are naturally transient signals.

    Output: [Batch, Window, 2 + 2*(n_features-1)] = [B, W, 10] for n_features=5.
    """
    def __init__(self, n_features: int = 5, delta_threshold: float = 0.005,
                 max_rate_input: float = 1.0, rate_index: int = 2,
                 rate_zero_point: float = 0.0):
        super().__init__()
        self.n_features = n_features
        self.delta = delta_threshold
        self.max_rate_input = max_rate_input
        self.rate_index = rate_index
        self.rate_zero_point = rate_zero_point
        self.register_buffer('prev_spike_value',
                             torch.zeros(n_features - 1))
        self.initialized = False

    def reset(self):
        self.initialized = False
        self.prev_spike_value = torch.zeros(self.n_features - 1)

    def forward(self, x):
        """
        :param x: [Batch, Window, n_features] -- feature 0 is error.
        :return:  [Batch, Window, 2*n_features] -- rate(error) + delta(rest).
        """
        batch_size, window, _ = x.shape
        n_rest = self.n_features - 1
        device = x.device
        ri = self.rate_index

        # Rate encoding for error (feature at rate_index):
        # Center around the scaled zero-point so the pos/neg split
        # reflects the true sign of the raw error, not the MinMax offset.
        centered = x[:, :, ri] - self.rate_zero_point  # [B, W]
        rate = (centered.abs() / self.max_rate_input).clamp(0.0, 1.0)
        err_pos_spikes = torch.where(centered > 0, rate, torch.zeros_like(rate))
        err_neg_spikes = torch.where(centered < 0, rate, torch.zeros_like(rate))

        # Delta encoding for all features except rate_index
        rest = torch.cat([x[:, :, :ri], x[:, :, ri+1:]], dim=-1)  # [B, W, n_rest]
        pos_spikes = torch.zeros(batch_size, window, n_rest, device=device)
        neg_spikes = torch.zeros(batch_size, window, n_rest, device=device)

        if not self.initialized:
            self.prev_spike_value = rest[:, 0, :].detach().clone()
            self.initialized = True

        prev = self.prev_spike_value
        for t in range(window):
            diff = rest[:, t, :] - prev
            pos_mask = diff > self.delta
            neg_mask = diff < -self.delta
            pos_spikes[:, t, :] = pos_mask.float()
            neg_spikes[:, t, :] = neg_mask.float()
            prev = (prev + pos_mask.float() * self.delta
                    - neg_mask.float() * self.delta)

        self.prev_spike_value = prev.detach()

        # [err+, err-, rest_pos0, rest_neg0, ...]
        delta_interleaved = torch.stack(
            [pos_spikes, neg_spikes], dim=-1).reshape(
                batch_size, window, 2 * n_rest)

        return torch.cat([
            err_pos_spikes.unsqueeze(-1),
            err_neg_spikes.unsqueeze(-1),
            delta_interleaved,
        ], dim=-1)  # [B, W, 2 + 2*n_rest] = [B, W, 10]


class AkidaExportable:
    """
    Mixin for SNN architectures that can be exported to the Akida neuromorphic
    hardware format via the Akida export pipeline.

    Implements to_keras_weights() using the abstract _output_layer property so
    that the shared weight-fusion logic is defined once. Subclasses only need
    to declare which nn.Linear layer serves as the output.

    Both HybridControlSNN and PopulationControlSNN implement this protocol.
    To make a custom architecture exportable, inherit from this mixin and
    implement _output_layer.
    """

    @property
    def _output_layer(self):
        """
        The nn.Linear layer whose weights and bias are used as the output stage.
        Must be implemented by each subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement the _output_layer property "
            "to be Akida-exportable."
        )

    def to_keras_weights(self) -> dict:
        """
        Return weight matrices in the format expected by the Akida export pipeline.

        Fuses the spike-input and context-input weight matrices into a single
        (7, hidden) matrix and packages the output layer weights and bias.
        The result is consumed by the exporter's _fuse_weights() to build the
        float Keras model before quantization.

        :returns: dict with keys:
            W_fused (7, hidden), W_output (out_features, hidden),
            B_output (out_features,) or None, out_features (int), hidden_size (int).
        """
        import numpy as np
        W_spike   = self.fc_input_spikes.weight.detach().cpu().numpy()  # (hidden, 2)
        W_context = self.fc_context.weight.detach().cpu().numpy()        # (hidden, 5)
        W_output  = self._output_layer.weight.detach().cpu().numpy()
        B_output  = (
            self._output_layer.bias.detach().cpu().numpy()
            if self._output_layer.bias is not None else None
        )
        return {
            'W_fused':      np.vstack([W_spike.T, W_context.T]),  # (7, hidden)
            'hidden_size':  W_spike.shape[0],
            'W_output':     W_output,
            'B_output':     B_output,
            'out_features': self._output_layer.out_features,
        }


class HybridControlSNN(AkidaExportable, nn.Module):
    """
    Hybrid SNN Architecture for Control - MCU deployment target.

    Combines delta spike encoding of error signal with context inputs.
    The architecture has two branches:
    - Spike branch: Delta-encoded error changes (2 channels: positive and negative spikes)
    - Context branch: 5 analog context features

    Context features: [setpoint, measurement, error, integral_error, derivative_error]

    Both branches converge at a shared LIF (Leaky Integrate-and-Fire) layer with
    configurable beta decay. The output is a raw linear scalar, which
    matches the fixed-point C firmware inference kernel exactly:
        u = (linear_output - 0.5) * output_scale
    """
    def __init__(self, hidden_size=128, beta = 0.92, delta_threshold=0.005):
        """
        Initializes the HybridControlSNN model.

        :param hidden_size: Number of hidden neurons in the shared LIF layer (default 128).
        :param beta: Membrane decay factor for the LIF neuron (default 0.92).
        :param delta_threshold: Minimum error change required to emit a spike in the DeltaEncoder (default 0.005).
        """
        if not torch_available or not snntorch_available:
            raise ImportError("Cannot instantiate HybridControlSNN: PyTorch or snnTorch is not available.")
        super().__init__()
        self.encoder = DeltaEncoder(delta_threshold=delta_threshold)
        
        self.fc_input_spikes = neucode.nn.layers.SNNSpikeInputLinear(2, hidden_size, bias=False)
        self.fc_context = neucode.nn.layers.SNNDirectInputLinear(5, hidden_size, bias=False)
        
        # We should make the parameters configurable in the future
        self.lif = snn.Leaky(
            beta=beta,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            init_hidden=False,
            output=True
        )
        self.output_scale = nn.Linear(hidden_size, 1, bias=False)
        
        self._init_weights()

    def forward(self, error_chunk, context_chunk, mem=None):
        """
        Forward pass through the Hybrid SNN.

        :param error_chunk: Tensor of shape [Batch, Window, 1] representing the error signal.
        :param context_chunk: Tensor of shape [Batch, Window, 5] representing context inputs.
        :param mem: Optional membrane potential state for the LIF neurons.
        :return: Tuple of (predicted control values of shape [Batch, Window, 1], updated membrane potential state)
        """
        spikes = self.encoder(error_chunk)
        if mem is None:
            mem = self.lif.init_leaky()

        outputs = []
        
        for t in range(spikes.shape[1]):
            spike_t = spikes[:, t, :]
            context_t = context_chunk[:, t, :]

            # Fuse Spikes + Context
            current = self.fc_input_spikes(spike_t) + self.fc_context(context_t)
            
            # Integrate-and-Fire
            spike, mem = self.lif(current, mem)
            
            # Raw linear output
            # Firmware (snn_inference.c) computes:
            #   out = raw_sum * scale_factor - 0.5 * OUTPUT_SCALE
            # Python inference computes:
            #   u = (predicted - 0.5) * output_scale
            # Both are identical when predicted = linear output of this layer.
            predicted = self.output_scale(spike)
            outputs.append(predicted)
            
        return torch.stack(outputs, dim=1), mem
    
    def _init_weights(self):
        """
        Initializes weights of the linear layers with small values to prevent saturation.
        """
        nn.init.uniform_(self.fc_input_spikes.weight, -0.5, 0.5) 
        nn.init.uniform_(self.fc_context.weight, -0.5, 0.5)
        nn.init.uniform_(self.output_scale.weight, -0.5, 0.5)

    def reset_states(self):
        """
        Reset encoder and LIF membrane state.
        """
        self.encoder.reset()
        utils.reset(self.lif)

    @property
    def _output_layer(self):
        return self.output_scale

class PopulationControlSNN(AkidaExportable, nn.Module):
    """
    Population-Coded SNN Architecture for Control.
    
    Similar to HybridControlSNN, but uses a population of output neurons instead of a single scalar output.
    The outputs of the population are averaged to produce the final continuous control effort.
    This resolves quantization limit cycles on edge NPUs by creating a high-resolution aggregate signal.
    """
    def __init__(self, hidden_size=128, population_size=64, beta=0.92,
                 max_rate_input=1.0):
        """
        Initializes the PopulationControlSNN model.

        :param hidden_size: Number of hidden neurons in the shared LIF layer (default 128).
        :param population_size: Number of output neurons to average (default 64).
        :param beta: Membrane decay factor for the LIF neuron (default 0.92).
        :param max_rate_input: Error magnitude that saturates the rate encoder (default 1.0).
        """
        if not torch_available or not snntorch_available:
            raise ImportError("Cannot instantiate PopulationControlSNN: PyTorch or snnTorch is not available.")
        super().__init__()
        self.encoder = ErrorRateEncoder(max_rate_input=max_rate_input)
        
        self.fc_input_spikes = neucode.nn.layers.SNNSpikeInputLinear(2, hidden_size, bias=False)
        self.fc_context = neucode.nn.layers.SNNDirectInputLinear(5, hidden_size, bias=False)
        
        self.lif = snn.Leaky(
            beta=beta,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            init_hidden=False,
            output=True
        )
        
        self.population_size = population_size
        self.output_population = nn.Linear(hidden_size, population_size, bias=True)

        self._init_weights()

    def forward(self, error_chunk, context_chunk, mem=None):
        """
        Forward pass through the Population-coded SNN.

        :param error_chunk: Tensor of shape [Batch, Window, 1] representing the error signal.
        :param context_chunk: Tensor of shape [Batch, Window, 5] representing context inputs.
        :param mem: Optional membrane potential state for the LIF neurons.
        :return: Tuple of (predicted control values of shape [Batch, Window, 1], updated membrane potential state)
        """
        spikes = self.encoder(error_chunk)
        if mem is None:
            mem = self.lif.init_leaky()

        outputs = []

        for t in range(spikes.shape[1]):
            spike_t = spikes[:, t, :]
            context_t = context_chunk[:, t, :]

            # Fuse Spikes + Context
            current = self.fc_input_spikes(spike_t) + self.fc_context(context_t)

            # Integrate-and-Fire
            spike, mem = self.lif(current, mem)

            # Population Decode: compute the mean across the population
            pop_out = self.output_population(spike)
            predicted = pop_out.mean(dim=-1, keepdim=True)

            outputs.append(predicted)

        return torch.stack(outputs, dim=1), mem

    def _init_weights(self):
        """
        Initializes weights of the linear layers with small values to prevent saturation.
        """
        nn.init.uniform_(self.fc_input_spikes.weight, -0.5, 0.5)
        nn.init.uniform_(self.fc_context.weight, -0.5, 0.5)
        nn.init.uniform_(self.output_population.weight, -0.1, 0.1)
        nn.init.constant_(self.output_population.bias, 0.5)

    def reset_states(self):
        """
        Reset encoder and LIF membrane state.
        """
        self.encoder.reset()
        utils.reset(self.lif)

    @property
    def _output_layer(self):
        return self.output_population


class SpikeControlSNN(nn.Module):
    """
    All-spike SNN architecture for control - NIR-exportable.

    All 5 context features are delta-encoded into 10 binary spike channels
    (pos/neg per feature).  No analog current injection -- the LIF layer
    receives only spikes, making the graph a standard Linear -> LIF -> Linear
    chain that maps directly to NIR and neuromorphic hardware backends.

    Uses population decoding (mean of *population_size* output neurons) to
    recover continuous control effort from binary spikes.
    """
    def __init__(self, hidden_size=256, population_size=64,
                 beta=0.92, delta_threshold=0.005,
                 max_rate_input=0.2, rate_zero_point=0.0):
        if not torch_available or not snntorch_available:
            raise ImportError("PyTorch and snnTorch are required.")
        super().__init__()

        self.hidden_size = hidden_size
        self.population_size = population_size
        self.n_spike_channels = 10  # 5 features x 2 (pos/neg)

        self.encoder = HybridSpikeEncoder(
            n_features=5, delta_threshold=delta_threshold,
            max_rate_input=max_rate_input,
            rate_zero_point=rate_zero_point)

        self.fc_input = neucode.nn.layers.SNNSpikeInputLinear(self.n_spike_channels, hidden_size, bias=False)
        self.lif = snn.Leaky(
            beta=beta,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            init_hidden=False,
            output=True,
        )
        self.output_population = nn.Linear(hidden_size, population_size, bias=True)

        self._init_weights()

    def forward(self, _error_chunk, context_chunk, mem=None):
        """
        :param _error_chunk: Unused (error is in context_chunk[:, :, 2]).
            Kept for trainer interface compatibility.
        :param context_chunk: [Batch, Window, 5] -- all 5 context features.
        :param mem: Optional LIF membrane state.
        :return: (predictions [Batch, Window, 1], updated membrane state)
        """
        all_spikes = self.encoder(context_chunk)

        if mem is None:
            mem = self.lif.init_leaky()

        outputs = []
        for t in range(all_spikes.shape[1]):
            current = self.fc_input(all_spikes[:, t, :])
            spike, mem = self.lif(current, mem)
            predicted = self.output_population(spike).mean(dim=-1, keepdim=True)
            outputs.append(predicted)

        return torch.stack(outputs, dim=1), mem

    def _init_weights(self):
        nn.init.uniform_(self.fc_input.weight, -0.5, 0.5)
        nn.init.uniform_(self.output_population.weight, -0.1, 0.1)
        nn.init.constant_(self.output_population.bias, 0.5)

    def reset_states(self):
        self.encoder.reset()
        utils.reset(self.lif)

