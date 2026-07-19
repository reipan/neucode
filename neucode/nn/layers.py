"""
Hardware-aware linear layer subclasses for the NeuCoDe SNN architecture.

SNNDirectInputLinear and SNNSpikeInputLinear are marker subclasses of nn.Linear
that the SNNExporter uses to distinguish the analog context branch from the
binary spike input branch during fixed-point quantisation.
"""
from neucode._torch_optional import nn

class SNNDirectInputLinear(nn.Linear):
    """
    Hardware-aware linear layer for continuous/analog inputs.
    - Fuses MinMaxScaler into the weights and bias.
    - Dictates the common accumulation alignment shift for the whole network.
    """
    pass

class SNNSpikeInputLinear(nn.Linear):
    """
    Hardware-aware linear layer for binary spiking inputs.
    - Forces int32 weights to align with the common accumulation alignment shift.
    """
    pass