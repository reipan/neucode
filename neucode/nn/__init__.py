"""
Custom neural network layer definitions for the NeuCoDe toolkit.

Exposes the hardware-aware linear layer variants used by HybridControlSNN:
SNNDirectInputLinear (analog context branch) and SNNSpikeInputLinear (spike branch).
"""
from .layers import SNNDirectInputLinear, SNNSpikeInputLinear

__all__ = [
    "SNNDirectInputLinear",
    "SNNSpikeInputLinear",
]