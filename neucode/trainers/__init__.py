"""
NeuCoDe Trainers Module

This module provides a unified interface and contains classes and
functions related to training algorithms for control systems.
"""
from .ann_replacement import ANNReplacementTrainer
from .snn_replacement import SNNReplacementTrainer
from .._torch_optional import torch_available

__all__ = [
    "ANNReplacementTrainer",
    "SNNReplacementTrainer",
    "torch_available",
]