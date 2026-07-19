"""
NeuCoDe Tuners Module

This module provides a unified interface and contains classes and
functions related to tuning algorithms for control systems.
"""
from .base import BaseTuner
from .classic import ZieglerNicholsReactionCurveTuner

__all__ = [
    "BaseTuner",
    "ZieglerNicholsReactionCurveTuner",
]

try:
    from .rl import RLTuner
    from .supervised import SupervisedTuner
    __all__.extend([
        "RLTuner",
        "SupervisedTuner"
    ])
except ImportError:
    print("Warning: PyTorch or stable-baselines3 not found. RL-based tuners will not be available.")