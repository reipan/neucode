"""
Hardware interface abstractions for the NeuCoDe communication layer.

Provides BaseInterface (abstract) and SerialInterface with its SerialConfig dataclass.
"""
from .base import BaseInterface
from .serial import SerialInterface, SerialConfig

__all__ = [
    "BaseInterface",
    "SerialInterface",
    "SerialConfig",
]