"""
Hardware communication sub-package for the NeuCoDe toolkit.

Exposes CommunicationClient, the parsed line types (TelemetryLine, LogLine),
and all interface classes (BaseInterface, SerialInterface, SerialConfig).
"""
from .client import CommunicationClient, LogLine, TelemetryLine
from .interface import *

__all__ = [
    'CommunicationClient',
    'LogLine',
    'TelemetryLine',
]

__all__.extend(interface.__all__)