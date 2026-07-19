"""
Model export sub-package for the NeuCoDe toolkit.

Exporters are grouped by execution paradigm:
  Von Neumann / MCU (C header output): ANNExporter, SNNExporter
  Neuromorphic / device-specific:      AKD1000Exporter
  Interchange / backend-agnostic:      NIRExporter

ANNExporter and SNNExporter are named by model type because their C-header
output is architecture-agnostic (any Cortex-M MCU). AKD1000Exporter is named
by hardware target because the AKD1000 fully determines the model type (SNN),
quantization scheme (4-bit), and output format (.fbz). NIRExporter outputs the
Neuromorphic Intermediate Representation (.nir) for import into any NIR-aware
framework (Lava-DL, Sinabs, Rockpool, etc.).
"""
from .base import BaseExporter

from .ann import ANNExporter
from .snn import SNNExporter
from .akida import AKD1000Exporter
from .nir import NIRExporter

__all__ = [
    'BaseExporter',
    'ANNExporter',
    'SNNExporter',
    'AKD1000Exporter',
    'NIRExporter',
]