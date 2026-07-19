"""
NeuCoDe Toolkit

A Python toolkit for neuromorphic control design and deployment,
featuring a C simulation core.
"""
from .experiment import Experiment
from .controllers import Controller, PIDController, ANNController, SNNController, BaseSNNController, KerasController, AkidaController
from .trainers import *
from .architectures import DefaultMLPArchitecture, NoContextMLPArchitecture, HybridControlSNN, PopulationControlSNN
from .plants import Plant, FOPDTPlant, FOIPDTPlant
from .signals import Setpoint, StepSetpoint, RampSetpoint, SinSetpoint, MultiStepSetpoint, Disturbance
from .harness import BaseHarness, SimulationHarness
from .results import ExperimentResult
from .scenarios import BaseScenario, BenchmarkScenario, StepBenchmark, RobustStepBenchmark, RampBenchmark, NoisyBenchmark, DisturbedBenchmark
from .generators import BasePlantGenerator, FOPDTPlantGenerator, RandomFOPDTPlantGenerator, FOIPDTPlantGenerator, RandomFOIPDTPlantGenerator
from .reporting import GraphPlotter, MetricsTable, GainsTable
from .evaluation import ControllerComparison
from .rewards import BaseReward, ISEOvershootReward, WeightedMetricsReward
from .datagen import TuningDatasetGenerator, ReplacementDatasetGenerator
from .profiles import TuningProfile
from .tuners import *
from .harness import *
from .communication import *
from .exporters import *
from .simcore import StandaloneMetrics, Simulation, PIDState
from .nn import SNNDirectInputLinear, SNNSpikeInputLinear

from .ui import get_progress_bar

__all__ = [
    'Experiment',
    'Controller',
    'PIDController',
    'ANNController',
    'SNNController',
    'BaseSNNController',
    'KerasController',
    'AkidaController',
    'DefaultMLPArchitecture',
    'NoContextMLPArchitecture',
    'HybridControlSNN',
    'PopulationControlSNN',
    'Plant',
    'FOPDTPlant',
    'FOIPDTPlant',
    'Setpoint',
    'StepSetpoint',
    'MultiStepSetpoint',
    'RampSetpoint',
    'SinSetpoint',
    'Disturbance',
    'ExperimentResult',
    'BaseScenario',
    'BenchmarkScenario',
    'StepBenchmark',
    'RobustStepBenchmark',
    'RampBenchmark',
    'NoisyBenchmark',
    'DisturbedBenchmark',
    'BasePlantGenerator',
    'FOPDTPlantGenerator',
    'RandomFOPDTPlantGenerator',
    'FOIPDTPlantGenerator',
    'RandomFOIPDTPlantGenerator',
    'BaseReward',
    'ISEOvershootReward',
    'WeightedMetricsReward',
    'TuningProfile',
    'GraphPlotter',
    'MetricsTable',
    'GainsTable',
    'ControllerComparison',
    'TuningDatasetGenerator',
    'ReplacementDatasetGenerator',
    'get_progress_bar',
    'SNNDirectInputLinear',
    'SNNSpikeInputLinear',
]

__all__.extend([
    'Simulation',
    'StandaloneMetrics',
    'PIDState',
])

__all__.extend(communication.__all__)
__all__.extend(tuners.__all__)
__all__.extend(harness.__all__)
__all__.extend(trainers.__all__)
__all__.extend(exporters.__all__)
    