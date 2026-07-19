"""
Abstract tuner base class for the NeuCoDe toolkit.
"""
from abc import ABC, abstractmethod
from neucode.plants import Plant
from neucode.controllers import PIDController

class BaseTuner(ABC):
    """
    Abstract base class defining the interface for all tuners.
    """
    @abstractmethod
    def tune(self, plant: Plant) -> PIDController:
        """
        Return a tuned PIDController for the given plant.

        :param plant: A Plant instance describing the process to be controlled.
        :returns: A PIDController with gains optimised for the given plant.
        """
        pass    