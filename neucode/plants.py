"""
Plant model definitions for the NeuCoDe simulation pipeline.

Provides the abstract Plant base class, FOPDTPlant (First-Order Plus Dead Time),
and FOIPDTPlant (First-Order Integrating Plus Dead Time) implementations.
"""
from abc import ABC, abstractmethod
import numpy as np

class Plant(ABC):
    """
    Abstract base class for plants.
    """
    pass

    @abstractmethod
    def get_observation_vector(self) -> np.ndarray:
        """
        Return the observation vector representing the plant's parameters.

        :returns: 1-D float32 array of plant parameters.
        :rtype: numpy.ndarray
        """
        pass

class FOPDTPlant(Plant):
    """
    First-Order Plus Dead Time (FOPDT) plant model.

    Encapsulates the three scalar parameters (K, tau, theta) that fully
    characterise a linear FOPDT process and exposes them as a fixed
    observation vector for use by plant generators and the simulation core.
    """
    def __init__(self, K: float, tau: float, theta: float, friction: float = 0.0):
        """
        Initialize an FOPDT plant with the given parameters.

        :param K: Process gain (steady-state output per unit input).
        :param tau: Time constant in seconds (must be positive).
        :param theta: Dead time in seconds (must be non-negative).
        :param friction: Coulomb friction [output units], opposing motion. 0 = disabled.
        :raises ValueError: If K is not a number.
        :raises ValueError: If tau is not a positive number.
        :raises ValueError: If theta is not a non-negative number.
        """
        if not isinstance(K, (int, float)):
            raise ValueError("Process gain K must be a number.")
        if not isinstance(tau, (int, float)) or tau <= 0:
            raise ValueError("Time constant tau must be a positive number.")
        if not isinstance(theta, (int, float)) or theta < 0:
            raise ValueError("Dead time theta must be a non-negative number.")
        if not isinstance(friction, (int, float)) or friction < 0:
            raise ValueError("Friction must be a non-negative number.")

        self.params = {
            'K': K,
            'tau': tau,
            'theta': theta,
            'friction': friction
        }

    def __repr__(self):
        """
        Return a human-readable string of the form '(FOPDT: K=..., tau=..., theta=...)'.
        """
        f = f", friction={self.params['friction']}" if self.params.get('friction', 0) > 0 else ""
        return f" (FOPDT: K={self.params['K']}, tau={self.params['tau']}, theta={self.params['theta']}{f})"
    
    def get_observation_vector(self) -> np.ndarray:
        """
        Return the [K, tau, theta] parameter vector.

        :returns: 1-D float32 array [K, tau, theta].
        :rtype: numpy.ndarray
        """
        return np.array([
            self.params['K'],
            self.params['tau'],
            self.params['theta']
        ], dtype=np.float32)


class FOIPDTPlant(Plant):
    """
    First-Order Integrating Plus Dead Time (FOIPDT) plant model.

    Models a velocity plant where the output (position) is the time-integral
    of the velocity response to the input:

        G(s) = Kv * e^{-theta*s} / (s * (tau*s + 1))

    The observation vector [Kv, tau, theta] has the same shape as
    FOPDTPlant's [K, tau, theta], so the supervised tuner is compatible
    with both plant types without architecture changes.
    """
    def __init__(self, Kv: float, tau: float, theta: float, friction: float = 0.0):
        """
        Initialize a FOIPDT plant.

        :param Kv: Velocity gain [output_unit/s per input_unit].
        :param tau: Velocity lag time constant in seconds (must be positive).
        :param theta: Dead time in seconds (must be non-negative).
        :param friction: Coulomb friction [input units], opposing motion. 0 = disabled.
        """
        if not isinstance(Kv, (int, float)):
            raise ValueError("Velocity gain Kv must be a number.")
        if not isinstance(tau, (int, float)) or tau <= 0:
            raise ValueError("Time constant tau must be a positive number.")
        if not isinstance(theta, (int, float)) or theta < 0:
            raise ValueError("Dead time theta must be a non-negative number.")
        if not isinstance(friction, (int, float)) or friction < 0:
            raise ValueError("Friction must be a non-negative number.")

        self.params = {'Kv': Kv, 'tau': tau, 'theta': theta, 'friction': friction}

    def __repr__(self):
        f = f", friction={self.params['friction']}" if self.params.get('friction', 0) > 0 else ""
        return f" (FOIPDT: Kv={self.params['Kv']}, tau={self.params['tau']}, theta={self.params['theta']}{f})"

    def get_observation_vector(self) -> np.ndarray:
        """
        Return the [Kv, tau, theta] parameter vector.

        :returns: 1-D float32 array [Kv, tau, theta].
        :rtype: numpy.ndarray
        """
        return np.array([
            self.params['Kv'],
            self.params['tau'],
            self.params['theta']
        ], dtype=np.float32)