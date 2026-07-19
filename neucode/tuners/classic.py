"""
Classical analytical PID tuning rules for the NeuCoDe toolkit.
"""
from .base import BaseTuner
from neucode.plants import Plant, FOPDTPlant
from neucode.controllers import PIDController

class ZieglerNicholsReactionCurveTuner(BaseTuner):
    """
    Ziegler-Nichols Reaction Curve Tuner (open-loop method).
    This is only applicable for FOPDT plants.
    """
    def __init__(self, dt: float):
        """
        Initialize the Ziegler-Nichols reaction curve tuner.

        :param dt: Sampling interval in seconds; used to convert the textbook kd gain
            to the pre-scaled industrial convention (kd_industrial = kd_textbook / dt).
        :raises ValueError: If dt is not positive.
        """
        if dt <= 0:
            raise ValueError("dt must be positive for kd gain calculation.")
        self.dt = dt

    def tune(self, plant: Plant) -> PIDController:
        """
        Calculate PID gains using the Ziegler-Nichols reaction curve (open-loop) method.

        :param plant: An FOPDTPlant instance with valid K, tau, and theta parameters.
        :returns: A PIDController with kp, ki, and kd gains (kd pre-scaled by 1/dt).
        :raises TypeError: If plant is not an FOPDTPlant.
        :raises ValueError: If plant parameters are invalid for ZN tuning (K=0 or theta=0).
        """
        if not isinstance(plant, FOPDTPlant):
            raise TypeError(f"{self.__class__.__name__} only supports FOPDT plants.")
        
        K = plant.params['K']
        tau = plant.params['tau']
        theta = plant.params['theta']

        if K == 0 or theta == 0:
            raise ValueError("Invalid plant parameters for Ziegler-Nichols tuning.")

        Kc = 1.2 * (tau / (K * theta))
        Ti = 2 * theta
        Td = 0.5 * theta

        kp = Kc
        ki = Kc / Ti
        
        kd_raw = Kc * Td
        # Remember that our PID implementation expects pre-scaled kd
        kd_industrial = kd_raw / self.dt

        return PIDController(kp=kp, ki=ki, kd=kd_industrial)