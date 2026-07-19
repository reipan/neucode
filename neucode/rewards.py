"""
Reward and cost strategies for evaluating controller performance in NeuCoDe.

Defines the abstract BaseReward interface and concrete implementations used
by tuners and the dataset generator to score simulation outcomes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import math
import numpy as np

from .results import ExperimentResult

class BaseReward(ABC):
    """
    Abstract base class for defining reward strategies.
    """
    @abstractmethod
    def calculate_cost(self, result: ExperimentResult) -> float:
        """
        Takes a SimulationResult and returns a cost value.

        :param result: ExperimentResult containing performance metrics from a simulation run.
        :returns: Scalar cost value (lower is better).
        """
        pass

    def calculate_reward(self, result: ExperimentResult) -> tuple[float, float]: 
        """
        Takes a SimulationResult, calculates and returns a reward and cost tuple.

        :param result: ExperimentResult containing performance metrics from a simulation run.
        :returns: Tuple of (reward, cost) where reward is a positive scalar (higher is better)
            and cost is a positive scalar (lower is better).
        """
        cost = self.calculate_cost(result)
        reward = 1.0 / (cost + 1e-6)
        return reward, cost

class ISEOvershootReward(BaseReward):
    """
    LEGACY (we should probably remove it!)
    Reward strategy that penalizes overshoot in the system response.
    * This is just the example strategy taken from the hard-coded one in tuners.py
    * Does not have test coverage yet (because it might be changed/removed later).
    """
    def __init__(self, overshoot_penalty_multiplier: float = 10.0, overshoot_threshold: float = 5.0):
        """
        Initialize the reward strategy.

        :param overshoot_penalty_multiplier: Scaling factor applied to the squared overshoot
            excess above ``overshoot_threshold``.
        :param overshoot_threshold: Overshoot percentage below which no penalty is applied.
        """
        self.overshoot_penalty_multiplier = overshoot_penalty_multiplier
        self.overshoot_threshold = overshoot_threshold

    def calculate_cost(self, result: ExperimentResult) -> float:
        """
        Calculate cost based on Integral of Squared Error (ISE) and a quadratic overshoot penalty.

        :param result: ExperimentResult containing performance metrics from a simulation run.
        :returns: Scalar cost value (lower is better).
        """
        ise = result.ise if (result.ise is not None and not math.isnan(result.ise)) else 1e6
        overshoot = result.overshoot_percent if (result.overshoot_percent is not None and not math.isnan(result.overshoot_percent)) else 0.0
        overshoot_penalty = max(0.0, overshoot - self.overshoot_threshold)**2
        cost = ise + (self.overshoot_penalty_multiplier * overshoot_penalty)
        return cost
    
class WeightedMetricsReward(BaseReward):
    """
    Cost strategy based on a weighted sum of classical time-domain performance indices.

    Combines ITAE, ISU, a quadratic overshoot penalty, and an optional
    failure penalty for steady-state error exceeding 10%.
    """
    def __init__(self, weight_tracking: float = 1.0, weight_effort: float = 0.01, weight_overshoot: float = 1.0, weight_speed: float = 0.0, overshoot_threshold: float = 5.0, failure_penalty: float = 1000.0, name: str = "WeightedMetrics"):
        """
        Initialize the weighted metrics reward.

        :param weight_tracking: Weight for the Integral of Time-weighted Absolute Error (ITAE).
        :param weight_effort: Weight for the Integral of Squared Control Effort (ISU).
        :param weight_overshoot: Weight for the quadratic overshoot penalty term.
        :param weight_speed: Weight for the normalized rise time penalty.
        :param overshoot_threshold: Overshoot percentage below which no penalty is applied.
        :param failure_penalty: Additional cost added when steady-state error exceeds 10%.
        :param name: Human-readable identifier for this reward configuration.
        """
        self.weight_tracking = weight_tracking
        self.weight_effort = weight_effort
        self.weight_overshoot = weight_overshoot
        self.weight_speed = weight_speed
        self.overshoot_threshold = overshoot_threshold
        self.failure_penalty = failure_penalty
        self.name = name

    def calculate_cost(self, result: ExperimentResult) -> float:
        """
        Calculate the weighted cost from a simulation result.

        Combines ITAE, ISU, normalized rise time, and a quadratic overshoot penalty.
        Adds ``failure_penalty`` if steady-state error exceeds 10%.

        :param result: ExperimentResult containing performance metrics from a simulation run.
        :returns: Scalar cost value (lower is better).
        """
        isu = result.isu if (result.isu is not None and not math.isnan(result.isu)) else 1e6
        itae = result.itae if (result.itae is not None and not math.isnan(result.itae)) else 1e6
        rise_time = result.rise_time if (result.rise_time is not None and not math.isnan(result.rise_time)) else 1e6
        
        # Normalize rise_time by total_time to bring it to a comparable scale
        normalized_rise_time = rise_time / result.total_time if result.total_time > 0 else 1e6

        cost = (self.weight_tracking * itae) + (self.weight_effort * isu) + (self.weight_speed * normalized_rise_time)

        overshoot = result.overshoot_percent if (result.overshoot_percent is not None and not math.isnan(result.overshoot_percent)) else 0.0
        overshoot_penalty = max(0.0, overshoot - self.overshoot_threshold)**2
        cost += (self.weight_overshoot * overshoot_penalty)

        sse_percent = result.steady_state_error_percent
        if sse_percent is not None and abs(sse_percent) > 10.0:
            cost += self.failure_penalty

        return cost