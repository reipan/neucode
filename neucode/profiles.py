"""
Pre-defined tuning profiles for the NeuCoDe reward system.

Provides the TuningProfile collection of WeightedMetricsReward instances covering
common industrial tuning strategies (aggressive, balanced, smooth, and disturbance-rejection).
"""
from .rewards import WeightedMetricsReward

class TuningProfile:
    """
    A collection of pre-defined WeightedMetricsReward instances representing
    common industrial tuning goals.
    """

    AGGRESSIVE = WeightedMetricsReward(
        name="Aggressive",
        weight_tracking=10.0,
        weight_effort=0.01,
        weight_overshoot=5.0,
        weight_speed=5.0,
        overshoot_threshold=10.0
    )
    """
    Fast setpoint tracking - allows for some overshoot and high control effort.
    """

    BALANCED = WeightedMetricsReward(
        name="Balanced",
        weight_tracking=1.0,
        weight_effort=0.1,
        weight_overshoot=10.0,
        weight_speed=1.0,
        overshoot_threshold=5.0
    )
    """
    A good compromise between speed and smoothness - the default for many applications.
    """

    SMOOTH = WeightedMetricsReward(
        name="Smooth",
        weight_tracking=0.1,
        weight_effort=1.0,
        weight_overshoot=20.0,
        weight_speed=0.1,
        overshoot_threshold=1.0
    )
    """
    Conservative tuning strategy focused on minimizing overshoot and reducing actuator stress.
    """

    DISTURBANCE_REJECTION = WeightedMetricsReward(
        name="DisturbanceRejection",
        weight_tracking=8.0,
        weight_effort=0.05,
        weight_overshoot=10.0,
        weight_speed=2.0,
        overshoot_threshold=5.0
    )
    """
    Tuned to quickly counteract external disturbances and return to the setpoint.
    """