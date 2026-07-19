from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class ExperimentResult:
    """
    Data container for the results of an experiment run.
    """
    success: bool
    final_value: float
    samples_written: int
    overshoot_percent: float = 0.0
    rise_time: float = 0.0
    ise: float = 0.0
    iae: float = 0.0
    itae: float = 0.0
    isu: float = 0.0
    peak_value: float = 0.0
    peak_time: float = 0.0
    steady_state_error: float = 0.0
    steady_state_error_percent: float = 0.0
    total_time: float = 0.0
    time_series: dict[str, np.ndarray] | None = None

    def __str__(self):
        """
        String representation of the most basic experiment result.
        """
        return (
            f"Experiment Result:\n"
            f"  - Success: {self.success}\n"
            f"  - Final Value: {self.final_value:.4f}\n"
            f"  - Overshoot: {self.overshoot_percent:.2f} %\n"
            f"  - Rise Time: {self.rise_time:.4f} s\n"
            f"  - ISE: {self.ise:.4f}\n"
            f"  - IAE: {self.iae:.4f}\n"
            f"  - ITAE: {self.itae:.4f}\n"
            f"  - ISU: {self.isu:.4f}\n"
        )
    
    def summary(self) -> str:
        """
        Detailed summary of the experiment result.
        """
        report = "--- Experiment Metrics Summary ---\n"
        for key, value in asdict(self).items():
            if isinstance(value, float):
                report += f"  - {key:<25}: {value:.4f}\n"
            else:
                report += f"  - {key:<25}: {value}\n"
        return report
    
    def to_dict(self) -> dict:
        """
        Convert the experiment result to a dictionary.
        """
        return asdict(self)