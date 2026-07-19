from abc import ABC, abstractmethod
import math
import matplotlib.pyplot as plt
import pandas as pd
import rich
from rich.console import Console
from rich.table import Table

from .results import ExperimentResult
from .controllers import PIDController
from .rewards import BaseReward

class BaseReporter(ABC):
    """
    Abstract base class for all reporters.
    """
    @abstractmethod
    def add_run(self, result: ExperimentResult, label: str):
        """
        Add a simulation result to the reporter with a given label.
        """
        pass
    @abstractmethod
    def report(self):
        """
        Generate the report from the accumulated runs.
        """
        pass

class GraphPlotter(BaseReporter):
    """
    Stateful utility class for plotting graphs of simulation results.
    """
    def __init__(self, title: str = "Simulation Results", style: str = 'seaborn-v0_8-whitegrid', fig: plt.Figure = None, alpha: float = 1.0, color_map: dict = None):
        plt.style.use(style)
        self.alpha = alpha
        self.color_map = color_map or {}
        if fig is None:
            self.fig, (self.ax1, self.ax2) = plt.subplots(
                nrows=2,
                ncols=1,
            figsize=(7.5, 5.5),
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}
        )
        else:
            self.fig = fig
            self.ax1, self.ax2 = fig.axes
        self.fig.suptitle(title, fontsize=16)
        # we need to track if we have already plotted setpoint to avoid duplicate legends
        self.setpoint_plotted = False
        
    def add_run(self, result: ExperimentResult, label: str):
        """
        Add a simulation result to the plotter with a given label.
        """
        if result.time_series is None:
            print(f"Warning: Skipping '{label}' because it does not contain time-series data.")
            return

        ts = result.time_series

        if not self.setpoint_plotted and 'setpoint' in ts:
            self.ax1.plot(ts['time'], ts['setpoint'], 'k--', label='Setpoint (r)')
            self.setpoint_plotted = True

        # Use color from map if available, otherwise let matplotlib choose
        plot_kwargs = {'linewidth': 1.5, 'label': label, 'alpha': self.alpha}
        if label in self.color_map:
            plot_kwargs['color'] = self.color_map[label]
        
        line, = self.ax1.plot(ts['time'], ts['measurement'], **plot_kwargs)
        color = line.get_color()
        self.ax2.plot(ts['time'], ts['control_effort'], color=color, alpha=self.alpha * 0.8)

    def _format_axes(self):
        """
        Internal helper to apply labels and legends right before showing/saving.
        """
        self.ax1.set_ylabel('System Output (y)', fontsize=10)
        self.ax1.legend(
            fontsize=9,
            loc='best', # Lets matplotlib find the spot with the least data overlap
            frameon=True,
            facecolor='white',
            framealpha=0.9
        )
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
        self.ax2.set_ylabel('Control Effort (u)', fontsize=10)
        self.ax2.set_xlabel('Time (s)', fontsize=10)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        self.fig.tight_layout()

    def save_pdf(self, filename: str):
        """
        Saves the plot as a vector PDF for LaTeX with tight bounding boxes.
        """
        self._format_axes()
        # bbox_inches='tight' removes excess white whitespace around the borders
        self.fig.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight')
        print(f"Saved plot to {filename}.pdf")

    def save_png(self, filename: str, dpi: int = 150):
        """
        Saves the plot as a PNG image.
        """
        self._format_axes()
        self.fig.savefig(f"{filename}.png", format='png', dpi=dpi, bbox_inches='tight')
        print(f"Saved plot to {filename}.png")

    def report(self):
        """
        Generate and display the plot of all added results.
        """
        self.ax1.set_ylabel('System Output (y)', fontsize=12)
        self.ax1.legend(
            fontsize=9,
            loc='best',
            frameon=True,
            facecolor='white',
            framealpha=0.9
        )
        self._format_axes()
        print("Showing plot... Close the plot window to exit.")
        plt.show()

class MetricsTable(BaseReporter):
    """
    Stateful utility class for displaying metrics in a table format.
    """
    def __init__(self, title: str, reward_strategy: BaseReward = None):
        self.results = []
        self.title = title
        self.reward_strategy = reward_strategy

    def add_run(self, result: ExperimentResult, label: str):
        """
        Add a simulation result to the table with a given label.
        """
        data = result.to_dict()
        data.pop('time_series', None)
        data['Controller'] = label

        if self.reward_strategy is not None:
            data['Cost'] = self.reward_strategy.calculate_cost(result)

        self.results.append(data)

    def to_dataframe(self):
        """
        Convert the metrics table to a pandas DataFrame.
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        wanted_columns = [
            'Controller',
            'success',
            'final_value',
            'samples_written',
            'overshoot_percent',
            'rise_time',
            'steady_state_error',
            'ise',
            'iae',
            'itae',
            'isu',
            'Cost'
        ]

        available_columns = [col for col in wanted_columns if col in df.columns]
        return df[available_columns]

    def report(self):
        """
        Display the metrics table.
        """
        df = self.to_dataframe()
        if df.empty:
            print("No results to display.")
            return
        
        console = Console()
        table = Table(title=f"\n--- {self.title} ---", 
                      show_header=True, 
                      header_style="bold magenta",
                      box=rich.box.ROUNDED)

        for col_name in df.columns:
            if col_name == "Cost":
                table.add_column(col_name, style="cyan", justify="right")
            elif col_name == "Controller":
                table.add_column(col_name, style="bold", justify="left")
            else:
                table.add_column(col_name, justify="right")

        for _, row in df.iterrows():
            row_values = []
            for col in df.columns:
                val = row[col]
                if isinstance(val, float):
                    if math.isnan(val):
                        row_values.append("[dim]nan[/dim]")
                    else:
                        row_values.append(f"{val:.3f}")
                else:
                    row_values.append(str(val))
            
            table.add_row(*row_values)
            
        console.print(table)

    def save_json(self, filename: str):
        """
        Save the metrics table as JSON.
        """
        df = self.to_dataframe()
        if df.empty:
            print("No results to save.")
            return
        
        df.to_json(f"{filename}.json", orient='records', indent=2)
        print(f"Saved metrics to {filename}.json")

class GainsTable:
    """
    A dedicated reporter that accumulates and displays the
    PID gains (parameters) for each controller in the experiment.
    """
    def __init__(self, title: str, dt: float = None):
        self.gains_list = []
        self.title = title
        self.dt = dt

    def add_run(self, controller: PIDController, label: str):
        """
        Adds a controller's gains to the table.
        """
        if not isinstance(controller, PIDController):
            print(f"Info: Skipping gains table for non-PID controller '{label}'.")
            return
            
        data = controller.params.copy()
        data['Controller'] = label

        if 'kd' in data and self.dt:
            kd_industrial = data.pop('kd')
            data['kd (pre-scaled)'] = kd_industrial
            data['kd_textbook'] = kd_industrial * self.dt
        self.gains_list.append(data)

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the collected gains as a pandas DataFrame."""
        if not self.gains_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.gains_list)
        
        # Re-order columns
        cols_to_show = [
            'Controller',
            'kp',
            'ki',
            'kd_textbook',
            'kd (pre-scaled)',
            'kd'
        ]
        final_cols = [col for col in cols_to_show if col in df.columns]
        other_cols = [col for col in df.columns if col not in final_cols and not col.startswith('i_')]
        return df[final_cols + other_cols]

    def report(self):
        """Prints a clean, formatted table of the pid gains."""
        df = self.to_dataframe()
        if df.empty:
            return
            
        console = Console()
        table = Table(title=f"\n--- {self.title} ---",
                      show_header=True,
                      header_style="bold cyan",
                      box=rich.box.ROUNDED)

        for col_name in df.columns:
            if col_name == "kd_textbook":
                table.add_column(col_name, style="bold yellow", justify="right")
            elif col_name == "Controller":
                table.add_column(col_name, style="bold", justify="left")
            else:
                table.add_column(col_name, justify="right")

        for _, row in df.iterrows():
            row_values = []
            for col in df.columns:
                val = row[col]
                if isinstance(val, float):
                    row_values.append(f"{val:.3f}")
                else:
                    row_values.append(str(val))
            
            table.add_row(*row_values)
            
        console.print(table)

    def save_json(self, filename: str):
        """
        Save the gains table as JSON.
        """
        df = self.to_dataframe()
        if df.empty:
            print("No gains to save.")
            return
        
        df.to_json(f"{filename}.json", orient='records', indent=2)
        print(f"Saved gains to {filename}.json")