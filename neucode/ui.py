"""
Logging and terminal UI utilities for the NeuCoDe toolkit.

Provides configure_logging() for setting up rich or stdlib logging, and
get_progress_bar() as a context manager for rich or no-op progress display.
"""
import logging
import sys
from contextlib import contextmanager

try:
    from rich.logging import RichHandler
    rich_available = True
except ImportError:
    rich_available = False

def configure_logging(level: str = "INFO", verbose: bool = False):
    """
    Configure the root logger for neucode and its dependencies.

    :param level: Logging level string (e.g. 'DEBUG', 'INFO', 'WARNING').
    :param verbose: If True, include timestamp, logger name, and line number in log output.
    """
    if verbose:
        log_format = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d]: %(message)s"
    else:
        log_format = "%(message)s"

    handlers = []
    if rich_available:
        handlers.append(RichHandler(rich_tracebacks=True, show_path=verbose, markup=True))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt="%H:%M:%S",
        handlers=handlers
    )
    
    logging.getLogger().setLevel(level)

@contextmanager
def get_progress_bar(description: str, total: int):
    """
    Context manager that yields a progress bar wrapper.

    Uses rich Progress when available, otherwise yields a no-op stub so call
    sites need no conditional logic.

    :param description: Label displayed to the left of the progress bar.
    :param total: Total number of steps to completion.
    :yields: A ProgressBarWrapper (rich) or DummyProgressBar (fallback) instance.
    """
    if rich_available:
        from rich.progress import (
            Progress,
            BarColumn,
            TextColumn,
            TimeRemainingColumn,
            MofNCompleteColumn
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[status]}[/dim]"),
            expand=True
        ) as progress:
            task = progress.add_task(description, total=total, status="")

            class ProgressBarWrapper:
                def __init__(self, progress_instance, task_id):
                    """
                    Wrap a rich Progress instance and a specific task ID.

                    :param progress_instance: The rich Progress object managing the display.
                    :param task_id: The task ID returned by Progress.add_task().
                    """
                    self._progress = progress_instance
                    self._task_id = task_id

                def update(self, advance: int = 1, description: str = None, status: str = None):
                    """
                    Advance the progress bar and optionally update its label or status text.

                    :param advance: Number of steps to advance the bar.
                    :param description: Optional new label for the bar.
                    :param status: Optional status string shown in the dim trailing column.
                    """
                    update_kwargs = {'advance': advance}
                    if description is not None:
                        update_kwargs['description'] = description
                    if status is not None:
                        update_kwargs['status'] = status
                    self._progress.update(self._task_id, **update_kwargs)
                    self._progress.refresh()

            yield ProgressBarWrapper(progress, task)
    else:
        class DummyProgressBar:
            """
            No-op progress bar used when rich is not installed.
            """
            def update(self, advance: int = 1, description: str = None):
                """
                Accept update calls silently and do nothing.
                """
                pass
        yield DummyProgressBar()