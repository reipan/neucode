"""
Abstract interface base class for NeuCoDe hardware communication.

Defines the read/write contract that all concrete interface implementations must satisfy.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import time

class BaseInterface(ABC):
    """
    Base class for hardware communication interfaces.
    """
    @abstractmethod
    def open(self) -> None:
        """
        Open the communication interface.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Close the communication interface.
        """
        ...

    @abstractmethod
    def write_line(self, line: str) -> None:
        """
        Write a single line to the interface.

        :param line: The string to transmit.
        """
        ...

    @abstractmethod
    def read_line(self, timeout: float = 0.5) -> Optional[str]:
        """
        Read a single line from the interface.

        :param timeout: Maximum time to wait in seconds.
        :returns: The received line string, or None if the timeout expires.
        """
        ...

    def write_lines(self, lines: list[str]) -> None:
        """
        Write multiple lines to the interface sequentially.

        :param lines: List of strings to transmit in order.
        """
        for line in lines:
            self.write_line(line)

    def empty_buffer(self, seconds: float = 0.2) -> list[str]:
        """
        Drain the input buffer by reading for a fixed duration.

        :param seconds: How long to read for in seconds.
        :returns: All lines consumed from the buffer during that window.
        """
        out: list[str] = []
        start_time = time.time()
        while time.time() - start_time < seconds:
            line = self.read_line(timeout=0.1)
            if line is None:
                break
            out.append(line)
        return out