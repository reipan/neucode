"""
High-level communication client for NeuCoDe hardware targets.

Wraps any BaseInterface with a typed command API and line parser that converts
raw serial output into TelemetryLine or LogLine instances.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import time

from neucode.communication.interface import BaseInterface

@dataclass(frozen=True)
class TelemetryLine:
    """
    A parsed hardware telemetry frame.

    Format: ``T,<timestamp>,<setpoint>,<measured_value>,<control_output>``

    :param t: Timestamp in seconds reported by the hardware.
    :param sp: Setpoint value at this tick.
    :param y: Measured process variable at this tick.
    :param u: Control output applied at this tick.
    :param raw: Original unparsed string for debugging.
    """
    t: float
    sp: float
    y: float
    u: float
    raw: str

@dataclass(frozen=True)
class LogLine:
    """
    A parsed hardware log message.

    :param msg: Message text with the leading ``L,`` prefix stripped.
    :param raw: Original unparsed string for debugging.
    """
    msg: str
    raw: str

ParsedLine = Union[TelemetryLine, LogLine]

class CommunicationClient:
    """
    Facade Client for communicating with hardware via a given interface.
    """
    def __init__(self, interface: BaseInterface):
        """
        Initialise the client with a hardware interface.

        :param interface: A BaseInterface instance (e.g. SerialInterface) that
            handles the physical byte transport.
        """
        self.interface = interface

    def open(self):
        """
        Open the communication interface.
        """
        self.interface.open()

    def close(self):
        """
        Close the communication interface.
        """
        self.interface.close()

    def empty_buffer(self, seconds: float = 0.2) -> list[str]:
        """
        Drain the input buffer by reading for a fixed duration.

        :param seconds: How long to read for in seconds.
        :returns: Lines consumed from the buffer.
        """
        self.interface.empty_buffer(seconds=seconds)

    def send(self, msg: str) -> None:
        """
        Send a raw message line to the hardware.

        :param msg: The message string to transmit (newline appended by the interface).
        """
        self.interface.write_line(msg)

    def read_raw(self, timeout: float = 0.5) -> Optional[ParsedLine]:
        """
        Read a raw (unparsed) line from the hardware.

        :param timeout: Maximum time to wait in seconds.
        :returns: The raw string, or None if no data arrives within the timeout.
        """
        return self.interface.read_line(timeout=timeout)
    
    def read(self, timeout: float = 0.5) -> Optional[ParsedLine]:
        """
        Read and parse a line from the hardware.

        :param timeout: Maximum time to wait in seconds.
        :returns: A TelemetryLine, LogLine, or None if no data arrives within the timeout.
        """
        raw_line = self.interface.read_line(timeout=timeout)
        if raw_line is None:
            return None
    
        return self.parse_line(raw_line)
    
    @staticmethod
    def parse_line(raw_line: str) -> ParsedLine:
        """
        Parse a raw hardware line into a typed object.

        Telemetry format: ``T,<timestamp>,<setpoint>,<measured_value>,<control_output>``
        Log format: ``L,<msg>``
        Unknown lines are wrapped in a LogLine with an ``(unparsed)`` prefix.

        :param raw_line: The raw string received from the interface.
        :returns: A TelemetryLine or LogLine instance.
        :raises ValueError: If a telemetry line is malformed or contains non-numeric fields.
        """
        if raw_line.startswith("T,"):
            parts = raw_line.split(",")
            if len(parts) < 5:
                raise ValueError(f"Malformed telemetry line: {raw_line!r}")
            try:
                return TelemetryLine(
                    t=float(parts[1]),
                    sp=float(parts[2]),
                    y=float(parts[3]),
                    u=float(parts[4]),
                    raw=raw_line,
                )
            except ValueError as e:
                raise ValueError(f"Invalid telemetry values: {raw_line!r}") from e

        # Log: L,<msg>
        if raw_line.startswith("L,"):
            return LogLine(msg=raw_line[2:], raw=raw_line)

        # Unknown -> treat as log
        return LogLine(msg=f"(unparsed) {raw_line}", raw=raw_line)

    def ping(self) -> None:
        """
        Wrapper for ping command.
        """
        self.send("ping")

    def show(self) -> None:
        """
        Wrapper for show command.
        """
        self.send("show")
        
    def pid(self, kp: float, ki: float, kd: float) -> None:
        """
        Send PID gain configuration to the hardware.

        :param kp: Proportional gain.
        :param ki: Integral gain.
        :param kd: Derivative gain.
        """
        self.send(f"pid {kp} {ki} {kd}")
    
    def setpoint_step(self, time: float, value: float) -> None:
        """
        Command the hardware to apply a step setpoint change.

        :param time: Time in seconds at which the step occurs.
        :param value: Target setpoint value after the step.
        """
        self.send(f"sp step {time} {value}")

    def setpoint_ramp(self, time: float, duration: float, a: float, b: float) -> None:
        """
        Command the hardware to apply a ramp setpoint.

        :param time: Start time of the ramp in seconds.
        :param duration: Duration of the ramp in seconds.
        :param a: Setpoint value at the start of the ramp.
        :param b: Setpoint value at the end of the ramp.
        """
        self.send(f"sp ramp {time} {duration} {a} {b}")

    def setpoint_sin(self, time: float, amp: float, freq: float) -> None:
        """
        Command the hardware to apply a sinusoidal setpoint.

        :param time: Start time of the sinusoid in seconds.
        :param amp: Amplitude of the sinusoidal variation.
        :param freq: Frequency of the sinusoid in Hz.
        """
        self.send(f"sp sin {time} {amp} {freq}")

    def mode(self, mode: str) -> None:
        """
        Set the controller mode on the hardware.

        :param mode: One of 'pid', 'ann', 'snn', or 'open'.
        :raises ValueError: If mode is not one of the accepted values.
        """
        mode = mode.strip().lower()
        if mode not in ("pid", "ann", "snn", "open"):
            raise ValueError(f"Invalid mode: {mode!r}")
        self.send(f"mode {mode}")

    def input_step(self, time: float, value: float) -> None:
        """
        Configure an open-loop input step waveform u(t).

        :param time: Time in seconds at which the step is applied.
        :param value: Input value after the step.
        """
        self.send(f"u step {time} {value}")

    def input_ramp(self, time: float, duration: float, a: float, b: float) -> None:
        """
        Configure an open-loop input ramp waveform u(t).

        :param time: Start time of the ramp in seconds.
        :param duration: Duration of the ramp in seconds.
        :param a: Input value at the start of the ramp.
        :param b: Input value at the end of the ramp.
        """
        self.send(f"u ramp {time} {duration} {a} {b}")

    def input_sin(self, time: float, amp: float, freq: float) -> None:
        """
        Configure an open-loop sinusoidal input waveform u(t).

        :param time: Start time of the sinusoid in seconds.
        :param amp: Amplitude of the sinusoidal input.
        :param freq: Frequency of the sinusoid in Hz.
        """
        self.send(f"u sin {time} {amp} {freq}")
    
    def exp_start(self, nozero: bool = False) -> None:
        """
        Wrapper for exp start command.

        :param nozero: If True, sends 'exp start nozero' which skips sensor
            zeroing on the firmware side. Use this for PID return-to-zero
            episodes so the original sensor reference is preserved.
        """
        if nozero:
            self.send("exp start nozero")
        else:
            self.send("exp start")

    def exp_stop(self) -> None:
        """
        Wrapper for exp stop command.
        """
        self.send("exp stop")

    def exp_dump(self, frame_timeout: float = 0.05) -> list[TelemetryLine]:
        """
        Request the firmware to stream its full 100 Hz experiment buffer.

        Must be called **after** ``exp_stop()``.  Sends ``exp dump``, reads
        the ``ok: exp dump N frames`` header, collects N ``TelemetryLine``
        frames (firmware streams them with the standard ``T,`` prefix), then
        waits for the ``ok: dump complete`` sentinel.

        :param frame_timeout: Per-frame read timeout in seconds (default 50 ms).
        :returns: List of TelemetryLine at the firmware loop rate (e.g. 100 Hz).
        :raises RuntimeError: If the header is not received within 2 s.
        """
        self.send("exp dump")

        # Wait for header: L,ok: exp dump N frames
        n_expected: int | None = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            msg = self.read(timeout=0.2)
            if msg is None:
                continue
            if isinstance(msg, LogLine) and msg.msg.startswith("ok: exp dump"):
                tokens = msg.msg.split()
                try:
                    n_expected = int(tokens[3])
                except (IndexError, ValueError) as exc:
                    raise RuntimeError(
                        f"exp dump: malformed header {msg.msg!r}"
                    ) from exc
                break

        if n_expected is None:
            raise RuntimeError("exp dump: no header received from firmware")

        # Collect exactly N telemetry frames
        frames: list[TelemetryLine] = []
        for _ in range(n_expected):
            msg = self.read(timeout=frame_timeout)
            if isinstance(msg, TelemetryLine):
                frames.append(msg)

        # Drain until sentinel or timeout
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            msg = self.read(timeout=0.1)
            if msg is None:
                break
            if isinstance(msg, LogLine) and "dump complete" in msg.msg:
                break

        return frames