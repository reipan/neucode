"""
Serial port interface implementation for NeuCoDe hardware communication.

Provides SerialConfig (connection parameters) and SerialInterface, which runs a
background reader thread to dequeue incoming lines without blocking the caller.
"""
from dataclasses import dataclass
from typing import Optional
from queue import Queue, Empty
import serial
import threading
import time

from .base import BaseInterface

@dataclass
class SerialConfig:
    """
    Configuration parameters for a serial port connection.

    :param port: Serial port identifier (e.g. '/dev/ttyUSB0' or 'COM3').
    :param baudrate: Communication baud rate in bps.
    :param timeout: Per-read timeout in seconds passed to pyserial.
    :param data_chunk_size: Number of bytes to read per chunk in the background thread.
    """
    port: str
    baudrate: int = 9600
    timeout: float = 1.0
    data_chunk_size: int = 128

class SerialInterface(BaseInterface):
    """
    Serial port implementation of BaseInterface.

    Opens the port, starts a background daemon thread that reads incoming bytes
    into a queue, and exposes blocking read_line() / non-blocking write_line() calls.
    """
    def __init__(self, config: SerialConfig):
        """
        Initialise the serial interface with the given configuration.

        :param config: SerialConfig instance specifying port, baudrate, and timeouts.
        """
        self.config = config
        self._serial: Optional['serial.Serial'] = None

        # Receive Queue
        self._receive_queue: Queue[str] = Queue()

        # Threading Stuff
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Status
        self._is_open = False

    def open(self) -> None:
        """
        Open the serial port and start the background reader thread.

        Idempotent: calling open() on an already-open interface has no effect.
        """
        with self._lock:
            # Additional bail if already open
            if self._is_open and self._serial is not None and self._serial.is_open:
                return
            
            self._serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
                dsrdtr=False, # do not toggle DTR on open -> prevents accidental MCU reset
                rtscts=False,
            )
            # Brief pause to let the UART settle after port open
            time.sleep(0.1)
            
            self._is_open = True

        if not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()

    def close(self) -> None:
        """
        Signal the reader thread to stop and close the serial port.
        """
        self._stop_event.set()
        try:
            # Wait for the thread to finish
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
        finally:
            with self._lock:
                if self._serial is not None:
                    try:
                        self._serial.close()
                    except Exception:
                        pass
                self._serial = None
                self._is_open = False

    def write_line(self, line: str) -> None:
        """
        Write a single line to the serial port, appending a newline if absent.

        :param line: The string to transmit.
        :raises RuntimeError: If the serial port is not open.
        """
        if not line.endswith('\n'):
            line += '\n'
        data = line.encode('utf-8')

        with self._lock:
            if self._serial is None or not self._serial.is_open:
                raise RuntimeError("Serial port is not open.")
            self._serial.write(data)

    def read_line(self, timeout = 0.5):
        """
        Block until a line is available or the timeout expires.

        :param timeout: Maximum time to wait in seconds.
        :returns: The next line from the receive queue, or None on timeout.
        """
        try:
            return self._receive_queue.get(timeout=timeout)
        except Empty:
            return None
        
    def _read_loop(self) -> None:
        """
        Background thread target: read bytes from the serial port and enqueue complete lines.
        """
        buffer = bytearray()
        while not self._stop_event.is_set():
            with self._lock:
                serial = self._serial

            if serial is None or not serial.is_open:
                time.sleep(0.05)
                continue

            try:
                data_chunk = serial.read(self.config.data_chunk_size)
            except Exception:
                time.sleep(0.05)
                continue

            if not data_chunk:
                time.sleep(0.05)
                continue

            buffer.extend(data_chunk)
            
            while True:
                newline_index = buffer.find(b'\n')
                if newline_index == -1:
                    break
                
                line_bytes = buffer[:newline_index + 1]
                buffer = buffer[newline_index + 1:]

                try:
                    line = line_bytes.decode('utf-8').rstrip('\r\n')
                    self._receive_queue.put(line)
                except UnicodeDecodeError:
                    continue
