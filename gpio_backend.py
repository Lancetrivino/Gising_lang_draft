"""
GPIOBackend - hardware abstraction so AlertManager can be unit tested on a
laptop (no Raspberry Pi, no RPi.GPIO) and then run unmodified on the actual
Pi.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class GPIOBackend(ABC):
    @abstractmethod
    def setup_output(self, pin: int) -> None: ...

    @abstractmethod
    def write(self, pin: int, high: bool) -> None: ...

    @abstractmethod
    def cleanup(self) -> None: ...


class FakeGPIOBackend(GPIOBackend):
    def __init__(self):
        self.pin_states: dict[int, bool] = {}
        self.history: list[tuple[int, bool]] = []
        self.cleaned_up = False

    def setup_output(self, pin: int) -> None:
        self.pin_states[pin] = False

    def write(self, pin: int, high: bool) -> None:
        self.pin_states[pin] = high
        self.history.append((pin, high))

    def cleanup(self) -> None:
        self.cleaned_up = True


class RaspberryPiGPIOBackend(GPIOBackend):
    def __init__(self, mode: str = "BCM"):
        try:
            import RPi.GPIO as GPIO
        except ImportError as exc:
            raise RuntimeError(
                "RPi.GPIO is not available. RaspberryPiGPIOBackend only runs "
                "on an actual Raspberry Pi. Use FakeGPIOBackend for laptop "
                "development and testing."
            ) from exc

        self._gpio = GPIO
        self._gpio.setmode(GPIO.BCM if mode == "BCM" else GPIO.BOARD)

    def setup_output(self, pin: int) -> None:
        self._gpio.setup(pin, self._gpio.OUT, initial=self._gpio.LOW)

    def write(self, pin: int, high: bool) -> None:
        self._gpio.output(pin, self._gpio.HIGH if high else self._gpio.LOW)

    def cleanup(self) -> None:
        self._gpio.cleanup()
