"""
AlertManager - drives the buzzer (GPIO18) and vibration motor (GPIO24) based
on the DrowsinessClassifier's output, per the GisingLang thesis (Chapter 3).
"""

from __future__ import annotations

import threading
import time

from drowsiness_classifier import AlertLevel, ClassificationResult
from gpio_backend import GPIOBackend

BUZZER_PIN = 18
MOTOR_PIN = 24


class AlertManager:
    def __init__(self, gpio: GPIOBackend, buzzer_pin: int = BUZZER_PIN, motor_pin: int = MOTOR_PIN):
        self.gpio = gpio
        self.buzzer_pin = buzzer_pin
        self.motor_pin = motor_pin
        self._threads: list[threading.Thread] = []
        self.gpio.setup_output(self.buzzer_pin)
        self.gpio.setup_output(self.motor_pin)

    def _pulse(self, pin: int, duration_ms: int) -> None:
        self.gpio.write(pin, True)
        time.sleep(duration_ms / 1000.0)
        self.gpio.write(pin, False)

    def trigger(self, classification: ClassificationResult, blocking: bool = False) -> None:
        if classification.alert_level is AlertLevel.NONE:
            return

        threads: list[threading.Thread] = []
        if classification.buzzer_duration_ms > 0:
            threads.append(threading.Thread(target=self._pulse, args=(self.buzzer_pin, classification.buzzer_duration_ms), daemon=True))
        if classification.motor_duration_ms > 0:
            threads.append(threading.Thread(target=self._pulse, args=(self.motor_pin, classification.motor_duration_ms), daemon=True))

        for t in threads:
            t.start()
        self._threads.extend(threads)

        if blocking:
            for t in threads:
                t.join()

    def shutdown(self) -> None:
        self.gpio.write(self.buzzer_pin, False)
        self.gpio.write(self.motor_pin, False)
        self.gpio.cleanup()
