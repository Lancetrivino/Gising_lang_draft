"""
Unit tests for AlertManager - implements test case IT-03 from the GisingLang
Software Testing document (Table 6):

    "DrowsinessClassifier -> AlertManager: Verify classified alert level
    correctly triggers the corresponding physical alert (buzzer / haptic
    motor / both). Correct actuator(s) activate within 3 seconds of
    classification for each alert level (LAYER1, LAYER2, COMBINED)."

Uses FakeGPIOBackend (gpio_backend.py) instead of real RPi.GPIO, so this
runs on a laptop with no Raspberry Pi or wiring attached. Once you're on the
Pi with Section 2 of the build guide wired up, swap FakeGPIOBackend for
RaspberryPiGPIOBackend - AlertManager itself doesn't change.

Run with:  pytest test_alert_manager.py -v
"""

import time

from alert_manager import BUZZER_PIN, MOTOR_PIN, AlertManager
from drowsiness_classifier import AlertLevel, ClassificationResult
from gpio_backend import FakeGPIOBackend


def test_layer1_pulses_buzzer_only():
    gpio = FakeGPIOBackend()
    manager = AlertManager(gpio)
    classification = ClassificationResult(
        alert_level=AlertLevel.LAYER1, buzzer_duration_ms=500, motor_duration_ms=0, sms_required=False
    )

    manager.trigger(classification, blocking=True)

    assert (BUZZER_PIN, True) in gpio.history
    assert (BUZZER_PIN, False) in gpio.history
    assert gpio.pin_states[BUZZER_PIN] is False  # pulse finished, back to LOW
    assert all(pin != MOTOR_PIN for pin, _ in gpio.history)  # motor never touched


def test_layer2_pulses_motor_only():
    gpio = FakeGPIOBackend()
    manager = AlertManager(gpio)
    classification = ClassificationResult(
        alert_level=AlertLevel.LAYER2, buzzer_duration_ms=0, motor_duration_ms=800, sms_required=False
    )

    manager.trigger(classification, blocking=True)

    assert (MOTOR_PIN, True) in gpio.history
    assert (MOTOR_PIN, False) in gpio.history
    assert all(pin != BUZZER_PIN for pin, _ in gpio.history)  # buzzer never touched


def test_combined_pulses_both_concurrently():
    gpio = FakeGPIOBackend()
    manager = AlertManager(gpio)
    classification = ClassificationResult(
        alert_level=AlertLevel.COMBINED, buzzer_duration_ms=1000, motor_duration_ms=1000, sms_required=True
    )

    start = time.monotonic()
    manager.trigger(classification, blocking=True)
    elapsed = time.monotonic() - start

    assert (BUZZER_PIN, True) in gpio.history
    assert (MOTOR_PIN, True) in gpio.history
    assert gpio.pin_states[BUZZER_PIN] is False
    assert gpio.pin_states[MOTOR_PIN] is False
    # Both actuators pulse concurrently (max ~1.0s), not sequentially
    # (which would take ~2.0s) - and well within the 3-second IT-03 budget.
    assert elapsed < 1.5, elapsed


def test_none_triggers_nothing():
    gpio = FakeGPIOBackend()
    manager = AlertManager(gpio)
    classification = ClassificationResult(
        alert_level=AlertLevel.NONE, buzzer_duration_ms=0, motor_duration_ms=0, sms_required=False
    )

    manager.trigger(classification, blocking=True)

    assert gpio.history == []


def test_non_blocking_trigger_returns_immediately():
    gpio = FakeGPIOBackend()
    manager = AlertManager(gpio)
    classification = ClassificationResult(
        alert_level=AlertLevel.COMBINED, buzzer_duration_ms=1000, motor_duration_ms=1000, sms_required=True
    )

    start = time.monotonic()
    manager.trigger(classification, blocking=False)  # production default
    elapsed = time.monotonic() - start

    assert elapsed < 0.1, elapsed  # returns almost instantly, doesn't block

    for t in manager._threads:  # let the daemon threads finish before the test ends
        t.join()
