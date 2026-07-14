"""
Unit tests for PERCLOSCalculator - implements test cases UT-02 and UT-03 from
the GisingLang Software Testing document (Table 5):

    UT-02: "Verify PERCLOS is computed correctly over a 60-second rolling
    window. Simulated EAR stream with a known ratio of closed-eye frames
    (e.g., 25% of frames below threshold). Returns PERCLOS value of 25%
    (+/-1%)."

    UT-03: "Verify drowsy classification triggers when PERCLOS exceeds 20%.
    Simulated PERCLOS value of 21%. Module returns DROWSY = True."

No camera, no MediaPipe, no Raspberry Pi required - just synthetic
timestamps and eye_closed booleans, so this can run anywhere Python runs.

Run with:  pytest test_perclos_calculator.py -v
"""

import math

from perclos_calculator import PERCLOSCalculator


def test_perclos_ratio_matches_ut02():
    calc = PERCLOSCalculator()

    # 20 samples spread over 57 seconds (fits inside the 60s window), 5 of
    # which are closed -> 5/20 = 25% closed.
    closed_flags = [True] * 5 + [False] * 15
    result = None
    for i, is_closed in enumerate(closed_flags):
        timestamp = i * 3.0  # 0, 3, 6, ..., 57
        result = calc.update(timestamp, is_closed)

    assert math.isclose(result.perclos, 0.25, abs_tol=0.01), result.perclos
    assert result.sample_count == 20


def test_drowsy_breach_at_21_percent_matches_ut03():
    calc = PERCLOSCalculator()

    # 100 samples spread over 59.4 seconds (fits inside the 60s window), 21
    # of which are closed -> 21/100 = 21% closed, above the 20% threshold.
    closed_flags = [True] * 21 + [False] * 79
    result = None
    for i, is_closed in enumerate(closed_flags):
        timestamp = i * 0.6
        result = calc.update(timestamp, is_closed)

    assert math.isclose(result.perclos, 0.21, abs_tol=0.01), result.perclos
    assert result.breached is True


def test_perclos_not_breached_below_threshold():
    calc = PERCLOSCalculator()

    # 100 samples, only 10% closed -> should NOT breach.
    closed_flags = [True] * 10 + [False] * 90
    result = None
    for i, is_closed in enumerate(closed_flags):
        result = calc.update(i * 0.6, is_closed)

    assert result.perclos < 0.20
    assert result.breached is False


def test_old_samples_are_pruned_after_60_seconds():
    calc = PERCLOSCalculator(window_seconds=60.0)

    # Fill the window with all-closed frames near t=0.
    for t in range(0, 11):  # t = 0..10 seconds
        calc.update(float(t), True)

    # Jump far into the future with a single open-eye frame. Everything
    # from the first batch should have aged out of the 60s window, leaving
    # only this one sample.
    result = calc.update(200.0, False)

    assert result.sample_count == 1
    assert result.perclos == 0.0
    assert result.breached is False


def test_empty_window_returns_zero():
    calc = PERCLOSCalculator()
    result = calc.update(0.0, False)

    assert result.perclos == 0.0
    assert result.sample_count == 1
