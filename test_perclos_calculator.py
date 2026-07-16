"""
Unit tests for PERCLOSCalculator - implements test cases UT-02 and UT-03
from the GisingLang Software Testing document (Table 5), plus coverage for
the release_seconds fast-release hysteresis added after real-testing showed
LAYER1 "sticking" for a long time after the driver reopened their eyes (see
the module docstring in perclos_calculator.py for the full rationale).

Run with:  pytest test_perclos_calculator.py -v
"""

from perclos_calculator import PERCLOSCalculator


def test_ut02_perclos_ratio_25_percent():
    calc = PERCLOSCalculator()
    calc.update(0.0, True)
    calc.update(1.0, False)
    calc.update(2.0, False)
    result = calc.update(3.0, False)

    assert result.sample_count == 4
    assert result.perclos == 0.25


def test_ut03_perclos_21_percent_breaches():
    # 5 closed out of 24 samples ~= 20.8% ("21%" in the thesis test case),
    # which crosses the default 20% breach_threshold. The window ends on a
    # closed frame so the fast-release hysteresis (which only ever applies
    # while the current frame is open) can't mask the breach here - that
    # behavior gets its own dedicated tests below.
    calc = PERCLOSCalculator()
    t = 0.0
    for _ in range(19):
        calc.update(t, False)
        t += 1.0
    result = None
    for _ in range(5):
        result = calc.update(t, True)
        t += 1.0

    assert result.sample_count == 24
    assert abs(result.perclos - (5 / 24)) < 1e-9
    assert result.breached is True


def test_below_threshold_does_not_breach():
    calc = PERCLOSCalculator()
    t = 0.0
    for _ in range(18):
        calc.update(t, False)
        t += 1.0
    result = calc.update(t, True)  # 1 closed out of 19 ~= 5%

    assert result.breached is False


def test_samples_older_than_window_are_pruned():
    calc = PERCLOSCalculator(window_seconds=2.0, release_seconds=100.0)
    calc.update(0.0, True)
    calc.update(0.5, True)
    early_result = calc.update(1.0, True)
    assert early_result.sample_count == 3
    assert early_result.perclos == 1.0

    later_result = calc.update(3.5, False)
    assert later_result.sample_count == 1
    assert later_result.perclos == 0.0


def test_reset_clears_history_for_the_next_window():
    calc = PERCLOSCalculator()
    for i in range(10):
        calc.update(float(i), True)

    calc.reset()

    result = calc.update(100.0, False)
    assert result.sample_count == 1
    assert result.perclos == 0.0


def test_default_release_seconds_is_1_5():
    calc = PERCLOSCalculator()
    assert calc.release_seconds == 1.5


def test_fast_release_clears_breach_after_sustained_open_eyes():
    calc = PERCLOSCalculator(window_seconds=10.0, breach_threshold=0.20, release_seconds=1.5)

    t = 0.0
    breached_result = None
    for _ in range(3):
        breached_result = calc.update(t, True)
        t += 1.0
    assert breached_result.breached is True

    just_opened = calc.update(t, False)
    assert just_opened.perclos >= 0.20
    assert just_opened.breached is True

    t += 1.0
    still_within_release = calc.update(t, False)
    assert still_within_release.breached is True

    t += 1.0
    released = calc.update(t, False)
    assert released.breached is False
    assert released.perclos >= 0.20


def test_release_streak_resets_on_any_closed_frame():
    calc = PERCLOSCalculator(window_seconds=10.0, breach_threshold=0.20, release_seconds=1.5)

    t = 0.0
    for _ in range(3):
        calc.update(t, True)
        t += 1.0

    t += 1.0
    calc.update(t, False)
    t += 1.0

    calc.update(t, True)
    t += 0.2
    resumed = calc.update(t, False)
    assert resumed.breached is True

    t += 1.6
    finally_released = calc.update(t, False)
    assert finally_released.breached is False


def test_no_release_needed_when_never_breached():
    calc = PERCLOSCalculator(window_seconds=10.0, breach_threshold=0.20, release_seconds=1.5)
    t = 0.0
    result = None
    for _ in range(5):
        result = calc.update(t, False)
        t += 1.0

    assert result.breached is False
    assert result.perclos == 0.0