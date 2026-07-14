"""
Unit tests for EARCalculator - implements test case UT-01 from the GisingLang
Software Testing document (Table 5):

    "Verify EAR value is computed correctly from six eye landmark coordinates.
    Simulated MediaPipe landmark set representing a fully open eye and a fully
    closed eye. Returns EAR ~ 0.30 for open eye and EAR ~ 0.05 for closed eye,
    within +/-0.02 tolerance."

No camera, no MediaPipe, and no Raspberry Pi required - just synthetic
landmark coordinates, so this can run anywhere Python runs.

Run with:  pytest test_ear_calculator.py -v
"""

import math
from types import SimpleNamespace

from ear_calculator import EARCalculator, LEFT_EYE_INDICES, RIGHT_EYE_INDICES

NUM_LANDMARKS = 468


def _make_landmarks(open_eye: bool) -> list:
    """Builds a 468-slot landmark list with only the 12 EAR indices (6 per
    eye) populated with meaningful coordinates; everything else is unused by
    EARCalculator so it's left at the origin."""
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(NUM_LANDMARKS)]

    # Vertical half-gap between the eyelid landmark pairs. Larger gap = eye
    # more open. These values are chosen so avg EAR lands on 0.30 (open) and
    # 0.05 (closed) exactly, matching UT-01's expected results.
    half_gap = 0.018 if open_eye else 0.003

    def fill(indices: dict) -> None:
        landmarks[indices["P1"]] = SimpleNamespace(x=0.30, y=0.50)  # outer corner
        landmarks[indices["P4"]] = SimpleNamespace(x=0.42, y=0.50)  # inner corner
        landmarks[indices["P2"]] = SimpleNamespace(x=0.34, y=0.50 - half_gap)  # upper eyelid
        landmarks[indices["P3"]] = SimpleNamespace(x=0.38, y=0.50 - half_gap)  # upper eyelid
        landmarks[indices["P5"]] = SimpleNamespace(x=0.38, y=0.50 + half_gap)  # lower eyelid
        landmarks[indices["P6"]] = SimpleNamespace(x=0.34, y=0.50 + half_gap)  # lower eyelid

    fill(LEFT_EYE_INDICES)
    fill(RIGHT_EYE_INDICES)
    return landmarks


def test_open_eye_ear_matches_ut01():
    calc = EARCalculator()
    landmarks = _make_landmarks(open_eye=True)

    result = calc.compute(landmarks, image_width=100, image_height=100)

    assert math.isclose(result.avg_ear, 0.30, abs_tol=0.02), result.avg_ear
    assert math.isclose(result.left_ear, 0.30, abs_tol=0.02), result.left_ear
    assert math.isclose(result.right_ear, 0.30, abs_tol=0.02), result.right_ear
    assert result.eye_closed is False


def test_closed_eye_ear_matches_ut01():
    calc = EARCalculator()
    landmarks = _make_landmarks(open_eye=False)

    result = calc.compute(landmarks, image_width=100, image_height=100)

    assert math.isclose(result.avg_ear, 0.05, abs_tol=0.02), result.avg_ear
    assert result.eye_closed is True


def test_zero_width_eye_does_not_crash():
    # Degenerate case: P1 and P4 coincide (horizontal distance = 0). Should
    # return 0.0 instead of raising a ZeroDivisionError.
    calc = EARCalculator()
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(NUM_LANDMARKS)]
    for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
        for key in indices.values():
            landmarks[key] = SimpleNamespace(x=0.30, y=0.50)

    result = calc.compute(landmarks, image_width=100, image_height=100)

    assert result.avg_ear == 0.0
    assert result.eye_closed is True
