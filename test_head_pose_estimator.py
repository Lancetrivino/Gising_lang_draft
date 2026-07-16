"""
Unit tests for HeadPoseEstimator - implements test cases UT-04 and UT-05 from
the GisingLang Software Testing document (Table 5), plus coverage for:

  - calibrate(): the baseline-offset feature added after real-webcam testing
    showed a systematic bias (a generic 3D face model + uncalibrated focal
    length doesn't read exactly 0 degrees for every real face/camera).
  - yaw_suppression_deg + min_breach_duration_seconds: added after real-
    webcam testing showed LAYER2 firing when the driver briefly turned to
    check a side mirror (yaw-dominant motion, not drowsy head droop). See
    the module docstring in head_pose_estimator.py for the full rationale.

Requires opencv-python and numpy (pip install opencv-python numpy) since
HeadPoseEstimator uses cv2.solvePnP / cv2.Rodrigues.

Run with:  pytest test_head_pose_estimator.py -v
"""

import math
from types import SimpleNamespace

import cv2
import numpy as np

from head_pose_estimator import HeadPoseEstimator, LANDMARK_INDICES, MODEL_POINTS

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NUM_LANDMARKS = 468

TRANSLATION = np.array([[0.0], [0.0], [600.0]])


def _camera_matrix() -> np.ndarray:
    focal_length = IMAGE_WIDTH
    center = (IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0)
    return np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _make_landmarks_for_pose(pitch_deg: float, yaw_deg: float, roll_deg: float) -> list:
    rotation_vector = np.array(
        [[math.radians(pitch_deg)], [math.radians(yaw_deg)], [math.radians(roll_deg)]]
    )
    camera_matrix = _camera_matrix()
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_points, _ = cv2.projectPoints(
        MODEL_POINTS, rotation_vector, TRANSLATION, camera_matrix, dist_coeffs
    )
    image_points = image_points.reshape(-1, 2)

    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(NUM_LANDMARKS)]
    ordered_names = (
        "nose_tip",
        "chin",
        "left_eye_outer",
        "right_eye_outer",
        "left_mouth",
        "right_mouth",
    )
    for name, (px, py) in zip(ordered_names, image_points):
        idx = LANDMARK_INDICES[name]
        landmarks[idx] = SimpleNamespace(x=px / IMAGE_WIDTH, y=py / IMAGE_HEIGHT)

    return landmarks


def test_pitch_recovered_matches_ut04():
    estimator = HeadPoseEstimator()
    landmarks = _make_landmarks_for_pose(pitch_deg=20.0, yaw_deg=0.0, roll_deg=0.0)

    result = estimator.compute(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert math.isclose(result.pitch, 20.0, abs_tol=2.0), result.pitch
    assert math.isclose(result.yaw, 0.0, abs_tol=2.0), result.yaw
    assert math.isclose(result.roll, 0.0, abs_tol=2.0), result.roll


def test_pitch_16_degrees_breaches_matches_ut05():
    estimator = HeadPoseEstimator()
    landmarks = _make_landmarks_for_pose(pitch_deg=16.0, yaw_deg=0.0, roll_deg=0.0)

    result = estimator.compute(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert result.breached is True


def test_roll_25_degrees_breaches():
    estimator = HeadPoseEstimator()
    landmarks = _make_landmarks_for_pose(pitch_deg=0.0, yaw_deg=0.0, roll_deg=25.0)

    result = estimator.compute(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert math.isclose(result.roll, 25.0, abs_tol=2.0), result.roll
    assert result.breached is True


def test_small_angles_do_not_breach():
    estimator = HeadPoseEstimator()
    landmarks = _make_landmarks_for_pose(pitch_deg=5.0, yaw_deg=5.0, roll_deg=5.0)

    result = estimator.compute(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert result.breached is False


def test_default_offsets_are_zero_backward_compatible():
    estimator = HeadPoseEstimator()
    assert estimator.pitch_offset_deg == 0.0
    assert estimator.roll_offset_deg == 0.0


def test_calibrate_cancels_a_systematic_bias():
    # Simulate a camera/face combo with a constant +8 degree pitch bias:
    # every "neutral" frame reads as pitch=8 instead of pitch=0, exactly
    # the kind of real-world bias that caused the false LAYER2 alerts.
    estimator = HeadPoseEstimator()
    neutral_landmarks = _make_landmarks_for_pose(pitch_deg=8.0, yaw_deg=0.0, roll_deg=0.0)

    uncalibrated = estimator.compute(neutral_landmarks, IMAGE_WIDTH, IMAGE_HEIGHT)
    assert math.isclose(uncalibrated.pitch, 8.0, abs_tol=1.0)
    assert uncalibrated.breached is False  # 8 deg alone doesn't cross 15 deg, but the bias is still there

    # Calibrate against that same neutral frame's raw reading.
    estimator.calibrate(pitch_offset_deg=uncalibrated.pitch, roll_offset_deg=uncalibrated.roll)

    recalibrated = estimator.compute(neutral_landmarks, IMAGE_WIDTH, IMAGE_HEIGHT)
    assert math.isclose(recalibrated.pitch, 0.0, abs_tol=0.5), recalibrated.pitch
    assert recalibrated.breached is False

    # And a genuine 20-degree tilt on top of that same bias should now
    # correctly read as ~20 degrees (bias cancelled out), not ~28.
    tilted_landmarks = _make_landmarks_for_pose(pitch_deg=28.0, yaw_deg=0.0, roll_deg=0.0)  # 8 (bias) + 20 (real tilt)
    tilted_result = estimator.compute(tilted_landmarks, IMAGE_WIDTH, IMAGE_HEIGHT)
    assert math.isclose(tilted_result.pitch, 20.0, abs_tol=1.0), tilted_result.pitch
    assert tilted_result.breached is True


def test_default_yaw_suppression_and_duration_are_set():
    estimator = HeadPoseEstimator()
    assert estimator.yaw_suppression_deg == 30.0
    assert estimator.min_breach_duration_seconds == 1.2


def test_side_mirror_glance_does_not_breach_even_with_pitch_drift():
    # Simulate the reported bug: turning to check a side mirror is yaw-
    # dominant (35 deg), but Euler-decomposition coupling also leaks into
    # the reported pitch (18 deg here, which alone would cross the 15 deg
    # threshold). The yaw gate should suppress the breach regardless,
    # since |yaw|=35 is past yaw_suppression_deg=30 - a real mirror check,
    # not a drowsy droop. No timestamp is passed, so this also confirms the
    # yaw gate applies even in the instantaneous (no-duration-gate) path.
    estimator = HeadPoseEstimator()
    landmarks = _make_landmarks_for_pose(pitch_deg=18.0, yaw_deg=35.0, roll_deg=0.0)

    result = estimator.compute(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert result.breached is False, (result.pitch, result.yaw, result.roll)


def test_forward_pitch_breach_still_fires_within_yaw_range():
    # Sanity check the yaw gate isn't over-broad: a genuine forward droop
    # with yaw well within range must still breach instantaneously.
    estimator = HeadPoseEstimator()
    landmarks = _make_landmarks_for_pose(pitch_deg=20.0, yaw_deg=5.0, roll_deg=0.0)

    result = estimator.compute(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT)

    assert result.breached is True


def test_duration_gate_requires_sustained_breach_before_firing():
    # A pitch breach that has only just started (elapsed < 1.2s) must not
    # yet report breached=True - this is what stops a single bad/borderline
    # frame from instantly triggering LAYER2.
    estimator = HeadPoseEstimator()
    landmarks = _make_landmarks_for_pose(pitch_deg=20.0, yaw_deg=0.0, roll_deg=0.0)

    first = estimator.compute(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=0.0)
    assert first.breached is False, "single frame should not breach instantly when timed"

    still_early = estimator.compute(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=0.5)
    assert still_early.breached is False, "0.5s of sustained tilt is still under the 1.2s gate"

    now_sustained = estimator.compute(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=1.3)
    assert now_sustained.breached is True, "1.3s of continuous tilt should finally breach"


def test_duration_gate_resets_on_brief_correction():
    # A brief return to normal head pose mid-tilt should reset the timer,
    # so a driver who over-corrects and then droops again doesn't get
    # credit for the earlier, interrupted tilt.
    estimator = HeadPoseEstimator()
    tilted = _make_landmarks_for_pose(pitch_deg=20.0, yaw_deg=0.0, roll_deg=0.0)
    neutral = _make_landmarks_for_pose(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0)

    estimator.compute(tilted, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=0.0)
    mid = estimator.compute(tilted, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=1.0)
    assert mid.breached is False  # only 1.0s in, under the 1.2s gate

    # Driver straightens up briefly, resetting the timer.
    estimator.compute(neutral, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=1.1)

    # Tilts again - even though 1.3s have passed since the *original* tilt
    # started, the timer restarted at the correction, so this should not
    # yet breach.
    resumed = estimator.compute(tilted, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=1.3)
    assert resumed.breached is False, "timer should have reset at the brief correction"

    finally_sustained = estimator.compute(tilted, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=2.5)
    assert finally_sustained.breached is True


def test_mirror_glance_with_timestamp_never_accumulates_toward_breach():
    # End-to-end style check: repeatedly "glancing" at a side mirror for
    # a couple of seconds (yaw-dominant, brief) should never breach, even
    # when timestamps are supplied and the glance is held for longer than
    # the duration gate would normally require.
    estimator = HeadPoseEstimator()
    glance = _make_landmarks_for_pose(pitch_deg=10.0, yaw_deg=35.0, roll_deg=0.0)

    for t in (0.0, 0.5, 1.0, 1.5, 2.0):
        result = estimator.compute(glance, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=t)
        assert result.breached is False, f"false breach at t={t}"