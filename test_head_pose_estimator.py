"""
Unit tests for HeadPoseEstimator - implements test cases UT-04 and UT-05 from
the GisingLang Software Testing document (Table 5), plus coverage for the
calibrate() baseline-offset feature added after real-webcam testing showed a
systematic bias (a generic 3D face model + uncalibrated focal length doesn't
read exactly 0 degrees for every real face/camera - see the comment at the
top of head_pose_estimator.py).

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