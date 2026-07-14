"""
Integration test for DetectionController - implements test case IT-08 from
the GisingLang Software Testing document (Table 6):

    "Full Pipeline (Camera -> Classifier -> Alert -> Cloud -> SMS): Verify
    an end-to-end simulated drowsiness event produces the correct alert, log
    entry, and notification in sequence."

(Cloud/SMS delivery are covered separately by test_cloud_logger.py and
test_cloud_notifier.py - this test covers the camera -> classifier -> alert
portion of the chain, since that's what DetectionController itself owns.)

Builds one synthetic landmark set representing BOTH closed eyes AND a
20-degree forward head tilt at the same time - i.e. a COMBINED-alert frame -
using the same round-trip projection technique as test_head_pose_estimator.py.
No webcam, no MediaPipe, no Raspberry Pi required.

Run with:  pytest test_detection_pipeline.py -v
"""

import math
from types import SimpleNamespace

import cv2
import numpy as np

from alert_manager import BUZZER_PIN, MOTOR_PIN, AlertManager
from detection_pipeline import DetectionController
from drowsiness_classifier import AlertLevel, DrowsinessClassifier
from ear_calculator import EARCalculator, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from gpio_backend import FakeGPIOBackend
from head_pose_estimator import HeadPoseEstimator, LANDMARK_INDICES, MODEL_POINTS
from perclos_calculator import PERCLOSCalculator

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NUM_LANDMARKS = 468
TRANSLATION = np.array([[0.0], [0.0], [600.0]])


def _camera_matrix() -> np.ndarray:
    focal_length = IMAGE_WIDTH
    center = (IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0)
    return np.array([[focal_length, 0.0, center[0]], [0.0, focal_length, center[1]], [0.0, 0.0, 1.0]], dtype=np.float64)


def _make_combined_breach_landmarks(pitch_deg: float, eyes_closed: bool) -> list:
    """
    Builds a synthetic 468-slot landmark list representing a driver tilted
    forward by pitch_deg with eyes closed (or open, if eyes_closed=False).

    The 6 head-pose landmarks (including the shared eye-corner indices 33
    and 263) are placed via a real cv2.projectPoints round-trip, exactly
    like test_head_pose_estimator.py, so HeadPoseEstimator recovers the
    correct pitch. The remaining EAR-only eyelid landmarks are then placed
    as small vertical offsets to represent open/closed eyes - EAR is driven
    almost entirely by that vertical gap, so this doesn't fight the head
    pose projection.
    """
    rotation_vector = np.array([[math.radians(pitch_deg)], [0.0], [0.0]])
    camera_matrix = _camera_matrix()
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_points, _ = cv2.projectPoints(MODEL_POINTS, rotation_vector, TRANSLATION, camera_matrix, dist_coeffs)
    image_points = image_points.reshape(-1, 2)

    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(NUM_LANDMARKS)]
    ordered_names = ("nose_tip", "chin", "left_eye_outer", "right_eye_outer", "left_mouth", "right_mouth")
    for name, (px, py) in zip(ordered_names, image_points):
        idx = LANDMARK_INDICES[name]
        landmarks[idx] = SimpleNamespace(x=px / IMAGE_WIDTH, y=py / IMAGE_HEIGHT)

    # left_eye_outer (362, EAR's P1) and right_eye_outer (133, EAR's P4)
    # aren't part of the head-pose 6-point set, so they're free to place
    # independently near the projected eye-corner points.
    right_p1 = landmarks[RIGHT_EYE_INDICES["P1"]]  # = landmarks[33], already set above
    left_p4 = landmarks[LEFT_EYE_INDICES["P4"]]  # = landmarks[263], already set above
    landmarks[RIGHT_EYE_INDICES["P4"]] = SimpleNamespace(x=right_p1.x + 0.05, y=right_p1.y)
    landmarks[LEFT_EYE_INDICES["P1"]] = SimpleNamespace(x=left_p4.x - 0.05, y=left_p4.y)

    half_gap = 0.018 if not eyes_closed else 0.003  # matches test_ear_calculator.py's open/closed values

    def set_eyelids(indices: dict, corner_x_a: float, corner_x_b: float, corner_y: float) -> None:
        mid_x_upper = (corner_x_a + corner_x_b) / 2.0 - 0.01
        mid_x_lower = (corner_x_a + corner_x_b) / 2.0 + 0.01
        landmarks[indices["P2"]] = SimpleNamespace(x=mid_x_upper, y=corner_y - half_gap)
        landmarks[indices["P3"]] = SimpleNamespace(x=mid_x_lower, y=corner_y - half_gap)
        landmarks[indices["P5"]] = SimpleNamespace(x=mid_x_lower, y=corner_y + half_gap)
        landmarks[indices["P6"]] = SimpleNamespace(x=mid_x_upper, y=corner_y + half_gap)

    set_eyelids(RIGHT_EYE_INDICES, landmarks[RIGHT_EYE_INDICES["P1"]].x, landmarks[RIGHT_EYE_INDICES["P4"]].x, right_p1.y)
    set_eyelids(LEFT_EYE_INDICES, landmarks[LEFT_EYE_INDICES["P1"]].x, landmarks[LEFT_EYE_INDICES["P4"]].x, left_p4.y)

    return landmarks


def _make_controller():
    gpio = FakeGPIOBackend()
    controller = DetectionController(
        ear_calculator=EARCalculator(),
        perclos_calculator=PERCLOSCalculator(),
        head_pose_estimator=HeadPoseEstimator(),
        classifier=DrowsinessClassifier(),
        alert_manager=AlertManager(gpio),
    )
    return controller, gpio


def test_combined_breach_fires_both_actuators_matches_it08():
    controller, gpio = _make_controller()

    # Feed enough closed-eye, tilted-head frames to push PERCLOS over 20%
    # within the 60s rolling window (a single frame isn't enough - PERCLOS
    # needs a history), then confirm the final frame produces COMBINED.
    landmarks = _make_combined_breach_landmarks(pitch_deg=20.0, eyes_closed=True)
    result = None
    for i in range(10):
        result = controller.process_frame(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=i * 0.5)

    assert result.ear_result.eye_closed is True
    assert result.perclos_result.breached is True
    assert result.head_pose_result.breached is True
    assert result.classification.alert_level is AlertLevel.COMBINED

    # AlertManager actually pulsed both GPIO pins.
    import time
    time.sleep(1.2)  # let the daemon threads finish their 1000ms pulse
    assert (BUZZER_PIN, True) in gpio.history
    assert (MOTOR_PIN, True) in gpio.history


def test_alert_face_produces_no_alert():
    controller, gpio = _make_controller()

    landmarks = _make_combined_breach_landmarks(pitch_deg=0.0, eyes_closed=False)
    result = controller.process_frame(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=0.0)

    assert result.ear_result.eye_closed is False
    assert result.head_pose_result.breached is False
    assert result.classification.alert_level is AlertLevel.NONE
    assert gpio.history == []
