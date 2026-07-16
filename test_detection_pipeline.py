"""
Integration test for DetectionController - implements test case IT-08 from
the GisingLang Software Testing document, plus two new regression tests for
the mirror-glance (LAYER2) and sticky-LAYER1 fixes at the full-pipeline
level (not just the unit level), confirming the timestamp is correctly
threaded through process_frame() into both PERCLOSCalculator and the new
HeadPoseEstimator duration gate.

NOTE ON SHARED LANDMARK INDICES: real MediaPipe topology has landmark 33
serve double duty as the right eye's outer corner (EARCalculator's
RIGHT_EYE P1) and as HeadPoseEstimator's "left_eye_outer" reference point;
landmark 263 is EARCalculator's LEFT_EYE P4 and HeadPoseEstimator's
"right_eye_outer" - the same physical point on a real face, correctly
shared. In these synthetic fixtures that means head pose must be applied
*before* the eye-openness landmarks are built, and the eye fixture must
anchor off wherever head pose actually placed landmarks 33/263, rather than
an independent fixed position - otherwise the two setup steps silently
clobber each other's landmarks and produce nonsense EAR/pose readings.

NOTE ON FakeGPIOBackend.history: this project's FakeGPIOBackend records
history as 2-tuples (pin, high) - not (action, pin, high). Assertions below
index accordingly (evt[0]=pin, evt[1]=high).

Run with:  pytest test_detection_pipeline.py -v
"""

import math
from types import SimpleNamespace

import cv2
import numpy as np

from alert_manager import AlertManager
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

EYE_CORNER_GAP = 0.12  # normalized horizontal separation between P1/P4
OPEN_VERTICAL_GAP = 0.05
CLOSED_VERTICAL_GAP = 0.003


def _camera_matrix():
    focal_length = IMAGE_WIDTH
    center = (IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0)
    return np.array(
        [[focal_length, 0.0, center[0]], [0.0, focal_length, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _base_landmarks():
    return [SimpleNamespace(x=0.5, y=0.5) for _ in range(NUM_LANDMARKS)]


def _set_head_pose(landmarks, pitch_deg, yaw_deg, roll_deg):
    rotation_vector = np.array(
        [[math.radians(pitch_deg)], [math.radians(yaw_deg)], [math.radians(roll_deg)]]
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        MODEL_POINTS, rotation_vector, TRANSLATION, _camera_matrix(), dist_coeffs
    )
    image_points = image_points.reshape(-1, 2)
    ordered_names = ("nose_tip", "chin", "left_eye_outer", "right_eye_outer", "left_mouth", "right_mouth")
    for name, (px, py) in zip(ordered_names, image_points):
        idx = LANDMARK_INDICES[name]
        landmarks[idx] = SimpleNamespace(x=px / IMAGE_WIDTH, y=py / IMAGE_HEIGHT)


def _set_eyes(landmarks, closed: bool):
    """Must be called AFTER _set_head_pose(). Anchors the eye shape off
    whichever pixel position head pose already assigned to the two shared
    corner landmarks (33 = right eye P1, 263 = left eye P4), and only
    writes to the remaining EAR landmarks, which head pose never touches."""
    vertical_gap = CLOSED_VERTICAL_GAP if closed else OPEN_VERTICAL_GAP

    p1 = landmarks[RIGHT_EYE_INDICES["P1"]]
    p4x, p4y = p1.x + EYE_CORNER_GAP, p1.y
    landmarks[RIGHT_EYE_INDICES["P4"]] = SimpleNamespace(x=p4x, y=p4y)
    mid_x = (p1.x + p4x) / 2.0
    landmarks[RIGHT_EYE_INDICES["P2"]] = SimpleNamespace(x=mid_x - 0.02, y=p1.y - vertical_gap)
    landmarks[RIGHT_EYE_INDICES["P3"]] = SimpleNamespace(x=mid_x + 0.02, y=p1.y - vertical_gap)
    landmarks[RIGHT_EYE_INDICES["P5"]] = SimpleNamespace(x=mid_x + 0.02, y=p1.y + vertical_gap)
    landmarks[RIGHT_EYE_INDICES["P6"]] = SimpleNamespace(x=mid_x - 0.02, y=p1.y + vertical_gap)

    p4 = landmarks[LEFT_EYE_INDICES["P4"]]
    p1x, p1y = p4.x - EYE_CORNER_GAP, p4.y
    landmarks[LEFT_EYE_INDICES["P1"]] = SimpleNamespace(x=p1x, y=p1y)
    mid_x = (p1x + p4.x) / 2.0
    landmarks[LEFT_EYE_INDICES["P2"]] = SimpleNamespace(x=mid_x - 0.02, y=p4.y - vertical_gap)
    landmarks[LEFT_EYE_INDICES["P3"]] = SimpleNamespace(x=mid_x + 0.02, y=p4.y - vertical_gap)
    landmarks[LEFT_EYE_INDICES["P5"]] = SimpleNamespace(x=mid_x + 0.02, y=p4.y + vertical_gap)
    landmarks[LEFT_EYE_INDICES["P6"]] = SimpleNamespace(x=mid_x - 0.02, y=p4.y + vertical_gap)


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


def test_it08_combined_breach_fires_both_actuators():
    controller, gpio = _make_controller()

    landmarks = _base_landmarks()
    _set_head_pose(landmarks, pitch_deg=20.0, yaw_deg=0.0, roll_deg=0.0)
    _set_eyes(landmarks, closed=True)

    result = None
    for i in range(30):  # ~1.5s at 20 samples/sec - past both duration gates
        t = i * 0.05
        result = controller.process_frame(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=t)

    import time

    time.sleep(1.2)  # let the daemon alert threads finish their pulses

    assert result.classification.alert_level is AlertLevel.COMBINED, result.classification
    assert gpio.pin_states[18] is False  # pulse already completed and released
    assert any(evt[0] == 18 and evt[1] is True for evt in gpio.history)
    assert any(evt[0] == 24 and evt[1] is True for evt in gpio.history)

    controller.alert_manager.shutdown()


def test_alert_face_returns_to_none():
    controller, gpio = _make_controller()

    landmarks = _base_landmarks()
    _set_head_pose(landmarks, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0)
    _set_eyes(landmarks, closed=False)

    result = controller.process_frame(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=0.0)

    assert result.classification.alert_level is AlertLevel.NONE, result.classification

    controller.alert_manager.shutdown()


def test_mirror_glance_never_reaches_layer2_through_full_pipeline():
    controller, gpio = _make_controller()

    landmarks = _base_landmarks()
    _set_head_pose(landmarks, pitch_deg=12.0, yaw_deg=35.0, roll_deg=0.0)
    _set_eyes(landmarks, closed=False)

    for i in range(40):  # ~2s of continuous "mirror glance"
        t = i * 0.05
        result = controller.process_frame(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=t)
        assert result.classification.alert_level is AlertLevel.NONE, (
            f"false LAYER2 at frame {i}: {result.classification} / pose={result.head_pose_result}"
        )

    controller.alert_manager.shutdown()


def test_layer1_clears_promptly_after_reopening_eyes_through_full_pipeline():
    controller, gpio = _make_controller()

    landmarks = _base_landmarks()
    _set_head_pose(landmarks, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0)

    _set_eyes(landmarks, closed=True)
    t = 0.0
    result = None
    for i in range(20):
        result = controller.process_frame(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=t)
        t += 0.1
    assert result.classification.alert_level is AlertLevel.LAYER1, result.classification

    _set_eyes(landmarks, closed=False)
    for i in range(40):
        result = controller.process_frame(landmarks, IMAGE_WIDTH, IMAGE_HEIGHT, timestamp=t)
        t += 0.1

    assert result.classification.alert_level is AlertLevel.NONE, (
        "LAYER1 should have cleared once eyes were sustained-open, "
        f"but still reports {result.classification.alert_level}"
    )

    controller.alert_manager.shutdown()