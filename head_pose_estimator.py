"""
HeadPoseEstimator - estimates head pitch/yaw/roll from facial landmarks
using an OpenCV solvePnP solver, per the GisingLang thesis (Chapter 3,
"Head Pose Estimation Algorithm").

    "The head pose estimator applies an OpenCV solvePnP solver to six
    reference facial landmarks - the nose tip (landmark 1), chin (landmark
    152), left eye outer corner (landmark 33), right eye outer corner
    (landmark 263), left mouth corner (landmark 61), and right mouth corner
    (landmark 291) - matched against a generic 3D face model with known
    metric coordinates. The PnP solver returns a rotation vector, which is
    converted to a rotation matrix via cv2.Rodrigues() and decomposed into
    Euler angles (pitch, yaw, roll) in degrees. A head alert is issued when
    the absolute pitch angle exceeds 15 degrees ... or when the absolute
    roll angle exceeds 20 degrees ..."

Unlike EARCalculator and PERCLOSCalculator, this module depends on OpenCV
and NumPy (pip install opencv-python numpy) since it uses cv2.solvePnP and
cv2.Rodrigues. It still has no camera dependency of its own - it consumes
landmark coordinates, so it's unit tested with a synthetic round-trip: a
known rotation is projected into fake 2D landmarks, then fed back in to
confirm the estimator recovers that same rotation (see
tests/test_head_pose_estimator.py). Corresponds to test cases UT-04 and
UT-05 in the Software Testing document.

CALIBRATION NOTE: A generic 3D face model plus an uncalibrated camera focal
length (see _camera_matrix below) will not read exactly 0 degrees for every
real face looking straight at the camera - the bias varies by webcam and
face shape, and was observed in practice during live_demo.py testing.
calibrate() lets a caller (typically live_demo.py's startup calibration
phase) record a real "neutral" baseline and subtract it from every reading
before the breach thresholds are applied, which corrects for this without
touching the underlying PnP math or its unit tests (offsets default to 0.0,
so existing synthetic round-trip tests are unaffected).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import cv2
import numpy as np


class LandmarkLike(Protocol):
    """Anything with normalized .x / .y attributes - matches MediaPipe's
    NormalizedLandmark, so real MediaPipe output and test fakes both work."""
    x: float
    y: float


# MediaPipe Face Mesh landmark indices for the 6-point PnP model, exactly as
# specified in Chapter 3.
LANDMARK_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# Generic 3D face model (arbitrary metric units, nose tip at the origin).
# Order must match LANDMARK_INDICES iteration order below.
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -330.0, -65.0),  # chin
        (-225.0, 170.0, -135.0),  # left eye outer corner
        (225.0, 170.0, -135.0),  # right eye outer corner
        (-150.0, -150.0, -125.0),  # left mouth corner
        (150.0, -150.0, -125.0),  # right mouth corner
    ],
    dtype=np.float64,
)

# Thesis Chapter 3: pitch > 15 deg or roll > 20 deg triggers the head-pose
# breach that feeds the DrowsinessClassifier's Layer 2 haptic alert.
PITCH_THRESHOLD_DEG = 15.0
ROLL_THRESHOLD_DEG = 20.0


@dataclass
class HeadPoseResult:
    pitch: float
    yaw: float
    roll: float
    breached: bool


def _rotation_matrix_to_euler_degrees(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    """Standard rotation-matrix -> Euler (X, Y, Z) decomposition, in degrees.
    X = pitch, Y = yaw, Z = roll."""
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0.0

    return math.degrees(x), math.degrees(y), math.degrees(z)


class HeadPoseEstimator:
    """Estimates pitch/yaw/roll from MediaPipe Face Mesh landmarks via
    OpenCV's solvePnP, and flags a breach per the thesis's alert thresholds."""

    def __init__(
        self,
        pitch_threshold_deg: float = PITCH_THRESHOLD_DEG,
        roll_threshold_deg: float = ROLL_THRESHOLD_DEG,
        pitch_offset_deg: float = 0.0,
        roll_offset_deg: float = 0.0,
    ):
        self.pitch_threshold_deg = pitch_threshold_deg
        self.roll_threshold_deg = roll_threshold_deg
        self.pitch_offset_deg = pitch_offset_deg
        self.roll_offset_deg = roll_offset_deg

    def calibrate(self, pitch_offset_deg: float, roll_offset_deg: float) -> None:
        """
        Records a neutral baseline to subtract from every future raw
        pitch/roll reading before comparing against the breach thresholds.
        Call this once at startup with the average raw pitch/roll observed
        while the driver is known to be facing the camera normally - see
        live_demo.py's calibration phase.
        """
        self.pitch_offset_deg = pitch_offset_deg
        self.roll_offset_deg = roll_offset_deg

    @staticmethod
    def _camera_matrix(image_width: int, image_height: int) -> np.ndarray:
        # No calibration data available on a low-cost prototype camera, so
        # approximate focal length as the image width (a standard fallback
        # for solvePnP when true calibration parameters are unavailable).
        focal_length = image_width
        center = (image_width / 2.0, image_height / 2.0)
        return np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def compute(
        self,
        landmarks: Sequence[LandmarkLike],
        image_width: int,
        image_height: int,
    ) -> HeadPoseResult:
        """
        landmarks: MediaPipe's face_landmarks.landmark list (normalized 0-1),
            i.e. results.multi_face_landmarks[0].landmark.
        image_width / image_height: source frame dimensions in pixels.
        """
        image_points = np.array(
            [
                (
                    landmarks[LANDMARK_INDICES[name]].x * image_width,
                    landmarks[LANDMARK_INDICES[name]].y * image_height,
                )
                for name in (
                    "nose_tip",
                    "chin",
                    "left_eye_outer",
                    "right_eye_outer",
                    "left_mouth",
                    "right_mouth",
                )
            ],
            dtype=np.float64,
        )

        camera_matrix = self._camera_matrix(image_width, image_height)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # assume no lens distortion

        success, rotation_vector, _translation_vector = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, breached=False)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        raw_pitch, yaw, raw_roll = _rotation_matrix_to_euler_degrees(rotation_matrix)

        pitch = raw_pitch - self.pitch_offset_deg
        roll = raw_roll - self.roll_offset_deg

        breached = abs(pitch) > self.pitch_threshold_deg or abs(roll) > self.roll_threshold_deg
        return HeadPoseResult(pitch=pitch, yaw=yaw, roll=roll, breached=breached)