"""
EARCalculator - computes the Eye Aspect Ratio (EAR) from MediaPipe Face Mesh
landmarks, per the GisingLang thesis (Chapter 3, "Eye Aspect Ratio and PERCLOS
Algorithm").

    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

Six points per eye:
    P1 = outer corner
    P2, P3 = upper eyelid
    P4 = inner corner
    P5, P6 = lower eyelid

This module has no camera or hardware dependency - it only consumes landmark
data, so it can be fully unit tested (see tests/test_ear_calculator.py) before
any camera or Raspberry Pi is involved. Corresponds to test case UT-01 in the
Software Testing document.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence


class LandmarkLike(Protocol):
    """Anything with normalized .x / .y attributes - matches MediaPipe's
    NormalizedLandmark, so real MediaPipe output and test fakes both work."""
    x: float
    y: float


# MediaPipe Face Mesh landmark indices for the 6-point EAR model, mapped onto
# the standard 468-point topology (P1=outer corner, P2/P3=upper eyelid,
# P4=inner corner, P5/P6=lower eyelid), per Chapter 3's landmark selection.
RIGHT_EYE_INDICES = {"P1": 33, "P2": 160, "P3": 158, "P4": 133, "P5": 153, "P6": 144}
LEFT_EYE_INDICES = {"P1": 362, "P2": 385, "P3": 387, "P4": 263, "P5": 373, "P6": 380}

# Thesis Chapter 3: "if the average EAR falls below the threshold of 0.20,
# the frame is classified as an eye-closed frame."
EAR_CLOSED_THRESHOLD = 0.20


@dataclass
class EARResult:
    left_ear: float
    right_ear: float
    avg_ear: float
    eye_closed: bool


class EARCalculator:
    """Computes per-frame Eye Aspect Ratio from MediaPipe Face Mesh landmarks."""

    def __init__(self, closed_threshold: float = EAR_CLOSED_THRESHOLD):
        self.closed_threshold = closed_threshold

    @staticmethod
    def _to_pixel(landmark: LandmarkLike, image_width: int, image_height: int) -> tuple[float, float]:
        # MediaPipe landmarks are normalized to [0, 1] relative to image width/
        # height separately, so we scale back to pixel space before measuring
        # distances - otherwise a non-square frame would skew the ratio.
        return landmark.x * image_width, landmark.y * image_height

    @staticmethod
    def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.dist(a, b)

    def _single_eye_ear(
        self,
        landmarks: Sequence[LandmarkLike],
        indices: dict,
        image_width: int,
        image_height: int,
    ) -> float:
        p1 = self._to_pixel(landmarks[indices["P1"]], image_width, image_height)
        p2 = self._to_pixel(landmarks[indices["P2"]], image_width, image_height)
        p3 = self._to_pixel(landmarks[indices["P3"]], image_width, image_height)
        p4 = self._to_pixel(landmarks[indices["P4"]], image_width, image_height)
        p5 = self._to_pixel(landmarks[indices["P5"]], image_width, image_height)
        p6 = self._to_pixel(landmarks[indices["P6"]], image_width, image_height)

        vertical = self._euclidean(p2, p6) + self._euclidean(p3, p5)
        horizontal = self._euclidean(p1, p4)
        if horizontal == 0:
            return 0.0
        return vertical / (2.0 * horizontal)

    def compute(
        self,
        landmarks: Sequence[LandmarkLike],
        image_width: int,
        image_height: int,
    ) -> EARResult:
        """
        landmarks: MediaPipe's face_landmarks.landmark list (468 points,
            normalized 0-1), i.e. results.multi_face_landmarks[0].landmark.
        image_width / image_height: source frame dimensions in pixels.
        """
        left = self._single_eye_ear(landmarks, LEFT_EYE_INDICES, image_width, image_height)
        right = self._single_eye_ear(landmarks, RIGHT_EYE_INDICES, image_width, image_height)
        avg = (left + right) / 2.0
        return EARResult(
            left_ear=left,
            right_ear=right,
            avg_ear=avg,
            eye_closed=avg < self.closed_threshold,
        )
