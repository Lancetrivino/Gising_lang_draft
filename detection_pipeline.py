"""
DetectionController's per-frame orchestration logic, per the GisingLang
thesis (Chapter 3):

    "The central class is the DetectionController, which acts as the main
    processor and coordinates camera frame capture and the full computer
    vision pipeline."

This module intentionally has NO camera, MediaPipe, or GPIO-library
dependency of its own - it just wires together the modules we've already
built and tested (EARCalculator, PERCLOSCalculator, HeadPoseEstimator,
DrowsinessClassifier, AlertManager) for a single frame's worth of landmarks.
That keeps it fully unit testable with synthetic landmarks, matching
integration test case IT-08 in the Software Testing document ("Full
Pipeline"), with no webcam required.

live_demo.py is the thin camera/display wrapper that calls process_frame()
in a loop using a real webcam and MediaPipe - that's the part you can't unit
test without a camera, so it stays as small as possible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from alert_manager import AlertManager
from drowsiness_classifier import ClassificationResult, DrowsinessClassifier
from ear_calculator import EARCalculator, EARResult
from head_pose_estimator import HeadPoseEstimator, HeadPoseResult
from perclos_calculator import PERCLOSCalculator, PERCLOSResult


class LandmarkLike(Protocol):
    x: float
    y: float


@dataclass
class FrameResult:
    ear_result: EARResult
    perclos_result: PERCLOSResult
    head_pose_result: HeadPoseResult
    classification: ClassificationResult


class DetectionController:
    """Coordinates one frame's worth of landmarks through the full
    EAR -> PERCLOS -> head pose -> classifier -> alert pipeline."""

    def __init__(
        self,
        ear_calculator: EARCalculator,
        perclos_calculator: PERCLOSCalculator,
        head_pose_estimator: HeadPoseEstimator,
        classifier: DrowsinessClassifier,
        alert_manager: AlertManager,
    ):
        self.ear_calculator = ear_calculator
        self.perclos_calculator = perclos_calculator
        self.head_pose_estimator = head_pose_estimator
        self.classifier = classifier
        self.alert_manager = alert_manager

    def process_frame(
        self,
        landmarks: Sequence[LandmarkLike],
        image_width: int,
        image_height: int,
        timestamp: float | None = None,
    ) -> FrameResult:
        """
        Runs one frame's landmarks through the full pipeline and fires the
        AlertManager if a breach is classified. Returns every intermediate
        result too, so callers (live_demo.py, tests) can inspect or display
        them without recomputing anything.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        ear_result = self.ear_calculator.compute(landmarks, image_width, image_height)
        perclos_result = self.perclos_calculator.update(timestamp, ear_result.eye_closed)
        head_pose_result = self.head_pose_estimator.compute(landmarks, image_width, image_height)
        classification = self.classifier.classify(perclos_result.breached, head_pose_result.breached)

        self.alert_manager.trigger(classification, blocking=False)

        return FrameResult(
            ear_result=ear_result,
            perclos_result=perclos_result,
            head_pose_result=head_pose_result,
            classification=classification,
        )
