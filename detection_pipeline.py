"""
DetectionController - orchestrates one full frame through the GisingLang
detection pipeline: EAR -> PERCLOS -> head pose -> classification -> alert.

Corresponds to test case IT-08 in the Software Testing document.

TIMESTAMP NOTE: `timestamp` is forwarded to both PERCLOSCalculator.update()
(required - it's how the rolling window works) and, as of the mirror-glance
fix, to HeadPoseEstimator.compute() as well. Without that second forward,
HeadPoseEstimator falls back to its instantaneous (no-duration-gate)
evaluation path, which would silently disable the new sustained-breach
requirement during real runs through live_demo.py - the duration gate would
still pass its own unit tests (which call it directly with timestamps) but
would never actually engage in the live pipeline. Pass a real, monotonically
non-decreasing timestamp (e.g. time.monotonic()) every frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from alert_manager import AlertManager
from drowsiness_classifier import ClassificationResult, DrowsinessClassifier
from ear_calculator import EARCalculator, EARResult
from head_pose_estimator import HeadPoseEstimator, HeadPoseResult
from perclos_calculator import PERCLOSCalculator, PERCLOSResult


@dataclass
class FrameResult:
    ear_result: EARResult
    perclos_result: PERCLOSResult
    head_pose_result: HeadPoseResult
    classification: ClassificationResult


class DetectionController:
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
        landmarks,
        image_width: int,
        image_height: int,
        timestamp: float = None,
    ) -> FrameResult:
        ear_result = self.ear_calculator.compute(landmarks, image_width, image_height)
        perclos_result = self.perclos_calculator.update(timestamp, ear_result.eye_closed)
        head_pose_result = self.head_pose_estimator.compute(
            landmarks, image_width, image_height, timestamp=timestamp
        )
        classification = self.classifier.classify(
            perclos_breached=perclos_result.breached,
            head_pose_breached=head_pose_result.breached,
        )
        self.alert_manager.trigger(classification, blocking=False)

        return FrameResult(
            ear_result=ear_result,
            perclos_result=perclos_result,
            head_pose_result=head_pose_result,
            classification=classification,
        )