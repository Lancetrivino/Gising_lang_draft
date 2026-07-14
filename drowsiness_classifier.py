"""
DrowsinessClassifier - fuses the PERCLOS breach flag (from PERCLOSCalculator)
and the head-pose breach flag (from HeadPoseEstimator) into one of four
alert states, per the GisingLang thesis (Chapter 3, "Three-Output
Notification Logic"):

    - Neither breached                        -> NONE      (no alert, no SMS)
    - PERCLOS breach only                     -> LAYER1    (buzzer, 500ms, no SMS)
    - Head-pose breach only                   -> LAYER2    (vibration motor, 800ms, no SMS)
    - Both PERCLOS and head-pose breached      -> COMBINED  (buzzer + motor, 1000ms, SMS sent)

This module has no camera or hardware dependency - it only consumes the two
boolean breach flags already computed by PERCLOSCalculator and
HeadPoseEstimator, so it's fully unit testable. Corresponds to test case
UT-06 in the Software Testing document.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    NONE = "NONE"
    LAYER1 = "LAYER1"
    LAYER2 = "LAYER2"
    COMBINED = "COMBINED"


# Thesis Chapter 3: pulse durations per alert level.
LAYER1_DURATION_MS = 500
LAYER2_DURATION_MS = 800
COMBINED_DURATION_MS = 1000


@dataclass
class ClassificationResult:
    alert_level: AlertLevel
    buzzer_duration_ms: int
    motor_duration_ms: int
    sms_required: bool


class DrowsinessClassifier:
    """Deterministic rule-based fusion of the PERCLOS and head-pose
    channels into a single alert decision."""

    def classify(self, perclos_breached: bool, head_pose_breached: bool) -> ClassificationResult:
        if perclos_breached and head_pose_breached:
            return ClassificationResult(
                alert_level=AlertLevel.COMBINED,
                buzzer_duration_ms=COMBINED_DURATION_MS,
                motor_duration_ms=COMBINED_DURATION_MS,
                sms_required=True,
            )

        if perclos_breached:
            return ClassificationResult(
                alert_level=AlertLevel.LAYER1,
                buzzer_duration_ms=LAYER1_DURATION_MS,
                motor_duration_ms=0,
                sms_required=False,
            )

        if head_pose_breached:
            return ClassificationResult(
                alert_level=AlertLevel.LAYER2,
                buzzer_duration_ms=0,
                motor_duration_ms=LAYER2_DURATION_MS,
                sms_required=False,
            )

        return ClassificationResult(
            alert_level=AlertLevel.NONE,
            buzzer_duration_ms=0,
            motor_duration_ms=0,
            sms_required=False,
        )
