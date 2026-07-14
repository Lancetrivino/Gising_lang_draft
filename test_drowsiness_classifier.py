"""
Unit tests for DrowsinessClassifier - implements test case UT-06 from the
GisingLang Software Testing document (Table 5):

    "Verify correct alert level (NONE / LAYER1 / LAYER2 / COMBINED) is
    assigned from EAR/PERCLOS and head-pose inputs. Simulated PERCLOS = 25%
    and HEAD_POSE_ALERT = True. Returns alert level COMBINED."

Also covers the other three branches of Chapter 3's Three-Output
Notification Logic table (NONE, LAYER1-only, LAYER2-only) for full coverage.

No camera, no MediaPipe, no OpenCV, no Raspberry Pi required - this module
only consumes two booleans.

Run with:  pytest test_drowsiness_classifier.py -v
"""

from drowsiness_classifier import AlertLevel, DrowsinessClassifier


def test_combined_alert_matches_ut06():
    # PERCLOS = 25% (>= 20% threshold, so breached=True) and
    # HEAD_POSE_ALERT = True -> both channels breached -> COMBINED.
    classifier = DrowsinessClassifier()

    result = classifier.classify(perclos_breached=True, head_pose_breached=True)

    assert result.alert_level is AlertLevel.COMBINED
    assert result.sms_required is True
    assert result.buzzer_duration_ms == 1000
    assert result.motor_duration_ms == 1000


def test_perclos_only_triggers_layer1():
    classifier = DrowsinessClassifier()

    result = classifier.classify(perclos_breached=True, head_pose_breached=False)

    assert result.alert_level is AlertLevel.LAYER1
    assert result.sms_required is False
    assert result.buzzer_duration_ms == 500
    assert result.motor_duration_ms == 0


def test_head_pose_only_triggers_layer2():
    classifier = DrowsinessClassifier()

    result = classifier.classify(perclos_breached=False, head_pose_breached=True)

    assert result.alert_level is AlertLevel.LAYER2
    assert result.sms_required is False
    assert result.buzzer_duration_ms == 0
    assert result.motor_duration_ms == 800


def test_no_breach_triggers_no_alert():
    classifier = DrowsinessClassifier()

    result = classifier.classify(perclos_breached=False, head_pose_breached=False)

    assert result.alert_level is AlertLevel.NONE
    assert result.sms_required is False
    assert result.buzzer_duration_ms == 0
    assert result.motor_duration_ms == 0
