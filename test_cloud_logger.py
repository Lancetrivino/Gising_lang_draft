"""
Unit tests for CloudLogger - implements test case UT-07 from the GisingLang
Software Testing document (Table 5):

    "Verify an event payload is correctly formatted for transmission to
    Supabase. Simulated event object (timestamp, EAR, PERCLOS, head pose,
    alert type, device ID). Payload matches the defined Supabase schema
    with no missing or malformed fields."

No live Supabase project or network access is used here: build_event_payload
is tested directly (pure function), and the HTTPS call is tested by mocking
urllib.request.urlopen so nothing actually leaves your machine. Once you
plug in your real Supabase project URL and service_role key, IT-05
(Table 6) is the real end-to-end check that a row actually lands in your
table - that one does need your live project.

Run with:  pytest test_cloud_logger.py -v
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from cloud_logger import CloudLogger, CloudLoggerConfig, build_event_payload
from drowsiness_classifier import AlertLevel, ClassificationResult
from ear_calculator import EARResult
from head_pose_estimator import HeadPoseResult
from perclos_calculator import PERCLOSResult

EXPECTED_KEYS = {
    "device_id",
    "event_timestamp",
    "alert_level",
    "ear_value",
    "perclos_value",
    "pitch",
    "yaw",
    "roll",
}


def _sample_results():
    ear_result = EARResult(left_ear=0.24, right_ear=0.22, avg_ear=0.23, eye_closed=False)
    perclos_result = PERCLOSResult(perclos=0.25, breached=True, sample_count=20)
    head_pose_result = HeadPoseResult(pitch=18.4, yaw=-2.1, roll=3.6, breached=True)
    classification = ClassificationResult(
        alert_level=AlertLevel.COMBINED, buzzer_duration_ms=1000, motor_duration_ms=1000, sms_required=True
    )
    return ear_result, perclos_result, head_pose_result, classification


def test_payload_matches_schema_matches_ut07():
    ear_result, perclos_result, head_pose_result, classification = _sample_results()
    timestamp = datetime(2026, 7, 13, 8, 30, 0, tzinfo=timezone.utc)

    payload = build_event_payload(
        device_id="pi-zero-001",
        ear_result=ear_result,
        perclos_result=perclos_result,
        head_pose_result=head_pose_result,
        classification=classification,
        timestamp=timestamp,
    )

    # No missing or extra fields - matches the drowsiness_events schema exactly.
    assert set(payload.keys()) == EXPECTED_KEYS

    # No malformed fields - every value is JSON-serializable.
    serialized = json.dumps(payload)
    assert isinstance(serialized, str)

    assert payload["device_id"] == "pi-zero-001"
    assert payload["event_timestamp"] == "2026-07-13T08:30:00+00:00"
    assert payload["alert_level"] == "COMBINED"
    assert payload["ear_value"] == 0.23
    assert payload["perclos_value"] == 0.25
    assert payload["pitch"] == 18.4
    assert payload["yaw"] == -2.1
    assert payload["roll"] == 3.6


def test_payload_defaults_timestamp_to_now_utc():
    ear_result, perclos_result, head_pose_result, classification = _sample_results()

    payload = build_event_payload(
        device_id="pi-zero-001",
        ear_result=ear_result,
        perclos_result=perclos_result,
        head_pose_result=head_pose_result,
        classification=classification,
    )

    # Should parse back as a valid ISO timestamp without raising.
    datetime.fromisoformat(payload["event_timestamp"])


@patch("cloud_logger.urllib.request.urlopen")
def test_post_sends_correct_request_without_hitting_network(mock_urlopen):
    mock_response = MagicMock()
    mock_response.status = 201
    mock_urlopen.return_value.__enter__.return_value = mock_response

    config = CloudLoggerConfig(supabase_url="https://example.supabase.co", api_key="test-service-role-key")
    logger = CloudLogger(config)
    payload = {"device_id": "pi-zero-001", "alert_level": "LAYER1"}

    logger.log_event(payload, blocking=True)

    assert mock_urlopen.called
    sent_request = mock_urlopen.call_args[0][0]
    assert sent_request.full_url == "https://example.supabase.co/rest/v1/drowsiness_events"
    assert sent_request.get_method() == "POST"
    assert sent_request.get_header("Apikey") == "test-service-role-key"
    assert sent_request.get_header("Authorization") == "Bearer test-service-role-key"
    assert json.loads(sent_request.data.decode("utf-8")) == payload
    assert logger.last_status == 201
    assert logger.last_error is None


@patch("cloud_logger.urllib.request.urlopen")
def test_network_failure_does_not_raise(mock_urlopen):
    import urllib.error

    mock_urlopen.side_effect = urllib.error.URLError("no network")

    config = CloudLoggerConfig(supabase_url="https://example.supabase.co", api_key="test-key")
    logger = CloudLogger(config)

    # Should not raise, even though the "network" is down.
    logger.log_event({"device_id": "pi-zero-001"}, blocking=True)

    assert logger.last_error is not None
    assert logger.last_status is None
