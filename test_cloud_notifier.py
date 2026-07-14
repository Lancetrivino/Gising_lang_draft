"""
Unit tests for CloudNotifier - implements test case UT-08 from the
GisingLang Software Testing document (Table 5):

    "Verify SMS payload is correctly constructed for a COMBINED alert.
    Simulated COMBINED alert event with two registered emergency contacts.
    Two correctly formatted Twilio API request payloads are generated, one
    per contact."

Also covers the 120-second per-contact cooldown and the "only COMBINED
triggers SMS" rule from Chapter 3.

No live Twilio account or network access required - urllib is mocked.

Run with:  pytest test_cloud_notifier.py -v
"""

import json
import urllib.parse
from unittest.mock import MagicMock, patch

from cloud_notifier import CloudNotifier, CloudNotifierConfig, EmergencyContact, build_sms_message
from drowsiness_classifier import AlertLevel, ClassificationResult

COMBINED = ClassificationResult(alert_level=AlertLevel.COMBINED, buzzer_duration_ms=1000, motor_duration_ms=1000, sms_required=True)
LAYER1 = ClassificationResult(alert_level=AlertLevel.LAYER1, buzzer_duration_ms=500, motor_duration_ms=0, sms_required=False)

CONTACT_A = EmergencyContact(contact_id="c1", user_id="u1", name="Maria Cruz", phone_number="+639171111111", relationship="Parent")
CONTACT_B = EmergencyContact(contact_id="c2", user_id="u1", name="Juan Cruz", phone_number="+639172222222", relationship="Spouse")


def _config():
    return CloudNotifierConfig(account_sid="ACtest", auth_token="test-token", from_number="+15005550006")


def test_message_is_bilingual_and_includes_required_fields():
    message = build_sms_message("Juan Dela Cruz", "COMBINED", 0.27)

    assert "Juan Dela Cruz" in message
    assert "COMBINED" in message
    assert "27%" in message
    assert "/" in message  # separates the English and Filipino halves
    assert "Babala" in message  # Filipino half present


@patch("cloud_notifier.urllib.request.urlopen")
def test_two_contacts_produce_two_twilio_requests_matches_ut08(mock_urlopen):
    mock_response = MagicMock()
    mock_response.status = 201
    mock_urlopen.return_value.__enter__.return_value = mock_response

    notifier = CloudNotifier(_config())
    sent_to = notifier.notify(
        classification=COMBINED,
        driver_name="Juan Dela Cruz",
        contacts=[CONTACT_A, CONTACT_B],
        perclos_value=0.27,
        blocking=True,
    )

    assert sorted(sent_to) == sorted([CONTACT_A.phone_number, CONTACT_B.phone_number])
    assert mock_urlopen.call_count == 2

    called_to_numbers = set()
    for call in mock_urlopen.call_args_list:
        request = call[0][0]
        assert request.full_url == "https://api.twilio.com/2010-04-01/Accounts/ACtest/Messages.json"
        assert request.get_method() == "POST"
        assert request.get_header("Authorization", "").startswith("Basic ")
        body = urllib.parse.parse_qs(request.data.decode("utf-8"))
        assert body["From"] == ["+15005550006"]
        assert "Juan Dela Cruz" in body["Body"][0]
        called_to_numbers.add(body["To"][0])

    assert called_to_numbers == {CONTACT_A.phone_number, CONTACT_B.phone_number}


@patch("cloud_notifier.urllib.request.urlopen")
def test_non_combined_alert_sends_no_sms(mock_urlopen):
    notifier = CloudNotifier(_config())

    sent_to = notifier.notify(
        classification=LAYER1,  # sms_required=False
        driver_name="Juan Dela Cruz",
        contacts=[CONTACT_A],
        perclos_value=0.22,
        blocking=True,
    )

    assert sent_to == []
    assert mock_urlopen.call_count == 0


@patch("cloud_notifier.urllib.request.urlopen")
def test_inactive_contact_is_skipped(mock_urlopen):
    mock_response = MagicMock()
    mock_response.status = 201
    mock_urlopen.return_value.__enter__.return_value = mock_response

    inactive_contact = EmergencyContact(
        contact_id="c3", user_id="u1", name="Inactive Person", phone_number="+639173333333",
        relationship="Friend", is_active=False,
    )
    notifier = CloudNotifier(_config())

    sent_to = notifier.notify(
        classification=COMBINED, driver_name="Juan Dela Cruz",
        contacts=[CONTACT_A, inactive_contact], perclos_value=0.27, blocking=True,
    )

    assert sent_to == [CONTACT_A.phone_number]
    assert mock_urlopen.call_count == 1


@patch("cloud_notifier.urllib.request.urlopen")
def test_cooldown_prevents_repeated_sms_within_120_seconds(mock_urlopen):
    mock_response = MagicMock()
    mock_response.status = 201
    mock_urlopen.return_value.__enter__.return_value = mock_response

    notifier = CloudNotifier(_config())

    first = notifier.notify(
        classification=COMBINED, driver_name="Juan Dela Cruz",
        contacts=[CONTACT_A], perclos_value=0.27, blocking=True,
    )
    # Immediately fires again, simulating a sustained drowsiness event.
    second = notifier.notify(
        classification=COMBINED, driver_name="Juan Dela Cruz",
        contacts=[CONTACT_A], perclos_value=0.29, blocking=True,
    )

    assert first == [CONTACT_A.phone_number]
    assert second == []  # suppressed by the 120s cooldown
    assert mock_urlopen.call_count == 1


@patch("cloud_notifier.urllib.request.urlopen")
def test_cooldown_expires_after_configured_window(mock_urlopen):
    mock_response = MagicMock()
    mock_response.status = 201
    mock_urlopen.return_value.__enter__.return_value = mock_response

    # A very short cooldown so the test doesn't need to sleep 120 real seconds.
    config = CloudNotifierConfig(account_sid="ACtest", auth_token="test-token", from_number="+15005550006", cooldown_seconds=0.05)
    notifier = CloudNotifier(config)

    first = notifier.notify(
        classification=COMBINED, driver_name="Juan Dela Cruz",
        contacts=[CONTACT_A], perclos_value=0.27, blocking=True,
    )

    import time
    time.sleep(0.1)  # let the short cooldown window pass

    second = notifier.notify(
        classification=COMBINED, driver_name="Juan Dela Cruz",
        contacts=[CONTACT_A], perclos_value=0.27, blocking=True,
    )

    assert first == [CONTACT_A.phone_number]
    assert second == [CONTACT_A.phone_number]
    assert mock_urlopen.call_count == 2
