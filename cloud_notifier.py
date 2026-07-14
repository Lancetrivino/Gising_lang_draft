"""
CloudNotifier - sends SMS notifications to a driver's pre-registered
emergency contacts via the Twilio API when a COMBINED alert is classified,
per the GisingLang thesis (Chapter 3):

    "The CloudNotifier module queries the EmergencyContacts table in
    Supabase to retrieve all active emergency contacts registered by the
    authenticated driver, then constructs and transmits HTTPS POST requests
    to the Twilio API endpoint for each contact number. The Twilio API
    authenticates requests using HTTP Basic Authentication with the
    account's SID and authentication token ..."

    "The SMS message includes the driver's name, alert level, timestamp,
    and PERCLOS value, formatted as a bilingual English and Filipino
    message ... A per-contact cooldown period of 120 seconds prevents
    repeated SMS messages during sustained drowsiness events."

    "... the only state that triggers the relative notification channel."
    (i.e. only ClassificationResult.sms_required == True, which only
    happens on AlertLevel.COMBINED - see drowsiness_classifier.py.)

Message building and cooldown logic are pure/testable without any network
access. The actual Twilio call is tested by mocking urllib, matching test
case UT-08 in the Software Testing document. A live Twilio account is only
needed for the real send-test script, not for these unit tests.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from drowsiness_classifier import ClassificationResult

DEFAULT_COOLDOWN_SECONDS = 120.0


@dataclass
class EmergencyContact:
    contact_id: str
    user_id: str
    name: str
    phone_number: str  # international format, e.g. "+639170000000"
    relationship: str
    is_active: bool = True


@dataclass
class CloudNotifierConfig:
    account_sid: str
    auth_token: str
    from_number: str  # your Twilio phone number, e.g. "+15005550006"
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS


def build_sms_message(
    driver_name: str,
    alert_level: str,
    perclos_value: float,
    timestamp: datetime | None = None,
) -> str:
    """
    Bilingual (English / Filipino) alert message, per thesis Chapter 3, "to
    maximize comprehension by Philippine relatives." Pure function, no
    network dependency.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    time_str = timestamp.strftime("%Y-%m-%d %H:%M UTC")
    perclos_pct = f"{perclos_value * 100:.0f}%"

    english = (
        f"GisingLang Alert: {driver_name} shows signs of severe drowsiness "
        f"({alert_level}) at {time_str}. PERCLOS: {perclos_pct}."
    )
    filipino = (
        f"Babala: Nagpapakita ng matinding pagkaantok si {driver_name} "
        f"({alert_level}) noong {time_str}. PERCLOS: {perclos_pct}."
    )
    return f"{english} / {filipino}"


def fetch_active_contacts(supabase_url: str, api_key: str, user_id: str, timeout_seconds: float = 5.0) -> list[EmergencyContact]:
    """
    Queries Supabase for a driver's active emergency contacts:
    GET /rest/v1/emergency_contacts?user_id=eq.<user_id>&is_active=eq.true
    """
    query = urllib.parse.urlencode({"user_id": f"eq.{user_id}", "is_active": "eq.true", "select": "*"})
    url = f"{supabase_url.rstrip('/')}/rest/v1/emergency_contacts?{query}"
    request = urllib.request.Request(
        url=url,
        method="GET",
        headers={"apikey": api_key, "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        rows = json.loads(response.read().decode("utf-8"))

    return [
        EmergencyContact(
            contact_id=row["contact_id"],
            user_id=row["user_id"],
            name=row["name"],
            phone_number=row["phone_number"],
            relationship=row.get("relationship", ""),
            is_active=row.get("is_active", True),
        )
        for row in rows
    ]


class CloudNotifier:
    """Sends Twilio SMS alerts to a driver's emergency contacts on
    COMBINED alerts, with a per-contact cooldown to prevent flooding."""

    def __init__(self, config: CloudNotifierConfig):
        self.config = config
        self._last_sent_at: dict[str, float] = {}  # phone_number -> monotonic time
        self._threads: list[threading.Thread] = []
        self.last_error: Exception | None = None

    def _in_cooldown(self, phone_number: str, now: float) -> bool:
        last_sent = self._last_sent_at.get(phone_number)
        return last_sent is not None and (now - last_sent) < self.config.cooldown_seconds

    def _send_sms(self, to_number: str, body: str) -> None:
        endpoint = f"https://api.twilio.com/2010-04-01/Accounts/{self.config.account_sid}/Messages.json"
        form_body = urllib.parse.urlencode({"To": to_number, "From": self.config.from_number, "Body": body}).encode("utf-8")
        credentials = base64.b64encode(f"{self.config.account_sid}:{self.config.auth_token}".encode("utf-8")).decode("ascii")

        request = urllib.request.Request(
            url=endpoint,
            data=form_body,
            method="POST",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10.0) as response:
                self.last_error = None
                _ = response.status
        except urllib.error.URLError as exc:
            self.last_error = exc

    def notify(
        self,
        classification: ClassificationResult,
        driver_name: str,
        contacts: list[EmergencyContact],
        perclos_value: float,
        timestamp: datetime | None = None,
        blocking: bool = False,
    ) -> list[str]:
        """
        Sends the bilingual alert SMS to every active, non-cooldown contact,
        but only if classification.sms_required is True (COMBINED alerts
        only - the DrowsinessClassifier is the single source of truth for
        that decision, per Chapter 3: "the only state that triggers the
        relative notification channel").

        Returns the list of phone numbers actually messaged this call
        (useful for tests and logging - numbers skipped due to cooldown or
        inactive status are excluded).
        """
        if not classification.sms_required:
            return []

        message = build_sms_message(driver_name, classification.alert_level.value, perclos_value, timestamp)
        now = time.monotonic()

        threads: list[threading.Thread] = []
        sent_to: list[str] = []
        for contact in contacts:
            if not contact.is_active:
                continue
            if self._in_cooldown(contact.phone_number, now):
                continue

            self._last_sent_at[contact.phone_number] = now
            sent_to.append(contact.phone_number)
            thread = threading.Thread(target=self._send_sms, args=(contact.phone_number, message), daemon=True)
            threads.append(thread)

        for t in threads:
            t.start()
        self._threads.extend(threads)

        if blocking:
            for t in threads:
                t.join()

        return sent_to
