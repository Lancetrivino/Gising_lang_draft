"""
CloudLogger - builds the drowsiness-event payload and POSTs it to the
Supabase REST API, per the GisingLang thesis (Chapter 3, Network Layer):

    "The CloudLogger module in the Python application constructs an HTTPS
    POST request containing the event payload - timestamp, device ID,
    drowsiness level, EAR value, PERCLOS value, head pose angles, and alert
    types triggered - and transmits it to the Supabase REST API endpoint."

    "The Raspberry Pi Zero 2W firmware communicates with Supabase by
    constructing an HTTPS POST request containing the event payload as a
    JSON object ... using Python's built-in urllib library."

    "Both network transmissions occur asynchronously in background Python
    threads to ensure that neither operation introduces any delay into the
    main detection loop."

Payload building (build_event_payload) is pure and camera/network-free, so
it's directly unit tested against UT-07 in the Software Testing document.
The actual HTTPS call is tested by mocking urllib - no live Supabase project
or network access required to run the tests. See supabase_schema.sql for the
matching table definition.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from drowsiness_classifier import ClassificationResult
from ear_calculator import EARResult
from head_pose_estimator import HeadPoseResult
from perclos_calculator import PERCLOSResult


def build_event_payload(
    device_id: str,
    ear_result: EARResult,
    perclos_result: PERCLOSResult,
    head_pose_result: HeadPoseResult,
    classification: ClassificationResult,
    timestamp: datetime | None = None,
) -> dict:
    """
    Assembles the drowsiness_events row exactly as described in Chapter 3's
    Network Layer paragraph: device ID, timestamp, drowsiness level, EAR
    value, PERCLOS value, head pose angles, and the triggered alert type.

    Pure function, no network or camera dependency - matches test case
    UT-07 (payload correctly formatted, no missing/malformed fields).
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    return {
        "device_id": device_id,
        "event_timestamp": timestamp.isoformat(),
        "alert_level": classification.alert_level.value,
        "ear_value": round(ear_result.avg_ear, 4),
        "perclos_value": round(perclos_result.perclos, 4),
        "pitch": round(head_pose_result.pitch, 2),
        "yaw": round(head_pose_result.yaw, 2),
        "roll": round(head_pose_result.roll, 2),
    }


@dataclass
class CloudLoggerConfig:
    supabase_url: str  # e.g. "https://xyzcompany.supabase.co"
    api_key: str  # service_role key - keep this only on the device, never in dashboard code
    table: str = "drowsiness_events"
    timeout_seconds: float = 5.0


class CloudLogger:
    """Sends drowsiness event payloads to Supabase over HTTPS, using
    urllib per the thesis's technology stack, in a background thread so the
    main detection loop is never blocked by network latency."""

    def __init__(self, config: CloudLoggerConfig):
        self.config = config
        self._threads: list[threading.Thread] = []
        self.last_error: Exception | None = None
        self.last_status: int | None = None

    def _endpoint(self) -> str:
        return f"{self.config.supabase_url.rstrip('/')}/rest/v1/{self.config.table}"

    def _post(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=self._endpoint(),
            data=body,
            method="POST",
            headers={
                "apikey": self.config.api_key,
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                self.last_status = response.status
                self.last_error = None
        except urllib.error.URLError as exc:
            # Network failure should never crash the detection loop - log
            # and move on. The event is simply not persisted this cycle.
            self.last_error = exc
            self.last_status = None

    def log_event(self, payload: dict, blocking: bool = False) -> None:
        """
        blocking=False (default, production behavior): POSTs in a daemon
            thread and returns immediately.
        blocking=True (test convenience): waits for the request to finish
            before returning.
        """
        thread = threading.Thread(target=self._post, args=(payload,), daemon=True)
        thread.start()
        self._threads.append(thread)
        if blocking:
            thread.join()
