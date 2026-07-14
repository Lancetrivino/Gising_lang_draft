"""
send_test_event.py - sends ONE real drowsiness event to your live Supabase
project, using the actual CloudLogger class (not a mock), to confirm the
end-to-end path works: Python -> HTTPS POST -> Supabase -> drowsiness_events
table.

This does NOT hardcode your Supabase URL or service_role key. Set them as
environment variables first (PowerShell, in the same terminal you'll run
this from):

    $env:SUPABASE_URL="https://your-project.supabase.co"
    $env:SUPABASE_SERVICE_KEY="your-service-role-key"
    python send_test_event.py

(cmd.exe instead: use `set SUPABASE_URL=...` / `set SUPABASE_SERVICE_KEY=...`)

Keeping the key out of any file means it's safe to commit this script to
Git/GitHub later - there's nothing secret in it.

After running, check Supabase: Table Editor > drowsiness_events. You should
see one new row with device_id "test-device-001".
"""

import os
import sys
from datetime import datetime, timezone

from cloud_logger import CloudLogger, CloudLoggerConfig, build_event_payload
from drowsiness_classifier import AlertLevel, ClassificationResult
from ear_calculator import EARResult
from head_pose_estimator import HeadPoseResult
from perclos_calculator import PERCLOSResult


def main() -> None:
    supabase_url = os.environ.get("SUPABASE_URL")
    api_key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not supabase_url or not api_key:
        print(
            "Missing environment variables. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_KEY in this terminal before running this "
            "script (see the docstring at the top of this file).",
            file=sys.stderr,
        )
        sys.exit(1)

    config = CloudLoggerConfig(supabase_url=supabase_url, api_key=api_key)
    logger = CloudLogger(config)

    # A representative COMBINED event, as if a real drowsy driving moment
    # had just been detected and classified.
    payload = build_event_payload(
        device_id="test-device-001",
        ear_result=EARResult(left_ear=0.09, right_ear=0.08, avg_ear=0.085, eye_closed=True),
        perclos_result=PERCLOSResult(perclos=0.27, breached=True, sample_count=60),
        head_pose_result=HeadPoseResult(pitch=18.2, yaw=1.4, roll=2.9, breached=True),
        classification=ClassificationResult(
            alert_level=AlertLevel.COMBINED, buzzer_duration_ms=1000, motor_duration_ms=1000, sms_required=True
        ),
        timestamp=datetime.now(timezone.utc),
    )

    print("Sending test event to:", config.supabase_url)
    print("Payload:", payload)

    logger.log_event(payload, blocking=True)  # blocking=True so we can check the result immediately

    if logger.last_error is not None:
        print("\nFAILED:", logger.last_error, file=sys.stderr)
        print(
            "Common causes: wrong URL, wrong key, the drowsiness_events "
            "table doesn't exist yet (run supabase_schema.sql first), or "
            "a column name mismatch.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nSUCCESS - Supabase responded with HTTP {logger.last_status}.")
    print("Check Table Editor > drowsiness_events in your Supabase dashboard for the new row.")


if __name__ == "__main__":
    main()
