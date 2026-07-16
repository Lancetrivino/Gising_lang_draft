"""
live_demo.py - runs the full GisingLang detection pipeline live on your
laptop's webcam: camera -> MediaPipe Face Landmarker -> EAR -> PERCLOS ->
head pose -> classifier -> alert (printed/overlaid, since your laptop has
no buzzer or vibration motor - AlertManager uses FakeGPIOBackend here).

This is the laptop stand-in for DetectionController running on the actual
Raspberry Pi Zero 2W with picamera2 + real GPIO. Swap FakeGPIOBackend for
RaspberryPiGPIOBackend and cv2.VideoCapture for picamera2 once you're on
the Pi - detection_pipeline.py itself doesn't change at all.

Uses MediaPipe's Tasks API (FaceLandmarker), not the older mp.solutions API -
Google removed mp.solutions starting in mediapipe 0.10.31.

DRIVER PROFILES: on startup you pick a saved driver (instant, uses their
stored calibration) or register a new one. Registering runs a two-phase
calibration:
  Phase A (~2s): sit normally, look at the camera - captures your head-pose
    baseline (corrects the generic 3D face model's systematic bias) and
    your open-eye EAR baseline.
  Phase B (~2s): close your eyes - captures your closed-eye EAR baseline.
Your personalized EAR threshold is the midpoint between the two, replacing
the generic 0.20 literature value with one tuned to your actual eyes and
camera. Everything is saved to driver_profiles/<slug>.json via
driver_profile.py, so next time you just pick your name from the menu.

Requires:  pip install mediapipe opencv-python numpy

On first run, this script auto-downloads the ~4MB face_landmarker.task
model file from Google's model server into the same folder (needs internet
access once; after that it's cached locally and works offline).

Run:  python live_demo.py
Quit: press 'q' with the video window focused.
"""

from __future__ import annotations

import os
import time
import urllib.request
from datetime import datetime, timezone

import cv2

import mediapipe as mp

from alert_manager import AlertManager
from detection_pipeline import DetectionController
from driver_profile import DriverProfile, list_profiles, save_profile, slugify
from drowsiness_classifier import AlertLevel, DrowsinessClassifier
from ear_calculator import EAR_CLOSED_THRESHOLD, EARCalculator
from gpio_backend import FakeGPIOBackend
from head_pose_estimator import HeadPoseEstimator
from perclos_calculator import PERCLOSCalculator

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

CALIBRATION_SECONDS = 10.0

ALERT_LABELS = {
    AlertLevel.NONE: ("No alert", (0, 200, 0)),
    AlertLevel.LAYER1: ("LAYER1 - buzzer (eyes)", (0, 165, 255)),
    AlertLevel.LAYER2: ("LAYER2 - vibration (head pose)", (0, 165, 255)),
    AlertLevel.COMBINED: ("COMBINED - buzzer + vibration + SMS", (0, 0, 255)),
}


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading face_landmarker.task model to " + MODEL_PATH + " ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def draw_overlay(frame, frame_result):
    ear = frame_result.ear_result
    perclos = frame_result.perclos_result
    pose = frame_result.head_pose_result
    label, color = ALERT_LABELS[frame_result.classification.alert_level]

    lines = [
        "EAR: %.3f  (closed: %s)" % (ear.avg_ear, ear.eye_closed),
        "PERCLOS: %.1f%%  (breached: %s)" % (perclos.perclos * 100, perclos.breached),
        "Pitch: %.1f deg  Roll: %.1f deg  (breached: %s)" % (pose.pitch, pose.roll, pose.breached),
        "Alert: " + label,
    ]
    for i, line in enumerate(lines):
        y = 30 + i * 28
        text_color = color if i == 3 else (255, 255, 255)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


class MonotonicMsClock:
    """Guarantees a strictly increasing millisecond timestamp for MediaPipe's
    VIDEO mode, which raises ValueError if two frames share a timestamp."""

    def __init__(self):
        self._start = time.monotonic()
        self._last_ms = -1

    def next_ms(self) -> int:
        elapsed_ms = int((time.monotonic() - self._start) * 1000)
        if elapsed_ms <= self._last_ms:
            elapsed_ms = self._last_ms + 1
        self._last_ms = elapsed_ms
        return elapsed_ms


def choose_driver_profile():
    """Console menu: pick an existing saved driver or register a new one.
    Returns an existing DriverProfile, or None if the user chose to
    register a new driver (caller runs calibration and saves a new one)."""
    profiles = list_profiles()

    print("\n=== GisingLang: select driver ===")
    for i, profile in enumerate(profiles, start=1):
        print("  %d) %s" % (i, profile.display_name))
    print("  N) Register new driver")

    while True:
        choice = input("Choice: ").strip().lower()
        if choice == "n":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(profiles):
            return profiles[int(choice) - 1]
        print("Not a valid choice, try again.")


def capture_window(cap, landmarker, clock, seconds, prompt_text, head_pose_estimator, ear_calculator):
    """Reads frames for `seconds`, showing `prompt_text` on screen, and
    returns (pitch_samples, roll_samples, ear_samples) collected from every
    frame where a face was detected."""
    pitch_samples, roll_samples, ear_samples = [], [], []
    window_start = time.monotonic()

    while time.monotonic() - window_start < seconds:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, clock.next_ms())
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            pose = head_pose_estimator.compute(landmarks, width, height)
            ear = ear_calculator.compute(landmarks, width, height)
            pitch_samples.append(pose.pitch)
            roll_samples.append(pose.roll)
            ear_samples.append(ear.avg_ear)

        remaining = seconds - (time.monotonic() - window_start)
        cv2.putText(frame, "%s (%.1fs)" % (prompt_text, max(remaining, 0.0)), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("GisingLang - live demo (press q to quit)", frame)
        cv2.waitKey(1)

    return pitch_samples, roll_samples, ear_samples


def register_new_driver(cap, landmarker, clock):
    """Runs the two-phase calibration and saves a new DriverProfile."""
    display_name = input("Enter the driver's name: ").strip() or "Driver"
    driver_id = slugify(display_name)

    raw_head_pose_estimator = HeadPoseEstimator()  # offsets=0, captures the true raw bias
    raw_ear_calculator = EARCalculator()  # threshold doesn't matter here, we only read avg_ear

    print("Phase A: sit normally and look at the camera...")
    pitch_samples, roll_samples, ear_open_samples = capture_window(
        cap, landmarker, clock, CALIBRATION_SECONDS, "Calibrating - look at camera", raw_head_pose_estimator, raw_ear_calculator
    )

    print("Phase B: now close your eyes...")
    _, _, ear_closed_samples = capture_window(
        cap, landmarker, clock, CALIBRATION_SECONDS, "Calibrating - close your eyes", raw_head_pose_estimator, raw_ear_calculator
    )

    pitch_offset = sum(pitch_samples) / len(pitch_samples) if pitch_samples else 0.0
    roll_offset = sum(roll_samples) / len(roll_samples) if roll_samples else 0.0
    ear_open_avg = sum(ear_open_samples) / len(ear_open_samples) if ear_open_samples else None
    ear_closed_avg = sum(ear_closed_samples) / len(ear_closed_samples) if ear_closed_samples else None

    if ear_open_avg is not None and ear_closed_avg is not None and ear_closed_avg < ear_open_avg:
        ear_threshold = (ear_open_avg + ear_closed_avg) / 2.0
        print("Personalized EAR threshold: %.3f (open avg %.3f, closed avg %.3f)" % (ear_threshold, ear_open_avg, ear_closed_avg))
    else:
        ear_threshold = EAR_CLOSED_THRESHOLD
        print("Could not reliably measure a closed-eye baseline - falling back to the default threshold (%.2f)." % ear_threshold)
        print("(Make sure your eyes were actually closed during Phase B, then re-register if you want a personalized value.)")

    profile = DriverProfile(
        driver_id=driver_id,
        display_name=display_name,
        pitch_offset_deg=pitch_offset,
        roll_offset_deg=roll_offset,
        ear_closed_threshold=ear_threshold,
        calibrated_at=datetime.now(timezone.utc).isoformat(),
    )
    save_profile(profile)
    print("Saved profile for %s. Next time, just pick them from the menu.\n" % display_name)
    return profile


def main():
    ensure_model_downloaded()

    base_options_cls = mp.tasks.BaseOptions
    face_landmarker_cls = mp.tasks.vision.FaceLandmarker
    face_landmarker_options_cls = mp.tasks.vision.FaceLandmarkerOptions
    running_mode = mp.tasks.vision.RunningMode

    options = face_landmarker_options_cls(
        base_options=base_options_cls(model_asset_path=MODEL_PATH),
        running_mode=running_mode.VIDEO,
        num_faces=1,
    )
    landmarker = face_landmarker_cls.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Is another app using the camera?")

    clock = MonotonicMsClock()

    try:
        selected_profile = choose_driver_profile()
        if selected_profile is None:
            selected_profile = register_new_driver(cap, landmarker, clock)
        else:
            print("Welcome back, %s. Loaded your saved calibration.\n" % selected_profile.display_name)

        head_pose_estimator = HeadPoseEstimator()
        head_pose_estimator.calibrate(selected_profile.pitch_offset_deg, selected_profile.roll_offset_deg)

        if selected_profile.ear_closed_threshold is not None:
            ear_calculator = EARCalculator(closed_threshold=selected_profile.ear_closed_threshold)
        else:
            ear_calculator = EARCalculator()

        gpio = FakeGPIOBackend()
        controller = DetectionController(
            ear_calculator=ear_calculator,
            perclos_calculator=PERCLOSCalculator(),
            head_pose_estimator=head_pose_estimator,
            classifier=DrowsinessClassifier(),
            alert_manager=AlertManager(gpio),
        )

        print("GisingLang live demo running. Press 'q' in the video window to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_timestamp_ms = clock.next_ms()
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                frame_result = controller.process_frame(landmarks, width, height, timestamp=time.monotonic())
                draw_overlay(frame, frame_result)

                if frame_result.classification.alert_level is not AlertLevel.NONE:
                    label, _ = ALERT_LABELS[frame_result.classification.alert_level]
                    print("[" + time.strftime("%H:%M:%S") + "] " + label)
            else:
                cv2.putText(frame, "No face detected", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("GisingLang - live demo (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()