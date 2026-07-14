"""
DriverProfile - persists a per-driver calibration profile (head-pose
baseline offset, personalized EAR closed-eye threshold) to a local JSON
file, so a returning driver doesn't need to recalibrate every session.

This is the practical, local-file version of what the thesis describes as
the "Driver (User)" role having a registered account (Chapter 3, Database
Design) - for now this lives on the device as a JSON file per driver rather
than a Supabase Users-table row, since driver login/auth hasn't been built
yet. It's a direct upgrade path: DriverProfile's fields are exactly what
you'd later store as columns on a Supabase profile row tied to
Users.UserID, so migrating this to Supabase later is a straightforward swap
of save_profile/load_profile's storage backend, not a redesign.

No camera or hardware dependency - just JSON file I/O - so it's fully unit
tested without a webcam (see test_driver_profile.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

DEFAULT_PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "driver_profiles")


@dataclass
class DriverProfile:
    driver_id: str  # filesystem-safe slug, e.g. "juan_dela_cruz" - used as the filename
    display_name: str
    pitch_offset_deg: float
    roll_offset_deg: float
    ear_closed_threshold: Optional[float] = None  # None = use EARCalculator's module default (0.20)
    calibrated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "DriverProfile":
        return DriverProfile(**data)


def slugify(display_name: str) -> str:
    """Turns 'Juan Dela Cruz' into 'juan_dela_cruz' for use as a filename /
    driver_id. Falls back to 'driver' if the name has no usable characters."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in display_name.strip().lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "driver"


def _profile_path(driver_id: str, directory: str) -> str:
    return os.path.join(directory, driver_id + ".json")


def save_profile(profile: DriverProfile, directory: str = DEFAULT_PROFILES_DIR) -> None:
    os.makedirs(directory, exist_ok=True)
    path = _profile_path(profile.driver_id, directory)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)


def load_profile(driver_id: str, directory: str = DEFAULT_PROFILES_DIR) -> DriverProfile:
    path = _profile_path(driver_id, directory)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DriverProfile.from_dict(data)


def list_profiles(directory: str = DEFAULT_PROFILES_DIR) -> list:
    """Returns every saved DriverProfile in `directory`, sorted by
    display_name. Skips any file that fails to parse rather than crashing -
    a corrupted profile shouldn't take down the whole selection menu."""
    if not os.path.isdir(directory):
        return []

    profiles = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        driver_id = filename[: -len(".json")]
        try:
            profiles.append(load_profile(driver_id, directory))
        except (json.JSONDecodeError, TypeError, KeyError, OSError):
            continue

    return sorted(profiles, key=lambda p: p.display_name.lower())
