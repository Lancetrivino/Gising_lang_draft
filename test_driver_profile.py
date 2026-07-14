"""
Unit tests for DriverProfile - save/load/list round trip and slugify. No
camera, no MediaPipe, no OpenCV required - uses a temp directory for
filesystem I/O so it never touches your real driver_profiles folder.

Run with:  pytest test_driver_profile.py -v
"""

import tempfile

from driver_profile import DriverProfile, list_profiles, load_profile, save_profile, slugify


def test_slugify_handles_spaces_and_case():
    assert slugify("Juan Dela Cruz") == "juan_dela_cruz"
    assert slugify("  Maria   Santos  ") == "maria_santos"
    assert slugify("") == "driver"
    assert slugify("!!!") == "driver"


def test_save_and_load_round_trip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        profile = DriverProfile(
            driver_id="juan_dela_cruz",
            display_name="Juan Dela Cruz",
            pitch_offset_deg=7.3,
            roll_offset_deg=-2.1,
            ear_closed_threshold=0.18,
            calibrated_at="2026-07-13T10:00:00+00:00",
        )
        save_profile(profile, directory=tmp_dir)

        loaded = load_profile("juan_dela_cruz", directory=tmp_dir)

        assert loaded == profile


def test_ear_closed_threshold_defaults_to_none():
    with tempfile.TemporaryDirectory() as tmp_dir:
        profile = DriverProfile(
            driver_id="maria_santos", display_name="Maria Santos",
            pitch_offset_deg=0.0, roll_offset_deg=0.0,
        )
        save_profile(profile, directory=tmp_dir)

        loaded = load_profile("maria_santos", directory=tmp_dir)

        assert loaded.ear_closed_threshold is None


def test_list_profiles_returns_all_sorted_by_display_name():
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_profile(
            DriverProfile(driver_id="zeus", display_name="Zeus Reyes", pitch_offset_deg=0, roll_offset_deg=0),
            directory=tmp_dir,
        )
        save_profile(
            DriverProfile(driver_id="ana", display_name="Ana Cruz", pitch_offset_deg=0, roll_offset_deg=0),
            directory=tmp_dir,
        )

        profiles = list_profiles(directory=tmp_dir)

        assert [p.display_name for p in profiles] == ["Ana Cruz", "Zeus Reyes"]


def test_list_profiles_on_missing_directory_returns_empty():
    assert list_profiles(directory="/nonexistent/path/for/sure") == []


def test_list_profiles_skips_corrupted_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_profile(
            DriverProfile(driver_id="good", display_name="Good Profile", pitch_offset_deg=0, roll_offset_deg=0),
            directory=tmp_dir,
        )
        with open(tmp_dir + "/corrupted.json", "w") as f:
            f.write("{not valid json")

        profiles = list_profiles(directory=tmp_dir)

        assert [p.display_name for p in profiles] == ["Good Profile"]
