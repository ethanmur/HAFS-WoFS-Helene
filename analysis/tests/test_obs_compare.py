"""Unit tests for obs_compare's pure logic (no network, no Hercules data).

Run directly:   python3 analysis/tests/test_obs_compare.py
Or via pytest:  pytest analysis/tests/test_obs_compare.py -v
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from obs_compare import (
    ObsCompareCase, from_yaml, swath_mask, case_swath_mask, track_segment,
    hourly_timestamps, check_cache_complete,
)
from hafs_common import mrms_s3_key
import aorc_common
import stage4_hourly


_TRACK = [
    (datetime(2024, 9, 26, 0), 28.0, -84.0),
    (datetime(2024, 9, 26, 6), 29.0, -84.0),
    (datetime(2024, 9, 26, 12), 30.0, -84.0),
]


def test_track_segment_interpolates_hourly():
    seg = track_segment(_TRACK, datetime(2024, 9, 26, 0),
                        datetime(2024, 9, 26, 6))
    assert len(seg) == 7
    assert np.isclose(seg[0][0], 28.0)
    assert np.isclose(seg[3][0], 28.5)   # halfway between 28 and 29
    assert np.isclose(seg[-1][0], 29.0)


def test_swath_mask_shrinks_with_radius():
    lat = np.linspace(20, 40, 41)
    lon = np.linspace(-95, -75, 41)
    big = swath_mask(_TRACK, 500.0, lat, lon,
                     datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 12))
    small = swath_mask(_TRACK, 50.0, lat, lon,
                       datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 12))
    assert big.shape == (41, 41)
    assert small.sum() < big.sum()
    assert small.sum() > 0   # the track itself is inside the domain


def test_swath_mask_accepts_1d_or_2d_grid():
    lat1d = np.linspace(20, 40, 21)
    lon1d = np.linspace(-95, -75, 21)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    m1 = swath_mask(_TRACK, 300.0, lat1d, lon1d,
                    datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 6))
    m2 = swath_mask(_TRACK, 300.0, lat2d, lon2d,
                    datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 6))
    assert np.array_equal(m1, m2)


def _minimal_case(**overrides):
    kwargs = dict(
        storm_name="Test Storm", best_track=Path("/tmp/bt.dat"),
        valid_start=datetime(2024, 9, 26, 0), valid_end=datetime(2024, 9, 26, 6),
        domain=(15.0, 42.0, -100.0, -60.0), mask_radius_km=500.0,
        grid_res=0.05, out_dir=Path("/tmp/out"),
        mrms_cache_dir=Path("/tmp/mrms"), stage4_cache_dir=Path("/tmp/st4"),
        aorc_cache_dir=Path("/tmp/aorc"),
    )
    kwargs.update(overrides)
    return ObsCompareCase(**kwargs)


def test_case_swath_mask_off_by_default_shows_everything():
    lat = np.linspace(20, 40, 21)
    lon = np.linspace(-95, -75, 21)
    case = _minimal_case()   # clip_outside_radius defaults to False
    assert case.clip_outside_radius is False
    mask = case_swath_mask(case, _TRACK, lat, lon,
                           datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 6))
    assert mask.all()   # nothing clipped -- full domain visible


def test_case_swath_mask_clips_when_enabled():
    lat = np.linspace(20, 40, 21)
    lon = np.linspace(-95, -75, 21)
    case = _minimal_case(clip_outside_radius=True)
    clipped = case_swath_mask(case, _TRACK, lat, lon,
                              datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 6))
    unclipped_equivalent = swath_mask(_TRACK, case.mask_radius_km, lat, lon,
                                      datetime(2024, 9, 26, 0),
                                      datetime(2024, 9, 26, 6))
    assert np.array_equal(clipped, unclipped_equivalent)
    assert not clipped.all()   # some points genuinely excluded


def test_hourly_timestamps_end_of_hour_convention():
    ts = hourly_timestamps(datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 3))
    assert ts == [datetime(2024, 9, 26, 1), datetime(2024, 9, 26, 2),
                 datetime(2024, 9, 26, 3)]


def test_from_yaml_round_trip_and_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "case.yaml"
        yaml_path.write_text(yaml.safe_dump({
            "storm_name": "Test Storm",
            "best_track": "/tmp/bt.dat",
            "valid_start": 2024092400,
            "valid_end": 2024092906,
            "domain": [15.0, 42.0, -100.0, -60.0],
            "skip_mrms": True,
            "skip_stage4": True,
        }))
        case = from_yaml(yaml_path)
        assert isinstance(case, ObsCompareCase)
        assert case.storm_name == "Test Storm"
        assert case.valid_start == datetime(2024, 9, 24, 0)
        assert case.valid_end == datetime(2024, 9, 29, 6)
        assert case.mask_radius_km == 500.0     # default
        assert case.grid_res == 0.05            # default
        assert case.skip_mrms is True
        assert case.skip_stage4 is True
        assert case.skip_aorc is False            # default
        assert case.clip_outside_radius is False  # default: show everything
        assert case.output_slug == "case_2024092400_2024092906"


def test_from_yaml_requires_core_fields():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "case.yaml"
        yaml_path.write_text(yaml.safe_dump({"storm_name": "Test Storm"}))
        try:
            from_yaml(yaml_path)
            assert False, "expected a KeyError for the missing fields"
        except KeyError:
            pass


def _touch_mrms(cache_dir, valid_dt):
    _, fname = mrms_s3_key(valid_dt)
    (Path(cache_dir) / fname.replace(".gz", "")).parent.mkdir(
        parents=True, exist_ok=True)
    (Path(cache_dir) / fname.replace(".gz", "")).touch()


def _touch_aorc(cache_dir, valid_dt):
    path = aorc_common.aorc_cache_path(cache_dir, valid_dt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_check_cache_complete_empty_when_everything_cached():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        case = _minimal_case(
            skip_stage4=True, mrms_cache_dir=base / "mrms",
            aorc_cache_dir=base / "aorc",
            valid_start=datetime(2024, 9, 26, 0),
            valid_end=datetime(2024, 9, 26, 2),
        )
        for t in hourly_timestamps(case.valid_start, case.valid_end):
            _touch_mrms(case.mrms_cache_dir, t)
            _touch_aorc(case.aorc_cache_dir, t)
        assert check_cache_complete(case) == []


def test_check_cache_complete_flags_missing_hours():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        case = _minimal_case(
            skip_stage4=True, mrms_cache_dir=base / "mrms",
            aorc_cache_dir=base / "aorc",
            valid_start=datetime(2024, 9, 26, 0),
            valid_end=datetime(2024, 9, 26, 2),
        )
        # Only cache one of the two needed hours, and only for MRMS.
        _touch_mrms(case.mrms_cache_dir, datetime(2024, 9, 26, 1))
        missing = check_cache_complete(case)
        assert len(missing) == 3   # MRMS h2, AORC h1, AORC h2
        assert any("MRMS" in m and "02Z" in m for m in missing)
        assert any("AORC" in m and "01Z" in m for m in missing)


def test_check_cache_complete_flags_missing_stage4_hours(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        case = _minimal_case(
            skip_stage4=False, skip_mrms=True, skip_aorc=True,
            stage4_cache_dir=Path(tmp),
            valid_start=datetime(2024, 9, 26, 0),
            valid_end=datetime(2024, 9, 26, 3),
        )
        # Only hour 2 is "in the cache" -- fake the parsed ST4.<day> index
        # so this test doesn't need a real GRIB fixture.
        monkeypatch.setattr(
            stage4_hourly, "index_stage4_hourly",
            lambda cache_dir_str: {datetime(2024, 9, 26, 2): Path(tmp) / "ST4.20240926"},
        )
        missing = check_cache_complete(case)
        assert len(missing) == 2   # hours 1 and 3 missing
        assert any("01Z" in m for m in missing)
        assert any("03Z" in m for m in missing)
        assert not any("02Z" in m for m in missing)


def test_check_cache_complete_empty_when_stage4_cached(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        case = _minimal_case(
            skip_stage4=False, skip_mrms=True, skip_aorc=True,
            stage4_cache_dir=Path(tmp),
            valid_start=datetime(2024, 9, 26, 0),
            valid_end=datetime(2024, 9, 26, 2),
        )
        fake_path = Path(tmp) / "ST4.20240926"
        monkeypatch.setattr(
            stage4_hourly, "index_stage4_hourly",
            lambda cache_dir_str: {
                datetime(2024, 9, 26, 1): fake_path,
                datetime(2024, 9, 26, 2): fake_path,
            },
        )
        assert check_cache_complete(case) == []


class _FakeMonkeypatch:
    """Minimal stand-in so this file also runs standalone (no pytest)."""

    def __init__(self):
        self._saved = []

    def setattr(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        needs_mp = "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]
        mp = _FakeMonkeypatch() if needs_mp else None
        try:
            fn(mp) if needs_mp else fn()
        finally:
            if mp is not None:
                mp.undo()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
