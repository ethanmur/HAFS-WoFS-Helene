"""Unit tests for aorc_common's accumulation logic (no network).

load_aorc_hour itself needs S3; these tests monkeypatch it with a synthetic
per-hour field to check sum_aorc_hours' windowing/accumulation math in
isolation, the same way a real run would sum real hours.

Run directly:   python3 analysis/tests/test_aorc_common.py
Or via pytest:  pytest analysis/tests/test_aorc_common.py -v
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import aorc_common


def _fake_load_aorc_hour(valid_dt, domain, cache_dir, download=True):
    lat = np.array([25.0, 26.0])
    lon = np.array([-85.0, -84.0])
    # 1 mm/hour everywhere, plus the hour-of-day so the sum is checkable.
    data = np.full((2, 2), float(valid_dt.hour))
    return lat, lon, data


def test_sum_aorc_hours_matches_manual_sum(monkeypatch):
    monkeypatch.setattr(aorc_common, "load_aorc_hour", _fake_load_aorc_hour)
    lat, lon, total = aorc_common.sum_aorc_hours(
        window_start=datetime(2024, 9, 26, 10),
        window_end=datetime(2024, 9, 26, 13), domain=(0, 0, 0, 0),
        cache_dir="/unused",
    )
    # hours summed are 11, 12, 13 (end-of-hour convention, same as MRMS)
    assert np.allclose(total, 11 + 12 + 13)
    assert lat.shape == (2,) and lon.shape == (2,)


def test_sum_aorc_hours_raises_when_nothing_loads(monkeypatch):
    def _always_fail(*args, **kwargs):
        raise RuntimeError("simulated S3 outage")
    monkeypatch.setattr(aorc_common, "load_aorc_hour", _always_fail)
    try:
        aorc_common.sum_aorc_hours(
            window_start=datetime(2024, 9, 26, 0),
            window_end=datetime(2024, 9, 26, 3), domain=(0, 0, 0, 0),
            cache_dir="/unused",
        )
        assert False, "expected a RuntimeError"
    except RuntimeError:
        pass


def test_sum_aorc_hours_reraises_file_not_found_without_swallowing(monkeypatch):
    """A cache miss under download=False must abort the whole sum, not be
    silently skipped like a transient fetch error -- a partial daily total
    would look valid but be quietly wrong."""
    def _missing(*args, **kwargs):
        raise FileNotFoundError("not cached")
    monkeypatch.setattr(aorc_common, "load_aorc_hour", _missing)
    try:
        aorc_common.sum_aorc_hours(
            window_start=datetime(2024, 9, 26, 0),
            window_end=datetime(2024, 9, 26, 3), domain=(0, 0, 0, 0),
            cache_dir="/unused", download=False,
        )
        assert False, "expected a FileNotFoundError"
    except FileNotFoundError:
        pass


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
    fns = [(k, v) for k, v in sorted(globals().items())
          if k.startswith("test_")]
    for name, fn in fns:
        mp = _FakeMonkeypatch()
        try:
            fn(mp)
        finally:
            mp.undo()
        print(f"PASS {name}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
