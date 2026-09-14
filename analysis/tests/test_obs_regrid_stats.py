"""Unit tests for the stats-regrid accumulators.

Run directly:   python3 analysis/tests/test_obs_regrid_stats.py
Or via pytest:  pytest analysis/tests/test_obs_regrid_stats.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from obs_regrid_stats import (BIN_EDGES, MIN_RAIN_MM, PairStats, SourceStats,
                              _pair_points, _points, hist_percentile)

_RNG = np.random.default_rng(7)


def test_source_stats_counts_dry_and_wet_cells():
    vals = np.array([0.0, 0.05, MIN_RAIN_MM, 1.0, 50.0])
    st = SourceStats()
    st.update(vals)
    assert st.n_valid == 5 and st.n_wet == 3
    assert st.dry_fraction == 2 / 5
    assert st.max_mm == 50.0
    assert st.counts.sum() == 3


def test_source_stats_merge_matches_single_pass():
    a, b = _RNG.gamma(0.7, 4.0, 500), _RNG.gamma(0.7, 4.0, 500)
    split = SourceStats()
    for chunk in (a, b):
        part = SourceStats()
        part.update(chunk)
        split.merge(part)
    whole = SourceStats()
    whole.update(np.concatenate([a, b]))
    assert split.row() == whole.row()
    assert np.array_equal(split.counts, whole.counts)


def test_source_stats_keeps_values_above_last_bin():
    st = SourceStats()
    st.update(np.array([BIN_EDGES[-1] * 5]))
    assert st.counts[-1] == 1 and st.max_mm == BIN_EDGES[-1] * 5


def test_pair_metrics_match_numpy():
    a, b = _RNG.gamma(0.7, 4.0, 4000), _RNG.gamma(0.7, 4.0, 4000)
    st = PairStats()
    half = a.size // 2
    for sl in (slice(0, half), slice(half, None)):
        part = PairStats()
        part.update(a[sl], b[sl])
        st.merge(part)
    m = st.metrics()
    assert st.n == a.size
    assert np.isclose(m["bias_mm"], np.mean(a - b), atol=1e-5)
    assert np.isclose(m["rmse_mm"], np.sqrt(np.mean((a - b) ** 2)), atol=1e-5)
    assert np.isclose(m["r"], np.corrcoef(a, b)[0, 1], atol=1e-4)


def test_pair_stats_empty_is_nan_not_error():
    m = PairStats().metrics()
    assert np.isnan(m["rmse_mm"]) and np.isnan(m["r"])


def test_hist_percentile_brackets_the_median():
    st = SourceStats()
    st.update(np.full(100, 5.0))
    median = hist_percentile(st.counts, 50.0)
    i = np.searchsorted(BIN_EDGES, 5.0) - 1
    assert BIN_EDGES[i] <= median <= BIN_EDGES[i + 1]
    assert np.isnan(hist_percentile(np.zeros_like(st.counts), 50.0))


def test_points_and_pair_points_respect_mask_and_missing():
    a = np.array([[1.0, np.nan], [3.0, 4.0]])
    b = np.array([[2.0, 2.0], [np.nan, 8.0]])
    mask = np.array([[True, True], [True, False]])
    assert sorted(_points(a, mask)) == [1.0, 3.0]
    assert sorted(_points(a, None)) == [1.0, 3.0, 4.0]
    pa, pb = _pair_points(a, b, mask)
    assert list(pa) == [1.0] and list(pb) == [2.0]
    pa, pb = _pair_points(a, b, None)
    assert list(pa) == [1.0, 4.0] and list(pb) == [2.0, 8.0]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
