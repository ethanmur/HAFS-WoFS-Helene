"""Local unit tests for the cycles product (no Hercules data needed).

Run directly:   python3 analysis/tests/test_cycles.py
Or via pytest:  pytest analysis/tests/test_cycles.py -v
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hafs_case import (
    CyclesCase, cycles_from_yaml, discover_inits, window_hours,
    cycle_eligibility, cycle_storm_case, from_yaml,
)


# ---------------------------------------------------------------------------
# Window math
# ---------------------------------------------------------------------------

def test_window_hours():
    f1, f2 = window_hours(datetime(2024, 9, 24, 0),
                          datetime(2024, 9, 26, 0),
                          datetime(2024, 9, 28, 0))
    assert (f1, f2) == (48, 96)
    # Init exactly at the window start -> f1 == 0.
    f1, f2 = window_hours(datetime(2024, 9, 26, 0),
                          datetime(2024, 9, 26, 0),
                          datetime(2024, 9, 28, 0))
    assert (f1, f2) == (0, 48)


def test_cycle_eligibility():
    vs, ve = datetime(2024, 9, 26, 0), datetime(2024, 9, 28, 0)
    ok, reason = cycle_eligibility(datetime(2024, 9, 24, 0), 126, vs, ve)
    assert ok and reason == ""
    # Init after window start is eligible and will use an init-clipped window.
    ok, reason = cycle_eligibility(datetime(2024, 9, 26, 6), 126, vs, ve)
    assert ok and reason == ""
    # An init at the common end has no verification period.
    ok, reason = cycle_eligibility(ve, 126, vs, ve)
    assert not ok and "at or after the window end" in reason
    # Run too short to reach window end -> ineligible.
    ok, reason = cycle_eligibility(datetime(2024, 9, 24, 0), 36, vs, ve)
    assert not ok and "before the window end" in reason
    # Boundary cases are eligible: init == valid_start, end == valid_end.
    assert cycle_eligibility(vs, 48, vs, ve)[0]
    assert cycle_eligibility(datetime(2024, 9, 26, 0), 48, vs, ve)[0]


# ---------------------------------------------------------------------------
# Init discovery
# ---------------------------------------------------------------------------

def test_discover_inits():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("2024092412", "2024092400", "notacycle", "20240924"):
            (root / name).mkdir()
        (root / "2024092418").write_text("a file, not a dir")
        assert discover_inits(root) == ["2024092400", "2024092412"]


def test_discover_inits_missing_root():
    try:
        discover_inits(Path("/nonexistent/dir/xyz"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# cycles_from_yaml
# ---------------------------------------------------------------------------

_CYCLES_YAML = """\
run_root: {root}
valid_start: 2024092600
valid_end: 2024092800
landfall_time: 202409270310
storm_name: Hurricane Helene
domain: [15.0, 42.0, -100.0, -60.0]
mask_radius_km: 500
out_dir: {out}
"""


def test_cycles_from_yaml_defaults_and_slug():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "helene_hfsa_cycles.yaml"
        yml.write_text(_CYCLES_YAML.format(root=tmp, out=tmp))
        cc = cycles_from_yaml(yml)
        assert cc.valid_start == datetime(2024, 9, 26, 0)
        assert cc.valid_end == datetime(2024, 9, 28, 0)
        assert cc.landfall_time == datetime(2024, 9, 27, 3, 10)
        assert cc.inits is None
        assert cc.ets_threshold_mm == 25.0
        assert cc.ets_bar_thresholds_in == list(range(2, 25, 2))
        assert cc.fss_thresholds_in == [1.0, 2.0]
        assert cc.fss_scales_cells == [1, 3, 5, 11, 21, 41]
        assert cc.best_track is None
        assert cc.track_step_hours == 6
        assert cc.headline_fss_threshold_in is None
        assert cc.headline_fss_scale_cells is None
        assert cc.object_threshold_mm == cc.ets_threshold_mm
        assert cc.object_smooth_cells == 5
        assert cc.object_min_area_cells == 25
        assert cc.ml_features is True
        assert cc.ml_features_csv.name == "ml_features.csv"
        assert cc.thresholds_mm[0] == 1
        assert cc.case_slug == "helene_hfsa_cycles"
        assert cc.output_slug == "helene_hfsa_cycles_2024092600_2024092800"
        glat, glon = cc.fixed_grid()
        assert glat.shape == glon.shape and glat.ndim == 2


def test_cycles_from_yaml_rejects_bad_window():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "bad.yaml"
        yml.write_text(_CYCLES_YAML.format(root=tmp, out=tmp).replace(
            "valid_end: 2024092800", "valid_end: 2024092600"))
        try:
            cycles_from_yaml(yml)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "valid_end" in str(e)


def test_cycles_from_yaml_rejects_per_init_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "case.yaml"
        yml.write_text(f"run_dir: {tmp}\ninit: 2024092400\n")
        try:
            cycles_from_yaml(yml)
            assert False, "expected KeyError"
        except KeyError as e:
            assert "run_root" in str(e)


def test_from_yaml_rejects_cycles_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "cyc.yaml"
        yml.write_text(_CYCLES_YAML.format(root=tmp, out=tmp))
        try:
            from_yaml(yml)
            assert False, "expected KeyError"
        except KeyError as e:
            assert "cycles" in str(e)


# ---------------------------------------------------------------------------
# cycle_storm_case
# ---------------------------------------------------------------------------

# Minimal 2-fix atcfunix (cols: basin, cy, init, technum, tech, tau, lat,
# lon, vmax, mslp, ty ... — parse_atcfunix needs >= 8 columns).
_ATCF = (
    "AL, 09, 2024092400, 03, HFSA, 000, 168N, 832W, 45, 1002, TS\n"
    "AL, 09, 2024092400, 03, HFSA, 048, 250N, 840W, 90, 960, HU\n"
)


def _tiny_cycles_case(root, out):
    return CyclesCase(
        run_root=Path(root),
        valid_start=datetime(2024, 9, 26, 0),
        valid_end=datetime(2024, 9, 28, 0),
        storm_name="Testorm", model_label="HAFS-A",
        domain=(0.0, 1.0, 0.0, 1.0), grid_res=0.5,
        mask_radius_km=500.0, display_radius_km=750.0,
        thresholds_mm=[1], ets_threshold_mm=1.0,
        out_dir=Path(out), mrms_cache_dir=Path("/tmp"),
        stage4_cache_dir=Path("/tmp"), inits=None,
        case_slug="testcycles",
        landfall_time=datetime(2024, 9, 27, 3, 10),
        ets_bar_thresholds_in=[1.0 / 25.4],
        fss_thresholds_in=[1.0 / 25.4], fss_scales_cells=[1, 3],
        make_animation=False,
        ml_features=False,
    )


def test_cycle_storm_case_builds_from_run_root():
    with tempfile.TemporaryDirectory() as tmp:
        cyc_dir = Path(tmp) / "2024092400"
        cyc_dir.mkdir()
        (cyc_dir / "storm09l.2024092400.hfsa.trak.atcfunix").write_text(_ATCF)
        cc = _tiny_cycles_case(tmp, tmp)
        case = cycle_storm_case(cc, "2024092400")
        assert case.run_dir == cyc_dir
        assert case.init_dt == datetime(2024, 9, 24, 0)
        assert case.init_str == "2024092400"
        assert case.storm_name == "Testorm"
        assert len(case.track) == 2
        assert case.mask_radius_km == 500.0


# ---------------------------------------------------------------------------
# Windowed observation loaders (Task 2)
# ---------------------------------------------------------------------------

def test_build_mrms_total_window_sums_requested_hours():
    """Patch the hour loader; check which hour-stamps are requested and
    that the window total is the plain sum, regridded."""
    import ets_score

    requested = []

    def fake_load(s3, hour_end_dt, cache_dir):
        requested.append(hour_end_dt)
        lat = np.linspace(0.0, 1.0, 5)
        lon = np.linspace(0.0, 1.0, 5)
        return lat, lon, np.ones((5, 5))

    orig = ets_score.load_mrms_hour
    ets_score.load_mrms_hour = fake_load
    try:
        glat, glon = np.meshgrid(np.linspace(0.2, 0.8, 3),
                                 np.linspace(0.2, 0.8, 3))[::-1]
        total = ets_score.build_mrms_total_window(
            datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 3),
            Path("/tmp"), glat, glon)
    finally:
        ets_score.load_mrms_hour = orig
    # Hour files are stamped by hour END: 01Z, 02Z, 03Z — not 00Z.
    assert requested == [datetime(2024, 9, 26, 1),
                         datetime(2024, 9, 26, 2),
                         datetime(2024, 9, 26, 3)]
    assert np.allclose(total, 3.0)


def test_build_mrms_total_window_creates_cache_directory():
    """A new /tmp-style cache path must exist before the hour loader runs."""
    import ets_score
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "new" / "mrms_cache"

        def fake_load(s3, hour_end_dt, cache_dir):
            assert Path(cache_dir).is_dir()
            axis = np.linspace(0.0, 1.0, 3)
            return axis, axis, np.ones((3, 3))

        original = ets_score.load_mrms_hour
        ets_score.load_mrms_hour = fake_load
        try:
            glon, glat = np.meshgrid(np.linspace(0.2, 0.8, 2),
                                     np.linspace(0.2, 0.8, 2))
            ets_score.build_mrms_total_window(
                datetime(2024, 9, 24, 0), datetime(2024, 9, 24, 1),
                cache, glat, glon)
        finally:
            ets_score.load_mrms_hour = original
        assert cache.is_dir()


def test_parent_path_at_fhour():
    from parent_qpf import parent_path_at_fhour
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        names = ["storm09l.2024092400.hfsa.parent.atm.f048.grb2",
                 "storm09l.2024092400.hfsa.parent.atm.f096.grb2"]
        for n in names:
            (run_dir / n).write_text("")
        case = _stub_case(run_dir)
        assert parent_path_at_fhour(case, 48).name == names[0]
        assert parent_path_at_fhour(case, 96).name == names[1]
        assert parent_path_at_fhour(case, 72) is None


def _stub_case(run_dir):
    """Minimal StormCase for glob-based tests."""
    from hafs_case import StormCase
    return StormCase(
        run_dir=Path(run_dir), init_dt=datetime(2024, 9, 24, 0),
        storm_name="Testorm", model_label="HAFS-A",
        domain=(0.0, 1.0, 0.0, 1.0), grid_res=0.5,
        mask_radius_km=500.0, display_radius_km=750.0,
        thresholds_mm=[1], out_dir=Path(run_dir),
        mrms_cache_dir=Path("/tmp"), stage4_cache_dir=Path("/tmp"),
        fhours_filter=None,
        track=[(datetime(2024, 9, 24, 0), 0.5, 0.5)],
        case_slug="testcase", init_str="2024092400",
    )


def test_stage4_label():
    from parent_qpf import stage4_label
    assert stage4_label(["20240926", "20240927"]) == \
        "~09-25 12Z – 09-27 12Z (2×24h)"


# ---------------------------------------------------------------------------
# Windowed parent fields + union swath (Task 3)
# ---------------------------------------------------------------------------


def test_parent_window_total_memoizes_cumulative_fields():
    """Sweeping consecutive windows must decode each fhour's cumulative
    field once, not twice (as one window's f2 and the next window's f1)."""
    import cycles

    decoded = []

    def fake_path(case, fh):
        return Path(f"parent.f{fh:03d}.grb2")

    def fake_records(path):
        # Encode the fhour in the record so the field equals fh everywhere.
        fh = int(path.stem.split(".f")[1])
        return [{"lats": None, "lons": None, "data": fh}]

    def fake_pick(records):
        return records[0]

    def fake_regrid(lats, lons, data, grid_lat, grid_lon):
        decoded.append(data)  # data == fhour; one entry per real decode
        return np.full(grid_lat.shape, float(data))

    saved = (cycles.parent_path_at_fhour, cycles.read_hafs_tp_records,
             cycles.pick_cumulative_record, cycles.regrid_2d_to_fixed)
    cycles.parent_path_at_fhour = fake_path
    cycles.read_hafs_tp_records = fake_records
    cycles.pick_cumulative_record = fake_pick
    cycles.regrid_2d_to_fixed = fake_regrid
    try:
        case = _stub_case("/tmp")
        grid_lat = grid_lon = np.zeros((3, 3))
        # Consecutive 6-h windows: f6, f6->f12, f12->f18.
        w1 = cycles.parent_window_total(case, 0, 6, grid_lat, grid_lon)
        w2 = cycles.parent_window_total(case, 6, 12, grid_lat, grid_lon)
        w3 = cycles.parent_window_total(case, 12, 18, grid_lat, grid_lon)
    finally:
        (cycles.parent_path_at_fhour, cycles.read_hafs_tp_records,
         cycles.pick_cumulative_record, cycles.regrid_2d_to_fixed) = saved

    # Each distinct fhour (6, 12, 18) decoded exactly once — not 5 decodes.
    assert sorted(decoded) == [6, 12, 18]
    # Difference math still correct: every 6-h window accumulates 6 mm.
    assert np.allclose(w1, 6.0)
    assert np.allclose(w2, 6.0)
    assert np.allclose(w3, 6.0)


def test_union_swath_is_union_of_single_masks():
    from cycles import union_swath
    glat, glon = np.meshgrid(np.linspace(0, 10, 21),
                             np.linspace(0, 10, 21))[::-1]
    pts_a = [(2.0, 2.0)]
    pts_b = [(8.0, 8.0)]
    m_a = union_swath(pts_a, 150.0, glat, glon)
    m_b = union_swath(pts_b, 150.0, glat, glon)
    m_ab = union_swath(pts_a + pts_b, 150.0, glat, glon)
    assert m_a.any() and m_b.any()
    assert not (m_a & m_b).any()          # disjoint circles at 150 km
    assert np.array_equal(m_ab, m_a | m_b)


def test_build_cycle_fields_requires_parent_not_nest():
    """A cycle with parent files is eligible even when no nest files exist."""
    import cycles

    ccase = _tiny_cycles_case("/tmp", "/tmp")
    ccase.inits = ["2024092400"]
    case = _stub_case("/tmp")
    grid = np.zeros(ccase.fixed_grid()[0].shape)
    seen_globs = []

    def fake_discover(run_dir, glob, fhours):
        seen_globs.append(glob)
        return [(96, Path("parent.f096.grb2"))]

    saved = (cycles.cycle_storm_case, cycles.discover_files,
             cycles.parent_window_total, cycles.build_mrms_total_window,
             cycles.stage4_total_window)
    cycles.cycle_storm_case = lambda *args: case
    cycles.discover_files = fake_discover
    cycles.parent_window_total = lambda *args: np.ones_like(grid)
    cycles.build_mrms_total_window = lambda *args: np.zeros_like(grid)
    cycles.stage4_total_window = lambda *args: (None, None, None, None)
    try:
        fields = cycles.build_cycle_fields(ccase)
    finally:
        (cycles.cycle_storm_case, cycles.discover_files,
         cycles.parent_window_total, cycles.build_mrms_total_window,
         cycles.stage4_total_window) = saved

    assert len(fields["cycles"]) == 1
    assert "parent_win" in fields["cycles"][0]
    assert "nest_win" not in fields["cycles"][0]
    assert seen_globs and all("parent.atm" in glob for glob in seen_globs)


def test_build_cycle_fields_clips_late_cycle_to_initialization():
    import dataclasses
    import cycles

    ccase = _tiny_cycles_case("/tmp", "/tmp")
    ccase.inits = ["2024092606"]
    case = dataclasses.replace(
        _stub_case("/tmp"), init_dt=datetime(2024, 9, 26, 6),
        init_str="2024092606")
    grid = np.zeros(ccase.fixed_grid()[0].shape)
    calls = {}

    saved = (cycles.cycle_storm_case, cycles.discover_files,
             cycles.parent_window_total, cycles.build_mrms_total_window,
             cycles.stage4_total_window)
    cycles.cycle_storm_case = lambda *args: case
    cycles.discover_files = lambda *args: [(42, Path("parent.f042.grb2"))]

    def fake_parent(case, f1, f2, *args):
        calls["forecast_hours"] = (f1, f2)
        return np.ones_like(grid)

    def fake_mrms(start, end, *args):
        calls["obs_window"] = (start, end)
        return np.zeros_like(grid)

    cycles.parent_window_total = fake_parent
    cycles.build_mrms_total_window = fake_mrms
    cycles.stage4_total_window = lambda *args: (None, None, None, None)
    try:
        fields = cycles.build_cycle_fields(ccase)
    finally:
        (cycles.cycle_storm_case, cycles.discover_files,
         cycles.parent_window_total, cycles.build_mrms_total_window,
         cycles.stage4_total_window) = saved

    cycle = fields["cycles"][0]
    assert cycle["valid_start"] == datetime(2024, 9, 26, 6)
    assert cycle["valid_end"] == datetime(2024, 9, 28, 0)
    assert calls["forecast_hours"] == (0, 42)
    assert calls["obs_window"] == (cycle["valid_start"], cycle["valid_end"])


def test_window_track_points_hourly_inclusive():
    from cycles import window_track_points
    case = _stub_case(Path("."))
    pts = window_track_points(case, datetime(2024, 9, 26, 0),
                              datetime(2024, 9, 26, 6))
    assert len(pts) == 7                  # 00Z..06Z inclusive, hourly
    # Single-fix track -> position clamps to that fix everywhere.
    assert all(p == (0.5, 0.5) for p in pts)


# ---------------------------------------------------------------------------
# compute_cycles scoring path (Task 4)
# ---------------------------------------------------------------------------

def _tiny_cycle_fields():
    glat, glon = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))[::-1]
    obs = np.full((4, 4), 10.0)
    return dict(
        grid_lat=glat, grid_lon=glon,
        mrms_win=obs,
        stage4_win=None, s4_label="unavailable",
        swath=np.ones((4, 4), dtype=bool),
        cycles=[
            dict(init_str="2024092400", init_dt=datetime(2024, 9, 24, 0),
                 f1=48, f2=96, parent_win=obs - 1.0),
            dict(init_str="2024092500", init_dt=datetime(2024, 9, 25, 0),
                 f1=24, f2=72, parent_win=obs.copy()),
        ],
    )


def test_compute_cycles_writes_csv_and_pngs():
    import csv as csvmod
    from cycles import SUMMARY_FIELDS, compute_cycles
    with tempfile.TemporaryDirectory() as tmp:
        ccase = _tiny_cycles_case(tmp, tmp)
        ccase.ml_features = True
        ccase.ml_features_csv = Path(tmp) / "ml_features.csv"
        slug = "testcycles_2024092600_2024092800"
        errors_png = Path(tmp) / f"cycles_errors_{slug}.png"
        maps_png = Path(tmp) / f"cycles_maps_{slug}.png"
        landfall_png = Path(tmp) / f"cycles_landfall_{slug}.png"
        objects_png = Path(tmp) / f"cycles_objects_{slug}.png"
        errors_png.write_bytes(b"obsolete")
        maps_png.write_bytes(b"obsolete")
        landfall_png.write_bytes(b"obsolete")
        objects_png.write_bytes(b"obsolete")
        compute_cycles(ccase, fields=_tiny_cycle_fields())
        csv_path = Path(tmp) / f"cycles_{slug}.csv"
        metrics_png = Path(tmp) / f"cycles_metrics_{slug}.png"
        ets_png = Path(tmp) / f"cycles_ets_heatmap_{slug}.png"
        ets_bars_png = Path(tmp) / f"cycles_ets_bars_{slug}.png"
        fss_png = Path(tmp) / f"cycles_fss_heatmap_{slug}.png"
        fss_csv = Path(tmp) / f"cycles_fss_{slug}.csv"
        dist_csv = Path(tmp) / f"cycles_dist_{slug}.csv"
        dist_png = Path(tmp) / f"cycles_dist_{slug}.png"
        percentiles_png = Path(tmp) / f"cycles_percentiles_{slug}.png"
        summary_csv = Path(tmp) / f"cycles_summary_{slug}.csv"
        track_csv = Path(tmp) / f"cycles_track_{slug}.csv"
        features_csv = Path(tmp) / "ml_features.csv"
        assert csv_path.exists(), "CSV not written"
        assert metrics_png.exists(), "metrics PNG not written"
        assert ets_png.exists(), "ETS lead-time plot not written"
        assert ets_bars_png.exists(), "ETS threshold bars not written"
        assert fss_png.exists(), "FSS lead-time plot not written"
        assert not errors_png.exists(), "obsolete error maps not removed"
        assert not maps_png.exists(), "obsolete QPF maps not removed"
        assert not landfall_png.exists(), "obsolete landfall plot not removed"
        assert not objects_png.exists(), "obsolete objects plot not removed"
        assert fss_csv.exists(), "FSS CSV not written"
        assert dist_csv.exists(), "distribution CSV not written"
        assert dist_png.exists(), "distribution PNG not written"
        assert percentiles_png.exists(), "percentile PNG not written"
        assert summary_csv.exists(), "summary CSV not written"
        assert not track_csv.exists(), "track CSV written without best track"
        assert features_csv.exists(), "ML feature CSV not written"
        with open(csv_path) as fh:
            rows = list(csvmod.DictReader(fh))
        # 2 cycles x parent forecast x 1 obs (Stage IV None) x 1 threshold.
        assert len(rows) == 2
        assert rows[0].keys() == {
            "init", "valid_start", "valid_end", "lead_hours_to_landfall",
            "forecast", "observation",
            "threshold", "n", "rmse",
            "mae", "bias_mm", "r", "a", "b", "c", "d", "ets", "bias",
            "pod", "far", "csi", "hss"}
        by_key = {(r["init"], r["forecast"]): r for r in rows}
        early_parent = by_key[("2024092400", "parent")]
        late_parent = by_key[("2024092500", "parent")]
        assert early_parent["valid_start"] == "2024092600"
        assert early_parent["valid_end"] == "2024092800"
        assert all(r["observation"] == "MRMS" for r in rows)
        assert all(r["forecast"] == "parent" for r in rows)
        # Constant offsets: rmse == |bias_mm| (positive bias = over-forecast).
        assert abs(float(early_parent["rmse"]) - 1.0) < 1e-9
        assert abs(float(early_parent["bias_mm"]) - (-1.0)) < 1e-9
        assert abs(float(late_parent["rmse"]) - 0.0) < 1e-9
        assert int(early_parent["n"]) == 16
        # Perfect >= 1mm coverage everywhere -> ETS-relevant counts: all hits.
        assert int(early_parent["a"]) == 16 and int(early_parent["c"]) == 0
        with open(summary_csv) as fh:
            summary = list(csvmod.DictReader(fh))
        assert list(summary[0]) == SUMMARY_FIELDS
        assert len(summary) == 2
        assert float(summary[0]["rmse"]) == 1.0
        assert summary[0]["mean_track_err_km"] == ""
        with open(features_csv) as fh:
            features = list(csvmod.DictReader(fh))
        assert len(features) == 2


def test_pooled_ets_by_threshold_sums_counts_before_scoring():
    from cycles import pooled_ets_by_threshold

    results = [
        dict(init_dt=datetime(2024, 9, 24, 0), forecast="parent",
             observation="MRMS",
             rows=[dict(threshold=50.8, a=4, b=1, c=2, d=3)]),
        dict(init_dt=datetime(2024, 9, 24, 6), forecast="parent",
             observation="MRMS",
             rows=[dict(threshold=50.8, a=2, b=2, c=1, d=5)]),
    ]

    row = pooled_ets_by_threshold(results, [2])[0]
    assert row["threshold_mm"] == 50.8
    assert row["n_cycles"] == 2
    assert (row["a"], row["b"], row["c"], row["d"]) == (6, 3, 3, 8)
    assert abs(row["ets"] - 0.24528301886792447) < 1e-12


def test_separate_cycle_animations_write_gifs():
    from cycles import (
        animate_cycle_difference, animate_cycle_observed, animate_cycle_qpf,
    )
    with tempfile.TemporaryDirectory() as tmp:
        fields = _tiny_cycle_fields()
        animators = {
            "cycles_qpf.gif": animate_cycle_qpf,
            "cycles_difference.gif": animate_cycle_difference,
            "cycles_observed.gif": animate_cycle_observed,
        }
        for filename, animator in animators.items():
            output = Path(tmp) / filename
            animator(_tiny_cycles_case(tmp, tmp), fields, output)
            assert output.exists() and output.stat().st_size > 0


def test_precipitation_error_levels_widen_beyond_two_inches():
    from cycles import precipitation_error_levels

    levels = precipitation_error_levels([np.asarray([0.2, 3.0, 40.0])])
    assert np.allclose(levels, -levels[::-1])
    assert levels[-1] >= 40.0
    positive = levels[levels >= 0]
    high_steps = np.diff(positive[positive >= 2.0])
    assert np.all(high_steps[1:] >= high_steps[:-1])
    assert 24.0 in positive
    assert 0.1 in positive and 0.25 in positive and 0.5 in positive


def test_animation_qpf_levels_preserve_native_subinch_detail():
    from hafs_common import QPF_LEVELS
    from cycles import _ANIMATION_QPF_LEVELS_IN

    assert np.allclose(_ANIMATION_QPF_LEVELS_IN,
                       np.asarray(QPF_LEVELS, dtype=float) / 25.4)
    assert np.any((_ANIMATION_QPF_LEVELS_IN > 0)
                  & (_ANIMATION_QPF_LEVELS_IN < 0.5))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
