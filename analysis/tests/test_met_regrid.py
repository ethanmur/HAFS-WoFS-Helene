"""Unit tests for met_regrid (no MET install, no network, no HPC data).

Synthetic GRIB2 is built with ecCodes, so message selection is tested
against real ecCodes decoding rather than a mock. regrid_data_plane itself
is replaced by tiny shell scripts where the subprocess plumbing is tested.

Run directly:   python3 analysis/tests/test_met_regrid.py
Or via pytest:  pytest analysis/tests/test_met_regrid.py -v
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eccodes
import netCDF4
import numpy as np

import met_regrid


def grib2_message(ni=6, nj=4, lat0=25.0, lon0=270.0, step=0.5, values=None,
                  valid_end=None, hours=1, unit=1, encoded_end=None):
    """One regular lat/lon GRIB2 message; an accumulation when valid_end set."""
    gid = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        for key, val in [("jScansPositively", 1), ("Ni", ni), ("Nj", nj),
                         ("latitudeOfFirstGridPointInDegrees", lat0),
                         ("longitudeOfFirstGridPointInDegrees", lon0),
                         ("latitudeOfLastGridPointInDegrees", lat0 + step * (nj - 1)),
                         ("longitudeOfLastGridPointInDegrees", lon0 + step * (ni - 1)),
                         ("iDirectionIncrementInDegrees", step),
                         ("jDirectionIncrementInDegrees", step)]:
            eccodes.codes_set(gid, key, val)
        if valid_end is not None:
            length_h = hours * {1: 1, 2: 24}[unit]
            ref = valid_end - timedelta(hours=length_h)
            end = encoded_end or valid_end
            for key, val in [("productDefinitionTemplateNumber", 8),
                             ("dataDate", int(f"{ref:%Y%m%d}")),
                             ("dataTime", int(f"{ref:%H%M}")),
                             ("indicatorOfUnitOfTimeRange", 1),
                             ("forecastTime", 0),
                             ("typeOfStatisticalProcessing", 1),
                             ("indicatorOfUnitForTimeRange", unit),
                             ("lengthOfTimeRange", hours),
                             ("yearOfEndOfOverallTimeInterval", end.year),
                             ("monthOfEndOfOverallTimeInterval", end.month),
                             ("dayOfEndOfOverallTimeInterval", end.day),
                             ("hourOfEndOfOverallTimeInterval", end.hour),
                             ("minuteOfEndOfOverallTimeInterval", 0),
                             ("secondOfEndOfOverallTimeInterval", 0)]:
                eccodes.codes_set(gid, key, val)
        vals = (np.arange(ni * nj, dtype=float) if values is None
                else np.asarray(values, dtype=float).ravel())
        eccodes.codes_set(gid, "bitmapPresent", 1)
        eccodes.codes_set(gid, "missingValue", 9999)
        eccodes.codes_set_values(gid, vals)
        return eccodes.codes_get_message(gid)
    finally:
        eccodes.codes_release(gid)


def _config(tmp, **overrides):
    kwargs = dict(grid_template=str(Path(tmp) / "*.grb2"),
                  cache_dir=Path(tmp) / "cache")
    kwargs.update(overrides)
    return met_regrid.RegridConfig(**kwargs)


def _executable(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def test_cell_areas_regular_grid_matches_spherical_band():
    step = 0.1
    lat1d = np.arange(29.05, 31.0, step)
    lon1d = np.arange(-85.95, -84.0, step)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    area = met_regrid.cell_areas_km2(lat2d, lon2d)
    r = met_regrid.R_EARTH_KM
    exact = (r ** 2 * np.radians(step)
             * (np.sin(np.radians(lat2d + step / 2))
                - np.sin(np.radians(lat2d - step / 2))))
    assert np.allclose(area, exact, rtol=1e-3)


def test_cell_areas_ignore_scan_direction_and_axis_order():
    lon2d, lat2d = np.meshgrid(np.arange(-90, -85, 0.5), np.arange(20, 25, 0.5))
    base = met_regrid.cell_areas_km2(lat2d, lon2d)
    assert np.allclose(met_regrid.cell_areas_km2(lat2d[::-1], lon2d[::-1]),
                       base[::-1])
    assert np.allclose(met_regrid.cell_areas_km2(lat2d.T, lon2d.T), base.T)
    assert (base > 0).all()


def _target_grid():
    lon2d, lat2d = np.meshgrid(np.arange(-85.975, -85.0, 0.05),
                               np.arange(25.025, 26.0, 0.05))
    return met_regrid.TargetGrid(lat2d, lon2d)


def test_target_assign_rejects_points_beyond_edge():
    target = _target_grid()
    cell = target.assign(np.array([25.51, 26.1, 40.0]),
                         np.array([-85.49, -85.5, -85.5]))
    assert cell[0] >= 0      # inside
    assert cell[1] == -1     # just past the north edge
    assert cell[2] == -1     # far outside (skipped by the bbox prefilter)


def test_target_assign_accepts_0_360_longitudes():
    target = _target_grid()
    west = target.assign([25.5], [-85.5])
    east = target.assign([25.5], [360.0 - 85.5])
    assert west[0] == east[0] >= 0


def test_fraction_outside():
    target = _target_grid()
    assert target.fraction_outside(25.1, 25.9, -85.9, -85.1) == 0.0
    assert 0.4 < target.fraction_outside(25.1, 26.9, -85.9, -85.1) < 0.6


# ----------------------------------------------------------------------------
# Conservation check
# ----------------------------------------------------------------------------

def _fine_native(seed=0):
    """0.01 deg field nested exactly 5x5 inside each 0.05 deg target cell."""
    rng = np.random.default_rng(seed)
    lon2d, lat2d = np.meshgrid(np.arange(-85.995, -85.0, 0.01),
                               np.arange(25.005, 26.0, 0.01))
    vals = rng.gamma(0.8, 3.0, size=lat2d.shape)
    return lat2d, lon2d, vals


def _block_mean(lat2d, vals):
    """Area-weighted 5x5 block means, computed independently of met_regrid."""
    w = np.cos(np.radians(lat2d))
    ny, nx = vals.shape
    block = lambda a: np.nansum(a.reshape(ny // 5, 5, nx // 5, 5), axis=(1, 3))
    return block(np.where(np.isfinite(vals), vals * w, 0)) / block(
        np.where(np.isfinite(vals), w, 0))


def test_budget_row_exact_box_average_is_not_flagged():
    target = _target_grid()
    lat, lon, vals = _fine_native()
    grid = met_regrid.NativeGrid.build(lat, lon, target)
    row = met_regrid.budget_row(target, _block_mean(lat, vals), vals, grid, 2.0)
    assert abs(row["mean_pct_diff"]) < 1e-3
    assert row["flag"] == ""
    assert row["native_max_mm"] >= row["regrid_max_mm"]   # peaks smooth out


def test_budget_row_flags_lost_rain():
    target = _target_grid()
    lat, lon, vals = _fine_native()
    grid = met_regrid.NativeGrid.build(lat, lon, target)
    row = met_regrid.budget_row(target, 0.95 * _block_mean(lat, vals), vals,
                                grid, 2.0)
    assert np.isclose(row["mean_pct_diff"], -5.0, atol=0.01)
    assert row["flag"] == "CHECK"


def test_budget_row_dry_hour_is_not_flagged():
    target = _target_grid()
    lat, lon, _ = _fine_native()
    grid = met_regrid.NativeGrid.build(lat, lon, target)
    native = np.zeros(lat.shape)
    native[0, 0] = 1e-4                          # a trace of rain somewhere
    row = met_regrid.budget_row(target, np.zeros(target.shape), native, grid, 2.0)
    assert row["flag"] == ""                     # tiny absolute miss: below floor


def test_budget_row_excludes_mostly_uncovered_cells():
    target = _target_grid()
    lat, lon, vals = _fine_native()
    vals[:, :3] = np.nan          # westmost target column only 2/5 covered
    grid = met_regrid.NativeGrid.build(lat, lon, target)
    full = met_regrid.budget_row(target, _block_mean(*_fine_native()[::2]),
                                 _fine_native()[2], grid, 2.0)
    row = met_regrid.budget_row(target, _block_mean(lat, vals), vals, grid, 2.0)
    assert row["common_area_km2"] < full["common_area_km2"]
    assert np.isclose(row["common_area_km2"] / full["common_area_km2"],
                      19 / 20, rtol=0.01)
    assert abs(row["mean_pct_diff"]) < 1e-3


# ----------------------------------------------------------------------------
# GRIB message selection (real ecCodes)
# ----------------------------------------------------------------------------

def test_extract_accumulation_takes_1h_finest_grid_never_24h():
    t = datetime(2024, 9, 24, 2)
    noon = datetime(2024, 9, 24, 12)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ST4.20240924"
        path.write_bytes(
            grib2_message(ni=3, nj=2, valid_end=t)                   # coarse 1h
            + grib2_message(ni=6, nj=4, valid_end=t)                 # fine 1h
            + grib2_message(ni=6, nj=4, valid_end=t, hours=6)        # 6h
            + grib2_message(ni=6, nj=4, valid_end=noon, unit=2)      # "0-1 day"
            + grib2_message(ni=6, nj=4, valid_end=t + timedelta(hours=1)))
        msg = met_regrid.extract_accumulation([path], t, hours=1)
        _, _, vals = met_regrid.read_grib_message(msg, want_latlon=False)
        assert vals.shape == (4, 6)
        # lengthOfTimeRange is 1 for the daily record too; it must not match.
        assert met_regrid.extract_accumulation([path], noon, hours=1) is None
        assert met_regrid.extract_accumulation([path], noon, hours=24) is not None


def test_accumulation_rejects_inconsistent_time_encoding():
    t = datetime(2024, 9, 24, 2)
    msg = grib2_message(valid_end=t, encoded_end=t - timedelta(hours=1))
    gid = eccodes.codes_new_from_message(msg)
    try:
        met_regrid.accumulation(gid)
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "inconsistent" in str(e)
    finally:
        eccodes.codes_release(gid)


def test_read_grib_message_masks_bitmap_and_negatives():
    vals = np.arange(24, dtype=float)
    vals[0] = 9999          # bitmap-missing
    vals[1] = -3            # MRMS-style no-coverage flag
    lat, lon, out = met_regrid.read_grib_message(grib2_message(values=vals))
    assert out.shape == lat.shape == lon.shape == (4, 6)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 1])
    assert out[0, 2] == 2
    assert np.isclose(lon[0, 0], -90.0)          # 270E wrapped to -90


def test_ensure_grid_template_copies_first_message_once():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "run" / "x.2024092400.parent.atm.f006.grb2"
        src.parent.mkdir()
        src.write_bytes(grib2_message(ni=6, nj=4) + grib2_message(ni=3, nj=2))
        cfg = _config(tmp, grid_template=str(Path(tmp) / "**" / "*parent.atm.f*.grb2"))
        small = met_regrid.ensure_grid_template(cfg)
        assert sum(1 for _ in met_regrid._grib_messages(small)) == 1
        _, _, vals = met_regrid.read_grib_field(small, want_latlon=False)
        assert vals.shape == (4, 6)
        assert str(src) in (cfg.grid_dir / "grid_template_source.txt").read_text()
        src.unlink()
        assert met_regrid.ensure_grid_template(cfg) == small   # no re-read


# ----------------------------------------------------------------------------
# NetCDF in and out
# ----------------------------------------------------------------------------

def test_write_cf_netcdf_has_cf_coordinates_and_fill():
    with tempfile.TemporaryDirectory() as tmp:
        data = np.array([[1.0, np.nan], [2.0, 3.0]])
        path = met_regrid.write_cf_netcdf([25.0, 25.01], [-85.0, -84.99], data,
                                          Path(tmp) / "aorc.nc")
        with netCDF4.Dataset(path) as nc:
            assert nc.variables["lat"].units == "degrees_north"
            assert nc.variables["lon"].units == "degrees_east"
            var = nc.variables["APCP_surface"]
            assert var._FillValue == met_regrid.MET_MISSING
            assert np.ma.is_masked(var[:][0, 1])


def test_read_regridded_opens_met_style_layout():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "met.nc"
        lon2d, lat2d = np.meshgrid([275.0, 276.0, 277.0], [25.0, 26.0])
        with netCDF4.Dataset(path, "w") as nc:
            nc.createDimension("lat", 2)
            nc.createDimension("lon", 3)
            # 2-D lat/lon named after their own dimensions, as MET writes them
            nc.createVariable("lat", "f4", ("lat", "lon"))[:] = lat2d
            nc.createVariable("lon", "f4", ("lat", "lon"))[:] = lon2d
            v = nc.createVariable("precip", "f4", ("lat", "lon"),
                                  fill_value=met_regrid.MET_MISSING)
            v[:] = np.array([[1.0, -9999.0, 2.0], [0.0, 3.0, 4.0]])
        lat, lon, vals = met_regrid.read_regridded(path)
        assert lat.shape == lon.shape == vals.shape == (2, 3)
        assert np.isnan(vals[0, 1]) and vals[1, 2] == 4.0
        assert np.isclose(lon[0, 0], -85.0)


# ----------------------------------------------------------------------------
# MET invocation
# ----------------------------------------------------------------------------

def test_regrid_command_argv():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp)
        cmd = met_regrid.regrid_command("rdp", "in.grb2", "grid.grb2", "out.nc",
                                        cfg.field_spec("stage4"), cfg)
    assert cmd[:4] == ["rdp", "in.grb2", "grid.grb2", "out.nc"]
    opts = dict(zip(cmd[4::2], cmd[5::2]))
    assert opts["-method"] == "BUDGET" and opts["-width"] == "2"
    assert opts["-vld_thresh"] == "0.5" and opts["-name"] == "precip"
    assert 'level="A1"' in opts["-field"] and "censor_thresh" in opts["-field"]


def test_run_regrid_success_is_atomic():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tool = _executable(tmp / "rdp", 'printf ok > "$3"\n')
        out = tmp / "cache" / "stage4" / "stage4_2024092402.nc"
        met_regrid.run_regrid(tool, tmp / "in", tmp / "grid", out, "f", _config(tmp))
        assert out.read_text() == "ok"
        assert [p.name for p in out.parent.iterdir()] == [out.name]


def test_run_regrid_failure_reports_met_log_and_leaves_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tool = _executable(tmp / "rdp", 'echo "ERROR  : no matching field" >&2\n'
                                        'printf partial > "$3"\nexit 1\n')
        out = tmp / "cache" / "stage4_2024092402.nc"
        try:
            met_regrid.run_regrid(tool, tmp / "in", tmp / "grid", out, "f",
                                  _config(tmp))
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "no matching field" in str(e) and "exit 1" in str(e)
        assert list(out.parent.iterdir()) == []


def test_met_tool_missing_mentions_module_load(monkeypatch):
    monkeypatch.setattr(met_regrid.shutil, "which", lambda name: None)
    try:
        met_regrid.met_tool("regrid_data_plane")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "module load met" in str(e)


def test_met_tool_prefers_bin_dir():
    with tempfile.TemporaryDirectory() as tmp:
        tool = _executable(Path(tmp) / "regrid_data_plane", "exit 0\n")
        assert met_regrid.met_tool("regrid_data_plane", tmp) == tool


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

def test_regrid_config_defaults_and_cache_layout():
    assert met_regrid.regrid_config_from_dict(None) is None
    cfg = met_regrid.regrid_config_from_dict(
        {"grid_template": "/x/*.grb2", "cache_dir": "/c", "method": "budget",
         "fields": {"mrms": 'name="X"; level="Z0";'}})
    assert cfg.method == "BUDGET" and cfg.width == 2 and cfg.vld_thresh == 0.5
    assert cfg.output_path("stage4", datetime(2024, 9, 24, 2)) == Path(
        "/c/hafs_parent_budget/stage4/stage4_2024092402.nc")
    assert cfg.field_spec("mrms") == 'name="X"; level="Z0";'
    assert cfg.field_spec("aorc") == met_regrid.DEFAULT_FIELDS["aorc"]


def test_regrid_config_rejects_unknown_field_source():
    try:
        met_regrid.regrid_config_from_dict(
            {"grid_template": "x", "cache_dir": "c", "fields": {"st4": "..."}})
        assert False, "expected KeyError"
    except KeyError as e:
        assert "st4" in str(e)


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
    os.environ.setdefault("ECCODES_LOG", "0")
    _run_all()
