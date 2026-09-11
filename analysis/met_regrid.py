"""Regrid observed precipitation onto a target grid with MET regrid_data_plane."""

import glob
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import eccodes
import netCDF4
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

R_EARTH_KM = 6371.0
MET_MISSING = -9999.0
OUTPUT_NAME = "precip"
SOURCES = ("mrms", "stage4", "aorc")
# Dry hours make percent differences explode from tiny absolute errors.
ABS_FLOOR_MM = 0.01

# Censoring turns negative no-coverage flags (MRMS uses -3) into missing
# instead of regridding them as negative rain.
_CENSOR = " censor_thresh=[ <0 ]; censor_val=[ -9999 ];"
DEFAULT_FIELDS = {
    "mrms": 'name="MultiSensor_QPE_01H_Pass2"; level="Z0";' + _CENSOR,
    "stage4": 'name="APCP"; level="A1";' + _CENSOR,
    "aorc": 'name="APCP_surface"; level="(*,*)";' + _CENSOR,
}


@dataclass
class RegridConfig:
    grid_template: str
    cache_dir: Path
    grid_name: str = "hafs_parent"
    method: str = "BUDGET"
    width: int = 2
    vld_thresh: float = 0.5
    tolerance_pct: float = 2.0
    met_bin_dir: Optional[Path] = None
    fields: dict = field(default_factory=dict)

    @property
    def grid_dir(self):
        # Grid and method in the path so a change never mixes two regrids.
        return self.cache_dir / f"{self.grid_name}_{self.method.lower()}"

    def field_spec(self, source):
        return self.fields.get(source, DEFAULT_FIELDS[source])

    def output_path(self, source, valid_dt):
        return self.grid_dir / source / f"{source}_{valid_dt:%Y%m%d%H}.nc"


def regrid_config_from_dict(cfg):
    """RegridConfig from a YAML `regrid:` block, or None when absent."""
    if not cfg:
        return None
    for key in ("grid_template", "cache_dir"):
        if key not in cfg:
            raise KeyError(f"'regrid.{key}' is required")
    fields = dict(cfg.get("fields") or {})
    unknown = sorted(set(fields) - set(SOURCES))
    if unknown:
        raise KeyError(f"regrid.fields has unknown source(s) {unknown}; "
                       f"expected any of {SOURCES}")
    return RegridConfig(
        grid_template=str(cfg["grid_template"]),
        cache_dir=Path(cfg["cache_dir"]),
        grid_name=str(cfg.get("grid_name", "hafs_parent")),
        method=str(cfg.get("method", "BUDGET")).upper(),
        width=int(cfg.get("width", 2)),
        vld_thresh=float(cfg.get("vld_thresh", 0.5)),
        tolerance_pct=float(cfg.get("tolerance_pct", 2.0)),
        met_bin_dir=Path(cfg["met_bin_dir"]) if cfg.get("met_bin_dir") else None,
        fields=fields,
    )


# =============================================================================
# MET invocation
# =============================================================================

def met_tool(name, bin_dir=None):
    """Absolute path to a MET executable, from bin_dir or else PATH."""
    if bin_dir is not None:
        path = Path(bin_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"{name} not found in met_bin_dir {bin_dir}")
        return path
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"{name} is not on PATH -- run `module load met/12.2.0` first, "
            f"or set regrid.met_bin_dir in the YAML")
    return Path(found)


def regrid_command(tool, input_path, grid_path, out_path, field_spec, config):
    return [str(tool), str(input_path), str(grid_path), str(out_path),
            "-field", field_spec,
            "-method", config.method,
            "-width", str(config.width),
            "-vld_thresh", str(config.vld_thresh),
            "-name", OUTPUT_NAME,
            "-v", "1"]


def _tmp_path(path):
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def run_regrid(tool, input_path, grid_path, out_path, field_spec, config):
    """Run regrid_data_plane, writing out_path only on success."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(out_path)
    cmd = regrid_command(tool, input_path, grid_path, tmp, field_spec, config)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not tmp.exists():
            log = (proc.stderr + proc.stdout).strip().splitlines()
            raise RuntimeError(
                f"regrid_data_plane failed (exit {proc.returncode}):\n  "
                + shlex.join(cmd) + "\n  " + "\n  ".join(log[-15:]))
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)


def ensure_grid_template(config):
    """Single-message copy of the template's grid, cut once and reused."""
    small = config.grid_dir / "grid_template.grb2"
    if small.exists():
        return small
    hits = sorted(glob.glob(config.grid_template, recursive=True))
    if not hits:
        raise FileNotFoundError(
            f"regrid.grid_template matched no files: {config.grid_template}")
    src = Path(hits[0])
    # MET only needs the grid; re-reading a multi-GB, hundreds-of-records
    # parent.atm file on every call would dominate the runtime.
    with open(src, "rb") as fh:
        gid = eccodes.codes_grib_new_from_file(fh)
    if gid is None:
        raise ValueError(f"{src} contains no GRIB messages")
    try:
        msg = eccodes.codes_get_message(gid)
    finally:
        eccodes.codes_release(gid)
    write_bytes(small, msg)
    (config.grid_dir / "grid_template_source.txt").write_text(f"{src}\n")
    return small


# =============================================================================
# Staging: one unambiguous field per file for MET
# =============================================================================

def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def _grib_messages(path):
    with open(path, "rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                return
            try:
                yield gid
            finally:
                eccodes.codes_release(gid)


def _grib_datetime(date, hhmm):
    return datetime.strptime(f"{date:08d}{hhmm:04d}", "%Y%m%d%H%M")


def accumulation(gid):
    """(valid_end, hours) of a GRIB2 accumulation message, else None."""
    try:
        if eccodes.codes_get(gid, "typeOfStatisticalProcessing") != 1:
            return None
    except eccodes.KeyValueNotFoundError:
        return None
    # For PDT 4.8 validityDate/Time is the separately encoded end-of-interval.
    # Read it before setting stepUnits, which rewrites those keys from
    # reference + step and would make the consistency check below vacuous.
    encoded = _grib_datetime(eccodes.codes_get(gid, "validityDate"),
                             eccodes.codes_get(gid, "validityTime"))
    # Hours, so a "0-1 day" record reads as 24 rather than matching 1h.
    eccodes.codes_set(gid, "stepUnits", 1)
    start = eccodes.codes_get(gid, "startStep")
    end = eccodes.codes_get(gid, "endStep")
    ref = _grib_datetime(eccodes.codes_get(gid, "dataDate"),
                         eccodes.codes_get(gid, "dataTime"))
    # reference + endStep is cfgrib's valid_time, which stage4_hourly indexes.
    valid_end = ref + timedelta(hours=end)
    if encoded != valid_end:
        raise ValueError(
            f"inconsistent GRIB time encoding: reference {ref} + {end}h = "
            f"{valid_end}, but end-of-interval keys say {encoded}")
    return valid_end, end - start


def extract_accumulation(paths, valid_end, hours):
    """Bytes of the finest-grid `hours` accumulation ending valid_end."""
    best = None
    for path in paths:
        for gid in _grib_messages(path):
            if accumulation(gid) != (valid_end, hours):
                continue
            # ST4.<day> files repeat each field on a second, coarser grid.
            npts = eccodes.codes_get(gid, "numberOfDataPoints")
            if best is None or npts > best[0]:
                best = (npts, eccodes.codes_get_message(gid))
    return None if best is None else best[1]


def write_cf_netcdf(lat1d, lon1d, data, out_path, var="APCP_surface"):
    """Lat/lon NetCDF with the CF coordinate attributes MET keys off."""
    ds = xr.Dataset(
        {var: (("lat", "lon"), np.asarray(data, dtype="float32"),
               {"units": "kg m-2",
                "long_name": "1-hour accumulated precipitation"})},
        coords={
            "lat": ("lat", np.asarray(lat1d, dtype=float),
                    {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", np.asarray(lon1d, dtype=float),
                    {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={"Conventions": "CF-1.6"},
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(out_path)
    ds.to_netcdf(tmp, encoding={var: {"_FillValue": MET_MISSING}})
    tmp.replace(out_path)
    return out_path


# =============================================================================
# Reading native and regridded fields
# =============================================================================

def _wrap_lon(lon):
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def _grid_shape(gid):
    npts = eccodes.codes_get(gid, "numberOfDataPoints")
    for rows, cols in (("Nj", "Ni"), ("Ny", "Nx")):
        if eccodes.codes_is_defined(gid, rows) and eccodes.codes_is_defined(gid, cols):
            shape = (eccodes.codes_get(gid, rows), eccodes.codes_get(gid, cols))
            if shape[0] * shape[1] == npts:
                return shape
    raise ValueError(f"can't determine a 2-D shape for a {npts}-point grid")


def read_grib_message(msg, want_latlon=True):
    """(lat2d, lon2d, values2d) from GRIB bytes; missing/negative -> NaN."""
    gid = eccodes.codes_new_from_message(msg)
    try:
        shape = _grid_shape(gid)
        values = np.asarray(eccodes.codes_get_values(gid), dtype=float)
        if eccodes.codes_get(gid, "bitmapPresent"):
            values[values == eccodes.codes_get(gid, "missingValue")] = np.nan
        values[values < 0] = np.nan
        lat = lon = None
        if want_latlon:
            lat = eccodes.codes_get_array(gid, "latitudes").reshape(shape)
            lon = _wrap_lon(eccodes.codes_get_array(gid, "longitudes")).reshape(shape)
        return lat, lon, values.reshape(shape)
    finally:
        eccodes.codes_release(gid)


def read_grib_field(path, want_latlon=True):
    """read_grib_message on the first message of a GRIB file."""
    with open(path, "rb") as fh:
        gid = eccodes.codes_grib_new_from_file(fh)
    if gid is None:
        raise ValueError(f"{path} contains no GRIB messages")
    try:
        msg = eccodes.codes_get_message(gid)
    finally:
        eccodes.codes_release(gid)
    return read_grib_message(msg, want_latlon)


def read_regridded(path, name=OUTPUT_NAME):
    """(lat2d, lon2d, values2d) from a regrid_data_plane output file."""
    # netCDF4 rather than xarray: MET names its 2-D lat/lon variables after
    # their own dimensions, which xarray refuses to open.
    with netCDF4.Dataset(path) as nc:
        values = np.ma.filled(nc.variables[name][:].astype(float), np.nan)
        lat = np.asarray(nc.variables["lat"][:], dtype=float)
        lon = np.asarray(nc.variables["lon"][:], dtype=float)
    values[values <= MET_MISSING + 1] = np.nan
    if lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    return lat, _wrap_lon(lon), values


# =============================================================================
# Conservation check
# =============================================================================

def cell_areas_km2(lat2d, lon2d):
    """Per-cell area (km^2) of any structured grid, regular or curvilinear."""
    lat = np.radians(np.asarray(lat2d, dtype=float))
    lon = np.radians(np.asarray(lon2d, dtype=float))
    lon = np.unwrap(np.unwrap(lon, axis=1), axis=0)
    dlat_r, dlat_c = np.gradient(lat)
    dlon_r, dlon_c = np.gradient(lon)
    coslat = np.cos(lat)
    east_r, north_r = R_EARTH_KM * coslat * dlon_r, R_EARTH_KM * dlat_r
    east_c, north_c = R_EARTH_KM * coslat * dlon_c, R_EARTH_KM * dlat_c
    return np.abs(east_r * north_c - east_c * north_r)


def _unit_vectors(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    coslat = np.cos(lat)
    return np.column_stack([coslat * np.cos(lon), coslat * np.sin(lon),
                            np.sin(lat)])


class TargetGrid:
    """The regrid target plus a point -> containing-cell lookup."""

    def __init__(self, lat2d, lon2d):
        self.shape = lat2d.shape
        self.area = cell_areas_km2(lat2d, lon2d).ravel()
        self._bbox = (lat2d.min(), lat2d.max(), lon2d.min(), lon2d.max())
        self._tree = cKDTree(_unit_vectors(lat2d.ravel(), lon2d.ravel()))

    def assign(self, lat, lon):
        """Target cell index per point, -1 for points outside the grid."""
        lat = np.ravel(np.asarray(lat, dtype=float))
        lon = _wrap_lon(np.ravel(lon))
        lat_min, lat_max, lon_min, lon_max = self._bbox
        margin = 1.0
        near = ((lat >= lat_min - margin) & (lat <= lat_max + margin)
                & (lon >= lon_min - margin) & (lon <= lon_max + margin))
        cell = np.full(lat.size, -1, dtype=np.int64)
        if near.any():
            dist, idx = self._tree.query(_unit_vectors(lat[near], lon[near]),
                                         workers=-1)
            # The nearest centre is the containing cell everywhere except
            # past the outer edge, where it's still an edge cell.
            half_diag = 0.75 * np.sqrt(self.area[idx]) / R_EARTH_KM
            cell[near] = np.where(dist <= half_diag, idx, -1)
        return cell

    def fraction_outside(self, lat_min, lat_max, lon_min, lon_max, n=60):
        lat, lon = np.meshgrid(np.linspace(lat_min, lat_max, n),
                               np.linspace(lon_min, lon_max, n))
        return float(np.mean(self.assign(lat, lon) < 0))


@dataclass
class NativeGrid:
    """A source grid's per-point area and target cell, built once per run."""
    shape: tuple
    area: np.ndarray
    cell: np.ndarray

    @classmethod
    def build(cls, lat2d, lon2d, target):
        return cls(tuple(lat2d.shape), cell_areas_km2(lat2d, lon2d).ravel(),
                   target.assign(lat2d, lon2d))


def budget_row(target, regridded, native, grid, tolerance_pct):
    """MET's area-weighted mean vs an exact box average of the native field."""
    ncell = target.area.size
    reg = np.asarray(regridded, dtype=float).ravel()
    vals = np.asarray(native, dtype=float).ravel()
    ok = np.isfinite(vals) & (grid.cell >= 0)
    cell = grid.cell[ok]
    cover = np.bincount(cell, weights=grid.area[ok], minlength=ncell)
    mass = np.bincount(cell, weights=vals[ok] * grid.area[ok], minlength=ncell)
    # Compare only where MET produced a value and the native field covers at
    # least half the cell, so partial edge cells don't masquerade as loss.
    common = np.isfinite(reg) & (cover >= 0.5 * target.area)
    area = target.area[common]
    total = float(area.sum())
    row = {
        "common_area_km2": round(total),
        "native_valid_points": int(ok.sum()),
        "regrid_valid_cells": int(np.isfinite(reg).sum()),
    }
    if total == 0:
        return {**row, "native_mean_mm": np.nan, "regrid_mean_mm": np.nan,
                "mean_diff_mm": np.nan, "mean_pct_diff": np.nan,
                "native_max_mm": np.nan, "regrid_max_mm": np.nan,
                "flag": "NO_OVERLAP"}
    native_mean = float((mass[common] / cover[common] * area).sum() / total)
    regrid_mean = float((reg[common] * area).sum() / total)
    diff = regrid_mean - native_mean
    if native_mean > 0:
        pct = 100.0 * diff / native_mean
    else:
        pct = 0.0 if regrid_mean == 0 else np.nan
    in_common = np.zeros(vals.size, dtype=bool)
    in_common[np.flatnonzero(ok)] = common[cell]
    flagged = abs(diff) > ABS_FLOOR_MM and not abs(pct) <= tolerance_pct
    return {
        **row,
        "native_mean_mm": round(native_mean, 5),
        "regrid_mean_mm": round(regrid_mean, 5),
        "mean_diff_mm": round(diff, 5),
        "mean_pct_diff": round(pct, 3) if np.isfinite(pct) else np.nan,
        "native_max_mm": round(float(vals[in_common].max()), 3)
                         if in_common.any() else np.nan,
        "regrid_max_mm": round(float(reg[common].max()), 3),
        "flag": "CHECK" if flagged else "",
    }
