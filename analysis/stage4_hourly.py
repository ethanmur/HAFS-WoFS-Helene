"""NCEP Stage IV hourly QPE access (``ST4.<YYYYMMDD>`` source files).

Distinct from parent_qpf.py's index_stage4_24h_conus/read_stage4, which read
the water.noaa.gov *daily* tarball's per-region 24h-only file
(``conus_YYYYMMDD_24h.grb2``). The files this module reads are NCEP's own
combined archive format: one file per day, bundling the day's 1h, 6h, and
24h APCP accumulations together, and repeating them on more than one grid.

cfgrib does NOT split those by accumulation length. It builds one sparse
``(time x step)`` hypercube per grid, with step holding [1h, 6h, 24h]
together and NaN filling every ``(time, step)`` cell that has no real
message -- so getting the hourly field means selecting the 1h step slice
out of a mixed dataset and discarding the NaN-fill cells, not looking for
a 1h-only dataset. A real file (ST4.20240924) yields three datasets: one
with steps [6h, 24h] and no hourly data at all, and two with [1h, 6h, 24h]
on grids of different sizes.

Hours are keyed by each message's GRIB-decoded `valid_time` (reference
time + step, i.e. accumulation END, per the GRIB2 spec -- the same
hour-end convention used everywhere else in this repo: MRMS, AORC, and
the 24h Stage IV file), so a hidden file-boundary convention can't
silently mismatch an hour; a file's hours are found wherever they
physically live, including the adjacent-day hours each file carries.

Download is NOT handled here -- unlike MRMS/AORC, these files are fetched
by the user outside this repo's tooling, so every function here is
cache-only and raises FileNotFoundError on a miss rather than trying to
fetch anything.
"""

import re
from functools import lru_cache
from pathlib import Path

import cfgrib
import numpy as np

STAGE4_HOURLY_FILE_RE = re.compile(r"ST4\.(\d{8})$")
ONE_HOUR = np.timedelta64(1, "h")


def stage4_hourly_files(cache_dir):
    """Every ``ST4.<YYYYMMDD>`` source file present in cache_dir, sorted by
    date. Pure path glob -- no I/O beyond a directory listing."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    return sorted(
        p for p in cache_dir.glob("ST4.*") if STAGE4_HOURLY_FILE_RE.match(p.name)
    )


def _to_pydatetime(dt64):
    return dt64.astype("datetime64[s]").item()


def _one_hour_slice(da):
    """The step==1h slice of a cfgrib DataArray, or None if it holds no
    hourly accumulation. Selects positionally so the timedelta comparison
    happens once, on plain numpy values."""
    step = da.coords.get("step")
    if step is None:
        return None
    steps = np.atleast_1d(step.values).astype("timedelta64[h]")
    hits = np.nonzero(steps == ONE_HOUR)[0]
    if hits.size == 0:
        return None
    if "step" in da.dims:
        da = da.isel(step=int(hits[0]))
    return da


@lru_cache(maxsize=64)
def _stage4_1h_valid_times(path_str):
    """Hour-END datetimes of every 1h cell in one ST4.<day> file.

    Coordinate metadata only -- never touches the data payload, so
    indexing a whole month of files stays cheap. Some of these cells are
    only NaN hypercube fill; that can't be told apart without the payload,
    so it's left to read time (see _stage4_1h_records).
    """
    out = set()
    for ds in cfgrib.open_datasets(path_str):
        for var in ds.data_vars:
            da = _one_hour_slice(ds[var])
            if da is None:
                continue
            for vt in np.atleast_1d(da.valid_time.values):
                out.add(_to_pydatetime(vt))
    return tuple(sorted(out))


@lru_cache(maxsize=2)
def _stage4_1h_records(path_str):
    """{valid_dt: (lat2d, lon2d, data_mm)} for one ST4.<day> file's real 1h
    accumulations.

    Loads the payload, hence a small cache -- a run works through all ~24
    hours of one day before moving to the next. Two things the raw
    hypercube requires: all-NaN cells are dropped (they are fill for
    (time, step) combinations that hold no message, and would otherwise
    become a field of zeros, i.e. a fabricated dry hour), and where the
    same hour appears on more than one grid the finest is kept, matching
    parent_qpf.pick_cumulative_record's convention.
    """
    best = {}
    for ds in cfgrib.open_datasets(path_str):
        for var in ds.data_vars:
            da = _one_hour_slice(ds[var])
            if da is None:
                continue
            lat = da.latitude.values
            lon = da.longitude.values
            lon = np.where(lon > 180, lon - 360, lon)
            vtimes = np.atleast_1d(da.valid_time.values)
            vals = da.values
            if vals.ndim == 2:
                vals = vals[np.newaxis, ...]
            for k, vt in enumerate(vtimes):
                raw = vals[k]
                if np.all(np.isnan(raw)):
                    continue
                key = _to_pydatetime(vt)
                if key in best and best[key][2].size >= raw.size:
                    continue
                data = np.where(np.isnan(raw) | (raw < 0), 0.0, raw)
                best[key] = (lat, lon, data.astype(np.float32, copy=False))
    return best


@lru_cache(maxsize=4)
def index_stage4_hourly(cache_dir_str):
    """Map valid datetime (hour-END) -> [candidate ST4.<day> paths], across
    every ST4.* file in cache_dir.

    An hour can be listed by more than one file (each carries some of the
    adjacent day), and a listed cell can still turn out to be NaN fill, so
    this keeps every candidate for load_stage4_hour to try in turn.
    Memoized per cache_dir -- the cache is read-only for the duration of a
    run, so it's built once and reused for every hour requested.
    """
    idx = {}
    for path in stage4_hourly_files(cache_dir_str):
        for vt in _stage4_1h_valid_times(str(path)):
            idx.setdefault(vt, []).append(path)
    return idx


def load_stage4_hour(valid_dt, cache_dir):
    """(lat2d, lon2d, data_mm) for the 1-hour Stage IV accumulation ending
    at valid_dt. Cache-only: raises FileNotFoundError if no cached ST4.*
    file holds that hour -- this module never downloads anything."""
    cache_dir = Path(cache_dir)
    candidates = index_stage4_hourly(str(cache_dir)).get(valid_dt, [])
    if not candidates:
        raise FileNotFoundError(
            f"Stage IV hourly record not cached for {valid_dt} "
            f"(expected inside an ST4.<day> file under {cache_dir})")
    for path in candidates:
        record = _stage4_1h_records(str(path)).get(valid_dt)
        if record is not None:
            return record
    raise FileNotFoundError(
        f"Stage IV {valid_dt} is listed by "
        f"{', '.join(p.name for p in candidates)} but holds no data in any "
        f"of them (empty hypercube cell, no real 1h message)")
