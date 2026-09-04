"""NCEP Stage IV hourly QPE access (``ST4.<YYYYMMDD>`` source files).

Distinct from parent_qpf.py's index_stage4_24h_conus/read_stage4, which read
the water.noaa.gov *daily* tarball's per-region 24h-only file
(``conus_YYYYMMDD_24h.grb2``). The files this module reads are NCEP's own
combined archive format: one file per day, bundling the day's 1h, 6h, and
24h APCP accumulations together as separate GRIB2 messages (confirmed via
``wgrib2 -s ST4.20240901``, which lists ``0-1 hour acc fcst`` /
``0-6 hour acc fcst`` / ``0-1 day acc fcst`` records side by side, and shows
each file leaking a few hours of the adjacent day too). There is no
per-hour filename to trust the way the 24h daily file's own date can be
trusted -- instead this reads each message's step length (1h) and
GRIB-decoded `valid_time` (reference time + step, i.e. accumulation END,
per the GRIB2 spec -- the same hour-end convention already used everywhere
else in this repo: MRMS, AORC, and the 24h Stage IV file) directly, so
mislabeling a hidden file-boundary convention can't silently mismatch an
hour.

Download is NOT handled here -- unlike MRMS/AORC, these files are fetched
by the user outside this repo's tooling, so every function here is
cache-only and raises FileNotFoundError on a miss rather than trying to
fetch anything.
"""

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cfgrib
import numpy as np

STAGE4_HOURLY_FILE_RE = re.compile(r"ST4\.(\d{8})$")


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


@lru_cache(maxsize=4)
def _stage4_1h_records(path_str):
    """[(valid_dt, lat2d, lon2d, data_mm), ...] for every 1-hour APCP
    message in one ST4.<day> file.

    cfgrib.open_datasets() automatically splits a file's incompatible
    step lengths (1h / 6h / 24h here) into separate internally-rectangular
    "hypercube" datasets -- we keep only the ones whose step is exactly
    1 hour and skip the 6h/24h messages bundled alongside them.
    Memoized (last 4 files) since a run touches the same day's file for
    every one of its ~24 hours.
    """
    path = Path(path_str)
    out = []
    for ds in cfgrib.open_datasets(str(path)):
        for var in ds.data_vars:
            da = ds[var]
            if "step" not in da.coords:
                continue
            steps = np.atleast_1d(da.step.values).astype("timedelta64[h]")
            if not np.all(steps == np.timedelta64(1, "h")):
                continue
            vtimes = np.atleast_1d(da.valid_time.values)
            lat = da.latitude.values
            lon = da.longitude.values
            lon = np.where(lon > 180, lon - 360, lon)
            vals = da.values
            if vals.ndim == 2:
                vals = vals[np.newaxis, ...]
            for k, vt in enumerate(vtimes):
                raw = vals[k]
                data = np.where(np.isnan(raw) | (raw < 0), 0.0, raw)
                out.append((_to_pydatetime(vt), lat, lon, data))
    return out


@lru_cache(maxsize=4)
def index_stage4_hourly(cache_dir_str):
    """Map valid datetime (hour-END) -> source ST4.<day> path, scanning
    every ST4.* file in cache_dir. Memoized per cache_dir -- obs-compare's
    cache is read-only for the duration of a run, so this is safe to
    build once and reuse across every hour requested.
    """
    idx = {}
    for path in stage4_hourly_files(cache_dir_str):
        for vt, _, _, _ in _stage4_1h_records(str(path)):
            idx.setdefault(vt, path)
    return idx


def load_stage4_hour(valid_dt, cache_dir):
    """(lat2d, lon2d, data_mm) for the 1-hour Stage IV accumulation ending
    at valid_dt. Cache-only: raises FileNotFoundError if no cached ST4.*
    file holds that hour -- this module never downloads anything."""
    cache_dir = Path(cache_dir)
    idx = index_stage4_hourly(str(cache_dir))
    path = idx.get(valid_dt)
    if path is None:
        raise FileNotFoundError(
            f"Stage IV hourly record not cached for {valid_dt} "
            f"(expected inside an ST4.<day> file under {cache_dir})")
    for vt, lat, lon, data in _stage4_1h_records(str(path)):
        if vt == valid_dt:
            return lat, lon, data
    raise FileNotFoundError(
        f"Stage IV hourly record for {valid_dt} indexed to {path} but not "
        f"found on re-read -- cache may have changed mid-run")
