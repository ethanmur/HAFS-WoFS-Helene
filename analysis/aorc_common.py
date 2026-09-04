"""NOAA AORC v1.1 precipitation access (1-km hourly).

The public bucket ``s3://noaa-nws-aorc-v1-1-1km`` holds one Zarr store per
year (``YYYY.zarr``) rather than per-timestep files -- opened lazily over
s3fs (anonymous, no AWS account needed), sliced to the requested domain and
hour, and cached locally as a small per-hour NetCDF so repeat runs don't
re-touch S3 at all.

``APCP_surface`` is in kg/m^2, numerically identical to mm -- same
convention as the HAFS/MRMS/Stage IV fields elsewhere in this repo, so no
unit conversion is needed anywhere downstream.

The S3 filesystem and per-year zarr stores are opened lazily and only on a
cache miss with `download=True` (the default) -- call sites that pass
`download=False` (obs_compare's comparison path, as opposed to its
download-obs path) never construct either, so they stay usable on a
no-internet compute node as long as the cache is already populated.

Registry: https://registry.opendata.aws/noaa-nws-aorc-v1-1-1km/
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr

AORC_BUCKET = "noaa-nws-aorc-v1-1-1km"
AORC_VAR = "APCP_surface"

_FS_CACHE = {}
_YEAR_STORE_CACHE = {}


def aorc_filesystem():
    """Anonymous S3 filesystem handle for the public AORC bucket, built
    once and reused."""
    if "fs" not in _FS_CACHE:
        import s3fs
        _FS_CACHE["fs"] = s3fs.S3FileSystem(anon=True)
    return _FS_CACHE["fs"]


def open_aorc_year(year):
    """Lazily open one year's AORC zarr store.

    The store is ~20 TB but dask-backed, so opening it only reads metadata;
    actual bytes are pulled per-slice in load_aorc_hour. Cached per year so
    a full-window run opens each touched year exactly once. This is the
    slow step (no consolidated metadata -- one HTTPS request per variable),
    so it's only ever called from a download=True path.
    """
    if year not in _YEAR_STORE_CACHE:
        print(f"  Opening AORC {year}.zarr store (no consolidated metadata "
             f"-- this is the slow step, one HTTPS request per variable, "
             f"only happens once per year touched)...", flush=True)
        store = aorc_filesystem().get_mapper(f"{AORC_BUCKET}/{year}.zarr")
        _YEAR_STORE_CACHE[year] = xr.open_zarr(store, consolidated=False)
        print(f"  AORC {year}.zarr store open.", flush=True)
    return _YEAR_STORE_CACHE[year]


def aorc_cache_path(cache_dir, valid_dt):
    """Expected per-hour cache path for one AORC hour -- pure path math, no
    I/O, safe to use for a completeness check."""
    return Path(cache_dir) / f"AORC_APCP_{valid_dt:%Y%m%d-%H%M%S}.nc"


def load_aorc_hour(valid_dt, domain, cache_dir, download=True):
    """(lat1d, lon1d, data_mm) for one AORC hour, cropped to `domain`.

    `domain` is (lat_min, lat_max, lon_min, lon_max). Cached as a per-hour
    NetCDF under `cache_dir`; a cache hit never touches S3, regardless of
    `download`. On a cache miss: fetches from S3 when `download` is True
    (the login-node download-obs path); raises FileNotFoundError instead of
    touching the network when `download` is False (the compute-node
    obs-compare path).
    """
    cache_dir = Path(cache_dir)
    cache_path = aorc_cache_path(cache_dir, valid_dt)

    if cache_path.exists():
        with xr.open_dataset(cache_path) as ds:
            lat = ds["latitude"].values.copy()
            lon = ds["longitude"].values.copy()
            data = ds[AORC_VAR].values.copy()
        return lat, lon, data

    if not download:
        raise FileNotFoundError(
            f"AORC hour not cached: {cache_path} -- obs-compare does not "
            f"download; run 'download-obs' for this case first.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    lat_min, lat_max, lon_min, lon_max = domain
    year_ds = open_aorc_year(valid_dt.year)
    da = year_ds[AORC_VAR].sel(
        time=np.datetime64(valid_dt), method="nearest",
        tolerance=np.timedelta64(30, "m"),
    ).sel(
        latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max),
    ).load()

    data = np.where(np.isfinite(da.values) & (da.values >= 0), da.values, 0.0)
    lat = da.latitude.values
    lon = da.longitude.values

    out = xr.Dataset(
        {AORC_VAR: (("latitude", "longitude"), data)},
        coords={"latitude": lat, "longitude": lon},
    )
    out.to_netcdf(cache_path)
    return lat, lon, data


def sum_aorc_hours(window_start, window_end, domain, cache_dir, label="",
                   download=True):
    """Sum AORC hourly precip over (window_start, window_end].

    Same end-of-hour convention as hafs_common.load_mrms_hour: an N-hour
    accumulation sums the N hourly fields stamped 1..N hours after the
    start, so an AORC daily sum lines up with a Stage IV 24h file's own
    12Z(D-1)->12Z(D) window when window_start/window_end are that window.
    Returns (lat1d, lon1d, total_mm); raises RuntimeError if nothing loaded,
    or FileNotFoundError immediately if download=False and any hour is
    missing (a partial daily sum would be silently wrong, so this does not
    skip-and-continue the way a missing single-hour comparison can).
    """
    n_hours = int(round((window_end - window_start).total_seconds() / 3600))
    total = lat = lon = None
    for h in range(1, n_hours + 1):
        t = window_start + timedelta(hours=h)
        try:
            la, lo, data = load_aorc_hour(t, domain, cache_dir,
                                          download=download)
        except FileNotFoundError:
            raise
        except Exception as e:
            print(f"  AORC {label}h{h:03d} unavailable: {e}")
            continue
        if total is None:
            lat, lon = la, lo
            total = np.zeros_like(data)
        elif data.shape != total.shape:
            print(f"  AORC {label}h{h:03d} shape mismatch, skipping")
            continue
        total += data
    if total is None:
        raise RuntimeError(f"No AORC hours could be loaded for {label}.")
    return lat, lon, total
