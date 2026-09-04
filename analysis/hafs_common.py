"""
Shared HAFS / MRMS plumbing for the QPF-comparison and ETS products.

This module holds the storm-agnostic building blocks the framework reuses:
  * HAFS GRIB2 tp (APCP) reading via eccodes, one message at a time
    (cfgrib chokes when a file carries tp on two grids — parent + storm nest).
  * Moving-nest event accumulation: pick the short per-interval BUCKET from each
    storm.atm file, reproject it onto a fixed lat/lon mesh, and sum across
    forecast hours so the total sticks to geography as the nest tracks inland.
  * MRMS MultiSensor QPE download/cache + domain crop, and a haversine helper
    for the TC-swath masks.
  * The shared QPF color scale (levels + colors).

Consumed by parent_qpf.py (QPF maps), ets_score.py (ETS helpers), and
ets_full.py (combined parent+nest ETS). Not a runnable script — use run.py.
"""

import logging
import re
import sys
import gzip
import io
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eccodes
import cfgrib
import numpy as np
from scipy.interpolate import griddata

warnings.filterwarnings("ignore")
for _log in ["cfgrib", "cfgrib.messages", "cfgrib.xarray_store", "cfgrib.dataset"]:
    logging.getLogger(_log).setLevel(logging.CRITICAL)


MRMS_BUCKET = "noaa-mrms-pds"
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"

QPF_LEVELS = [0, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
QPF_COLORS = [
    "#ffffff", "#c8f0f0", "#64d2ff", "#3296ff",
    "#02fd02", "#01c501", "#008e00", "#fdf802",
    "#e5bc00", "#fd9500", "#fd0000", "#d40000",
]


# =============================================================================
# File discovery
# =============================================================================

def parse_fhour(filepath):
    m = re.search(r"\.f(\d{3})\.grb2$", filepath.name)
    return int(m.group(1)) if m else None


def discover_files(run_dir, glob, fhours_filter=None):
    files = sorted(run_dir.glob(glob))
    pairs = [(parse_fhour(f), f) for f in files]
    pairs = [(h, f) for h, f in pairs if h is not None]
    if fhours_filter:
        pairs = [(h, f) for h, f in pairs if h in fhours_filter]
    pairs.sort()
    return pairs


# =============================================================================
# HAFS loader + fixed-grid reprojection
# =============================================================================

def grid_latlon(lat0, lon0, lat1, lon1, nj, ni):
    """2-D (lats, lons in −180…180) meshes for a regular_ll GRIB grid.

    Latitudes span lat0→lat1 directly. Longitudes are reconstructed by walking
    east from lon0: a global grid whose first point sits east of its last (e.g.
    the HAFS-M multistorm parent, 210 E → 10 E) crosses the 0/360 seam, so the
    stored last longitude has already wrapped below the first. Unwrapping it
    (+360) before the linear span keeps the axis monotonic; the result is then
    folded back into −180…180. Regional grids (lon0 < lon1) are unaffected.
    """
    if lon1 < lon0:
        lon1 = lon1 + 360.0
    lats_1d = np.linspace(lat0, lat1, nj)
    lons_1d = np.linspace(lon0, lon1, ni)
    lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)
    lons_2d = np.where(lons_2d > 180, lons_2d - 360, lons_2d)
    return lats_2d, lons_2d

def read_hafs_tp_records(filepath):
    """Return every APCP/tp record in a HAFS GRIB2 file with grid + accumulation
    metadata, as a list of dicts.

    Reads directly via eccodes rather than cfgrib so we can pick messages one
    at a time.  cfgrib fails when the file has tp on two different grids (parent
    + storm nest) because it tries to concatenate them without an index file.

    Each dict: npoints, lats, lons (−180…180), data (mm, fill→NaN),
    start_step, end_step, step_type.
    """
    records = []
    with open(str(filepath), "rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                try:
                    sn = eccodes.codes_get(gid, "shortName", ktype=str)
                except Exception:
                    continue
                if sn != "tp":
                    continue
                nj = eccodes.codes_get(gid, "Nj")
                ni = eccodes.codes_get(gid, "Ni")
                lat0 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
                lon0 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
                lat1 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
                lon1 = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")

                def _opt(key, cast=int):
                    try:
                        return cast(eccodes.codes_get(gid, key))
                    except Exception:
                        return None

                start_step = _opt("startStep")
                end_step = _opt("endStep")
                step_type = _opt("stepType", cast=str)

                vals = eccodes.codes_get_values(gid)
                lats_2d, lons_180 = grid_latlon(lat0, lon0, lat1, lon1, nj, ni)
                data = vals.reshape(nj, ni)
                missing = eccodes.codes_get(gid, "missingValue")
                data = np.where(np.abs(data - missing) < 1.0, np.nan, data)
                data = np.where(data < 0, np.nan, data)
                records.append({
                    "npoints": ni * nj,
                    "lats": lats_2d,
                    "lons": lons_180,
                    "data": data,
                    "start_step": start_step,
                    "end_step": end_step,
                    "step_type": step_type,
                })
            except Exception:
                pass
            finally:
                eccodes.codes_release(gid)
    return records


def read_hafs_env_records(filepath, wanted):
    """Read selected environmental fields from one HAFS GRIB2 file.

    ``wanted`` contains ``(short_name, level)`` pairs.  A numeric level means
    an ``isobaricInhPa`` pressure level; ``None`` selects a single-level field.
    The file is traversed once and the first matching message wins.
    """
    wanted = list(wanted)
    wanted_set = set(wanted)
    records = {}
    with open(str(filepath), "rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                def _opt(key, cast=int):
                    try:
                        return cast(eccodes.codes_get(gid, key))
                    except Exception:
                        return None

                short_name = _opt("shortName", cast=str)
                type_of_level = _opt("typeOfLevel", cast=str)
                level = _opt("level")
                if type_of_level == "isobaricInhPa":
                    key = (short_name, level)
                else:
                    key = (short_name, None)
                if key not in wanted_set or key in records:
                    continue

                nj = eccodes.codes_get(gid, "Nj")
                ni = eccodes.codes_get(gid, "Ni")
                lat0 = eccodes.codes_get(
                    gid, "latitudeOfFirstGridPointInDegrees")
                lon0 = eccodes.codes_get(
                    gid, "longitudeOfFirstGridPointInDegrees")
                lat1 = eccodes.codes_get(
                    gid, "latitudeOfLastGridPointInDegrees")
                lon1 = eccodes.codes_get(
                    gid, "longitudeOfLastGridPointInDegrees")
                vals = eccodes.codes_get_values(gid)
                lats_2d, lons_180 = grid_latlon(lat0, lon0, lat1, lon1, nj, ni)
                data = vals.reshape(nj, ni)
                missing = eccodes.codes_get(gid, "missingValue")
                data = np.where(np.abs(data - missing) < 1.0, np.nan, data)
                records[key] = {
                    "npoints": ni * nj,
                    "lats": lats_2d,
                    "lons": lons_180,
                    "data": data,
                    "start_step": _opt("startStep"),
                    "end_step": _opt("endStep"),
                    "step_type": _opt("stepType", cast=str),
                    "units": _opt("units", cast=str),
                }
            except Exception:
                pass
            finally:
                eccodes.codes_release(gid)
    return records


def pick_total_record(records):
    """Pick the per-interval precip BUCKET from a moving-nest file.

    HAFS storm.atm files carry two tp records on the nest grid: a per-interval
    bucket (e.g. 60->63h) and a 0->fhour "cumulative" total.  The cumulative
    record is a TRAP on a storm-following nest: precip accumulates per grid
    CELL, and a cell that stays under the eyewall racks up rain for the whole
    run, giving physically impossible 0->126h totals (~2400 mm) in storm-
    relative space rather than at any fixed point on the ground.

    The short bucket is geographically valid — over one output step the nest
    barely moves — so we take the SHORTEST positive window and SUM the buckets
    (each reprojected to its own lat/lon) across frames to build the true
    geographic storm total.

    Returns (record, "incremental"), or (None, None) for a zero-length window
    (e.g. F000) which contributes nothing.
    """
    finest = max(r["npoints"] for r in records)
    cand = [r for r in records if r["npoints"] == finest]
    buckets = [r for r in cand
               if (r["end_step"] or 0) > (r["start_step"] or 0)]
    if not buckets:
        return None, None
    window = lambda r: (r["end_step"] or 0) - (r["start_step"] or 0)
    return min(buckets, key=window), "incremental"


def regrid_hafs(src_lats, src_lons, src_data, grid_lat, grid_lon):
    """
    Reproject HAFS tp from the moving storm grid onto the fixed lat/lon mesh.
    Uses linear interpolation which only fills points inside the storm grid
    convex hull — NaN outside, so no rectangular-edge stripe artifacts when
    the running max accumulates across frames.
    """
    pts = np.column_stack([src_lons.ravel(), src_lats.ravel()])
    vals = src_data.ravel()
    valid = np.isfinite(vals) & np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    if valid.sum() < 4:
        return np.full(grid_lat.shape, np.nan)
    return griddata(
        pts[valid], vals[valid],
        (grid_lon, grid_lat),
        method="linear",
        fill_value=np.nan,
    )


def accumulate_hafs_step(running, interp, mode):
    """Fold one frame's regridded tp into the running event total on the fixed grid.

    - cumulative: tp is monotonic since init, so fmax stamps the latest total
      at each fixed point as the storm nest sweeps over it.
    - incremental: each file holds only its own interval's precip, so the
      increments are summed across forecast hours.
    """
    interp = np.nan_to_num(interp, nan=0.0)
    if mode == "incremental":
        return running + interp
    return np.fmax(running, interp)


def hafs_event_total(file_pairs, grid_lat, grid_lon, verbose=True):
    """Full-event accumulated HAFS precip (mm) on the fixed grid.

    Selects the shortest positive accumulation bucket in each moving-nest
    frame, regrids it geographically, and adds it to the running total.
    Returns (total, mode).
    """
    total = np.zeros(grid_lat.shape)
    mode_seen = None
    n_files = len(file_pairs)
    for i, (fhour, fp) in enumerate(file_pairs, start=1):
        try:
            recs = read_hafs_tp_records(fp)
            if not recs:
                continue
            r, mode = pick_total_record(recs)
            if r is None:
                continue
        except Exception as e:
            if verbose:
                print(f"  F{fhour:03d} HAFS read failed: {e}")
            continue
        if mode_seen is None:
            mode_seen = mode
            if verbose:
                print(f"  APCP record: {mode} "
                      f"(window {r['start_step']}->{r['end_step']}h)")
        interp = regrid_hafs(r["lats"], r["lons"], r["data"], grid_lat, grid_lon)
        total = accumulate_hafs_step(total, interp, mode)
        # Per-file progress — the griddata reprojection is slow and otherwise
        # silent, so without this the loop looks hung on large grids.
        if verbose:
            print(f"  nest F{fhour:03d} [{i}/{n_files}] "
                  f"running total {np.nanmax(total):.0f} mm", flush=True)
    return total, (mode_seen or "cumulative")


# =============================================================================
# Distance helper
# =============================================================================

def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    """Great-circle distance (km) from (lat1,lon1) to each point in arrays."""
    R = 6371.0
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * \
        np.cos(np.radians(lat2_arr)) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


# =============================================================================
# MRMS downloader / cache
# =============================================================================

def mrms_s3_key(hour_end_dt):
    date_str = hour_end_dt.strftime("%Y%m%d")
    time_str = hour_end_dt.strftime("%Y%m%d-%H%M%S")
    fname = f"MRMS_{MRMS_PRODUCT}_{time_str}.grib2.gz"
    return f"CONUS/{MRMS_PRODUCT}/{date_str}/{fname}", fname


def load_mrms_hour(s3, hour_end_dt, cache_dir, download=True):
    """(lat, lon, data) for one MRMS hour. On a cache miss: downloads from
    S3 when `download` is True (the default); raises FileNotFoundError
    instead of touching `s3` when `download` is False, for callers that
    must not download (e.g. a no-internet compute node)."""
    cache_dir = Path(cache_dir)
    key, fname = mrms_s3_key(hour_end_dt)
    cache_path = cache_dir / fname.replace(".gz", "")
    if not cache_path.exists():
        if not download:
            raise FileNotFoundError(
                f"MRMS hour not cached: {cache_path} -- obs-compare does "
                f"not download; run 'download-obs' for this case first.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        gz_buf = io.BytesIO()
        s3.download_fileobj(MRMS_BUCKET, key, gz_buf)
        gz_buf.seek(0)
        raw = gzip.decompress(gz_buf.read())
        cache_path.write_bytes(raw)
    datasets = cfgrib.open_datasets(str(cache_path))
    for ds in datasets:
        for var in ds.data_vars:
            da = ds[var]
            data = np.where(da.values < 0, 0.0, da.values)
            return da.latitude.values, da.longitude.values, data
    raise RuntimeError(f"No variable in {cache_path}")


def crop_to_domain(lats, lons, data, lat_min, lat_max, lon_min, lon_max):
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    ri = np.where(lat_mask)[0]
    ci = np.where(lon_mask)[0]
    if ri.size == 0 or ci.size == 0:
        return lats, lons, data
    return (lats[ri[0]:ri[-1]+1],
            lons[ci[0]:ci[-1]+1],
            data[ri[0]:ri[-1]+1, ci[0]:ci[-1]+1])
