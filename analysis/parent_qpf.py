"""
HAFS QPF (nest + parent) vs MRMS QPE vs Stage IV QPE — static 4-panel.

The parent panel is the field operational viewers (e.g. Tropical Tidbits'
"HAFS-A Parent Model — Total Accumulated Precip") draw.  The 6-km parent domain
is FIXED, so its 0->fhour cumulative APCP (tp) accumulates geographically
correctly — no moving-nest storm-relative inflation, no bucket-summing, no
eyewall scallops.  The whole-run storm total is just the cumulative tp in the
last forecast-hour file.

Two observed-QPE references are shown for comparison of rainfall amounts:
  * MRMS MultiSensor QPE (Pass2), 1-hourly, from the noaa-mrms-pds S3 bucket.
  * NCEP Stage IV QPE, 6-hourly, from the NOAA water.noaa.gov daily tarballs.
    Stage IV is gauge+radar and is the classic verification benchmark; it is
    CONUS-only, so Helene's Gulf/Caribbean rain won't appear in that panel.

The 2-km moving NEST is higher-res but its cumulative tp is storm-relative
(so geographically valid per-interval buckets are regridded and summed); the
parent domain is coarser (lower peaks) but geographically honest and viewer-
consistent. The reconstructed nest total (the same field ets_full.py scores,
via hafs_common.hafs_event_total) is shown as the first panel.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/run.py storms/helene_hfsa.yaml parent

Reuses the GRIB2 / MRMS plumbing from hafs_common.py.
"""

import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta

# Make the sibling module importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import cfgrib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from hafs_common import (
    QPF_LEVELS, QPF_COLORS,
    read_hafs_tp_records, haversine_km, load_mrms_hour, crop_to_domain,
    discover_files, hafs_event_total,
)
from hafs_case import from_yaml
from plot_units import inches

# Stage IV QPE — NOAA water.noaa.gov daily source tarballs (GRIB2 since 2020).
# Each daily tar is valid 12Z->12Z and holds 1h/6h/24h accumulation files.
STAGE4_BASE = "https://water.noaa.gov/resources/downloads/precip/stageIV"


# =============================================================================
# Parent file discovery + cumulative-APCP selection
# =============================================================================

def default_parent_path(case):
    """Highest forecast-hour parent.atm file for the configured run."""
    hits = sorted(case.run_dir.glob(case.parent_glob()))
    if not hits:
        return None

    def fhour(p):
        m = re.search(r"\.f(\d{3})\.grb2$", p.name)
        return int(m.group(1)) if m else -1

    return max(hits, key=fhour)


def parent_path_at_fhour(case, fhour):
    """The parent.atm file for one exact forecast hour, or None if absent."""
    suffix = f".f{fhour:03d}.grb2"
    hits = [p for p in sorted(case.run_dir.glob(case.parent_glob()))
            if p.name.endswith(suffix)]
    return hits[0] if hits else None


def pick_cumulative_record(records):
    """Pick the 0->fhour cumulative APCP record on the (fixed) parent grid.

    On a fixed grid the cumulative total IS the geographic storm total, so we
    take the longest 0-> window directly — no fmax, no bucket summing.
    """
    if not records:
        return None
    finest = max(r["npoints"] for r in records)
    cand = [r for r in records if r["npoints"] == finest]
    cumulative = [r for r in cand if r["start_step"] in (0, None)]
    pool = cumulative or cand
    return max(pool, key=lambda r: (r["end_step"] or 0))


# =============================================================================
# Stage IV QPE downloader / accumulator
# =============================================================================

def stage4_tar_url(day):
    return (f"{STAGE4_BASE}/{day:%Y}/{day:%m}/{day:%d}/"
            f"ncep_stage_iv_source_files_{day:%Y%m%d}.tar")


def ensure_stage4_files(start_dt, end_dt, cache_dir):
    """Download + extract the daily Stage IV tarballs covering [start, end].

    Tarballs are valid 12Z->12Z, so we fetch from the day BEFORE start through
    end to be safe.  A per-day marker file makes this idempotent (cached).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    day = (start_dt - timedelta(days=1)).date()
    last = end_dt.date()
    while day <= last:
        marker = cache_dir / f".{day:%Y%m%d}.done"
        if not marker.exists():
            url = stage4_tar_url(day)
            tar_path = cache_dir / f"st4_{day:%Y%m%d}.tar"
            try:
                urllib.request.urlretrieve(url, tar_path)
                with tarfile.open(tar_path) as tf:
                    tf.extractall(cache_dir)
                marker.write_text("ok")
            except Exception as e:
                print(f"  Stage IV tar {day} failed: {e}")
            finally:
                tar_path.unlink(missing_ok=True)
        day += timedelta(days=1)


def index_stage4_24h_conus(cache_dir):
    """Map date string (YYYYMMDD) -> CONUS 24h Stage IV grib path.

    The water.noaa.gov daily source tar holds only 24h accumulations per region
    (conus_YYYYMMDD_24h.grb2, valid 12Z->12Z); we use the CONUS one.
    """
    idx = {}
    for p in cache_dir.rglob("*conus*24h*.grb2"):
        m = re.search(r"(\d{8})_24h", p.name)
        if m:
            idx[m.group(1)] = p
    return idx


def read_stage4(path):
    """Return (lat2d, lon2d, data_mm) for a Stage IV file, UNMODIFIED except
    that GRIB missing/negative flags are set to 0 so the array is plottable.
    No precip values are clipped — raw amounts are preserved for diagnosis."""
    for ds in cfgrib.open_datasets(str(path)):
        for var in ds.data_vars:
            da = ds[var]
            raw = da.values
            data = np.where(np.isnan(raw) | (raw < 0), 0.0, raw)
            lat = da.latitude.values
            lon = da.longitude.values
            lon = np.where(lon > 180, lon - 360, lon)
            return lat, lon, data
    raise RuntimeError(f"no variable in {path}")


def stage4_sum_days(cache_dir, start_dt, end_dt):
    """Sum the daily CONUS 24h Stage IV files whose date the window touches.

    Returns (lat2d, lon2d, total_mm, used_keys) with the total UNMASKED;
    (None, None, None, []) when nothing is available.
    """
    ensure_stage4_files(start_dt, end_dt, cache_dir)
    idx = index_stage4_24h_conus(cache_dir)
    if not idx:
        return None, None, None, []
    total = lat2d = lon2d = None
    used = []
    day = start_dt.date()
    while day <= end_dt.date():
        key = day.strftime("%Y%m%d")
        path = idx.get(key)
        if path is not None:
            try:
                lat, lon, data = read_stage4(path)
            except Exception as e:
                print(f"  Stage IV read failed {key}: {e}")
                day += timedelta(days=1)
                continue
            if total is None:
                lat2d, lon2d = lat, lon
                total = np.zeros_like(data)
            total += data
            used.append(key)
            # Per-file diagnostic: where is the daily max, how many extreme px?
            j, i = np.unravel_index(np.argmax(data), data.shape)
            print(f"  {key} 24h: max {data[j, i]:6.0f} mm at "
                  f"({lat[j, i]:.2f}, {lon[j, i]:.2f})  "
                  f">300mm:{int((data > 300).sum())}  >600mm:{int((data > 600).sum())}")
        day += timedelta(days=1)
    if total is None:
        return None, None, None, []
    return lat2d, lon2d, total, used


def stage4_label(used):
    """Window label from the used day keys (24h file D covers 12Z(D-1)->12Z(D))."""
    d0 = datetime.strptime(used[0], "%Y%m%d") - timedelta(hours=12)
    d1 = datetime.strptime(used[-1], "%Y%m%d")
    return f"~{d0:%m-%d %HZ} – {d1:%m-%d 12Z} ({len(used)}×24h)"


def stage4_total(case, end_fhour):
    """Sum the daily CONUS 24h Stage IV files spanning the forecast window,
    then mask the total to the full-track TC swath (same footprint as HAFS).

    24h files are valid 12Z->12Z so they don't align exactly with the 00Z-init
    0->end_fhour window; we sum every day the event touches.  Returns
    (lat2d, lon2d, total_mm, label) or (None, None, None, None) if unavailable.
    """
    valid_end = case.init_dt + timedelta(hours=end_fhour)
    lat2d, lon2d, total, used = stage4_sum_days(
        case.stage4_cache_dir, case.init_dt, valid_end)
    if total is None:
        return None, None, None, None

    # Mask the summed total to the union of circles along the track.
    swath = np.zeros(lat2d.shape, dtype=bool)
    for h in range(0, end_fhour + 1):
        tlat, tlon = case.position_at(case.init_dt + timedelta(hours=h))
        swath |= haversine_km(tlat, tlon, lat2d, lon2d) <= case.display_radius_km
    total = np.where(swath, total, 0.0)

    # Total diagnostic: locate the event-total max so you can judge if it's a
    # real accumulation or an artifact (e.g. a stuck pixel across days).
    j, i = np.unravel_index(np.argmax(total), total.shape)
    print(f"  Stage IV total: max {total[j, i]:.0f} mm at "
          f"({lat2d[j, i]:.2f}, {lon2d[j, i]:.2f})  "
          f">500mm:{int((total > 500).sum())}  >800mm:{int((total > 800).sum())}")

    return lat2d, lon2d, total, stage4_label(used)


def stage4_total_window(cache_dir, valid_start, valid_end, track_points,
                        radius_km):
    """Stage IV touched-days total for an absolute window, masked to the
    union of circles of radius_km around track_points [(lat, lon), ...].

    Returns (lat2d, lon2d, total_mm, label) or (None, None, None, None).
    """
    lat2d, lon2d, total, used = stage4_sum_days(cache_dir, valid_start,
                                                valid_end)
    if total is None:
        return None, None, None, None
    swath = np.zeros(lat2d.shape, dtype=bool)
    for tlat, tlon in track_points:
        swath |= haversine_km(tlat, tlon, lat2d, lon2d) <= radius_km
    total = np.where(swath, total, 0.0)
    return lat2d, lon2d, total, stage4_label(used)


# =============================================================================
# Plotting
# =============================================================================

def qpf_cmap():
    cmap = mcolors.ListedColormap(QPF_COLORS)
    norm = mcolors.BoundaryNorm(QPF_LEVELS, cmap.N)
    return cmap, norm


def swath_masked(field, lats, lons, case, end_fhour):
    """Zero `field` outside the TC display swath; NaN->0 inside it.

    The swath is the union of circles of `case.display_radius_km` around the
    best-track position at each hour 0..end_fhour.  `lats`/`lons` are 2-D
    arrays of the same shape as `field` (any grid: HAFS-native or fixed mesh).
    """
    mask = np.zeros(np.shape(lats), dtype=bool)
    for h in range(0, end_fhour + 1):
        tlat, tlon = case.position_at(case.init_dt + timedelta(hours=h))
        mask |= haversine_km(tlat, tlon, lats, lons) <= case.display_radius_km
    return np.where(mask, np.nan_to_num(field, nan=0.0), 0.0)


def plot_compare(case, panels, end_fhour, out_path):
    """panels: list of (lons, lats, data_mm, title); data may be None."""
    lat_min, lat_max, lon_min, lon_max = case.domain
    cmap, _ = qpf_cmap()
    levels_in = np.asarray(inches(np.asarray(QPF_LEVELS, dtype=float)))
    norm = mcolors.BoundaryNorm(levels_in, cmap.N)

    fig, axes = plt.subplots(
        1, len(panels), figsize=(8 * len(panels), 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if len(panels) == 1:
        axes = [axes]

    cf = None
    for ax, (lons, lats, data, title) in zip(axes, panels):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                          linestyle="--", alpha=0.5)
        gl.top_labels = gl.right_labels = False
        if data is not None:
            cf = ax.contourf(
                lons, lats, inches(data), levels=levels_in, cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(), extend="max",
            )
        else:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center",
                    transform=ax.transAxes)
        # Cartopy GeoAxes titles set via ax.set_title() are frequently
        # clipped by savefig(bbox_inches="tight") because get_tightbbox()
        # under-measures GeoAxes decorations. A plain text artist in axes
        # coordinates is measured correctly and never gets cut off.
        ax.text(0.5, 1.05, title, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11)
        ax.text(0.5, -0.09, "Longitude", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        ax.text(-0.1, 0.5, "Latitude", transform=ax.transAxes,
                ha="center", va="center", rotation=90, fontsize=9)

    if cf is not None:
        plt.colorbar(cf, ax=axes, label="Accumulated precipitation (inches)",
                     ticks=levels_in[::2], shrink=0.7, fraction=0.02,
                     format="%g")
    valid_dt = case.init_dt + timedelta(hours=end_fhour)
    fig.suptitle(
        f"{case.storm_name} — {case.model_label} QPF: nest vs parent vs MRMS vs Stage IV "
        f"(init {case.init_dt:%Y-%m-%d %HZ}, 0–{end_fhour}h, "
        f"valid {valid_dt:%Y-%m-%d %HZ})",
        fontsize=13, y=1.01,
    )
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# Nest field
# =============================================================================

def compute_nest_field(case, grid_lat, grid_lon, end_fhour):
    """Moving-nest incremental APCP total on the fixed grid, masked to the swath.

    Same field ets_full.py scores (hafs_event_total over the storm.atm files).
    Returns None when no nest files are found so the figure can still render
    its other panels.
    """
    file_pairs = discover_files(case.run_dir, case.storm_glob(),
                                case.fhours_filter)
    if not file_pairs:
        print("  No storm.atm (nest) files found — nest panel unavailable.")
        return None
    nest_total, mode = hafs_event_total(file_pairs, grid_lat, grid_lon)
    print(f"  nest APCP mode: {mode}, max {np.nanmax(nest_total):.0f} mm")
    return swath_masked(nest_total, grid_lat, grid_lon, case, end_fhour)


# =============================================================================
# Main
# =============================================================================

def generate_parent_figure(case):
    case.out_dir.mkdir(parents=True, exist_ok=True)
    case.mrms_cache_dir.mkdir(parents=True, exist_ok=True)

    path = default_parent_path(case)
    if path is None or not path.exists():
        print(f"No parent.atm file found under {case.run_dir} "
              f"(matching {case.parent_glob()})")
        return
    print(f"Parent file: {path}")

    # ------------------------------------------------------------------
    # HAFS parent-domain cumulative APCP (fixed grid -> geographic total).
    # ------------------------------------------------------------------
    records = read_hafs_tp_records(path)
    if not records:
        print("\nNo 'tp' (APCP) records in the parent file.")
        print(f"Run  python analysis/inspect_grib.py {path}  to list its fields.")
        return
    print(f"\nFound {len(records)} tp record(s):")
    for i, r in enumerate(records):
        print(f"  [{i}] window {r['start_step']}->{r['end_step']}h "
              f"stepType={r['step_type']} npoints={r['npoints']:,} "
              f"max={np.nanmax(r['data']):.0f} mm")

    rec = pick_cumulative_record(records)
    end_fhour = rec["end_step"] or 0
    hafs_lats, hafs_lons, hafs_mm = rec["lats"], rec["lons"], rec["data"]
    print(f"\nUsing 0->{end_fhour}h cumulative record, grid {hafs_lats.shape}, "
          f"max {np.nanmax(hafs_mm):.0f} mm ({np.nanmax(hafs_mm)/25.4:.1f} in)")

    valid_end = case.init_dt + timedelta(hours=end_fhour)

    # TC rainfall swath mask on the parent grid (union of circles along
    # the best track for hours 0..end_fhour) — same footprint as the QPE below.
    hafs_display = swath_masked(hafs_mm, hafs_lats, hafs_lons, case, end_fhour)

    # ------------------------------------------------------------------
    # MRMS total over the same 0->end_fhour window, masked to the swath.
    # ------------------------------------------------------------------
    lat_min, lat_max, lon_min, lon_max = case.domain
    print(f"\nAccumulating MRMS 1H QPE over hours 1–{end_fhour} ...")
    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    mrms_lats = mrms_lons = mrms_total = None
    for h in range(1, end_fhour + 1):
        t = case.init_dt + timedelta(hours=h)
        try:
            lat, lon, data = load_mrms_hour(s3, t, case.mrms_cache_dir)
            clat, clon, cdata = crop_to_domain(lat, lon, data,
                                               lat_min, lat_max, lon_min, lon_max)
        except Exception as e:
            print(f"  h{h:03d} unavailable: {e}")
            continue
        if mrms_total is None:
            mrms_lats, mrms_lons = clat, clon
            mrms_total = np.zeros((clat.size, clon.size))
        tlat, tlon = case.position_at(t)
        mlon2d, mlat2d = np.meshgrid(mrms_lons, mrms_lats)
        dist = haversine_km(tlat, tlon, mlat2d, mlon2d)
        mrms_total += np.where(dist <= case.display_radius_km, cdata, 0.0)
        if h % 12 == 0 or h == end_fhour:
            print(f"  cached h{h:03d}/{end_fhour} ({t:%Y-%m-%d %HZ})")

    # ------------------------------------------------------------------
    # Stage IV total over the same window (6-hourly), masked to the swath.
    # ------------------------------------------------------------------
    print(f"\nAccumulating Stage IV 24H CONUS QPE spanning {case.init_dt:%Y-%m-%d %HZ} "
          f"-> {valid_end:%Y-%m-%d %HZ} ...")
    s4_lats, s4_lons, s4_total, s4_label = stage4_total(case, end_fhour)
    if s4_total is None:
        print("  Stage IV unavailable (download/extract failed).")
        s4_label = "unavailable"
    else:
        print(f"  Stage IV window: {s4_label}")

    # ------------------------------------------------------------------
    # Moving-nest total on the fixed verification grid (same field ETS
    # scores), masked to the display swath.
    # ------------------------------------------------------------------
    print("\nAccumulating HAFS nest (storm.atm) incremental APCP ...")
    grid_lat, grid_lon = case.fixed_grid()
    nest_display = compute_nest_field(case, grid_lat, grid_lon, end_fhour)

    # ------------------------------------------------------------------
    # Plot 4-panel.
    # ------------------------------------------------------------------
    out_png = case.out_dir / f"parent_qpf_{case.output_slug}.png"
    panels = [
        (grid_lon, grid_lat, nest_display,
         f"{case.model_label} Nest APCP (moving 1.2-mile, summed intervals)\n"
         f"0–{end_fhour}h"),
        (hafs_lons, hafs_lats, hafs_display,
         f"{case.model_label} Parent APCP\n0–{end_fhour}h (valid {valid_end:%Y-%m-%d %HZ})"),
        (mrms_lons, mrms_lats, mrms_total,
         f"MRMS MultiSensor QPE (Pass2)\n{end_fhour}h accumulation"),
        (s4_lons, s4_lats, s4_total,
         f"NCEP Stage IV QPE (CONUS)\n{s4_label}"),
    ]
    plot_compare(case, panels, end_fhour, out_png)

    def _mx(a):
        return float(np.nanmax(a)) if a is not None else float("nan")
    print(f"\nSaved {out_png}")
    print(f"  HAFS nest max {_mx(nest_display):.0f} mm | "
          f"HAFS parent max {_mx(hafs_display):.0f} mm | "
          f"MRMS max {_mx(mrms_total):.0f} mm | "
          f"Stage IV max {_mx(s4_total):.0f} mm")


if __name__ == "__main__":
    generate_parent_figure(from_yaml(sys.argv[1]))
