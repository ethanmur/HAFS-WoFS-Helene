"""Observation-vs-observation rainfall comparison: MRMS, Stage IV, AORC.

No HAFS forecast involved -- this validates the observational datasets
against each other before they're trusted as verification truth elsewhere
in this repo. Two tiers, both restricted to the TC footprint (union of
circles of `mask_radius_km` around the best track over the whole window):

  hourly  MRMS vs AORC, both natively hourly.
  daily   Stage IV (native 24h, 12Z->12Z) vs MRMS and AORC summed to match
          each Stage IV file's own window -- a fair comparison, since Stage
          IV's accumulation window never aligns to calendar days.

Every download is individually skippable (skip_mrms / skip_stage4 /
skip_aorc in the YAML), so a partial run (e.g. AORC only, reusing an
already-cached MRMS) still produces whatever comparisons that leaves
possible; nothing errors on a skipped/missing source, it's just left out
of that day's/hour's panels and pairings.

Usage:
    python analysis/run.py storms/helene_obs_compare.yaml obs-compare
"""

import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import yaml
from botocore import UNSIGNED
from botocore.config import Config
import boto3

from hafs_common import QPF_LEVELS, haversine_km, load_mrms_hour
from hafs_case import make_fixed_grid, position_on_track
from best_track import parse_bdeck
from parent_qpf import (
    ensure_stage4_files, index_stage4_24h_conus, read_stage4, qpf_cmap,
)
from ets_full import regrid_2d_to_fixed
from cycles import _precipitation_error_scale
from compare import _add_us_geography
from skill_metrics import continuous_scores
from plot_units import inches
import aorc_common


# =============================================================================
# Config
# =============================================================================

@dataclass
class ObsCompareCase:
    storm_name: str
    best_track: Path
    valid_start: datetime
    valid_end: datetime
    domain: tuple                 # (lat_min, lat_max, lon_min, lon_max)
    mask_radius_km: float
    grid_res: float
    out_dir: Path
    mrms_cache_dir: Path
    stage4_cache_dir: Path
    aorc_cache_dir: Path
    skip_mrms: bool = False
    skip_stage4: bool = False
    skip_aorc: bool = False
    clip_outside_radius: bool = False
    case_slug: str = "obs_compare"

    def fixed_grid(self):
        return make_fixed_grid(self.domain, self.grid_res)

    @property
    def output_slug(self):
        return (f"{self.case_slug}_{self.valid_start:%Y%m%d%H}_"
                f"{self.valid_end:%Y%m%d%H}")


def from_yaml(yaml_path):
    yaml_path = Path(yaml_path)
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    for key in ("best_track", "valid_start", "valid_end", "domain"):
        if key not in cfg:
            raise KeyError(f"'{key}' is required in an obs-compare YAML "
                           f"({yaml_path})")
    out_dir = (Path(cfg["out_dir"]) if cfg.get("out_dir")
              else Path("analysis/output") / yaml_path.stem)
    valid_start = datetime.strptime(str(cfg["valid_start"]), "%Y%m%d%H")
    valid_end = datetime.strptime(str(cfg["valid_end"]), "%Y%m%d%H")
    if valid_end <= valid_start:
        raise ValueError(f"valid_end must be after valid_start in {yaml_path}")
    return ObsCompareCase(
        storm_name=cfg.get("storm_name", "Storm"),
        best_track=Path(cfg["best_track"]),
        valid_start=valid_start,
        valid_end=valid_end,
        domain=tuple(cfg["domain"]),
        mask_radius_km=float(cfg.get("mask_radius_km", 500.0)),
        grid_res=float(cfg.get("grid_res", 0.05)),
        out_dir=out_dir,
        mrms_cache_dir=Path(cfg.get("mrms_cache_dir", "/tmp/mrms_cache")),
        stage4_cache_dir=Path(cfg.get("stage4_cache_dir", "/tmp/stage4_cache")),
        aorc_cache_dir=Path(cfg.get("aorc_cache_dir", "/tmp/aorc_cache")),
        skip_mrms=bool(cfg.get("skip_mrms", False)),
        skip_stage4=bool(cfg.get("skip_stage4", False)),
        skip_aorc=bool(cfg.get("skip_aorc", False)),
        clip_outside_radius=bool(cfg.get("clip_outside_radius", False)),
        case_slug=yaml_path.stem,
    )


# =============================================================================
# Track / footprint
# =============================================================================

def _as_2d(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lat2d, lon2d
    return lat, lon


def swath_mask(track, radius_km, lat, lon, t0, t1, step_hours=1):
    """Boolean mask (shape of `lat`/`lon`): within radius_km of the track at
    any hour in [t0, t1]. `lat`/`lon` may be 1-D axes or 2-D meshes."""
    lat2d, lon2d = _as_2d(lat, lon)
    mask = np.zeros(lat2d.shape, dtype=bool)
    n_hours = int(round((t1 - t0).total_seconds() / 3600))
    for h in range(0, n_hours + 1, step_hours):
        t = t0 + timedelta(hours=h)
        tlat, tlon = position_on_track(track, t)
        mask |= haversine_km(tlat, tlon, lat2d, lon2d) <= radius_km
    return mask


def case_swath_mask(case, track, lat, lon, t0, t1, step_hours=1):
    """swath_mask gated by case.clip_outside_radius.

    When clipping is off (the default), returns an all-True mask of the
    right shape -- the track is still drawn on every map, but data outside
    the 500-km circle is shown rather than blanked, for a broader sanity
    check against the raw product.
    """
    if not case.clip_outside_radius:
        lat2d, lon2d = _as_2d(lat, lon)
        return np.ones(lat2d.shape, dtype=bool)
    return swath_mask(track, case.mask_radius_km, lat, lon, t0, t1,
                      step_hours=step_hours)


def track_segment(track, t0, t1, step_hours=1):
    """[(lat, lon), ...] of the best track sampled hourly over [t0, t1]."""
    n_hours = int(round((t1 - t0).total_seconds() / 3600))
    return [position_on_track(track, t0 + timedelta(hours=h))
            for h in range(0, n_hours + 1, step_hours)]


# =============================================================================
# Plotting
# =============================================================================

def _draw_track(ax, track_line):
    lats = [p[0] for p in track_line]
    lons = [p[1] for p in track_line]
    ax.plot(lons, lats, color="black", lw=1.4, transform=ccrs.PlateCarree(),
            zorder=5)
    ax.plot(lons[-1], lats[-1], marker="o", color="red", markersize=5,
            transform=ccrs.PlateCarree(), zorder=6)


def _panel_frame(ax, domain, track_line, title):
    lat_min, lat_max, lon_min, lon_max = domain
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    _add_us_geography(ax)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--",
                      alpha=0.5)
    gl.top_labels = gl.right_labels = False
    if track_line:
        _draw_track(ax, track_line)
    # ax.set_title() on a GeoAxes is frequently clipped by
    # savefig(bbox_inches="tight"); a plain text artist in axes coordinates
    # is measured correctly and never gets cut off.
    ax.text(0.5, 1.05, title, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=11)
    ax.text(0.5, -0.09, "Longitude", transform=ax.transAxes, ha="center",
            va="top", fontsize=9)
    ax.text(-0.1, 0.5, "Latitude", transform=ax.transAxes, ha="center",
            va="center", rotation=90, fontsize=9)


def spatial_multipanel(panels, domain, track_line, suptitle, out_path):
    """panels: [(name, lat, lon, data_mm_or_None), ...], each on its own
    native grid; data already swath-masked to NaN outside the footprint."""
    cmap, _ = qpf_cmap()
    levels_in = np.asarray(inches(np.asarray(QPF_LEVELS, dtype=float)))
    norm = mcolors.BoundaryNorm(levels_in, cmap.N)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 6.5),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    if n == 1:
        axes = [axes]
    cf = None
    for ax, (name, lat, lon, data) in zip(axes, panels):
        _panel_frame(ax, domain, track_line, name)
        if data is not None:
            lat2d, lon2d = _as_2d(lat, lon)
            cf = ax.contourf(lon2d, lat2d, inches(data), levels=levels_in,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                             extend="max")
        else:
            ax.text(0.5, 0.5, "unavailable (skipped)", ha="center",
                    va="center", transform=ax.transAxes)
    if cf is not None:
        fig.colorbar(cf, ax=axes, label="Precipitation (inches)",
                     ticks=levels_in[::2], shrink=0.7, fraction=0.02,
                     format="%g")
    fig.suptitle(suptitle, fontsize=13, y=1.04)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def anomaly_multipanel(pairs, grid_lat, grid_lon, domain, track_line,
                       suptitle, out_path):
    """pairs: [(label, field_a_minus_b_mm_or_None), ...] on the common grid."""
    fields = [p[1] for p in pairs if p[1] is not None]
    if not fields:
        return False
    cmap, norm, ticks = _precipitation_error_scale(fields)
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 6.5),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    if n == 1:
        axes = [axes]
    mesh = None
    for ax, (label, diff) in zip(axes, pairs):
        _panel_frame(ax, domain, track_line, label)
        if diff is not None:
            mesh = ax.pcolormesh(grid_lon, grid_lat, inches(diff), cmap=cmap,
                                 norm=norm, shading="auto",
                                 transform=ccrs.PlateCarree())
        else:
            ax.text(0.5, 0.5, "unavailable (skipped)", ha="center",
                    va="center", transform=ax.transAxes)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, label="Difference (inches)", ticks=ticks,
                     shrink=0.7, fraction=0.02, format="%g")
    fig.suptitle(suptitle, fontsize=13, y=1.04)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def pooled_heatmap(pairs, suptitle, out_path):
    """pairs: [(label_a, label_b, values_a, values_b), ...], already the
    pooled finite, footprint-masked point clouds (any units, mm here).
    One hexbin panel per pair with a 1:1 line and RMSE/bias/r annotated."""
    pairs = [p for p in pairs if p[2].size and p[3].size]
    if not pairs:
        return False
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (label_a, label_b, va, vb) in zip(axes, pairs):
        a_in, b_in = inches(va), inches(vb)
        limit = max(1.0, float(np.nanmax(a_in)), float(np.nanmax(b_in)))
        hb = ax.hexbin(b_in, a_in, gridsize=60, cmap="viridis",
                       bins="log", mincnt=1, extent=(0, limit, 0, limit))
        ax.plot([0, limit], [0, limit], color="#d94f4f", lw=1.4, ls="--",
                label="1:1")
        stats = continuous_scores(va, vb)
        ax.text(0.03, 0.97,
               f"n={stats['n']:,}\nRMSE={inches(stats['rmse']):.2f} in\n"
               f"bias={inches(stats['bias']):+.2f} in\nr={stats['r']:.2f}",
               transform=ax.transAxes, va="top", fontsize=9,
               bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.set_xlabel(f"{label_b} (in)")
        ax.set_ylabel(f"{label_a} (in)")
        ax.set_title(f"{label_a} vs {label_b}")
        ax.set_aspect("equal")
        fig.colorbar(hb, ax=ax, label="log10(count)", shrink=0.85)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# =============================================================================
# Hourly: MRMS vs AORC
# =============================================================================

def hourly_timestamps(t0, t1):
    n_hours = int(round((t1 - t0).total_seconds() / 3600))
    return [t0 + timedelta(hours=h) for h in range(1, n_hours + 1)]


def sum_mrms_hours(s3, window_start, window_end, cache_dir, label=""):
    """Sum MRMS hourly QPE over (window_start, window_end] on its native
    grid. Same end-of-hour convention as load_mrms_hour; mirrors
    aorc_common.sum_aorc_hours so a Stage IV 24h window can be matched
    exactly by both. Returns (lat1d, lon1d, total_mm)."""
    n_hours = int(round((window_end - window_start).total_seconds() / 3600))
    total = lat = lon = None
    for h in range(1, n_hours + 1):
        t = window_start + timedelta(hours=h)
        try:
            la, lo, data = load_mrms_hour(s3, t, cache_dir)
        except Exception as e:
            print(f"  MRMS {label}h{h:03d} unavailable: {e}")
            continue
        if total is None:
            lat, lon = la, lo
            total = np.zeros_like(data)
        elif data.shape != total.shape:
            print(f"  MRMS {label}h{h:03d} shape mismatch, skipping")
            continue
        total += data
    if total is None:
        raise RuntimeError(f"No MRMS hours could be loaded for {label}.")
    return lat, lon, total


def run_hourly_comparison(case):
    print("\n" + "=" * 78)
    print("HOURLY: MRMS vs AORC")
    print("=" * 78)
    if case.skip_mrms and case.skip_aorc:
        print("Both MRMS and AORC skipped -- nothing to compare hourly.")
        return
    hourly_dir = case.out_dir / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)

    track = parse_bdeck(case.best_track)
    grid_lat, grid_lon = case.fixed_grid()
    grid_swath = case_swath_mask(case, track, grid_lat, grid_lon,
                                 case.valid_start, case.valid_end)

    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    aorc_fs = aorc_common.aorc_filesystem() if not case.skip_aorc else None

    pooled_mrms, pooled_aorc = [], []
    csv_rows = []
    for t in hourly_timestamps(case.valid_start, case.valid_end):
        mlat = mlon = mdata = None
        alat = alon = adata = None
        if not case.skip_mrms:
            try:
                mlat, mlon, mdata = load_mrms_hour(s3, t, case.mrms_cache_dir)
            except Exception as e:
                print(f"  {t:%Y-%m-%d %HZ} MRMS unavailable: {e}")
        if not case.skip_aorc:
            try:
                alat, alon, adata = aorc_common.load_aorc_hour(
                    aorc_fs, t, case.domain, case.aorc_cache_dir)
            except Exception as e:
                print(f"  {t:%Y-%m-%d %HZ} AORC unavailable: {e}")
        if mdata is None and adata is None:
            continue

        track_line = track_segment(track, case.valid_start, t)
        tag = f"{t:%Y%m%d%H}"

        def _native_masked(lat, lon, data):
            if data is None:
                return None
            mask = case_swath_mask(case, track, lat, lon, t, t, step_hours=1)
            return np.where(mask, data, np.nan)

        spatial_multipanel(
            [("MRMS", mlat, mlon, _native_masked(mlat, mlon, mdata)),
             ("AORC", alat, alon, _native_masked(alat, alon, adata))],
            case.domain, track_line,
            f"{case.storm_name} — hourly precipitation, {t:%Y-%m-%d %HZ}",
            hourly_dir / f"obs_hourly_map_{tag}.png",
        )

        mreg = (regrid_2d_to_fixed(mlat, mlon, mdata, grid_lat, grid_lon)
               if mdata is not None else None)
        areg = (regrid_2d_to_fixed(alat, alon, adata, grid_lat, grid_lon)
               if adata is not None else None)
        if mreg is not None:
            mreg = np.where(grid_swath, mreg, np.nan)
        if areg is not None:
            areg = np.where(grid_swath, areg, np.nan)

        if mreg is not None and areg is not None:
            anomaly_multipanel(
                [("MRMS − AORC", mreg - areg)],
                grid_lat, grid_lon, case.domain, track_line,
                f"{case.storm_name} — MRMS minus AORC, {t:%Y-%m-%d %HZ}",
                hourly_dir / f"obs_hourly_anomaly_{tag}.png",
            )
            valid = np.isfinite(mreg) & np.isfinite(areg)
            pooled_mrms.append(mreg[valid])
            pooled_aorc.append(areg[valid])
            stats = continuous_scores(mreg[valid], areg[valid])
            csv_rows.append({"valid": f"{t:%Y-%m-%d %H:%M}", **stats})
        print(f"  {t:%Y-%m-%d %HZ}  MRMS={'ok' if mdata is not None else '--'}"
             f"  AORC={'ok' if adata is not None else '--'}")

    if pooled_mrms:
        pooled_heatmap(
            [("MRMS", "AORC", np.concatenate(pooled_mrms),
             np.concatenate(pooled_aorc))],
            f"{case.storm_name} — MRMS vs AORC, pooled hourly "
            f"({case.valid_start:%Y-%m-%d %HZ}–{case.valid_end:%Y-%m-%d %HZ})",
            case.out_dir / f"obs_hourly_heatmap_{case.output_slug}.png",
        )
    if csv_rows:
        out_csv = case.out_dir / f"obs_hourly_stats_{case.output_slug}.csv"
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Saved table: {out_csv}")


# =============================================================================
# Daily: Stage IV vs MRMS-summed vs AORC-summed
# =============================================================================

def daily_windows(cache_dir, t0, t1):
    """[(day_key, window_start, window_end, stage4_path), ...] for every
    Stage IV 24h file whose calendar date falls in [t0.date(), t1.date()].
    window is that file's own 12Z(D-1)->12Z(D), independent of t0/t1's
    exact hour -- summing MRMS/AORC over the SAME window is what makes the
    three-way daily comparison fair."""
    idx = index_stage4_24h_conus(cache_dir)
    out = []
    day = t0.date()
    while day <= t1.date():
        key = day.strftime("%Y%m%d")
        path = idx.get(key)
        if path is not None:
            window_end = datetime(day.year, day.month, day.day, 12)
            window_start = window_end - timedelta(hours=24)
            out.append((key, window_start, window_end, path))
        day += timedelta(days=1)
    return out


def run_daily_comparison(case):
    print("\n" + "=" * 78)
    print("DAILY: Stage IV vs MRMS (summed) vs AORC (summed)")
    print("=" * 78)
    if case.skip_stage4:
        print("skip_stage4 is set -- no Stage IV 24h files to anchor the "
             "daily windows on, so daily comparison is skipped entirely.")
        return
    if case.skip_mrms and case.skip_aorc:
        print("Both MRMS and AORC skipped -- Stage IV would have nothing "
             "to compare against.")
        return
    daily_dir = case.out_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    track = parse_bdeck(case.best_track)
    grid_lat, grid_lon = case.fixed_grid()
    grid_swath = case_swath_mask(case, track, grid_lat, grid_lon,
                                 case.valid_start, case.valid_end)

    ensure_stage4_files(case.valid_start, case.valid_end, case.stage4_cache_dir)
    windows = daily_windows(case.stage4_cache_dir, case.valid_start,
                            case.valid_end)
    if not windows:
        print("No Stage IV 24h files found for this window.")
        return

    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    aorc_fs = aorc_common.aorc_filesystem() if not case.skip_aorc else None
    pair_names = [("Stage IV", "MRMS"), ("Stage IV", "AORC"), ("MRMS", "AORC")]
    pooled = {p: ([], []) for p in pair_names}
    csv_rows = []

    for key, w0, w1, path in windows:
        slat = slon = sdata = None
        try:
            slat, slon, sdata = read_stage4(path)
        except Exception as e:
            print(f"  {key} Stage IV read failed: {e}")

        mlat = mlon = mdata = None
        if not case.skip_mrms:
            try:
                mlat, mlon, mdata = sum_mrms_hours(
                    s3, w0, w1, case.mrms_cache_dir, label=f"{key} ")
            except Exception as e:
                print(f"  {key} MRMS 24h sum unavailable: {e}")

        adata = None
        alat = alon = None
        if not case.skip_aorc:
            try:
                alat, alon, adata = aorc_common.sum_aorc_hours(
                    aorc_fs, w0, w1, case.domain, case.aorc_cache_dir,
                    label=f"{key} ")
            except Exception as e:
                print(f"  {key} AORC 24h sum unavailable: {e}")

        if sdata is None and mdata is None and adata is None:
            continue

        track_line = track_segment(track, w0, w1)
        window_label = f"{w0:%Y-%m-%d %HZ}–{w1:%Y-%m-%d %HZ}"

        def _native_masked(lat, lon, data):
            if data is None:
                return None
            mask = case_swath_mask(case, track, lat, lon, w0, w1)
            return np.where(mask, data, np.nan)

        spatial_multipanel(
            [("Stage IV", slat, slon, _native_masked(slat, slon, sdata)),
             ("MRMS (24h sum)", mlat, mlon, _native_masked(mlat, mlon, mdata)),
             ("AORC (24h sum)", alat, alon, _native_masked(alat, alon, adata))],
            case.domain, track_line,
            f"{case.storm_name} — daily precipitation, {window_label}",
            daily_dir / f"obs_daily_map_{key}.png",
        )

        regridded = {}
        for name, lat, lon, data in (("Stage IV", slat, slon, sdata),
                                     ("MRMS", mlat, mlon, mdata),
                                     ("AORC", alat, alon, adata)):
            if data is None:
                regridded[name] = None
                continue
            reg = regrid_2d_to_fixed(lat, lon, data, grid_lat, grid_lon)
            regridded[name] = np.where(grid_swath, reg, np.nan)

        anomaly_pairs = []
        for a, b in pair_names:
            fa, fb = regridded[a], regridded[b]
            diff = fa - fb if (fa is not None and fb is not None) else None
            anomaly_pairs.append((f"{a} − {b}", diff))
            if diff is not None:
                valid = np.isfinite(fa) & np.isfinite(fb)
                pooled[(a, b)][0].append(fa[valid])
                pooled[(a, b)][1].append(fb[valid])
                stats = continuous_scores(fa[valid], fb[valid])
                csv_rows.append({"day": key, "pair": f"{a} vs {b}", **stats})
        if any(d is not None for _, d in anomaly_pairs):
            anomaly_multipanel(
                anomaly_pairs, grid_lat, grid_lon, case.domain, track_line,
                f"{case.storm_name} — daily differences, {window_label}",
                daily_dir / f"obs_daily_anomaly_{key}.png",
            )
        print(f"  {key} ({window_label})  Stage IV={'ok' if sdata is not None else '--'}"
             f"  MRMS={'ok' if mdata is not None else '--'}"
             f"  AORC={'ok' if adata is not None else '--'}")

    heatmap_pairs = [
        (a, b, np.concatenate(va) if va else np.array([]),
        np.concatenate(vb) if vb else np.array([]))
        for (a, b), (va, vb) in pooled.items()
    ]
    if any(p[2].size for p in heatmap_pairs):
        pooled_heatmap(
            heatmap_pairs,
            f"{case.storm_name} — pooled daily comparison "
            f"({case.valid_start:%Y-%m-%d}–{case.valid_end:%Y-%m-%d})",
            case.out_dir / f"obs_daily_heatmap_{case.output_slug}.png",
        )
    if csv_rows:
        out_csv = case.out_dir / f"obs_daily_stats_{case.output_slug}.csv"
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Saved table: {out_csv}")


# =============================================================================
# Entry point
# =============================================================================

def run_obs_compare(case):
    case.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Obs compare: {case.storm_name}")
    print(f"Window: {case.valid_start:%Y-%m-%d %HZ} -> "
         f"{case.valid_end:%Y-%m-%d %HZ}")
    print(f"skip_mrms={case.skip_mrms}  skip_stage4={case.skip_stage4}  "
         f"skip_aorc={case.skip_aorc}")
    run_hourly_comparison(case)
    run_daily_comparison(case)


if __name__ == "__main__":
    run_obs_compare(from_yaml(sys.argv[1]))
