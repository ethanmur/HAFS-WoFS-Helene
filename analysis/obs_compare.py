"""Observation-vs-observation rainfall comparison: MRMS, Stage IV, AORC.

No HAFS forecast involved -- this validates the observational datasets
against each other before they're trusted as verification truth elsewhere
in this repo. All three are compared natively hourly, restricted to the TC
footprint (union of circles of `mask_radius_km` around the best track over
the whole window): MRMS and AORC are natively hourly already; Stage IV is
read from NCEP's own ST4.<day> archive files (see stage4_hourly.py), which
bundle a 1-hour APCP message alongside the 6h/24h ones for every hour of
the day.

Every source is individually skippable (skip_mrms / skip_stage4 /
skip_aorc in the YAML), so a partial run (e.g. AORC only, reusing an
already-cached MRMS) still produces whatever comparisons that leaves
possible; nothing errors on a skipped/missing source, it's just left out
of that hour's panels and pairings.

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
from matplotlib.transforms import blended_transform_factory
import cartopy.crs as ccrs
import yaml
from botocore import UNSIGNED
from botocore.config import Config
import boto3

from hafs_common import QPF_LEVELS, haversine_km, load_mrms_hour, mrms_s3_key
from hafs_case import make_fixed_grid, position_on_track
from best_track import parse_bdeck
from parent_qpf import qpf_cmap
from ets_full import regrid_2d_to_fixed
from cycles import _precipitation_error_scale
from compare import _add_us_geography
from skill_metrics import continuous_scores
from plot_units import inches
import aorc_common
import stage4_hourly


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
    stage4_cache_dir: Path        # holds ST4.<YYYYMMDD> hourly source files
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
    """Draw one map panel's frame. Returns the text artists, which the
    caller MUST pass to savefig(bbox_extra_artists=...): they sit outside
    the axes, and GeoAxes.get_tightbbox() doesn't report them, so
    bbox_inches="tight" crops them off otherwise."""
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
    return [
        ax.text(0.5, 1.05, title, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=11),
        ax.text(0.5, -0.09, "Longitude", transform=ax.transAxes, ha="center",
                va="top", fontsize=9),
        ax.text(-0.1, 0.5, "Latitude", transform=ax.transAxes, ha="center",
                va="center", rotation=90, fontsize=9),
    ]


def _figure_title(fig, axes, text):
    """Figure-wide title as a text artist just above the panel titles.

    fig.suptitle() places itself in figure coordinates, which on these
    wide, short figures leaves it stranded far above the maps. Blending
    figure-x with axes-y instead keeps it horizontally centred on the
    figure while pinning it to the top of the panels. Returned so the
    caller can include it in bbox_extra_artists.
    """
    transform = blended_transform_factory(fig.transFigure, axes[0].transAxes)
    return axes[0].text(0.5, 1.13, text, transform=transform, ha="center",
                        va="bottom", fontsize=13)


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
    labels = []
    for ax, (name, lat, lon, data) in zip(axes, panels):
        labels += _panel_frame(ax, domain, track_line, name)
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
    labels.append(_figure_title(fig, axes, suptitle))
    # bbox_extra_artists names the out-of-axes labels explicitly (older
    # cartopy's GeoAxes.get_tightbbox omits child text, which is what
    # sheared the Latitude/Longitude labels off earlier figures);
    # pad_inches then leaves margin regardless of what tight-bbox measured.
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white",
                bbox_extra_artists=labels, pad_inches=0.4)
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
    labels = []
    for ax, (label, diff) in zip(axes, pairs):
        labels += _panel_frame(ax, domain, track_line, label)
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
    labels.append(_figure_title(fig, axes, suptitle))
    # bbox_extra_artists names the out-of-axes labels explicitly (older
    # cartopy's GeoAxes.get_tightbbox omits child text, which is what
    # sheared the Latitude/Longitude labels off earlier figures);
    # pad_inches then leaves margin regardless of what tight-bbox measured.
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white",
                bbox_extra_artists=labels, pad_inches=0.4)
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
# Hourly: MRMS vs Stage IV vs AORC
# =============================================================================

def hourly_timestamps(t0, t1):
    n_hours = int(round((t1 - t0).total_seconds() / 3600))
    return [t0 + timedelta(hours=h) for h in range(1, n_hours + 1)]


def run_hourly_comparison(case):
    print("\n" + "=" * 78)
    print("HOURLY: MRMS vs Stage IV vs AORC")
    print("=" * 78)
    if case.skip_mrms and case.skip_stage4 and case.skip_aorc:
        print("MRMS, Stage IV, and AORC all skipped -- nothing to compare.")
        return
    hourly_dir = case.out_dir / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)

    track = parse_bdeck(case.best_track)
    print(f"Best track: {len(track)} fixes from {case.best_track}", flush=True)
    grid_lat, grid_lon = case.fixed_grid()
    print(f"Common grid: {grid_lat.shape} at {case.grid_res} deg", flush=True)
    grid_swath = case_swath_mask(case, track, grid_lat, grid_lon,
                                 case.valid_start, case.valid_end)

    timestamps = hourly_timestamps(case.valid_start, case.valid_end)
    print(f"Reading {len(timestamps)} cached hour(s): "
         f"{timestamps[0]:%Y-%m-%d %HZ} -> {timestamps[-1]:%Y-%m-%d %HZ}",
         flush=True)

    pair_names = [("Stage IV", "MRMS"), ("Stage IV", "AORC"), ("MRMS", "AORC")]
    pooled = {p: ([], []) for p in pair_names}
    csv_rows = []
    for t in timestamps:

        print(f'Timestep {t}')

        mlat = mlon = mdata = None
        slat = slon = sdata = None
        alat = alon = adata = None
        if not case.skip_mrms:
            try:
                mlat, mlon, mdata = load_mrms_hour(
                    None, t, case.mrms_cache_dir, download=False)
                print('MRMS Loaded')
            except Exception as e:
                print(f"  {t:%Y-%m-%d %HZ} MRMS unavailable: {e}", flush=True)

        if not case.skip_stage4:
            try:
                slat, slon, sdata = stage4_hourly.load_stage4_hour(
                    t, case.stage4_cache_dir)
                print('Stage IV Loaded')
            except Exception as e:
                print(f"  {t:%Y-%m-%d %HZ} Stage IV unavailable: {e}",
                     flush=True)

        if not case.skip_aorc:
            try:
                alat, alon, adata = aorc_common.load_aorc_hour(
                    t, case.domain, case.aorc_cache_dir, download=False)
                print('AORC Loaded')
            except Exception as e:
                print(f"  {t:%Y-%m-%d %HZ} AORC unavailable: {e}", flush=True)

        if mdata is None and sdata is None and adata is None:
            continue

        print('all data loaded')

        track_line = track_segment(track, case.valid_start, t)
        tag = f"{t:%Y%m%d%H}"

        print('track line created')

        def _native_masked(lat, lon, data):
            if data is None:
                return None
            mask = case_swath_mask(case, track, lat, lon, t, t, step_hours=1)
            return np.where(mask, data, np.nan)

        print('data masked')

        spatial_multipanel(
            [("Stage IV", slat, slon, _native_masked(slat, slon, sdata)),
             ("MRMS", mlat, mlon, _native_masked(mlat, mlon, mdata)),
             ("AORC", alat, alon, _native_masked(alat, alon, adata))],
            case.domain, track_line,
            f"{case.storm_name} — hourly precipitation, {t:%Y-%m-%d %HZ}",
            hourly_dir / f"obs_hourly_map_{tag}.png",
        )

        print('spatial multipanel created')

        regridded = {}
        for name, lat, lon, data in (("Stage IV", slat, slon, sdata),
                                     ("MRMS", mlat, mlon, mdata),
                                     ("AORC", alat, alon, adata)):
            if data is None:
                regridded[name] = None
                continue

            print(f'regrid run for {name}')
            reg = regrid_2d_to_fixed(lat, lon, data, grid_lat, grid_lon)
            print(f'regrid complete for {name}')
            regridded[name] = np.where(grid_swath, reg, np.nan)

        print('data regridded')

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
                csv_rows.append({"valid": f"{t:%Y-%m-%d %H:%M}",
                                 "pair": f"{a} vs {b}", **stats})
        if any(d is not None for _, d in anomaly_pairs):
            anomaly_multipanel(
                anomaly_pairs, grid_lat, grid_lon, case.domain, track_line,
                f"{case.storm_name} — hourly differences, {t:%Y-%m-%d %HZ}",
                hourly_dir / f"obs_hourly_anomaly_{tag}.png",
            )
        print(f"  {t:%Y-%m-%d %HZ}  Stage IV={'ok' if sdata is not None else '--'}"
             f"  MRMS={'ok' if mdata is not None else '--'}"
             f"  AORC={'ok' if adata is not None else '--'}", flush=True)

    print('heatmap start')

    heatmap_pairs = [
        (a, b, np.concatenate(va) if va else np.array([]),
        np.concatenate(vb) if vb else np.array([]))
        for (a, b), (va, vb) in pooled.items()
    ]
    if any(p[2].size for p in heatmap_pairs):
        pooled_heatmap(
            heatmap_pairs,
            f"{case.storm_name} — pooled hourly comparison "
            f"({case.valid_start:%Y-%m-%d %HZ}–{case.valid_end:%Y-%m-%d %HZ})",
            case.out_dir / f"obs_hourly_heatmap_{case.output_slug}.png",
        )

    print('heatmap end')

    if csv_rows:
        out_csv = case.out_dir / f"obs_hourly_stats_{case.output_slug}.csv"
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Saved table: {out_csv}")


# =============================================================================
# Download (login node) -- data only, no regridding, no plotting
# =============================================================================

def download_obs(case):
    """Login-node command: populate the raw MRMS/AORC caches only.

    Stage IV hourly data is NOT fetched here -- it comes from NCEP's own
    ST4.<day> archive, obtained outside this repo's tooling, so this
    command has nothing to do for it beyond what skip_stage4 already
    documents in the YAML. No regridding, no masking, no plotting, no
    matplotlib work -- just sequential (never concurrent) requests to
    MRMS / AORC, so this stays gentle on a login node. obs-compare (the
    compute-node command) never downloads on its own; run this first.
    """
    print(f"Download obs: {case.storm_name}")
    print(f"Window: {case.valid_start:%Y-%m-%d %HZ} -> "
         f"{case.valid_end:%Y-%m-%d %HZ}")
    print(f"skip_mrms={case.skip_mrms}  skip_stage4={case.skip_stage4}  "
         f"skip_aorc={case.skip_aorc}")

    if not case.skip_stage4:
        print("\nStage IV: not fetched by this command -- place ST4.<day> "
             f"files under {case.stage4_cache_dir} yourself before running "
             "obs-compare.", flush=True)

    if case.skip_mrms and case.skip_aorc:
        print("\nskip_mrms and skip_aorc both set -- nothing more to fetch.")
        return

    timestamps = hourly_timestamps(case.valid_start, case.valid_end)
    print(f"\nMRMS/AORC: fetching {len(timestamps)} hour(s) "
         f"{timestamps[0]:%Y-%m-%d %HZ} -> {timestamps[-1]:%Y-%m-%d %HZ}",
         flush=True)

    s3 = (boto3.client("s3", region_name="us-east-1",
                       config=Config(signature_version=UNSIGNED))
         if not case.skip_mrms else None)
    for t in timestamps:
        if case.skip_mrms:
            mstatus = "skipped"
        else:
            try:
                load_mrms_hour(s3, t, case.mrms_cache_dir, download=True)
                mstatus = "ok"
            except Exception as e:
                mstatus = f"FAILED ({e})"
        if case.skip_aorc:
            astatus = "skipped"
        else:
            try:
                aorc_common.load_aorc_hour(t, case.domain,
                                           case.aorc_cache_dir, download=True)
                astatus = "ok"
            except Exception as e:
                astatus = f"FAILED ({e})"
        print(f"  {t:%Y-%m-%d %HZ}  MRMS={mstatus}  AORC={astatus}",
             flush=True)

    print("\nDownload complete.")


# =============================================================================
# Entry point (compute node) -- reads the cache only, never downloads
# =============================================================================

def check_cache_complete(case):
    """List of human-readable missing-item strings; empty means every file
    obs-compare will need is already cached. Never touches the network, so
    it's safe on a no-internet compute node. The Stage IV check parses the
    cached ST4.<day> files' GRIB metadata (stage4_hourly.index_stage4_hourly)
    rather than a pure path check, since one file holds many hours and
    there's no filename-only way to know which hours it covers -- that
    parsing is memoized and gets reused by the real read afterwards, so
    it's not wasted work."""
    missing = []

    stage4_idx = ({} if case.skip_stage4
                 else stage4_hourly.index_stage4_hourly(str(case.stage4_cache_dir)))

    for t in hourly_timestamps(case.valid_start, case.valid_end):
        if not case.skip_mrms:
            _, fname = mrms_s3_key(t)
            path = case.mrms_cache_dir / fname.replace(".gz", "")
            if not path.exists():
                missing.append(f"MRMS hour {t:%Y-%m-%d %HZ} "
                               f"(expected {path})")
        if not case.skip_stage4 and t not in stage4_idx:
            missing.append(f"Stage IV hourly record for {t:%Y-%m-%d %HZ} "
                           f"(expected inside an ST4.<day> file under "
                           f"{case.stage4_cache_dir})")
        if not case.skip_aorc:
            path = aorc_common.aorc_cache_path(case.aorc_cache_dir, t)
            if not path.exists():
                missing.append(f"AORC hour {t:%Y-%m-%d %HZ} "
                               f"(expected {path})")
    return missing


def run_obs_compare(case):
    case.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Obs compare: {case.storm_name}")
    print(f"Window: {case.valid_start:%Y-%m-%d %HZ} -> "
         f"{case.valid_end:%Y-%m-%d %HZ}")
    print(f"skip_mrms={case.skip_mrms}  skip_stage4={case.skip_stage4}  "
         f"skip_aorc={case.skip_aorc}")

    missing = check_cache_complete(case)
    if missing:
        print("\nERROR: required data is not cached, and obs-compare does "
             "not download it. Run this first (on a node with internet):\n"
             "  python analysis/run.py <yaml> download-obs\n")
        for item in missing[:25]:
            print(f"  missing: {item}")
        if len(missing) > 25:
            print(f"  ... and {len(missing) - 25} more")
        raise SystemExit(1)

    run_hourly_comparison(case)


if __name__ == "__main__":
    run_obs_compare(from_yaml(sys.argv[1]))
