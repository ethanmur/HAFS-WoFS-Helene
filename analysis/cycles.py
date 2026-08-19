"""
Cycle comparison for HAFS QPF: score every initialization through a common
valid end. Cycles initialized after the configured start use an init-clipped
window and matching observations over a shared union-of-tracks footprint.

Uses the fixed parent domain exclusively. Produces landfall-relative metric
curves, QPF small multiples, ETS/FSS lead-time plots, separate
forecast/observed/error animations, and long-format
categorical/continuous and FSS CSVs.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/run.py storms/helene_hfsa_cycles.yaml cycles
"""

import sys
import csv
from pathlib import Path
from datetime import timedelta

# Make sibling analysis modules importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, colors, ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    import seaborn as sns
except ImportError:  # Optional at runtime; matplotlib remains a supported fallback.
    sns = None

from hafs_common import (
    QPF_LEVELS, discover_files, haversine_km, read_hafs_tp_records,
)
from hafs_case import (
    cycles_from_yaml, cycle_storm_case, discover_inits, window_hours,
    cycle_eligibility,
)
from ets_score import contingency_scores, build_mrms_total_window
from ets_full import regrid_2d_to_fixed, _OBS_COLOR, _FCST_STYLE
from parent_qpf import (
    parent_path_at_fhour, pick_cumulative_record, stage4_total_window,
    qpf_cmap,
)
from skill_metrics import continuous_scores, fractions_skill_score
from precip_structure import (
    distribution_stats, object_comparison, plot_distributions,
    plot_pattern_r, plot_percentiles_by_cycle,
)
from rmse_scatter import valid_points
from best_track import parse_bdeck_full
from track_skill import (
    cycle_track_summary, landfall_metrics, mean_displacement_deg,
    plot_shifted_skill, plot_track_error, plot_track_precip,
    motion_vector, score_shifted, track_error_rows,
)
from plot_units import format_inches, format_miles, inches, miles


SUMMARY_FIELDS = [
    "init", "init_dt", "lead_hours_to_landfall", "valid_start", "valid_end",
    "ets_headline", "fss_headline", "rmse", "mae", "bias_mm", "pattern_r",
    "mean_track_err_km", "max_track_err_km", "mean_along_km",
    "mean_cross_km", "vmax_bias_kt", "mslp_bias_hpa",
    "landfall_timing_err_h", "landfall_pos_err_km", "mean_dlat_deg",
    "mean_dlon_deg", "mean_displacement_km",
    "ets_shifted", "fss_shifted", "rmse_shifted", "pattern_r_shifted",
    "fcst_p90", "fcst_p95", "fcst_p99", "obs_p90", "obs_p95", "obs_p99",
    "fcst_volume_km3", "obs_volume_km3", "volume_ratio",
    "obj_area_fcst_km2", "obj_area_obs_km2", "obj_area_ratio",
    "obj_centroid_err_km", "obj_centroid_along_km",
    "obj_centroid_cross_km", "obj_angle_diff_deg", "obj_mean_ratio",
    "obj_max_ratio",
]


# =============================================================================
# Union footprint
# =============================================================================

def window_track_points(case, valid_start, valid_end):
    """Hourly (lat, lon) track positions of one cycle inside the window,
    endpoints inclusive."""
    pts = []
    t = valid_start
    while t <= valid_end:
        pts.append(case.position_at(t))
        t += timedelta(hours=1)
    return pts


def union_swath(track_points, radius_km, grid_lat, grid_lon):
    """Boolean mask: grid points within radius_km of ANY (lat, lon) point."""
    swath = np.zeros(grid_lat.shape, dtype=bool)
    for tlat, tlon in track_points:
        swath |= haversine_km(tlat, tlon, grid_lat, grid_lon) <= radius_km
    return swath


# =============================================================================
# Windowed parent forecast fields
# =============================================================================


def parent_window_total(case, f1, f2, grid_lat, grid_lon):
    """Parent QPF for the window: cumulative APCP at f2 minus at f1.

    The 6-km parent domain is fixed, so its 0->fhour cumulative record is
    geographically valid and the window total is a clean difference.
    Interpolation noise can leave tiny negatives; those are floored at 0
    (NaN outside the parent hull is preserved). Raises RuntimeError when a
    needed parent file or record is missing.

    Sweeping consecutive windows (…f6, f6->f12, f12->f18…) asks for each
    cumulative field twice: once as a window's f2, then as the next window's
    f1. The regridded cumulative field is memoized on the case (keyed by
    fhour + grid shape) so that repeated decode+regrid work is done once per
    fhour per case. The fixed mesh is constant for a case's lifetime, so the
    grid-shape key only guards against accidental cross-grid reuse.
    """
    cache = getattr(case, "_parent_cumulative_cache", None)
    if cache is None:
        cache = {}
        try:
            case._parent_cumulative_cache = cache
        except AttributeError:  # slotted/frozen case: run without memoization
            cache = None

    def cumulative_at(fh):
        key = (fh, grid_lat.shape)
        if cache is not None and key in cache:
            return cache[key]
        path = parent_path_at_fhour(case, fh)
        if path is None:
            raise RuntimeError(
                f"no parent.atm file at f{fh:03d} for init {case.init_str}")
        rec = pick_cumulative_record(read_hafs_tp_records(path))
        if rec is None:
            raise RuntimeError(
                f"no APCP record in parent f{fh:03d} for init {case.init_str}")
        field = regrid_2d_to_fixed(rec["lats"], rec["lons"], rec["data"],
                                   grid_lat, grid_lon)
        if cache is not None:
            cache[key] = field
        return field

    # cumulative_at returns the cached array by reference; the arithmetic
    # below (subtraction, np.where) always allocates new arrays, so callers
    # never receive a handle that could mutate a cached field in place.
    total = cumulative_at(f2)
    if f1 > 0:
        total = total - cumulative_at(f1)
    return np.where(total < 0, 0.0, total)


# =============================================================================
# Field building (once per cycles case)
# =============================================================================

def build_cycle_fields(ccase):
    """Build everything the cycles product scores and plots.

    Each cycle contains its effective valid_start, the common valid_end, and
    matching parent, MRMS, and optional Stage IV accumulations. Raises when no
    cycle is eligible or every eligible cycle fails field extraction.
    """
    init_strs = ccase.inits or discover_inits(ccase.run_root)
    if not init_strs:
        raise RuntimeError(
            f"No YYYYMMDDHH cycle directories under {ccase.run_root} "
            f"and no 'inits' list given.")
    print(f"Window {ccase.valid_start:%Y-%m-%d %HZ} -> "
          f"{ccase.valid_end:%Y-%m-%d %HZ} | candidate inits: "
          f"{', '.join(init_strs)}")

    grid_lat, grid_lon = ccase.fixed_grid()
    print(f"Fixed grid: {grid_lat.shape[0]}x{grid_lat.shape[1]} "
          f"@ {ccase.grid_res}deg")

    # Pass 1: load cases; keep cycles that reach the common window end.
    cases = []
    for init_str in init_strs:
        try:
            case = cycle_storm_case(ccase, init_str)
        except (FileNotFoundError, ValueError) as e:
            print(f"  skip {init_str}: {e}")
            continue
        file_pairs = discover_files(case.run_dir, case.parent_glob(), None)
        if not file_pairs:
            print(f"  skip {init_str}: no parent files matching "
                  f"{case.parent_glob()}")
            continue
        max_fhour = file_pairs[-1][0]
        ok, reason = cycle_eligibility(case.init_dt, max_fhour,
                                       ccase.valid_start, ccase.valid_end)
        if not ok:
            print(f"  skip {init_str}: {reason}")
            continue
        cases.append(case)
    if not cases:
        raise RuntimeError(
            f"No eligible cycles for window "
            f"{ccase.valid_start:%Y%m%d%H}->{ccase.valid_end:%Y%m%d%H} "
            f"(inspected: {', '.join(init_strs)})")

    # Per-cycle forecast windows.
    cycles = []
    survivors = []
    for case in cases:
        effective_start = max(ccase.valid_start, case.init_dt)
        f1, f2 = window_hours(case.init_dt, effective_start,
                              ccase.valid_end)
        print(f"\nCycle {case.init_str} (valid {effective_start:%Y-%m-%d %HZ}"
              f" -> {ccase.valid_end:%Y-%m-%d %HZ}; f{f1:03d}-f{f2:03d})")
        try:
            parent_win = parent_window_total(case, f1, f2,
                                             grid_lat, grid_lon)
        except RuntimeError as e:
            print(f"  skip {case.init_str}: {e}")
            continue
        cycles.append(dict(init_str=case.init_str, init_dt=case.init_dt,
                           valid_start=effective_start,
                           valid_end=ccase.valid_end,
                           f1=f1, f2=f2, parent_win=parent_win,
                           track_fixes=case.track_fixes, _case=case))
        survivors.append(case)
    if not cycles:
        raise RuntimeError("Every eligible cycle failed field extraction.")

    # Shared footprint: union of every surviving cycle's in-window track.
    print("Union verification swath ...")
    all_pts = []
    for case, cycle in zip(survivors, cycles):
        all_pts.extend(window_track_points(case, cycle["valid_start"],
                                           cycle["valid_end"]))
    swath = union_swath(all_pts, ccase.mask_radius_km, grid_lat, grid_lon)
    print(f"  swath: {int(swath.sum()):,} grid points from "
          f"{len(survivors)} track(s)")

    # Matching observations for each init-clipped forecast window.
    for case, cycle in zip(survivors, cycles):
        start, end = cycle["valid_start"], cycle["valid_end"]
        print(f"MRMS total for {case.init_str}: {start:%m-%d %HZ} -> "
              f"{end:%m-%d %HZ} ...")
        cycle["mrms_win"] = build_mrms_total_window(
            start, end, ccase.mrms_cache_dir, grid_lat, grid_lon)
        points = window_track_points(case, start, end)
        print(f"Stage IV total for {case.init_str} ...")
        s4_lat, s4_lon, s4_native, s4_label = stage4_total_window(
            ccase.stage4_cache_dir, start, end, points,
            ccase.display_radius_km)
        if s4_native is None:
            cycle["stage4_win"], cycle["s4_label"] = None, "unavailable"
        else:
            cycle["stage4_win"] = regrid_2d_to_fixed(
                s4_lat, s4_lon, s4_native, grid_lat, grid_lon)
            cycle["s4_label"] = s4_label

    if all(cycle["stage4_win"] is None for cycle in cycles):
        print("  Stage IV unavailable — scoring MRMS only.")
    # Retain representative top-level observation keys for compatibility with
    # callers that display the earliest cycle's fields.
    first = cycles[0]
    return dict(grid_lat=grid_lat, grid_lon=grid_lon,
                mrms_win=first["mrms_win"],
                stage4_win=first["stage4_win"],
                s4_label=first["s4_label"], swath=swath, cycles=cycles)


# =============================================================================
# Scoring + outputs
# =============================================================================

_PLOT_TYPOGRAPHY = {
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.titleweight": "bold",
}


def _plot_theme():
    """Use seaborn when available while retaining a dependency-light fallback."""
    if sns is not None:
        sns.set_theme(context="notebook", style="whitegrid", font_scale=1.0,
                      rc=_PLOT_TYPOGRAPHY)
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(_PLOT_TYPOGRAPHY)


def _inches_to_mm(value):
    """Stable inch-to-mm conversion for threshold lookup and de-duplication."""
    return round(float(value) * 25.4, 6)


def _csv_value(value):
    """Render missing and non-finite summary values as empty CSV cells."""
    if value is None:
        return ""
    try:
        if not np.isfinite(value):
            return ""
    except TypeError:
        pass
    return value


def hours_before_landfall(ccase, init_dt):
    """Hours from initialization to landfall, or None when not configured."""
    if ccase.landfall_time is None:
        return None
    return (ccase.landfall_time - init_dt).total_seconds() / 3600.0


def cycle_label(ccase, init_dt):
    """Compact cycle label with optional landfall-relative lead time."""
    lead = hours_before_landfall(ccase, init_dt)
    if lead is None:
        return init_dt.strftime("%m-%d %HZ")
    return f"{init_dt:%m-%d %HZ}\n{lead:.0f} h pre-LF"


# Preserve every native QPF color boundary, including the 0.2- and 0.4-inch
# bands that carry important swath detail. Only the displayed tick text is
# rounded to friendly inch values.
_ANIMATION_QPF_LEVELS_IN = (
    np.asarray(QPF_LEVELS, dtype=float) / 25.4
)


def cycle_window_label(cycle):
    """Compact effective accumulation window for a cycle panel."""
    start = cycle.get("valid_start")
    end = cycle.get("valid_end")
    if start is None or end is None:
        return ""
    return f"{start:%m-%d %HZ}–{end:%m-%d %HZ}"


def _cycle_observation(cycle, fields, name):
    """Get a per-cycle observation with legacy top-level fallback."""
    key = "mrms_win" if name == "MRMS" else "stage4_win"
    return cycle.get(key, fields.get(key))


def cycles_caveat(fields, ccase):
    """Figure-footer caveat describing the Stage IV window (or its absence)."""
    if not any(_cycle_observation(cycle, fields, "Stage IV") is not None
               for cycle in fields["cycles"]):
        return "Stage IV unavailable — not scored."
    return (f"Stage IV: CONUS-only, 24h 12Z–12Z files summed over touched "
            f"days; later initializations use init-clipped windows ending "
            f"{ccase.valid_end:%m-%d %HZ}.")


def plot_metrics(ccase, results, out_path, caveat=""):
    """RMSE and bias versus initialization/landfall lead.

    results: list of dicts {init_str, init_dt, forecast, observation,
    cont, rows} — one per cycle x pair.
    """
    _plot_theme()
    inits = sorted({r["init_dt"] for r in results})
    pairs = sorted({(r["forecast"], r["observation"]) for r in results},
                   key=lambda p: (p[1], p[0]))
    use_lead = ccase.landfall_time is not None
    x = ([hours_before_landfall(ccase, init_dt) for init_dt in inits]
         if use_lead else inits)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)

    def series(fname, oname, getter):
        by_init = {r["init_dt"]: r for r in results
                   if r["forecast"] == fname and r["observation"] == oname}
        return [getter(by_init[i]) if i in by_init else np.nan
                for i in inits]

    panels = [
        (axes[0], lambda res: inches(res["cont"]["rmse"]), "RMSE (inches)"),
        (axes[1], lambda res: inches(res["cont"]["bias"]), "bias (inches)"),
    ]
    for ax, getter, label in panels:
        for fname, oname in pairs:
            style = _FCST_STYLE.get(fname, dict(ls="-", marker="o"))
            ax.plot(x, series(fname, oname, getter),
                    color=_OBS_COLOR.get(oname, "gray"), lw=2, **style,
                    label=f"{fname} vs {oname}")
        ax.set_ylabel(label)
        ax.grid(True, ls=":", alpha=0.4)
    axes[1].axhline(0, color="gray", ls=":", lw=0.8)
    axes[0].legend(loc="best", fontsize=9)
    if use_lead:
        axes[-1].set_xlabel("Hours before landfall (forecast initialization)")
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([f"{value:.0f}" for value in x])
        axes[-1].invert_xaxis()
    else:
        axes[-1].set_xlabel("initialization")
        axes[-1].set_xticks(inits)
        axes[-1].set_xticklabels([i.strftime("%m-%d %HZ") for i in inits],
                                 rotation=45, ha="right")
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} QPF Errors by "
        "Initialization")
    if caveat:
        fig.text(0.5, -0.01, caveat, ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_maps(ccase, fields, out_path):
    """Small-multiple parent-QPF maps per cycle + observed MRMS panel.

    Shared color scale and extent; the union verification swath is
    outlined on every panel. Wraps at 4 columns.
    """
    panels = [(f"{cycle_label(ccase, c['init_dt'])}\n"
               f"{cycle_window_label(c)}", c["parent_win"])
              for c in fields["cycles"]]
    panels.append((f"MRMS observed\n{cycle_window_label(fields['cycles'][0])}",
                   _cycle_observation(fields["cycles"][0], fields, "MRMS")))
    n = len(panels)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    lat_min, lat_max, lon_min, lon_max = ccase.domain
    cmap, _ = qpf_cmap()
    levels_in = np.asarray(QPF_LEVELS, dtype=float) / 25.4
    norm = colors.BoundaryNorm(levels_in, cmap.N)
    grid_lat, grid_lon = fields["grid_lat"], fields["grid_lon"]

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 4.6 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    flat = axes.ravel()
    cf = None
    for ax, (title, data) in zip(flat, panels):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
        cf = ax.contourf(grid_lon, grid_lat,
                         np.nan_to_num(data / 25.4, nan=0.0),
                         levels=levels_in, cmap=cmap, norm=norm,
                         transform=ccrs.PlateCarree(), extend="max")
        ax.contour(grid_lon, grid_lat, fields["swath"].astype(float),
                   levels=[0.5], colors="k", linewidths=1.0,
                   transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=10)
    for ax in flat[n:]:
        ax.set_visible(False)
    if cf is not None:
        fig.colorbar(cf, ax=axes, label="Accumulated precipitation (inches)",
                     ticks=levels_in[::2], shrink=0.7, fraction=0.02,
                     format="%g")
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} parent QPF by "
        f"initialization\naccumulations begin at the later of "
        f"{ccase.valid_start:%Y-%m-%d %HZ} or initialization and end "
        f"{ccase.valid_end:%Y-%m-%d %HZ} | swath outline "
        f"≤{miles(ccase.mask_radius_km):.0f} miles", y=1.02)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ets_leadtime(ccase, results, out_path):
    """ETS heatmap with rainfall thresholds by cycle initialization."""
    _plot_theme()
    rows = [r for r in results
            if r["observation"] == "MRMS" and r["forecast"] == "parent"]
    inits = sorted({r["init_dt"] for r in rows})
    configured = sorted(set(ccase.thresholds_mm) | {ccase.ets_threshold_mm})
    available = {item["threshold"] for r in rows for item in r["rows"]}
    thresholds = [threshold for threshold in configured
                  if any(np.isclose(threshold, value) for value in available)]
    if not inits or not thresholds:
        raise ValueError("No parent-vs-MRMS ETS rows to plot.")

    by_init = {r["init_dt"]: {item["threshold"]: item["ets"]
                              for item in r["rows"]}
               for r in rows}
    values = np.asarray(
        [[by_init.get(init_dt, {}).get(threshold, np.nan) for init_dt in inits]
         for threshold in thresholds], dtype=float)
    xlabels = [init_dt.strftime("%m-%d %HZ") for init_dt in inits]
    ylabels = [f"≥ {format_inches(threshold)}" for threshold in thresholds]
    annotate = values.size <= 80
    fig, ax = plt.subplots(
        figsize=(max(8.0, 1.35 * len(inits) + 3.0),
                 max(4.2, 0.68 * len(thresholds) + 2.4)))
    colorbar = None
    if sns is not None:
        cmap = sns.diverging_palette(240, 15, s=80, l=45, as_cmap=True)
        ax.set_facecolor("#eceff1")
        sns.heatmap(
            values, ax=ax, cmap=cmap, vmin=-0.2, vmax=1.0, center=0.0,
            mask=~np.isfinite(values), annot=annotate, fmt=".2f",
            linewidths=0.8, linecolor="white", square=False,
            xticklabels=xlabels, yticklabels=ylabels,
            cbar_kws={"label": "Equitable Threat Score (ETS)",
                      "shrink": 0.88},
        )
        colorbar = ax.collections[0].colorbar
        for row, col in np.argwhere(~np.isfinite(values)):
            ax.text(col + 0.5, row + 0.5, "—", ha="center", va="center",
                    color="#7a7a7a", fontsize=9)
    else:
        norm = colors.TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=1.0)
        image = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
        colorbar = fig.colorbar(image, ax=ax,
                                 label="Equitable Threat Score (ETS)",
                                 shrink=0.88)
        ax.set_xticks(np.arange(len(inits)), labels=xlabels)
        ax.set_yticks(np.arange(len(thresholds)), labels=ylabels)
        if annotate:
            for row, col in np.argwhere(np.isfinite(values)):
                ax.text(col, row, f"{values[row, col]:.2f}",
                        ha="center", va="center", fontsize=8)
        for row, col in np.argwhere(~np.isfinite(values)):
            ax.text(col, row, "—", ha="center", va="center",
                    color="#7a7a7a", fontsize=9)

    ax.set_xlabel("Forecast Initialization (UTC)", labelpad=10,
                  fontsize=13, fontweight="semibold")
    ax.set_ylabel("Rainfall Threshold (in)", fontsize=13,
                  fontweight="semibold")
    ax.tick_params(axis="x", labelrotation=0, labelsize=11)
    ax.tick_params(axis="y", labelrotation=0, labelsize=11)
    if colorbar is not None:
        colorbar.set_label("Equitable Threat Score (ETS)", fontsize=13,
                           fontweight="semibold")
        colorbar.ax.tick_params(labelsize=11)
    ax.set_title(
        f"{ccase.storm_name} — {ccase.model_label} Parent ETS Progression",
        fontsize=18, fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pooled_ets_by_threshold(results, thresholds_in):
    """Pool contingency counts across parent-vs-MRMS cycles, then compute ETS.

    Each cycle is a separate forecast of the shared valid window. Pooling its
    hits, false alarms, misses, and correct negatives mirrors a model-level
    FULL summary while preserving the existing common swath and observations.
    """
    rows = [r for r in results
            if r["observation"] == "MRMS" and r["forecast"] == "parent"]
    summary = []
    for threshold_in in thresholds_in:
        threshold_mm = _inches_to_mm(threshold_in)
        matched = [(r, item) for r in rows for item in r["rows"]
                   if np.isclose(item["threshold"], threshold_mm)]
        counts = {key: sum(int(item[key]) for _, item in matched)
                  for key in ("a", "b", "c", "d")}
        n = sum(counts.values())
        a_ref = ((counts["a"] + counts["b"])
                 * (counts["a"] + counts["c"]) / n) if n else 0.0
        denominator = counts["a"] + counts["b"] + counts["c"] - a_ref
        ets = ((counts["a"] - a_ref) / denominator
               if denominator != 0 else np.nan)
        summary.append({
            "threshold_in": float(threshold_in),
            "threshold_mm": threshold_mm,
            "n_cycles": len({r["init_dt"] for r, _ in matched}),
            "ets": float(ets),
            **counts,
        })
    return summary


def plot_ets_threshold_bars(ccase, results, out_path):
    """Paper-style model ETS bars at even-inch rainfall thresholds."""
    _plot_theme()
    summary = pooled_ets_by_threshold(results, ccase.ets_bar_thresholds_in)
    if not summary:
        raise ValueError("No ETS bar thresholds configured.")

    x = np.arange(len(summary))
    values = np.asarray([row["ets"] for row in summary], dtype=float)
    palette = (sns.color_palette("crest", len(summary)) if sns is not None
               else plt.cm.viridis(np.linspace(0.25, 0.85, len(summary))))
    fig, ax = plt.subplots(figsize=(12.2, 6.2))
    fig.patch.set_facecolor("#f7f9fb")
    ax.set_facecolor("#f7f9fb")
    bars = ax.bar(x, np.nan_to_num(values, nan=0.0), width=0.72,
                  color=palette, edgecolor="white", linewidth=1.0)

    finite = values[np.isfinite(values)]
    ymin = min(-0.05, float(finite.min()) - 0.04) if finite.size else -0.05
    ymax = max(0.4, float(finite.max()) + 0.08) if finite.size else 0.4
    ymax = min(1.0, ymax)
    ax.set_ylim(ymin, ymax)
    offset = 0.015 * (ymax - ymin)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value + (offset if value >= 0 else -offset),
                    f"{value:.2f}", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=8)
        else:
            bar.set_facecolor("#d9d9d9")
            bar.set_hatch("//")
            ax.text(bar.get_x() + bar.get_width() / 2, offset, "N/A",
                    ha="center", va="bottom", fontsize=8, color="#666666")

    n_cycles = max((row["n_cycles"] for row in summary), default=0)
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, [f"{row['threshold_in']:g}" for row in summary])
    ax.set_xlabel("Rainfall Accumulation Threshold (in)")
    ax.set_ylabel("Equitable Threat Score (ETS)")
    ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        f"{ccase.storm_name} · {ccase.model_label} Parent Rainfall Skill "
        f"Average ETS Across {n_cycles} Forecast Cycles vs MRMS",
        fontsize=14, fontweight="semibold", pad=15, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fss_leadtime(ccase, fss_rows, out_path):
    """FSS heatmaps by cycle, faceted across rainfall thresholds."""
    _plot_theme()
    rows = [r for r in fss_rows
            if r["observation"] == "MRMS" and r["forecast"] == "parent"]
    inits = sorted({r["init_dt"] for r in rows})
    thresholds = sorted({r.get("threshold_in", r["threshold"] / 25.4)
                         for r in rows})
    scales = sorted({r["scale_km"] for r in rows})
    if not inits or not thresholds or not scales:
        raise ValueError("No parent-vs-MRMS FSS rows to plot.")

    # A label for every six-hourly cycle is unreadable once the suite reaches
    # ten-plus initializations. Keep every other label while retaining every
    # heatmap column.
    xlabels = [f"{init_dt:%m-%d}\n{init_dt:%HZ}" if index % 2 == 0 else ""
               for index, init_dt in enumerate(inits)]
    ylabels = [format_miles(scale) for scale in scales]
    fig = plt.figure(figsize=(max(11.5, 0.72 * len(inits) + 4.5),
                              max(4.8, 0.56 * len(scales) + 2.8)))
    grid = fig.add_gridspec(
        1, len(thresholds) + 1,
        width_ratios=[1.0] * len(thresholds) + [0.055], wspace=0.18)
    axes = [fig.add_subplot(grid[0, index])
            for index in range(len(thresholds))]
    cbar_ax = fig.add_subplot(grid[0, -1])
    fallback_image = None
    for index, (ax, threshold) in enumerate(zip(axes, thresholds)):
        lookup = {(r["init_dt"], r["scale_km"]): r["fss"] for r in rows
                  if np.isclose(r.get("threshold_in",
                                      r["threshold"] / 25.4), threshold)}
        values = np.asarray(
            [[lookup.get((init_dt, scale), np.nan) for init_dt in inits]
             for scale in scales], dtype=float)
        if sns is not None:
            ax.set_facecolor("#eceff1")
            sns.heatmap(
                values, ax=ax, cmap="mako", vmin=0.0, vmax=1.0,
                mask=~np.isfinite(values), annot=False,
                linewidths=0.8, linecolor="white", square=False,
                xticklabels=xlabels,
                yticklabels=ylabels if index == 0 else False,
                cbar=index == len(thresholds) - 1, cbar_ax=cbar_ax,
                cbar_kws={"label": "Fractions Skill Score (FSS)"},
            )
        else:
            fallback_image = ax.imshow(values, cmap="viridis", vmin=0.0,
                                       vmax=1.0, aspect="auto")
            ax.set_xticks(np.arange(len(inits)), labels=xlabels)
            ax.set_yticks(np.arange(len(scales)), labels=ylabels)
        ax.set_title(f"Rainfall ≥ {threshold:g} in", fontsize=14,
                     fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=0, labelsize=10, pad=4)
        ax.tick_params(axis="y", labelrotation=0, labelsize=11)
    if fallback_image is not None:
        fig.colorbar(fallback_image, cax=cbar_ax,
                     label="Fractions Skill Score (FSS)")

    fig.supxlabel("Forecast Initialization (UTC)", y=0.04, fontsize=13,
                  fontweight="semibold")
    fig.supylabel("Neighborhood Scale (mi)", x=0.02, fontsize=13,
                  fontweight="semibold")
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} Parent FSS Progression",
        fontsize=18, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.95, bottom=0.20, top=0.80,
                        wspace=0.10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _map_context(ax, ccase):
    lat_min, lat_max, lon_min, lon_max = ccase.domain
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")


_BASE_ERROR_LEVELS_IN = np.asarray(
    [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
     3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0],
    dtype=float,
)


def precipitation_error_levels(error_fields):
    """Symmetric nonlinear inch bounds for forecast-minus-observation maps.

    Small errors receive fine color steps. Above two inches, each successive
    bin widens so unusually large tropical-cyclone rainfall errors remain on
    the scale without sacrificing detail near zero. The base scale reaches
    24 inches and grows geometrically when a field exceeds that range.
    """
    positive = list(_BASE_ERROR_LEVELS_IN)
    finite = [np.abs(np.asarray(field, dtype=float)[np.isfinite(field)])
              for field in error_fields]
    finite = [values for values in finite if values.size]
    maximum = max((float(np.max(values)) for values in finite), default=0.0)
    step = positive[-1] - positive[-2]
    while positive[-1] < maximum:
        step *= 1.5
        positive.append(positive[-1] + step)
    positive = np.asarray(positive)
    return np.concatenate((-positive[:0:-1], positive))


def _precipitation_error_scale(error_fields):
    levels = precipitation_error_levels(error_fields)
    cmap = plt.get_cmap("RdBu_r", len(levels) - 1)
    norm = colors.BoundaryNorm(levels, cmap.N)
    preferred = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    positive = levels[levels >= 0]
    ticks = [value for value in preferred
             if np.any(np.isclose(positive, value))]
    if positive[-1] > preferred[-1]:
        ticks.append(float(positive[-1]))
    ticks = np.asarray(ticks)
    ticks = np.concatenate((-ticks[:0:-1], ticks))
    return cmap, norm, ticks


def _animate_cycle_field(ccase, fields, out_path, data_by_cycle, cmap, norm,
                         colorbar_label, panel_title, colorbar_ticks=None,
                         colorbar_spacing=None):
    """Animate one cycle field so each GIF remains readable on its own."""
    cycles_data = fields["cycles"]
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(7.6, 5.8),
                           subplot_kw={"projection": projection})
    fig.subplots_adjust(left=0.06, right=0.82, bottom=0.08, top=0.76)
    colorbar_kwargs = {
        "ax": ax,
        "shrink": 0.94,
        "fraction": 0.052,
        "pad": 0.035,
        "label": colorbar_label,
    }
    if colorbar_ticks is not None:
        colorbar_kwargs["ticks"] = colorbar_ticks
    if colorbar_spacing is not None:
        colorbar_kwargs["spacing"] = colorbar_spacing
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar_map, **colorbar_kwargs)
    colorbar.ax.tick_params(labelsize=10, pad=5)
    colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda value, _: (
            f"{int(round(value))}"
            if np.isclose(value, round(value), atol=0.03)
            else (f"{value:.1f}" if abs(value) < 1.0 else f"{value:.0f}")
        )
    ))
    colorbar.set_label(colorbar_label, fontsize=11, fontweight="bold",
                       labelpad=12)

    def update(frame):
        cycle = cycles_data[frame]
        ax.clear()
        _map_context(ax, ccase)
        masked = np.where(fields["swath"], data_by_cycle[frame], np.nan)
        ax.pcolormesh(fields["grid_lon"], fields["grid_lat"], masked,
                      cmap=cmap, norm=norm, shading="auto",
                      transform=projection)
        ax.contour(fields["grid_lon"], fields["grid_lat"],
                   fields["swath"].astype(float), levels=[0.5],
                   colors="#333333", linewidths=0.7,
                   transform=projection)
        ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=6)
        init_label = cycle_label(ccase, cycle["init_dt"]).replace(
            "\n", " · ")
        fig.suptitle(
            f"{ccase.storm_name} — {ccase.model_label}\n"
            f"{init_label} | {cycle_window_label(cycle)}",
            fontsize=13, fontweight="bold", linespacing=1.35, y=0.97)
        return (ax,)

    movie = animation.FuncAnimation(fig, update, frames=len(cycles_data),
                                    interval=1200, repeat=True, blit=False)
    movie.save(out_path, writer=animation.PillowWriter(fps=1), dpi=140)
    plt.close(fig)


def animate_cycle_qpf(ccase, fields, out_path):
    """Animate parent-domain forecast QPF across initializations."""
    qpf_cmap_obj, _ = qpf_cmap()
    qpf_norm = colors.BoundaryNorm(
        _ANIMATION_QPF_LEVELS_IN, qpf_cmap_obj.N)
    forecasts = [cycle["parent_win"] / 25.4
                 for cycle in fields["cycles"]]
    _animate_cycle_field(
        ccase, fields, out_path, forecasts, qpf_cmap_obj, qpf_norm,
        "Accumulated Precipitation (in)", "Parent Forecast",
        colorbar_ticks=_ANIMATION_QPF_LEVELS_IN,
        colorbar_spacing="uniform")


def animate_cycle_difference(ccase, fields, out_path):
    """Animate parent forecast minus MRMS across initializations."""
    errors = [(cycle["parent_win"]
              - _cycle_observation(cycle, fields, "MRMS")) / 25.4
              for cycle in fields["cycles"]]
    error_cmap, error_norm, error_ticks = _precipitation_error_scale(errors)
    _animate_cycle_field(
        ccase, fields, out_path, errors, error_cmap, error_norm,
        "Forecast − MRMS (in)",
        "Parent − MRMS", colorbar_ticks=error_ticks,
        colorbar_spacing="uniform")


def animate_cycle_observed(ccase, fields, out_path):
    """Animate cycle-matched MRMS QPE (static while its window is unchanged)."""
    qpf_cmap_obj, _ = qpf_cmap()
    qpf_norm = colors.BoundaryNorm(
        _ANIMATION_QPF_LEVELS_IN, qpf_cmap_obj.N)
    observations = [
        _cycle_observation(cycle, fields, "MRMS") / 25.4
        for cycle in fields["cycles"]
    ]
    _animate_cycle_field(
        ccase, fields, out_path, observations, qpf_cmap_obj, qpf_norm,
        "Accumulated Precipitation (in)", "MRMS Observed",
        colorbar_ticks=_ANIMATION_QPF_LEVELS_IN,
        colorbar_spacing="uniform")


def compute_cycles(ccase, fields=None):
    if fields is None:
        fields = build_cycle_fields(ccase)
    swath = fields["swath"]
    # Make sure the headline ETS threshold is actually scored.
    thresholds = list(ccase.thresholds_mm)
    if ccase.ets_threshold_mm not in thresholds:
        thresholds = sorted(set(thresholds) | {ccase.ets_threshold_mm})
    thresholds = sorted(set(thresholds) | {
        _inches_to_mm(value) for value in ccase.ets_bar_thresholds_in})

    results = []
    fss_rows = []
    dist_rows = []
    bdeck_full = parse_bdeck_full(ccase.best_track) if ccase.best_track else None
    fss_thresholds_in = list(ccase.fss_thresholds_in)
    if (ccase.headline_fss_threshold_in is not None
            and ccase.headline_fss_threshold_in not in fss_thresholds_in):
        fss_thresholds_in.append(ccase.headline_fss_threshold_in)
    fss_scales_cells = list(ccase.fss_scales_cells)
    if (ccase.headline_fss_scale_cells is not None
            and ccase.headline_fss_scale_cells not in fss_scales_cells):
        fss_scales_cells.append(ccase.headline_fss_scale_cells)
    print("\n" + "=" * 84)
    for cyc in fields["cycles"]:
        observations = [("MRMS", _cycle_observation(cyc, fields, "MRMS"))]
        stage4_win = _cycle_observation(cyc, fields, "Stage IV")
        if stage4_win is not None:
            observations.append(("Stage IV", stage4_win))
        for fname, fgrid in (("parent", cyc["parent_win"]),):
            for oname, ogrid in observations:
                fcst, obs = valid_points(fgrid, ogrid, swath)
                cont = continuous_scores(fcst, obs)
                rows = [contingency_scores(fcst, obs, thr)
                        for thr in thresholds]
                results.append(dict(init_str=cyc["init_str"],
                                    init_dt=cyc["init_dt"],
                                    valid_start=cyc.get("valid_start",
                                                        ccase.valid_start),
                                    valid_end=cyc.get("valid_end",
                                                      ccase.valid_end),
                                    forecast=fname, observation=oname,
                                    cont=cont, rows=rows))
                print(f"{cyc['init_str']} {fname:>7} vs {oname:<9} "
                      f"n={cont['n']:>9,} RMSE={cont['rmse']:>7.2f} "
                      f"MAE={cont['mae']:>7.2f} bias={cont['bias']:>+7.2f} "
                      f"r={cont['r']:>5.2f}")
                if oname == "MRMS":
                    valid = (swath & np.isfinite(fgrid)
                             & np.isfinite(ogrid))
                    clean_fcst = np.nan_to_num(fgrid, nan=0.0)
                    clean_obs = np.nan_to_num(ogrid, nan=0.0)
                    for threshold_in in fss_thresholds_in:
                        threshold = _inches_to_mm(threshold_in)
                        for scale in fss_scales_cells:
                            fss_rows.append({
                                "init": cyc["init_str"],
                                "init_dt": cyc["init_dt"],
                                "lead_hours_to_landfall":
                                    hours_before_landfall(ccase, cyc["init_dt"]),
                                "forecast": fname,
                                "observation": oname,
                                "threshold": threshold,
                                "threshold_in": threshold_in,
                                "scale_cells": scale,
                                "scale_km": round(
                                    scale * ccase.grid_res * 111.0, 1),
                                "fss": fractions_skill_score(
                                    clean_fcst, clean_obs, threshold,
                                    scale, valid),
                            })
    print("=" * 84)

    for cyc in fields["cycles"]:
        sources = [
            ("parent", cyc["parent_win"]),
            ("MRMS", _cycle_observation(cyc, fields, "MRMS")),
        ]
        stage4_win = _cycle_observation(cyc, fields, "Stage IV")
        if stage4_win is not None:
            sources.append(("StageIV", stage4_win))
        for source, field in sources:
            dist_rows.append({
                "init": cyc["init_str"],
                "source": source,
                **distribution_stats(field, swath, fields["grid_lat"],
                                     ccase.grid_res),
            })

    ccase.out_dir.mkdir(parents=True, exist_ok=True)
    slug = ccase.output_slug
    out_csv = ccase.out_dir / f"cycles_{slug}.csv"

    fieldnames = ["init", "valid_start", "valid_end",
                  "lead_hours_to_landfall", "forecast",
                  "observation", "threshold", "n",
                  "rmse", "mae", "bias_mm", "r", "a", "b", "c", "d",
                  "ets", "bias", "pod", "far", "csi", "hss"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for res in results:
            cont = res["cont"]
            for r in res["rows"]:
                w.writerow({"init": res["init_str"],
                            "valid_start": res["valid_start"].strftime(
                                "%Y%m%d%H"),
                            "valid_end": res["valid_end"].strftime(
                                "%Y%m%d%H"),
                            "lead_hours_to_landfall": hours_before_landfall(
                                ccase, res["init_dt"]),
                            "forecast": res["forecast"],
                            "observation": res["observation"],
                            "n": cont["n"], "rmse": cont["rmse"],
                            "mae": cont["mae"], "bias_mm": cont["bias"],
                            "r": cont["r"], **r})
    print(f"\nSaved table: {out_csv}")

    out_fss_csv = ccase.out_dir / f"cycles_fss_{slug}.csv"
    with open(out_fss_csv, "w", newline="") as fh:
        fieldnames = ["init", "lead_hours_to_landfall", "forecast",
                      "observation", "threshold", "threshold_in", "scale_cells",
                      "scale_km", "fss"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fss_rows)
    print(f"Saved table: {out_fss_csv}")

    out_dist_csv = ccase.out_dir / f"cycles_dist_{slug}.csv"
    dist_fields = ["init", "source", "p50", "p90", "p95", "p99",
                   "max_mm", "volume_km3", "wet_frac"]
    with open(out_dist_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=dist_fields)
        writer.writeheader()
        for row in dist_rows:
            writer.writerow({key: _csv_value(row.get(key))
                             for key in dist_fields})
    print(f"Saved table: {out_dist_csv}")

    track_rows_by_init = {}
    track_summaries_by_init = {}
    if bdeck_full is not None:
        for cyc in fields["cycles"]:
            fixes = cyc.get("track_fixes")
            if not fixes:
                continue
            rows = track_error_rows(
                fixes, bdeck_full,
                cyc.get("valid_start", ccase.valid_start),
                cyc.get("valid_end", ccase.valid_end),
                ccase.track_step_hours)
            landfall = landfall_metrics(fixes, bdeck_full,
                                        ccase.landfall_time)
            track_rows_by_init[cyc["init_dt"]] = rows
            track_summaries_by_init[cyc["init_dt"]] = cycle_track_summary(
                rows, landfall)
        out_track_csv = ccase.out_dir / f"cycles_track_{slug}.csv"
        track_fields = [
            "init", "lead_hours_to_landfall", "valid", "fhr",
            "pos_err_km", "along_km", "cross_km", "dlat_deg", "dlon_deg",
            "vmax_err_kt", "mslp_err_hpa",
        ]
        with open(out_track_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=track_fields)
            writer.writeheader()
            for init_dt, rows in sorted(track_rows_by_init.items()):
                for row in rows:
                    output = {key: _csv_value(row.get(key))
                              for key in track_fields}
                    output["init"] = init_dt.strftime("%Y%m%d%H")
                    output["lead_hours_to_landfall"] = _csv_value(
                        hours_before_landfall(ccase, init_dt))
                    output["valid"] = row["valid"].strftime("%Y%m%d%H")
                    writer.writerow(output)
        print(f"Saved table: {out_track_csv}")
    else:
        print("Best track unavailable — track CSV and plots not produced.")

    # The default headline FSS uses the first threshold and middle scale.
    headline_threshold_in = (
        ccase.headline_fss_threshold_in
        if ccase.headline_fss_threshold_in is not None
        else ccase.fss_thresholds_in[0])
    headline_scale_cells = (
        ccase.headline_fss_scale_cells
        if ccase.headline_fss_scale_cells is not None
        else ccase.fss_scales_cells[len(ccase.fss_scales_cells) // 2])
    summary_rows = []
    for cyc in fields["cycles"]:
        init_dt = cyc["init_dt"]
        precip = next((r for r in results
                       if r["init_dt"] == init_dt
                       and r["forecast"] == "parent"
                       and r["observation"] == "MRMS"), None)
        ets = np.nan
        if precip is not None:
            ets = next((r["ets"] for r in precip["rows"]
                        if r["threshold"] == ccase.ets_threshold_mm), np.nan)
        fss = next((r["fss"] for r in fss_rows
                    if r["init_dt"] == init_dt
                    and r["forecast"] == "parent"
                    and r["observation"] == "MRMS"
                    and r["threshold_in"] == headline_threshold_in
                    and r["scale_cells"] == headline_scale_cells), np.nan)
        cont = precip["cont"] if precip is not None else {}
        row = {
            "init": cyc["init_str"],
            "init_dt": init_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "lead_hours_to_landfall": hours_before_landfall(ccase, init_dt),
            "valid_start": cyc.get("valid_start", ccase.valid_start).strftime(
                "%Y%m%d%H"),
            "valid_end": cyc.get("valid_end", ccase.valid_end).strftime(
                "%Y%m%d%H"),
            "ets_headline": ets,
            "fss_headline": fss,
            "rmse": cont.get("rmse"),
            "mae": cont.get("mae"),
            "bias_mm": cont.get("bias"),
            "pattern_r": cont.get("r"),
            "_init_dt": init_dt,
        }
        dist_by_source = {item["source"]: item for item in dist_rows
                          if item["init"] == cyc["init_str"]}
        fcst_dist = dist_by_source.get("parent", {})
        obs_dist = dist_by_source.get("MRMS", {})
        for percentile in (90, 95, 99):
            row[f"fcst_p{percentile}"] = fcst_dist.get(f"p{percentile}", np.nan)
            row[f"obs_p{percentile}"] = obs_dist.get(f"p{percentile}", np.nan)
        row["fcst_volume_km3"] = fcst_dist.get("volume_km3", np.nan)
        row["obs_volume_km3"] = obs_dist.get("volume_km3", np.nan)
        obs_volume = row["obs_volume_km3"]
        row["volume_ratio"] = (
            row["fcst_volume_km3"] / obs_volume
            if np.isfinite(row["fcst_volume_km3"])
            and np.isfinite(obs_volume) and obs_volume != 0 else np.nan)
        midpoint = (cyc.get("valid_start", ccase.valid_start)
                    + (cyc.get("valid_end", ccase.valid_end)
                       - cyc.get("valid_start", ccase.valid_start)) / 2)
        motion_unit = (motion_vector(bdeck_full, midpoint)
                       if bdeck_full is not None else None)
        row.update(object_comparison(
            cyc["parent_win"], _cycle_observation(cyc, fields, "MRMS"),
            swath, fields["grid_lat"], fields["grid_lon"],
            ccase.object_threshold_mm, ccase.object_smooth_cells,
            ccase.object_min_area_cells, motion_unit))
        row.update(track_summaries_by_init.get(init_dt, {}))
        if bdeck_full is not None:
            dlat, dlon = mean_displacement_deg(
                track_rows_by_init.get(init_dt, []))
            shifted = score_shifted(
                cyc["parent_win"], _cycle_observation(cyc, fields, "MRMS"),
                swath, ccase.ets_threshold_mm,
                _inches_to_mm(headline_threshold_in), headline_scale_cells,
                dlat, dlon, ccase.grid_res)
            row.update(shifted)
            print(f"  shifted ETS {ets:.3f} -> "
                  f"{shifted['ets_shifted']:.3f}")
        summary_rows.append(row)
    out_summary_csv = ccase.out_dir / f"cycles_summary_{slug}.csv"
    with open(out_summary_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: _csv_value(row.get(key))
                             for key in SUMMARY_FIELDS})
    print(f"Saved table: {out_summary_csv}")

    caveat = cycles_caveat(fields, ccase)
    print(caveat)
    out_metrics = ccase.out_dir / f"cycles_metrics_{slug}.png"
    # Keep the established filenames for download and gallery compatibility.
    out_ets_leadtime = ccase.out_dir / f"cycles_ets_heatmap_{slug}.png"
    out_ets_bars = ccase.out_dir / f"cycles_ets_bars_{slug}.png"
    out_fss_leadtime = ccase.out_dir / f"cycles_fss_heatmap_{slug}.png"
    obsolete_plots = [
        ccase.out_dir / f"cycles_errors_{slug}.png",
        ccase.out_dir / f"cycles_maps_{slug}.png",
        ccase.out_dir / f"cycles_landfall_{slug}.png",
        ccase.out_dir / f"cycles_objects_{slug}.png",
    ]
    for obsolete_plot in obsolete_plots:
        if obsolete_plot.exists():
            obsolete_plot.unlink()
            print(f"Removed obsolete plot: {obsolete_plot}")
    structure_plots = [
        (ccase.out_dir / f"cycles_dist_{slug}.png",
         lambda path: plot_distributions(ccase, fields, path)),
        (ccase.out_dir / f"cycles_percentiles_{slug}.png",
         lambda path: plot_percentiles_by_cycle(ccase, summary_rows, path)),
        (ccase.out_dir / f"cycles_pattern_r_{slug}.png",
         lambda path: plot_pattern_r(ccase, summary_rows, path)),
    ]
    for path, plotter in structure_plots:
        if plotter(path):
            print(f"Saved plot : {path}")
        else:
            print(f"Skipped plot: {path.name} (no finite data)")
    plot_metrics(ccase, results, out_metrics, caveat=caveat)
    print(f"Saved plot : {out_metrics}")
    plot_ets_leadtime(ccase, results, out_ets_leadtime)
    print(f"Saved plot : {out_ets_leadtime}")
    plot_ets_threshold_bars(ccase, results, out_ets_bars)
    print(f"Saved plot : {out_ets_bars}")
    plot_fss_leadtime(ccase, fss_rows, out_fss_leadtime)
    print(f"Saved plot : {out_fss_leadtime}")
    if bdeck_full is not None and track_rows_by_init:
        out_track_plot = ccase.out_dir / f"cycles_track_error_{slug}.png"
        plot_track_error(track_rows_by_init, ccase, out_track_plot)
        print(f"Saved plot : {out_track_plot}")
    if bdeck_full is not None:
        out_shifted_plot = ccase.out_dir / f"cycles_shifted_ets_{slug}.png"
        out_track_precip = ccase.out_dir / f"cycles_track_precip_{slug}.png"
        plot_shifted_skill(summary_rows, ccase, out_shifted_plot)
        print(f"Saved plot : {out_shifted_plot}")
        plot_track_precip(summary_rows, ccase, out_track_precip)
        print(f"Saved plot : {out_track_precip}")
    if ccase.make_animation:
        animations = [
            (ccase.out_dir / f"cycles_qpf_{slug}.gif",
             animate_cycle_qpf),
            (ccase.out_dir / f"cycles_difference_{slug}.gif",
             animate_cycle_difference),
            (ccase.out_dir / f"cycles_observed_{slug}.gif",
             animate_cycle_observed),
        ]
        for out_animation, animator in animations:
            try:
                animator(ccase, fields, out_animation)
            except (ImportError, RuntimeError) as exc:
                print(f"Animation unavailable ({out_animation.name}): {exc}")
            else:
                print(f"Saved movie: {out_animation}")

    if ccase.ml_features:
        try:
            from ml_features import append_features, extract_cycle_features
            feature_rows = []
            for cyc, summary_row in zip(fields["cycles"], summary_rows):
                case = cyc.get("_case")
                if case is None:
                    try:
                        case = cycle_storm_case(ccase, cyc["init_str"])
                    except Exception as exc:
                        print(f"  feature case {cyc['init_str']}: {exc}")
                feature_rows.append(extract_cycle_features(
                    ccase, case, cyc, summary_row, bdeck_full))
            append_features(feature_rows, ccase.ml_features_csv)
            print(f"features: appended {len(feature_rows)} rows to "
                  f"{ccase.ml_features_csv}")
        except Exception as exc:
            print(f"features: extraction failed: {exc}")


if __name__ == "__main__":
    compute_cycles(cycles_from_yaml(sys.argv[1]))
