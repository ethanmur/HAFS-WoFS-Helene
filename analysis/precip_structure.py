"""Precipitation-distribution, pattern, and object verification helpers."""

import numpy as np
from scipy.ndimage import label, uniform_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skill_metrics import cell_area_km2
from track_skill import along_cross_km
from plot_units import cubic_miles, inches, miles, square_miles


DIST_FIELDS = ("p50", "p90", "p95", "p99", "max_mm", "volume_km3",
               "wet_frac")
OBJECT_FIELDS = (
    "obj_area_fcst_km2", "obj_area_obs_km2", "obj_area_ratio",
    "obj_centroid_err_km", "obj_centroid_along_km",
    "obj_centroid_cross_km", "obj_angle_diff_deg", "obj_mean_ratio",
    "obj_max_ratio",
)

_PLOT_TYPOGRAPHY = {
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.titleweight": "bold",
}


def _apply_plot_typography():
    """Keep cycle-structure text consistently bold and legible."""
    plt.rcParams.update(_PLOT_TYPOGRAPHY)


def _nan_dict(keys):
    return {key: np.nan for key in keys}


def _safe_ratio(numerator, denominator):
    if (not np.isfinite(numerator) or not np.isfinite(denominator)
            or denominator == 0):
        return np.nan
    return float(numerator / denominator)


def distribution_stats(field, swath, grid_lat, grid_res):
    """Distribution and volume statistics over finite points in ``swath``."""
    field = np.asarray(field, dtype=float)
    valid = np.asarray(swath, dtype=bool) & np.isfinite(field)
    if not valid.any():
        return _nan_dict(DIST_FIELDS)
    values = field[valid]
    percentiles = np.percentile(values, [50, 90, 95, 99])
    area = cell_area_km2(grid_lat, grid_res)
    return {
        "p50": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "max_mm": float(np.max(values)),
        "volume_km3": float(np.sum(field[valid] * area[valid]) * 1e-6),
        "wet_frac": float(np.count_nonzero(values >= 1.0) / values.size),
    }


def qq_percentiles(fcst, obs, swath, q=np.arange(1, 100)):
    """Forecast and observation percentiles over their common valid mask."""
    fcst = np.asarray(fcst, dtype=float)
    obs = np.asarray(obs, dtype=float)
    valid = (np.asarray(swath, dtype=bool) & np.isfinite(fcst)
             & np.isfinite(obs))
    q = np.asarray(q, dtype=float)
    if not valid.any():
        empty = np.full(q.shape, np.nan, dtype=float)
        return empty.copy(), empty
    return np.percentile(fcst[valid], q), np.percentile(obs[valid], q)


def _pooled_values(cycles, swath, key, fallback=None):
    arrays = []
    for cycle in cycles:
        field = cycle.get(key, fallback)
        if field is None:
            continue
        field = np.asarray(field, dtype=float)
        valid = np.asarray(swath, dtype=bool) & np.isfinite(field)
        if valid.any():
            arrays.append(field[valid])
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=float)


def _cdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, values.size + 1, dtype=float) / values.size


def _cycle_x(ccase, rows):
    rows = sorted(rows, key=lambda row: row["_init_dt"])
    inits = [row["_init_dt"] for row in rows]
    if ccase.landfall_time is not None:
        x = [(ccase.landfall_time - init_dt).total_seconds() / 3600.0
             for init_dt in inits]
    else:
        x = inits
    return rows, inits, x


def _format_cycle_axis(ax, ccase, inits, x):
    ax.set_xticks(x)
    if ccase.landfall_time is not None:
        ax.set_xlabel("Hours Before Landfall (Forecast Initialization)")
        ax.set_xticklabels([f"{value:.0f}" for value in x])
        ax.invert_xaxis()
    else:
        ax.set_xlabel("Initialization")
        ax.set_xticklabels([value.strftime("%m-%d %HZ") for value in inits],
                           rotation=45, ha="right")


def plot_distributions(ccase, fields, out_path):
    """Plot pooled PDF/CDF and per-cycle plus pooled forecast-MRMS Q-Q."""
    _apply_plot_typography()
    cycles = fields["cycles"]
    swath = fields["swath"]
    sources = [
        ("Forecast", "parent_win", "#2563a6"),
        ("MRMS", "mrms_win", "#222222"),
        ("Stage IV", "stage4_win", "#d97941"),
    ]
    pooled = [(name, _pooled_values(cycles, swath, key, fields.get(key)), color)
              for name, key, color in sources]
    if not any(values.size for _, values, _ in pooled):
        return False

    finite = np.concatenate([inches(values) for _, values, _ in pooled
                             if values.size])
    upper = max(1.0, float(np.percentile(finite, 99.9)), float(np.max(finite)))
    bins = np.linspace(0.0, upper, 51)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for name, values, color in pooled:
        if not values.size:
            continue
        values_in = inches(values)
        axes[0].hist(values_in, bins=bins, density=True, histtype="step", lw=2,
                     color=color, label=name)
        vx, vy = _cdf(values_in)
        axes[1].plot(vx, vy, lw=2, color=color, label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Window-total precipitation (inches)")
    axes[0].set_ylabel("Probability density")
    axes[1].set_xlabel("Window-total precipitation (inches)")
    axes[1].set_ylabel("Cumulative probability")
    for cycle in cycles:
        fcst_q, obs_q = qq_percentiles(
            cycle["parent_win"], cycle.get("mrms_win", fields.get("mrms_win")),
            swath)
        if np.isfinite(fcst_q).any():
            axes[2].plot(inches(obs_q), inches(fcst_q), color="#9aa0a6",
                         lw=0.8, alpha=0.7)
    common_fcst = []
    common_obs = []
    for cycle in cycles:
        fgrid = np.asarray(cycle["parent_win"], dtype=float)
        ogrid = np.asarray(cycle.get("mrms_win", fields.get("mrms_win")),
                           dtype=float)
        valid = swath & np.isfinite(fgrid) & np.isfinite(ogrid)
        if valid.any():
            common_fcst.append(fgrid[valid])
            common_obs.append(ogrid[valid])
    if common_fcst:
        fq = inches(np.percentile(np.concatenate(common_fcst), np.arange(1, 100)))
        oq = inches(np.percentile(np.concatenate(common_obs), np.arange(1, 100)))
        axes[2].plot(oq, fq, color="#2563a6", lw=2.5, label="pooled")
        limit = max(float(np.nanmax(fq)), float(np.nanmax(oq)), 1.0)
        axes[2].plot([0, limit], [0, limit], color="#555555", ls=":",
                     lw=1.2, label="1:1")
    axes[2].set_xlabel("MRMS percentile (inches)")
    axes[2].set_ylabel("Forecast percentile (inches)")
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    for ax, title in zip(axes, ("Pooled PDF", "Pooled CDF", "Forecast–MRMS Q–Q")):
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_title(title)
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} precipitation distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_percentiles_by_cycle(ccase, summary_rows, out_path):
    """Plot upper percentiles and precipitation volume by cycle."""
    _apply_plot_typography()
    rows = [row for row in summary_rows
            if any(np.isfinite(row.get(key, np.nan))
                   for key in ("fcst_p90", "obs_p90", "fcst_volume_km3",
                               "obs_volume_km3"))]
    if not rows:
        return False
    rows, inits, x = _cycle_x(ccase, rows)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True)
    colors = {90: "#2a9d78", 95: "#e9a23b", 99: "#c43d4d"}
    for percentile in (90, 95, 99):
        axes[0].plot(x, [inches(row.get(f"fcst_p{percentile}", np.nan))
                         for row in rows],
                     color=colors[percentile], marker="o", lw=2,
                     label=f"{ccase.model_label} {percentile}%")
        axes[0].plot(x, [inches(row.get(f"obs_p{percentile}", np.nan))
                         for row in rows],
                     color=colors[percentile], marker="s", lw=1.4, ls="--",
                     label=f"MRMS {percentile}%")
    axes[1].plot(x, [cubic_miles(row.get("fcst_volume_km3", np.nan))
                     for row in rows],
                 color="#2563a6", marker="o", lw=2,
                 label=ccase.model_label)
    axes[1].plot(x, [cubic_miles(row.get("obs_volume_km3", np.nan))
                     for row in rows],
                 color="#222222", marker="s", lw=2, ls="--", label="MRMS")
    axes[0].set_ylabel("Precipitation (in)")
    axes[1].set_ylabel("Volume (Cubic Miles)")
    axes[0].legend(ncols=2, fontsize=8, frameon=False)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, ls=":", alpha=0.4)
    _format_cycle_axis(axes[-1], ccase, inits, x)
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} Precipitation Structure "
        "by Cycle")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_pattern_r(ccase, summary_rows, out_path):
    """Plot unshifted and optional track-shifted pattern correlation."""
    _apply_plot_typography()
    rows = [row for row in summary_rows
            if (np.isfinite(row.get("pattern_r", np.nan))
                or np.isfinite(row.get("pattern_r_shifted", np.nan)))]
    if not rows:
        return False
    rows, inits, x = _cycle_x(ccase, rows)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(x, [row.get("pattern_r", np.nan) for row in rows], marker="o",
            lw=2.2, color="#2563a6", label="Forecast")
    shifted = np.asarray([row.get("pattern_r_shifted", np.nan) for row in rows])
    if np.isfinite(shifted).any():
        ax.plot(x, shifted, marker="s", lw=2, ls="--", color="#d97941",
                label="Best-Track Shifted")
    ax.set_ylabel("Pearson Pattern Correlation")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False)
    _format_cycle_axis(ax, ccase, inits, x)
    ax.set_title(
        f"{ccase.storm_name} — {ccase.model_label} Pattern Correlation vs MRMS")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def largest_object(field, swath, threshold_mm, smooth_cells, min_area_cells):
    """Largest qualifying connected precipitation object, or ``None``."""
    field = np.asarray(field, dtype=float)
    swath = np.asarray(swath, dtype=bool)
    size = max(1, int(smooth_cells))
    smooth = uniform_filter(np.nan_to_num(field, nan=0.0), size=size,
                            mode="constant", cval=0.0)
    objects, count = label((smooth >= float(threshold_mm)) & swath)
    if count == 0:
        return None
    sizes = np.bincount(objects.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    if sizes[largest] < int(min_area_cells):
        return None
    return objects == largest


def _grid_resolution(grid_lat, grid_lon):
    candidates = []
    for grid in (np.asarray(grid_lat, dtype=float),
                 np.asarray(grid_lon, dtype=float)):
        for axis in (0, 1):
            values = np.abs(np.diff(grid, axis=axis))
            values = values[np.isfinite(values) & (values > 0)]
            if values.size:
                candidates.append(float(np.median(values)))
    return float(np.median(candidates)) if candidates else np.nan


def object_properties(obj_mask, raw_field, grid_lat, grid_lon):
    """Area, centroid, orientation, and intensity of one object mask."""
    mask = np.asarray(obj_mask, dtype=bool)
    if not mask.any():
        return _nan_dict(("area_km2", "centroid_lat", "centroid_lon",
                          "angle_deg", "mean_mm", "max_mm"))
    grid_lat = np.asarray(grid_lat, dtype=float)
    grid_lon = np.asarray(grid_lon, dtype=float)
    raw_field = np.asarray(raw_field, dtype=float)
    grid_res = _grid_resolution(grid_lat, grid_lon)
    area_grid = cell_area_km2(grid_lat, grid_res)
    weights = area_grid[mask]
    lat = grid_lat[mask]
    lon = grid_lon[mask]
    area = float(np.sum(weights))
    centroid_lat = float(np.average(lat, weights=weights))
    centroid_lon = float(np.average(lon, weights=weights))
    mean_lat = float(np.mean(lat))
    x = lon * 111.0 * np.cos(np.radians(mean_lat))
    y = lat * 111.0
    angle = np.nan
    if x.size >= 2:
        covariance = np.cov(np.vstack((x, y)), bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if np.isfinite(eigenvalues[-1]) and eigenvalues[-1] > 0:
            vector = eigenvectors[:, -1]
            angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)
    raw = raw_field[mask]
    raw = raw[np.isfinite(raw)]
    return {
        "area_km2": area,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "angle_deg": angle,
        "mean_mm": float(np.mean(raw)) if raw.size else np.nan,
        "max_mm": float(np.max(raw)) if raw.size else np.nan,
    }


def object_comparison(fcst, obs, swath, grid_lat, grid_lon, threshold_mm,
                      smooth_cells, min_area_cells, motion_unit):
    """Compare the largest forecast and observed precipitation objects."""
    try:
        fcst_obj = largest_object(fcst, swath, threshold_mm, smooth_cells,
                                  min_area_cells)
        obs_obj = largest_object(obs, swath, threshold_mm, smooth_cells,
                                 min_area_cells)
        if fcst_obj is None or obs_obj is None:
            return _nan_dict(OBJECT_FIELDS)
        fp = object_properties(fcst_obj, fcst, grid_lat, grid_lon)
        op = object_properties(obs_obj, obs, grid_lat, grid_lon)
        mean_lat = 0.5 * (fp["centroid_lat"] + op["centroid_lat"])
        east = ((fp["centroid_lon"] - op["centroid_lon"]) * 111.0
                * np.cos(np.radians(mean_lat)))
        north = (fp["centroid_lat"] - op["centroid_lat"]) * 111.0
        along, cross = along_cross_km(east, north, motion_unit)
        angle_diff = np.nan
        if np.isfinite(fp["angle_deg"]) and np.isfinite(op["angle_deg"]):
            delta = abs(fp["angle_deg"] - op["angle_deg"]) % 180.0
            angle_diff = float(min(delta, 180.0 - delta))
        return {
            "obj_area_fcst_km2": fp["area_km2"],
            "obj_area_obs_km2": op["area_km2"],
            "obj_area_ratio": _safe_ratio(fp["area_km2"], op["area_km2"]),
            "obj_centroid_err_km": float(np.hypot(east, north)),
            "obj_centroid_along_km": np.nan if along is None else along,
            "obj_centroid_cross_km": np.nan if cross is None else cross,
            "obj_angle_diff_deg": angle_diff,
            "obj_mean_ratio": _safe_ratio(fp["mean_mm"], op["mean_mm"]),
            "obj_max_ratio": _safe_ratio(fp["max_mm"], op["max_mm"]),
        }
    except (TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        return _nan_dict(OBJECT_FIELDS)


def plot_objects(ccase, summary_rows, out_path):
    """Plot object areas/ratios and centroid displacement components."""
    _apply_plot_typography()
    rows = [row for row in summary_rows
            if any(np.isfinite(row.get(key, np.nan)) for key in OBJECT_FIELDS)]
    if not rows:
        return False
    rows, inits, x_values = _cycle_x(ccase, rows)
    x = np.arange(len(rows), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    width = 0.36
    axes[0].bar(x - width / 2,
                [square_miles(row.get("obj_area_fcst_km2", np.nan))
                 for row in rows],
                width, color="#2563a6", label="forecast area")
    axes[0].bar(x + width / 2,
                [square_miles(row.get("obj_area_obs_km2", np.nan))
                 for row in rows],
                width, color="#555555", label="MRMS area")
    ratio_ax = axes[0].twinx()
    ratio_ax.plot(x, [row.get("obj_area_ratio", np.nan) for row in rows],
                  color="#d97941", marker="o", lw=1.8, label="area ratio")
    ratio_ax.axhline(1.0, color="#d97941", ls=":", lw=0.9)
    ratio_ax.set_ylabel("Forecast / MRMS area")
    axes[0].set_ylabel("Object area (square miles)")
    axes[0].legend(loc="upper left", frameon=False)
    ratio_ax.legend(loc="upper right", frameon=False)
    axes[1].bar(x - width / 2,
                [miles(row.get("obj_centroid_along_km", np.nan))
                 for row in rows],
                width, color="#2a9d78", label="along-track")
    axes[1].bar(x + width / 2,
                [miles(row.get("obj_centroid_cross_km", np.nan))
                 for row in rows],
                width, color="#c43d4d", label="cross-track")
    axes[1].plot(x, [miles(row.get("obj_centroid_err_km", np.nan))
                     for row in rows],
                 color="#222222", marker="o", lw=2, label="total error")
    axes[1].axhline(0.0, color="#777777", ls=":", lw=0.9)
    axes[1].set_ylabel("Centroid displacement (miles)")
    axes[1].legend(frameon=False, ncols=3)
    for ax in axes:
        ax.grid(True, axis="y", ls=":", alpha=0.4)
    axes[-1].set_xticks(x)
    if ccase.landfall_time is not None:
        axes[-1].set_xticklabels([f"{value:.0f}" for value in x_values])
        axes[-1].set_xlabel("Hours before landfall (forecast initialization)")
        axes[-1].invert_xaxis()
    else:
        axes[-1].set_xticklabels([value.strftime("%m-%d %HZ") for value in inits],
                                 rotation=45, ha="right")
        axes[-1].set_xlabel("initialization")
    fig.suptitle(f"{ccase.storm_name} — {ccase.model_label} precipitation objects")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True
