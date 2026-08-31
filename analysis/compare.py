"""HFSA-vs-HFSB head-to-head comparison over a shared best-track swath.

Loaded via run.py:  python analysis/run.py storms/<name>_compare.yaml compare
"""

import sys
import csv
import warnings
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RegularGridInterpolator
import yaml
from ets_full import score_pair
from ets_score import contingency_scores
from skill_metrics import cell_area_km2, continuous_scores, fractions_skill_score
from hafs_case import from_yaml, position_on_track
from best_track import parse_bdeck, parse_bdeck_fixes
from skill_metrics import swath_from_track
from hafs_common import discover_files, hafs_event_total
from ets_full import hafs_parent_total, stage4_on_fixed
from ets_score import build_mrms_total
from plot_units import format_inches, inches, miles, square_miles

_DEFAULT_FSS_SCALES = [1, 3, 5, 11, 21, 41]
_DEFAULT_FSS_PLOT_THR = [10, 25, 50]


def load_comparison(path):
    """Parse + validate a comparison YAML and fill defaults (no GRIB loading)."""
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    cases = cfg.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError(f"'cases' must list exactly 2 case YAMLs in {path}")
    if "best_track" not in cfg:
        raise KeyError(f"'best_track' is required in {path}")
    return {
        "label": cfg.get("label", path.stem),
        "case_paths": [str(c) for c in cases],
        "best_track": str(cfg["best_track"]),
        "out_dir": Path(cfg["out_dir"]) if cfg.get("out_dir")
                   else Path("analysis/output") / path.stem,
        "thresholds_mm": cfg.get("thresholds_mm"),
        "fss_scales_cells": cfg.get("fss_scales_cells", list(_DEFAULT_FSS_SCALES)),
        "fss_plot_thresholds": cfg.get("fss_plot_thresholds",
                                       list(_DEFAULT_FSS_PLOT_THR)),
    }


def score_matrix(models, swath, thresholds, fss_scales, grid_res):
    """Score every model x forecast x obs over the shared swath, restricted to
    the COMMON coverage across models so n is identical for all models in a
    given (forecast, observation) pair (fair head-to-head). Returns
    (cat_rows, fss_rows). A (forecast, obs) pair is skipped if ANY model's obs
    for it is None (e.g. Stage IV unavailable for one model)."""
    cat_rows, fss_rows = [], []
    fnames = list(models[0]["forecasts"].keys())
    onames = list(models[0]["obs"].keys())
    for fname in fnames:
        for oname in onames:
            ogrids = [m["obs"].get(oname) for m in models]
            if any(o is None for o in ogrids):
                continue
            # Common coverage: swath AND every model's forecast finite AND obs finite.
            common = swath.copy()
            for m in models:
                common &= np.isfinite(m["forecasts"][fname])
            for o in ogrids:
                common &= np.isfinite(o)
            for m in models:
                fgrid = m["forecasts"][fname]
                ogrid = m["obs"][oname]
                rows, _ = score_pair(fgrid, ogrid, common, thresholds,
                                     contingency_scores)
                for r in rows:
                    cat_rows.append({"model": m["name"], "forecast": fname,
                                     "observation": oname, **r})
                ff = np.nan_to_num(fgrid, nan=0.0)
                oo = np.nan_to_num(ogrid, nan=0.0)
                for thr in thresholds:
                    for sc in fss_scales:
                        fss_rows.append({
                            "model": m["name"], "forecast": fname,
                            "observation": oname, "threshold": thr,
                            "scale_cells": sc,
                            "scale_km": round(sc * grid_res * 111.0, 1),
                            "fss": fractions_skill_score(ff, oo, thr, sc, common),
                        })
    return cat_rows, fss_rows


def continuous_matrix(models, swath):
    """Continuous scores on one common model/observation footprint per pair."""
    output = []
    fnames = list(models[0]["forecasts"].keys())
    onames = list(models[0]["obs"].keys())
    for fname in fnames:
        for oname in onames:
            ogrids = [model["obs"].get(oname) for model in models]
            if any(grid is None for grid in ogrids):
                continue
            common = np.asarray(swath, dtype=bool).copy()
            for model in models:
                common &= np.isfinite(model["forecasts"][fname])
            for grid in ogrids:
                common &= np.isfinite(grid)
            for model in models:
                scores = continuous_scores(
                    model["forecasts"][fname][common],
                    model["obs"][oname][common],
                )
                output.append({"model": model["name"], "forecast": fname,
                               "observation": oname, **scores})
    return output


# Distinct color per model, assigned dynamically so colors appear regardless of
# how the model is labelled ("HAFS-A"/"HAFS-B", "HFSA"/"HFSB", etc.).
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
_FCST_STYLE = {"parent": dict(ls="-", marker="o"),
               "nest": dict(ls="--", marker="s")}


def _model_colors(models):
    """Map each model name to a distinct palette color (sorted for stability)."""
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(sorted(models))}


def plot_categorical_compare(cat_rows, label, out_path, observation="MRMS"):
    """3 panels (ETS, CSI, freq bias) vs threshold; HFSA/HFSB x parent/nest."""
    rows = [r for r in cat_rows if r["observation"] == observation]
    metrics = [("ets", "Equitable Threat Score"),
               ("csi", "Critical Success Index"),
               ("bias", "Frequency bias")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    models = sorted({r["model"] for r in rows})
    forecasts = sorted({r["forecast"] for r in rows})
    colors = _model_colors(models)
    for ax, (key, title) in zip(axes, metrics):
        for mdl in models:
            for fc in forecasts:
                sub = sorted((r for r in rows
                              if r["model"] == mdl and r["forecast"] == fc),
                             key=lambda r: r["threshold"])
                if not sub:
                    continue
                ax.plot([inches(r["threshold"]) for r in sub],
                        [r[key] for r in sub],
                        color=colors[mdl],
                        **_FCST_STYLE.get(fc, dict(ls="-", marker="o")),
                        lw=2, label=f"{mdl} {fc}")
        ax.set_xscale("log")
        ax.set_xlabel("Rainfall threshold (inches)")
        ax.set_ylabel(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if key == "bias":
            ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        else:
            ax.axhline(0.0, color="gray", ls=":", lw=0.8)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{label} — {' vs '.join(models)} categorical skill "
                 f"(vs {observation})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fss_compare(fss_rows, label, out_path, observation="MRMS",
                     forecast="parent", plot_thresholds=(10, 25, 50)):
    """FSS vs neighborhood scale; one line per (model, threshold)."""
    rows = [r for r in fss_rows if r["observation"] == observation
            and r["forecast"] == forecast and r["threshold"] in plot_thresholds]
    fig, ax = plt.subplots(figsize=(9, 6))
    models = sorted({r["model"] for r in rows})
    thrs = sorted({r["threshold"] for r in rows})
    colors = _model_colors(models)
    dashes = {t: (None if i == 0 else (4 + 2 * i, 2))
              for i, t in enumerate(thrs)}
    for mdl in models:
        for t in thrs:
            sub = sorted((r for r in rows
                          if r["model"] == mdl and r["threshold"] == t),
                         key=lambda r: r["scale_km"])
            if not sub:
                continue
            line, = ax.plot([miles(r["scale_km"]) for r in sub],
                            [r["fss"] for r in sub],
                            color=colors[mdl],
                            lw=2, marker="o",
                            label=f"{mdl}  {format_inches(t)} in")
            if dashes[t] is not None:
                line.set_dashes(dashes[t])
    ax.set_xlabel("Neighborhood scale (miles)")
    ax.set_ylabel("Fractions Skill Score (FSS)")
    ax.set_ylim(0, 1)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"{label} — {' vs '.join(models)} FSS "
                 f"({forecast} vs {observation})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_rmse_compare(rows, label, out_path):
    """Grouped common-footprint RMSE bars for each forecast/observation pair."""
    pairs = sorted({(row["forecast"], row["observation"]) for row in rows})
    models = sorted({row["model"] for row in rows})
    fig, ax = plt.subplots(figsize=(max(8.5, 2.0 * len(pairs) + 3.5), 6))
    x = np.arange(len(pairs), dtype=float)
    width = 0.78 / max(len(models), 1)
    colors = _model_colors(models)
    for index, model in enumerate(models):
        values = []
        for forecast, observation in pairs:
            row = next((item for item in rows
                        if item["model"] == model
                        and item["forecast"] == forecast
                        and item["observation"] == observation), None)
            values.append(inches(row["rmse"]) if row is not None else np.nan)
        offset = (index - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width * 0.9, color=colors[model],
                      label=model)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.annotate(f"{value:.2f}",
                            (bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"{forecast}\nvs {observation}"
                      for forecast, observation in pairs])
    ax.set_ylabel("Storm-total RMSE (inches)")
    ax.set_title(f"{label}\nContinuous rainfall error on the common footprint",
                 loc="left")
    ax.grid(axis="y", ls=":", alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def csi_from_pod_sr(pod, sr):
    """CSI reconstructed from POD and success ratio (SR = 1 - FAR).

    From the contingency identity 1/CSI = 1/SR + 1/POD - 1. Vectorizes over
    numpy meshes so it can shade the CSI contours on the performance diagram;
    NaN/inf where POD or SR is 0 (the plot clips those at the axes)."""
    pod = np.asarray(pod, dtype=float)
    sr = np.asarray(sr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / (1.0 / sr + 1.0 / pod - 1.0)


def performance_points(cat_rows, observation="MRMS", forecast="parent"):
    """(success_ratio, pod, csi, bias) per threshold for one forecast/obs.

    Success ratio = 1 - FAR. Pulls straight from the contingency rows compare
    already computes, so the performance diagram adds no new scoring."""
    pts = []
    for r in cat_rows:
        if r.get("observation") != observation or r.get("forecast") != forecast:
            continue
        pts.append({"model": r["model"], "threshold": r["threshold"],
                    "success_ratio": 1.0 - r["far"], "pod": r["pod"],
                    "csi": r["csi"], "bias": r["bias"]})
    return pts


_PERF_MARKERS = ["o", "s", "^", "D", "v", "P"]
_PERF_BIAS_RAYS = (0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0)


def plot_performance_diagram(cat_rows, label, out_path, observation="MRMS",
                             forecast="parent"):
    """Roebber performance diagram: POD vs success ratio, CSI shaded, frequency
    bias as diagonal rays. One marker per model, annotated by threshold; the
    perfect forecast sits in the top-right corner."""
    pts = performance_points(cat_rows, observation, forecast)
    fig, ax = plt.subplots(figsize=(7.5, 7))

    axis = np.linspace(0.001, 1.0, 200)
    sr_mesh, pod_mesh = np.meshgrid(axis, axis)
    csi = csi_from_pod_sr(pod_mesh, sr_mesh)
    shaded = ax.contourf(sr_mesh, pod_mesh, csi, levels=np.arange(0.1, 1.01, 0.1),
                         cmap="Blues", alpha=0.55)
    cbar = fig.colorbar(shaded, ax=ax, shrink=0.85)
    cbar.set_label("Critical Success Index (CSI)")
    lines = ax.contour(sr_mesh, pod_mesh, csi, levels=np.arange(0.1, 1.0, 0.1),
                       colors="#4a4a4a", linewidths=0.5, alpha=0.6)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.1f")

    # Frequency-bias rays through the origin: POD = bias * SR. Each ray exits
    # the unit square at the right edge (bias <= 1) or the top edge (bias > 1).
    for b in _PERF_BIAS_RAYS:
        if b <= 1.0:
            ax.plot([0, 1], [0, b], color="gray", ls="--", lw=0.7, alpha=0.7)
            ax.text(1.006, b, f"{b:g}", fontsize=7, color="gray", va="center")
        else:
            ax.plot([0, 1.0 / b], [0, 1.0], color="gray", ls="--", lw=0.7, alpha=0.7)
            ax.text(1.0 / b, 1.008, f"{b:g}", fontsize=7, color="gray", ha="center")

    models = sorted({p["model"] for p in pts})
    colors = _model_colors(models)
    markers = {m: _PERF_MARKERS[i % len(_PERF_MARKERS)]
               for i, m in enumerate(models)}
    for m in models:
        mp = sorted((p for p in pts if p["model"] == m),
                    key=lambda p: p["threshold"])
        ax.plot([p["success_ratio"] for p in mp], [p["pod"] for p in mp],
                color=colors[m], marker=markers[m], ms=8, lw=1.2,
                markeredgecolor="white", zorder=5, label=m)
        for p in mp:
            ax.annotate(f"{format_inches(p['threshold'])} in",
                        (p["success_ratio"], p["pod"]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6, color=colors[m])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Success Ratio (1 - FAR)  →  fewer false alarms")
    ax.set_ylabel("Probability of Detection (POD)  →  more hits")
    ax.set_title(f"{label} — performance diagram ({forecast} vs {observation})\n"
                 "filled = CSI, dashed rays = frequency bias, perfect = top-right",
                 fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def accumulation_stats(field, swath, grid_lat, grid_res, thresholds_mm):
    """Storm-total summary over the swath: peak accumulation and the land/area
    (km^2) receiving at least each threshold. Points outside the swath or with
    non-finite values are ignored."""
    field = np.asarray(field, dtype=float)
    valid = swath & np.isfinite(field)
    area = cell_area_km2(grid_lat, grid_res)
    max_mm = float(field[valid].max()) if valid.any() else float("nan")
    area_km2 = {float(t): float(area[valid & (field >= t)].sum())
                for t in thresholds_mm}
    return {"max_mm": max_mm, "area_km2": area_km2}


# Storm totals reach hundreds of mm, so use a wider ladder than the 6-h QPF one.
_STORM_TOTAL_LEVELS = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
# Default exceedance thresholds: 1, 3, 5, 10 inches in mm.
_STORM_TOTAL_THRESHOLDS_MM = (25.4, 76.2, 127.0, 254.0)


def _add_us_geography(ax):
    """Overlay US coastline, state, and national borders on a cartopy axis.
    Best-effort: a no-op if the Natural Earth data is unavailable (e.g. an
    offline test host), so the map still renders."""
    try:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
    except Exception:
        pass


def plot_storm_total(sources, swath, grid_lat, grid_lon, grid_res, label,
                     out_path, thresholds_mm=_STORM_TOTAL_THRESHOLDS_MM):
    """Whole-storm accumulation maps (obs first, then each model) over the
    top row, with an area-exceeding-threshold bar chart beneath. ``sources``
    is an ordered list of (name, total_field). Annotates each map with its
    peak total; the bars quantify the flood-relevant footprint per source."""
    n = len(sources)
    fig = plt.figure(figsize=(5 * n, 8.5))
    gs = fig.add_gridspec(2, n, height_ratios=[2.1, 1.0], hspace=0.35)
    cmap = plt.get_cmap("turbo")
    levels_in = inches(np.asarray(_STORM_TOTAL_LEVELS, dtype=float))
    norm = BoundaryNorm(levels_in, cmap.N, extend="max")

    proj = ccrs.PlateCarree()
    lon_min, lon_max = float(np.min(grid_lon)), float(np.max(grid_lon))
    lat_min, lat_max = float(np.min(grid_lat)), float(np.max(grid_lat))
    map_axes, mesh, stats = [], None, {}
    for i, (name, field) in enumerate(sources):
        ax = fig.add_subplot(gs[0, i], projection=proj)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        _add_us_geography(ax)
        masked = np.where(swath & np.isfinite(field), inches(field), np.nan)
        mesh = ax.pcolormesh(grid_lon, grid_lat, masked, cmap=cmap, norm=norm,
                             shading="auto", transform=proj)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                          alpha=0.4, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        if i > 0:
            gl.left_labels = False
        st = accumulation_stats(field, swath, grid_lat, grid_res, thresholds_mm)
        stats[name] = st
        # ax.set_title() on a GeoAxes is frequently clipped by
        # savefig(bbox_inches="tight"); a plain text artist in axes
        # coordinates is measured correctly and never gets cut off.
        ax.text(0.5, 1.10, f"{name}\npeak {inches(st['max_mm']):.1f} in",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=11)
        ax.text(0.5, -0.12, "Longitude", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        if i == 0:
            ax.text(-0.14, 0.5, "Latitude", transform=ax.transAxes,
                    ha="center", va="center", rotation=90, fontsize=9)
        map_axes.append(ax)
    fig.colorbar(mesh, ax=map_axes, shrink=0.85, extend="max",
                 label="Storm-total precipitation (inches)")

    axb = fig.add_subplot(gs[1, :])
    colors = _model_colors([n for n, _ in sources[1:]])
    x = np.arange(len(thresholds_mm))
    width = 0.8 / n
    for j, (name, _) in enumerate(sources):
        areas = [square_miles(stats[name]["area_km2"][float(t)])
                 for t in thresholds_mm]
        color = "0.4" if j == 0 else colors.get(name)
        axb.bar(x + j * width, areas, width, label=name, color=color)
    axb.set_xticks(x + width * (n - 1) / 2)
    axb.set_xticklabels([f"{format_inches(t)} in" for t in thresholds_mm])
    axb.set_xlabel("Storm-total accumulation threshold")
    axb.set_ylabel("Area exceeding (square miles)")
    axb.grid(True, axis="y", ls=":", alpha=0.4)
    axb.legend(fontsize=9)

    fig.suptitle(f"{label} — storm-total precipitation and exceedance area "
                 "(vs MRMS)", fontsize=13)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _interp_center_rmw(fixes, valid_time, fallback_km):
    """(lat, lon, rmw_km, used_fallback) interpolated to valid_time from
    best-track fixes [(t, lat, lon, rmw_or_None), ...]. Position comes from the
    track; RMW is linearly interpolated among fixes that report one, clamped at
    the ends, and replaced by fallback_km (flagged) when none do."""
    basic = [(t, lat, lon) for t, lat, lon, _ in fixes]
    lat, lon = position_on_track(basic, valid_time)
    have = [(t, rmw) for t, _, _, rmw in fixes if rmw is not None]
    if not have:
        return lat, lon, float(fallback_km), True
    if valid_time <= have[0][0]:
        return lat, lon, have[0][1], False
    if valid_time >= have[-1][0]:
        return lat, lon, have[-1][1], False
    for (t0, r0), (t1, r1) in zip(have, have[1:]):
        if t0 <= valid_time <= t1:
            frac = (valid_time - t0).total_seconds() / (t1 - t0).total_seconds()
            return lat, lon, r0 + frac * (r1 - r0), False
    return lat, lon, have[-1][1], False


def storm_relative_field(field, grid_lat, grid_lon, center, rmw_km,
                         radius_rmw=6.0, res_rmw=0.2):
    """Resample a regular lat/lon field onto an RMW-normalized Cartesian grid
    centered on the storm. Returns (x, y, values) with x/y in units of RMW and
    values NaN beyond radius_rmw, so composites tie rainfall structure to storm
    size rather than absolute geography (Newman et al. 2024, TC-RMW)."""
    axis = np.arange(-radius_rmw, radius_rmw + res_rmw / 2, res_rmw)
    x, y = np.meshgrid(axis, axis)
    clat, clon = center
    qlat = clat + y * rmw_km / 111.0
    coslat = max(np.cos(np.radians(clat)), 0.1)
    qlon = clon + x * rmw_km / (111.0 * coslat)
    lat_axis = np.asarray(grid_lat[:, 0], dtype=float)
    lon_axis = np.asarray(grid_lon[0, :], dtype=float)
    values = np.asarray(field, dtype=float)
    if lat_axis[0] > lat_axis[-1]:               # interpolator needs ascending axes
        lat_axis, values = lat_axis[::-1], values[::-1, :]
    if lon_axis[0] > lon_axis[-1]:
        lon_axis, values = lon_axis[::-1], values[:, ::-1]
    interp = RegularGridInterpolator((lat_axis, lon_axis), values,
                                     bounds_error=False, fill_value=np.nan)
    out = interp(np.column_stack([qlat.ravel(), qlon.ravel()])).reshape(x.shape)
    out[np.hypot(x, y) > radius_rmw] = np.nan
    return x, y, out


def radial_profile(stack, radius_rmw=6.0, res_rmw=0.2, bin_rmw=0.4):
    """Percentile distribution per radial bin over a stack of RMW-normalized
    fields. Returns one dict per bin with p05/p25/median/p75/p95 and count."""
    axis = np.arange(-radius_rmw, radius_rmw + res_rmw / 2, res_rmw)
    x, y = np.meshgrid(axis, axis)
    r = np.hypot(x, y)
    edges = np.arange(0, radius_rmw + bin_rmw, bin_rmw)
    arr = np.stack(stack)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = arr[:, (r >= lo) & (r < hi)].ravel()
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        p05, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
        rows.append({"r_lo": float(lo), "r_hi": float(hi),
                     "r_mid": float((lo + hi) / 2), "n": int(vals.size),
                     "p05": p05, "p25": p25, "median": p50,
                     "p75": p75, "p95": p95})
    return rows


def storm_relative_composite(cases, best_track_path, grid_lat, grid_lon,
                             max_fhour, accumulation_hours=6,
                             rmw_fallback_km=50.0, radius_rmw=6.0, res_rmw=0.2):
    """Composite RMW-normalized 6-h rain over every lead window for MRMS and
    each model. Each model field is centered on its OWN forecast track (so
    track error is removed from the structure) while every source is normalized
    by the common best-track RMW. Windows that lack a parent file or an MRMS
    hour are skipped with a printed reason. Returns
    (composites, radial_by_source, used_fallback_rmw)."""
    from cycles import parent_window_total
    from ets_score import build_mrms_total_window

    fixes = parse_bdeck_fixes(best_track_path)
    init_dt = cases[0].init_dt
    mrms_cache_dir = cases[0].mrms_cache_dir
    order = ["MRMS"] + [c.model_label for c in cases]
    stacks = {name: [] for name in order}
    used_fallback = False

    for lead in range(accumulation_hours, max_fhour + 1, accumulation_hours):
        f1, f2 = lead - accumulation_hours, lead
        valid_start = init_dt + timedelta(hours=f1)
        valid_end = init_dt + timedelta(hours=f2)
        blat, blon, rmw, fb = _interp_center_rmw(fixes, valid_end, rmw_fallback_km)
        used_fallback = used_fallback or fb

        try:
            obs = build_mrms_total_window(valid_start, valid_end, mrms_cache_dir,
                                          grid_lat, grid_lon)
            _, _, sr = storm_relative_field(obs, grid_lat, grid_lon,
                                            (blat, blon), rmw, radius_rmw, res_rmw)
            stacks["MRMS"].append(sr)
        except RuntimeError as exc:
            print(f"  storm-relative skip MRMS F{lead:03d}: {exc}")

        for case in cases:
            try:
                fcst = parent_window_total(case, f1, f2, grid_lat, grid_lon)
            except RuntimeError as exc:
                print(f"  storm-relative skip {case.model_label} F{lead:03d}: {exc}")
                continue
            mlat, mlon = position_on_track(case.track, valid_end)
            _, _, sr = storm_relative_field(fcst, grid_lat, grid_lon,
                                            (mlat, mlon), rmw, radius_rmw, res_rmw)
            stacks[case.model_label].append(sr)

    composites, radial_by_source = [], {}
    for name in order:
        if not stacks[name]:
            continue
        # Pixels beyond the RMW radius are NaN in every window, so nanmean warns
        # about all-NaN slices there; that is expected and the result stays NaN.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_field = np.nanmean(np.stack(stacks[name]), axis=0)
        composites.append((name, mean_field))
        radial_by_source[name] = radial_profile(stacks[name], radius_rmw, res_rmw)
    return composites, radial_by_source, used_fallback


_STORM_REL_LEVELS = [0.5, 1, 2, 4, 8, 15, 25, 40, 60, 90]


def plot_storm_relative(composites, radial_by_source, label, out_path,
                        radius_rmw=6.0, res_rmw=0.2):
    """RMW-normalized mean-rain composites (top, obs first) with radial median
    +/- IQR distributions beneath. ``composites`` is an ordered list of
    (name, mean_field); ``radial_by_source`` maps name -> radial_profile rows."""
    sources = [name for name, _ in composites]
    n = len(sources)
    fig = plt.figure(figsize=(4.8 * n, 9))
    gs = fig.add_gridspec(2, n, height_ratios=[1.25, 1.0], hspace=0.28)
    axis = np.arange(-radius_rmw, radius_rmw + res_rmw / 2, res_rmw)
    x, y = np.meshgrid(axis, axis)
    cmap = plt.get_cmap("turbo")
    levels_in = inches(np.asarray(_STORM_REL_LEVELS, dtype=float))
    norm = BoundaryNorm(levels_in, cmap.N, extend="max")

    map_axes, mesh = [], None
    for i, (name, field) in enumerate(composites):
        ax = fig.add_subplot(gs[0, i])
        mesh = ax.pcolormesh(x, y, inches(field), cmap=cmap, norm=norm,
                             shading="auto")
        for ring in range(1, int(radius_rmw) + 1):
            ax.add_patch(plt.Circle((0, 0), ring, fill=False, color="#555",
                                    lw=0.6, alpha=0.7))
        ax.axhline(0, color="#777", lw=0.4)
        ax.axvline(0, color="#777", lw=0.4)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xlabel("x / RMW")
        if i == 0:
            ax.set_ylabel("y / RMW")
        map_axes.append(ax)
    fig.colorbar(mesh, ax=map_axes, shrink=0.85, extend="max",
                 label="Mean 6-h precipitation (inches)")

    model_names = [name for name, _ in composites[1:]]
    colors = _model_colors(model_names)
    axr = fig.add_subplot(gs[1, :])
    for name in sources:
        rows = sorted(radial_by_source.get(name, []), key=lambda r: r["r_mid"])
        if not rows:
            continue
        rmid = [r["r_mid"] for r in rows]
        color = "0.4" if name == sources[0] else colors.get(name)
        axr.plot(rmid, [inches(r["median"]) for r in rows],
                 color=color, lw=2, label=name)
        axr.fill_between(rmid, [inches(r["p25"]) for r in rows],
                         [inches(r["p75"]) for r in rows],
                         color=color, alpha=0.18)
    axr.set_xlim(0, radius_rmw)
    axr.set_xlabel("Distance from center (RMW)")
    axr.set_ylabel("6-h precipitation (inches)")
    axr.grid(True, ls=":", alpha=0.4)
    axr.legend(fontsize=9)
    axr.set_title("Radial distribution — median with 25–75% band")

    fig.suptitle(f"{label} — storm-relative composite (vs MRMS)", fontsize=13)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _slug(label):
    return label.lower().replace(" ", "_")


def _check_same_init(cases, case_paths):
    """Raise ValueError naming both if the two comparison cases' inits differ."""
    a, b = cases
    if a.init_dt != b.init_dt:
        raise ValueError(
            f"comparison cases must share an init: {case_paths[0]} is "
            f"{a.init_str}, {case_paths[1]} is {b.init_str}")


def _init_tag(label, init_dt):
    """(slug, title) tagged by init. slug = '<label-slug>_<YYYYMMDDHH>'
    (de-duplicated); title = '<label> (init YYYY-MM-DD HHZ)'."""
    init_str = init_dt.strftime("%Y%m%d%H")
    base = _slug(label)
    slug = base if init_str in base else f"{base}_{init_str}"
    title = f"{label} (init {init_dt:%Y-%m-%d %HZ})"
    return slug, title


_CAT_NUM = ("threshold", "a", "b", "c", "d", "ets", "csi", "bias",
            "pod", "far", "hss")
_FSS_NUM = ("threshold", "scale_cells", "scale_km", "fss")
_CONT_NUM = ("n", "rmse", "mae", "bias", "r")


def _read_rows(path, numeric):
    """Read a comparison CSV, casting numeric columns to float (NaN-aware)."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for k in numeric:
                if k in r:
                    try:
                        r[k] = float(r[k])
                    except (TypeError, ValueError):
                        r[k] = float("nan")
            rows.append(r)
    return rows


def _plot_comparison(out_dir, slug, title, fss_plot_thresholds):
    """Read the init-tagged CSVs and (re)draw the two PNGs. No case loading."""
    cat_rows = _read_rows(out_dir / f"compare_categorical_{slug}.csv", _CAT_NUM)
    fss_rows = _read_rows(out_dir / f"compare_fss_{slug}.csv", _FSS_NUM)
    plot_categorical_compare(cat_rows, title,
                             out_dir / f"compare_categorical_{slug}.png",
                             observation="MRMS")
    plot_fss_compare(fss_rows, title,
                     out_dir / f"compare_fss_{slug}.png",
                     observation="MRMS", forecast="parent",
                     plot_thresholds=tuple(float(t) for t in fss_plot_thresholds))
    plot_performance_diagram(cat_rows, title,
                             out_dir / f"compare_performance_{slug}.png",
                             observation="MRMS", forecast="parent")
    continuous_path = out_dir / f"compare_continuous_{slug}.csv"
    if continuous_path.exists():
        continuous_rows = _read_rows(continuous_path, _CONT_NUM)
        plot_rmse_compare(continuous_rows, title,
                          out_dir / f"compare_rmse_{slug}.png")


def replot_from_csv(cfg):
    """Regenerate the comparison figures from the existing CSVs — no GRIB work.

    Loads the two case YAMLs (cheap; only parses the .atcfunix) to learn the
    shared init, then redraws the init-tagged figures.
    """
    cases = [from_yaml(p) for p in cfg["case_paths"]]
    _check_same_init(cases, cfg["case_paths"])
    slug, title = _init_tag(cfg["label"], cases[0].init_dt)
    _plot_comparison(cfg["out_dir"], slug, title, cfg["fss_plot_thresholds"])
    print(f"Replotted: {cfg['out_dir']}/compare_categorical_{slug}.png")
    print(f"Replotted: {cfg['out_dir']}/compare_fss_{slug}.png")
    rmse_path = cfg["out_dir"] / f"compare_rmse_{slug}.png"
    if rmse_path.exists():
        print(f"Replotted: {rmse_path}")


def _build_model_fields(case, grid_lat, grid_lon, max_fhour):
    """Parent + nest forecast totals and MRMS + Stage IV obs on the common grid."""
    file_pairs = discover_files(case.run_dir, case.storm_glob(),
                                case.fhours_filter)
    print(f"  {case.model_label}: {len(file_pairs)} storm files, nest total ...")
    nest_total, _ = hafs_event_total(file_pairs, grid_lat, grid_lon)
    print(f"  {case.model_label}: parent total ...")
    parent_total = hafs_parent_total(case, grid_lat, grid_lon)
    print(f"  {case.model_label}: MRMS total ...")
    mrms = build_mrms_total(case, max_fhour, grid_lat, grid_lon)
    print(f"  {case.model_label}: Stage IV total ...")
    stage4, _ = stage4_on_fixed(case, max_fhour, grid_lat, grid_lon)
    return {
        "name": case.model_label,
        "forecasts": {"parent": parent_total, "nest": nest_total},
        "obs": {"MRMS": mrms, "Stage IV": stage4},
    }


def generate_comparison(cfg):
    """Score both cases over one best-track swath and write figures + CSVs."""
    cases = [from_yaml(p) for p in cfg["case_paths"]]
    _check_same_init(cases, cfg["case_paths"])
    slug, title = _init_tag(cfg["label"], cases[0].init_dt)
    init_str = cases[0].init_str
    a, b = cases
    if a.domain != b.domain or a.grid_res != b.grid_res:
        raise ValueError(
            f"cases must share domain/grid_res: {cfg['case_paths'][0]} has "
            f"{a.domain}@{a.grid_res}, {cfg['case_paths'][1]} has {b.domain}@{b.grid_res}")
    if a.mask_radius_km != b.mask_radius_km:
        raise ValueError(
            f"cases must share mask_radius_km ({a.mask_radius_km} vs {b.mask_radius_km})")
    # None or empty list -> fall back to the cases' default thresholds.
    thresholds = cfg["thresholds_mm"] or a.thresholds_mm
    grid_lat, grid_lon = a.fixed_grid()

    print(f"Best track: {cfg['best_track']}")
    track = parse_bdeck(cfg["best_track"])

    fa = discover_files(a.run_dir, a.storm_glob(), a.fhours_filter)
    fb = discover_files(b.run_dir, b.storm_glob(), b.fhours_filter)
    if not fa or not fb:
        print("No storm files found for one of the cases; aborting.")
        return
    max_fhour = min(fa[-1][0], fb[-1][0])
    print(f"Shared swath: best track, 0-{max_fhour}h, "
          f"<= {a.mask_radius_km:.0f} km")
    swath = swath_from_track(track, grid_lat, grid_lon, a.mask_radius_km,
                             a.init_dt, max_fhour)

    models = [_build_model_fields(c, grid_lat, grid_lon, max_fhour) for c in cases]
    cat_rows, fss_rows = score_matrix(models, swath, thresholds,
                                      cfg["fss_scales_cells"], a.grid_res)
    continuous_rows = continuous_matrix(models, swath)

    for r in cat_rows:
        r["init"] = init_str
    for r in fss_rows:
        r["init"] = init_str
    for r in continuous_rows:
        r["init"] = init_str

    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_cols = ["init", "model", "forecast", "observation", "threshold",
                "a", "b", "c", "d", "ets", "csi", "bias", "pod", "far", "hss"]
    cat_csv = out_dir / f"compare_categorical_{slug}.csv"
    with open(cat_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cat_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(cat_rows)

    fss_cols = ["init", "model", "forecast", "observation", "threshold",
                "scale_cells", "scale_km", "fss"]
    fss_csv = out_dir / f"compare_fss_{slug}.csv"
    with open(fss_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fss_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(fss_rows)

    continuous_cols = ["init", "model", "forecast", "observation", "n",
                       "rmse", "mae", "bias", "r"]
    continuous_csv = out_dir / f"compare_continuous_{slug}.csv"
    with open(continuous_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=continuous_cols,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(continuous_rows)

    cat_png = out_dir / f"compare_categorical_{slug}.png"
    fss_png = out_dir / f"compare_fss_{slug}.png"
    perf_png = out_dir / f"compare_performance_{slug}.png"
    rmse_png = out_dir / f"compare_rmse_{slug}.png"
    storm_total_png = out_dir / f"compare_storm_total_{slug}.png"
    plot_categorical_compare(cat_rows, title, cat_png, observation="MRMS")
    plot_fss_compare(fss_rows, title, fss_png, observation="MRMS",
                     forecast="parent",
                     plot_thresholds=tuple(cfg["fss_plot_thresholds"]))
    plot_performance_diagram(cat_rows, title, perf_png, observation="MRMS",
                             forecast="parent")
    plot_rmse_compare(continuous_rows, title, rmse_png)
    # Storm-total maps need the 2-D total fields, so they are drawn here (not in
    # replot, which only has the CSVs). MRMS is identical across models (shared
    # window), so take it from the first.
    storm_total_sources = [("MRMS", models[0]["obs"]["MRMS"])] + \
        [(m["name"], m["forecasts"]["parent"]) for m in models]
    plot_storm_total(storm_total_sources, swath, grid_lat, grid_lon, a.grid_res,
                     title, storm_total_png)

    # Storm-relative RMW composite: pool 6-h windows over all lead times, each
    # model centered on its own track, all normalized by the best-track RMW.
    print("Storm-relative RMW composite (per-lead windows) ...")
    storm_rel_png = out_dir / f"compare_storm_relative_{slug}.png"
    composites, radial_by_source, used_fallback = storm_relative_composite(
        cases, cfg["best_track"], grid_lat, grid_lon, max_fhour)
    if used_fallback:
        print(f"  note: best track had no RMW column; used "
              f"{50.0:.0f} km fallback (distances are ~km/50, not true RMW)")
    if composites:
        plot_storm_relative(composites, radial_by_source, title, storm_rel_png)
    else:
        storm_rel_png = None
        print("  storm-relative: no usable windows; skipped")

    print(f"\nSaved: {cat_csv}")
    print(f"Saved: {fss_csv}")
    print(f"Saved: {continuous_csv}")
    print(f"Saved: {cat_png}")
    print(f"Saved: {fss_png}")
    print(f"Saved: {perf_png}")
    print(f"Saved: {rmse_png}")
    print(f"Saved: {storm_total_png}")
    if storm_rel_png is not None:
        print(f"Saved: {storm_rel_png}")
    print("\nETS (parent vs MRMS) at 25 / 50 mm:")
    for mdl in sorted({r["model"] for r in cat_rows}):
        e25 = next((r["ets"] for r in cat_rows if r["model"] == mdl
                    and r["forecast"] == "parent" and r["observation"] == "MRMS"
                    and r["threshold"] == 25), float("nan"))
        e50 = next((r["ets"] for r in cat_rows if r["model"] == mdl
                    and r["forecast"] == "parent" and r["observation"] == "MRMS"
                    and r["threshold"] == 50), float("nan"))
        print(f"  {mdl}: ETS25={e25:.3f}  ETS50={e50:.3f}")
    print("\nRMSE (parent vs MRMS):")
    for row in continuous_rows:
        if row["forecast"] == "parent" and row["observation"] == "MRMS":
            print(f"  {row['model']}: {inches(row['rmse']):.2f} in "
                  f"(n={row['n']:,})")
