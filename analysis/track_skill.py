"""Track, intensity, and landfall verification helpers for cycle products."""

from datetime import datetime, timedelta

import numpy as np

from plot_units import inches, miles


# Kept local so importing this module does not load eccodes through hafs_common.
def _haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2)
    return float(2.0 * radius_km * np.arcsin(np.sqrt(a)))


def _interp_scalar(pairs, t):
    """Linearly interpolate non-None scalar values, clamped at the ends."""
    values = sorted((time, float(value)) for time, value in pairs
                    if value is not None)
    if not values:
        return None
    if t <= values[0][0]:
        return values[0][1]
    if t >= values[-1][0]:
        return values[-1][1]
    for (t0, v0), (t1, v1) in zip(values, values[1:]):
        if t0 <= t <= t1:
            frac = (t - t0).total_seconds() / (t1 - t0).total_seconds()
            return v0 + frac * (v1 - v0)
    return values[-1][1]


def bdeck_state(bdeck_full, valid_dt):
    """Best-track position and scalar state interpolated to ``valid_dt``."""
    return {
        "lat": _interp_scalar([(r["t"], r["lat"]) for r in bdeck_full],
                              valid_dt),
        "lon": _interp_scalar([(r["t"], r["lon"]) for r in bdeck_full],
                              valid_dt),
        "vmax": _interp_scalar([(r["t"], r.get("vmax_kt"))
                                for r in bdeck_full], valid_dt),
        "mslp": _interp_scalar([(r["t"], r.get("mslp_hpa"))
                                for r in bdeck_full], valid_dt),
        "rmw": _interp_scalar([(r["t"], r.get("rmw_km"))
                               for r in bdeck_full], valid_dt),
    }


def motion_vector(bdeck_full, valid_dt, dt_hours=3):
    """Unit (east, north) best-track motion from centered/one-sided motion."""
    if not bdeck_full or dt_hours <= 0:
        return None
    first = min(r["t"] for r in bdeck_full)
    last = max(r["t"] for r in bdeck_full)
    t0 = max(first, valid_dt - timedelta(hours=dt_hours))
    t1 = min(last, valid_dt + timedelta(hours=dt_hours))
    if t1 <= t0:
        return None
    s0 = bdeck_state(bdeck_full, t0)
    s1 = bdeck_state(bdeck_full, t1)
    if any(s0[k] is None or s1[k] is None for k in ("lat", "lon")):
        return None
    mean_lat = 0.5 * (s0["lat"] + s1["lat"])
    east = (s1["lon"] - s0["lon"]) * 111.0 * np.cos(np.radians(mean_lat))
    north = (s1["lat"] - s0["lat"]) * 111.0
    magnitude = float(np.hypot(east, north))
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return None
    return east / magnitude, north / magnitude


def along_cross_km(err_east_km, err_north_km, motion_unit):
    """Project an error vector along motion and across it, positive right."""
    if motion_unit is None:
        return None, None
    east, north = motion_unit
    along = err_east_km * east + err_north_km * north
    cross = err_east_km * north - err_north_km * east
    return float(along), float(cross)


def _forecast_state(fcst_fixes, valid_dt):
    return {
        "lat": _interp_scalar([(r[0], r[1]) for r in fcst_fixes], valid_dt),
        "lon": _interp_scalar([(r[0], r[2]) for r in fcst_fixes], valid_dt),
        "vmax": _interp_scalar([(r[0], r[3]) for r in fcst_fixes], valid_dt),
        "mslp": _interp_scalar([(r[0], r[4]) for r in fcst_fixes], valid_dt),
    }


def track_error_rows(fcst_fixes, bdeck_full, valid_start, valid_end,
                     step_hours):
    """Track/intensity errors through a valid-time window.

    ``dlat_deg`` and ``dlon_deg`` are observed minus forecast, so their sign
    gives the displacement needed to move the forecast onto the observed
    track. Intensity errors and the along/cross error vector are forecast
    minus observed.
    """
    if not fcst_fixes or not bdeck_full or step_hours <= 0:
        return []
    init_dt = min(r[0] for r in fcst_fixes)
    rows = []
    valid = valid_start
    while valid <= valid_end:
        fcst = _forecast_state(fcst_fixes, valid)
        obs = bdeck_state(bdeck_full, valid)
        if all(fcst[k] is not None and obs[k] is not None
               for k in ("lat", "lon")):
            dx_km = ((fcst["lon"] - obs["lon"]) * 111.0
                     * np.cos(np.radians(obs["lat"])))
            dy_km = (fcst["lat"] - obs["lat"]) * 111.0
            along, cross = along_cross_km(
                dx_km, dy_km, motion_vector(bdeck_full, valid))
            rows.append({
                "valid": valid,
                "fhr": int(round((valid - init_dt).total_seconds() / 3600.0)),
                "pos_err_km": _haversine_km(
                    fcst["lat"], fcst["lon"], obs["lat"], obs["lon"]),
                "along_km": along,
                "cross_km": cross,
                "dlat_deg": obs["lat"] - fcst["lat"],
                "dlon_deg": obs["lon"] - fcst["lon"],
                "vmax_err_kt": (fcst["vmax"] - obs["vmax"]
                                if fcst["vmax"] is not None
                                and obs["vmax"] is not None else None),
                "mslp_err_hpa": (fcst["mslp"] - obs["mslp"]
                                  if fcst["mslp"] is not None
                                  and obs["mslp"] is not None else None),
                "_best_lat": obs["lat"],
            })
        valid += timedelta(hours=step_hours)
    return rows


def landfall_metrics(fcst_fixes, bdeck_full, landfall_time):
    """Forecast timing and position errors relative to the truth landfall."""
    if landfall_time is None:
        return None
    truth = bdeck_state(bdeck_full, landfall_time)
    if (not fcst_fixes or truth["lat"] is None or truth["lon"] is None):
        return {"timing_err_h": np.nan, "pos_err_km": np.nan,
                "closest_approach_km": np.nan}
    forecast_at_truth = _forecast_state(fcst_fixes, landfall_time)
    pos_error = _haversine_km(
        forecast_at_truth["lat"], forecast_at_truth["lon"],
        truth["lat"], truth["lon"])
    start = min(r[0] for r in fcst_fixes)
    end = max(r[0] for r in fcst_fixes)
    candidates = []
    valid = start
    while valid <= end:
        state = _forecast_state(fcst_fixes, valid)
        candidates.append((_haversine_km(
            state["lat"], state["lon"], truth["lat"], truth["lon"]), valid))
        valid += timedelta(hours=1)
    if candidates[-1][1] < end:
        state = _forecast_state(fcst_fixes, end)
        candidates.append((_haversine_km(
            state["lat"], state["lon"], truth["lat"], truth["lon"]), end))
    closest, forecast_time = min(candidates, key=lambda item: item[0])
    return {
        "timing_err_h": (forecast_time - landfall_time).total_seconds() / 3600.0,
        "pos_err_km": pos_error,
        "closest_approach_km": closest,
    }


def _finite_mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None
              and np.isfinite(row[key])]
    return float(np.mean(values)) if values else np.nan


def cycle_track_summary(rows, landfall):
    """Condense one cycle's track, intensity, and landfall errors."""
    mean_dlat = _finite_mean(rows, "dlat_deg")
    mean_dlon = _finite_mean(rows, "dlon_deg")
    mean_lat = _finite_mean(rows, "_best_lat")
    if all(np.isfinite(v) for v in (mean_dlat, mean_dlon, mean_lat)):
        displacement = _haversine_km(
            mean_lat, 0.0, mean_lat + mean_dlat, mean_dlon)
    else:
        displacement = np.nan
    pos = [float(r["pos_err_km"]) for r in rows
           if r.get("pos_err_km") is not None and np.isfinite(r["pos_err_km"])]
    landfall = landfall or {}
    return {
        "mean_track_err_km": float(np.mean(pos)) if pos else np.nan,
        "max_track_err_km": float(np.max(pos)) if pos else np.nan,
        "mean_along_km": _finite_mean(rows, "along_km"),
        "mean_cross_km": _finite_mean(rows, "cross_km"),
        "vmax_bias_kt": _finite_mean(rows, "vmax_err_kt"),
        "mslp_bias_hpa": _finite_mean(rows, "mslp_err_hpa"),
        "landfall_timing_err_h": landfall.get("timing_err_h", np.nan),
        "landfall_pos_err_km": landfall.get("pos_err_km", np.nan),
        "mean_dlat_deg": mean_dlat,
        "mean_dlon_deg": mean_dlon,
        "mean_displacement_km": displacement,
    }


def mean_displacement_deg(rows):
    """Mean observed-minus-forecast latitude and longitude displacement."""
    return _finite_mean(rows, "dlat_deg"), _finite_mean(rows, "dlon_deg")


def shift_field_cells(field, dlat_deg, dlon_deg, grid_res):
    """Shift a field by a geographic displacement on the fixed grid.

    ``hafs_case.make_fixed_grid`` produces ascending-latitude rows, so positive
    ``dlat_deg`` moves values toward positive row indices and positive
    ``dlon_deg`` moves them toward positive column indices.
    """
    from scipy.ndimage import shift

    return shift(np.nan_to_num(field, nan=0.0),
                 (dlat_deg / grid_res, dlon_deg / grid_res),
                 order=1, mode="constant", cval=0.0)


def score_shifted(parent_win, mrms_win, swath, ets_threshold_mm,
                  fss_threshold_mm, fss_scale_cells, dlat_deg, dlon_deg,
                  grid_res):
    """Score a track-shifted forecast on the unshifted valid footprint."""
    keys = ("ets_shifted", "fss_shifted", "rmse_shifted",
            "pattern_r_shifted")
    if not (np.isfinite(dlat_deg) and np.isfinite(dlon_deg)):
        return {key: np.nan for key in keys}

    # Lazy imports preserve this module's lightweight optional-dependency path.
    from ets_score import contingency_scores
    from skill_metrics import continuous_scores, fractions_skill_score

    valid = (np.asarray(swath, dtype=bool) & np.isfinite(mrms_win)
             & np.isfinite(parent_win))
    shifted = shift_field_cells(parent_win, dlat_deg, dlon_deg, grid_res)
    clean_obs = np.nan_to_num(mrms_win, nan=0.0)
    fcst_points = shifted[valid]
    obs_points = clean_obs[valid]
    categorical = contingency_scores(
        fcst_points, obs_points, ets_threshold_mm)
    continuous = continuous_scores(fcst_points, obs_points)
    return {
        "ets_shifted": categorical["ets"],
        "fss_shifted": fractions_skill_score(
            shifted, clean_obs, fss_threshold_mm, fss_scale_cells, valid),
        "rmse_shifted": continuous["rmse"],
        "pattern_r_shifted": continuous["r"],
    }


def correlation_annotation(x, y):
    """Return robust Pearson/Spearman annotation text for finite pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    n = int(finite.sum())
    if n < 3:
        return "n<3"
    try:
        from scipy.stats import pearsonr, spearmanr
        with np.errstate(all="ignore"):
            pearson = float(pearsonr(x[finite], y[finite]).statistic)
            spearman = float(spearmanr(x[finite], y[finite]).statistic)
    except (ValueError, FloatingPointError):
        pearson = spearman = np.nan
    return f"Pearson r={pearson:.2f}\nSpearman ρ={spearman:.2f}\nn={n}"


def _summary_x(rows, ccase):
    """Return ordered summary rows, x coordinates, and axis mode."""
    def init_value(row):
        value = row.get("_init_dt", row.get("init_dt", row.get("init")))
        if isinstance(value, datetime):
            return value
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H"):
            try:
                return datetime.strptime(str(value), fmt)
            except (TypeError, ValueError):
                pass
        return datetime.min

    ordered = sorted(rows, key=init_value)
    use_lead = ccase.landfall_time is not None
    x = ([float(row["lead_hours_to_landfall"]) for row in ordered]
         if use_lead else np.arange(len(ordered), dtype=float))
    return ordered, np.asarray(x, dtype=float), use_lead


def plot_shifted_skill(summary_rows, ccase, out_path):
    """Compare unshifted and best-track-shifted ETS/FSS with line graphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_theme()
    rows, x, use_lead = _summary_x(summary_rows, ccase)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8), sharex=True)
    panels = (("ets_headline", "ets_shifted", "ETS"),
              ("fss_headline", "fss_shifted", "FSS"))
    for ax, (raw_key, shifted_key, label) in zip(axes, panels):
        raw = [row.get(raw_key, np.nan) for row in rows]
        shifted = [row.get(shifted_key, np.nan) for row in rows]
        ax.plot(x, raw, color="#607d9b", marker="o", lw=2.2,
                label="Unshifted")
        ax.plot(x, shifted, color="#d97941", marker="s", lw=2.2,
                ls="--", label="Best-Track Shifted")
        ax.set_ylabel(label)
        ax.grid(axis="y", ls=":", alpha=0.4)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xticks(x)
    if use_lead:
        axes[-1].set_xlabel("Hours Before Landfall (Forecast Initialization)")
        axes[-1].set_xticklabels([f"{value:.0f}" for value in x])
        axes[-1].invert_xaxis()
    else:
        axes[-1].set_xlabel("Initialization")
        axes[-1].set_xticklabels([str(row.get("init", "")) for row in rows],
                                  rotation=45, ha="right")
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} Best-Track Shifted Skill")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_track_precip(summary_rows, ccase, out_path):
    """Scatter mean track error against headline precipitation metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_theme()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    track = miles(np.asarray([row.get("mean_track_err_km", np.nan)
                              for row in summary_rows], dtype=float))
    lead = np.asarray([row.get("lead_hours_to_landfall", np.nan)
                       for row in summary_rows], dtype=float)
    panels = (("ets_headline", "ETS", False),
              ("fss_headline", "FSS", False),
              ("rmse", "RMSE (in)", True))
    scatter = None
    for ax, (key, label, convert_rain) in zip(axes, panels):
        values = np.asarray([row.get(key, np.nan) for row in summary_rows],
                            dtype=float)
        if convert_rain:
            values = inches(values)
        finite = np.isfinite(track) & np.isfinite(values)
        colors = lead[finite] if np.isfinite(lead[finite]).any() else None
        scatter = ax.scatter(track[finite], values[finite], c=colors,
                             cmap="viridis", s=48, edgecolor="white")
        ax.set_xlabel("Mean Track Error (mi)")
        ax.set_ylabel(label)
        ax.grid(True, ls=":", alpha=0.4)
    if (scatter is not None and scatter.get_array() is not None
            and scatter.get_array().size):
        fig.colorbar(scatter, ax=axes, label="Hours Before Landfall",
                     fraction=0.025, pad=0.03)
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} Track vs Precipitation Skill")
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.14, top=0.86,
                        wspace=0.32)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_theme():
    typography = {
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.titleweight": "bold",
    }
    try:
        import seaborn as sns
        sns.set_theme(context="notebook", style="whitegrid", font_scale=1.0,
                      rc=typography)
    except ImportError:
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(typography)


def plot_track_error(track_rows_by_init, ccase, out_path):
    """Plot position, along/cross, and intensity errors by valid time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _plot_theme()
    inits = sorted(track_rows_by_init)
    cmap = plt.get_cmap("viridis")
    colors = {init: cmap(i / max(len(inits) - 1, 1))
              for i, init in enumerate(inits)}
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10), sharex=True)
    for init in inits:
        rows = track_rows_by_init[init]
        valid = [r["valid"] for r in rows]
        label = init.strftime("%m-%d %HZ") if hasattr(init, "strftime") else str(init)
        axes[0].plot(valid, [miles(r["pos_err_km"]) for r in rows],
                     color=colors[init],
                     marker="o", lw=1.8, label=label)
        axes[1].plot(valid, [miles(r["along_km"]) for r in rows],
                     color=colors[init],
                     lw=1.8)
        axes[1].plot(valid, [miles(r["cross_km"]) for r in rows],
                     color=colors[init],
                     ls="--", lw=1.8)
        axes[2].plot(valid, [r["vmax_err_kt"] for r in rows], color=colors[init],
                     marker="o", lw=1.8, label=label)
    axes[0].set_ylabel("Position Error (mi)")
    axes[1].set_ylabel("Track-Relative Error (mi)")
    axes[2].set_ylabel("Vmax Error (kt)")
    axes[2].set_xlabel("Valid Time")
    axes[1].axhline(0, color="gray", ls=":", lw=0.8)
    axes[2].axhline(0, color="gray", ls=":", lw=0.8)
    axes[0].legend(fontsize=8, ncol=2)
    direction_key = [
        Line2D([], [], color="#333333", lw=2, ls="-", label="Along-track"),
        Line2D([], [], color="#333333", lw=2, ls="--", label="Cross-track"),
    ]
    axes[1].legend(handles=direction_key, loc="upper center", ncol=2,
                   fontsize=8, frameon=True)
    for ax in axes:
        ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} Track Skill by Initialization")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_landfall(summaries_by_init, ccase, out_path):
    """Plot landfall timing and position errors by initialization lead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_theme()
    inits = sorted(summaries_by_init)
    use_lead = ccase.landfall_time is not None
    x = ([(ccase.landfall_time - init).total_seconds() / 3600.0
          for init in inits] if use_lead else inits)
    timing = [summaries_by_init[i]["landfall_timing_err_h"] for i in inits]
    position = [miles(summaries_by_init[i]["landfall_pos_err_km"])
                for i in inits]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)
    axes[0].plot(x, timing, color="#d95f02", marker="o", lw=2)
    axes[1].plot(x, position, color="#1b9e77", marker="o", lw=2)
    axes[0].axhline(0, color="gray", ls=":", lw=0.8)
    axes[0].set_ylabel("Landfall timing error (h)\npositive = late")
    axes[1].set_ylabel("Position error at landfall (miles)")
    axes[1].set_xticks(x)
    if use_lead:
        axes[1].set_xlabel("Hours before landfall (forecast initialization)")
        axes[1].set_xticklabels([f"{value:.0f}" for value in x])
        axes[1].invert_xaxis()
    else:
        axes[1].set_xlabel("Initialization")
        axes[1].set_xticklabels([i.strftime("%m-%d %HZ") for i in inits],
                                rotation=45, ha="right")
    for ax in axes:
        ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle(f"{ccase.storm_name} — {ccase.model_label} landfall skill")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
