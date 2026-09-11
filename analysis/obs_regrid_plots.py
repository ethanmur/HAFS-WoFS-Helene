"""Maps of the MET-regridded obs written by regrid-obs (plot-regrid).

For every hour in the case window, into case.regrid_plot_dir:
  compare-regrid/<source>_{full,zoom}_<YYYYMMDDHH>.png
      native vs regridded, 1x2, full domain and case.zoom_domain
  compare-products/products_full_<YYYYMMDDHH>.png
      MRMS | Stage IV | AORC, all on the regrid grid
  compare-anomaly/anomaly_full_<YYYYMMDDHH>.png
      AORC | AORC - MRMS | AORC - Stage IV, AORC taken as truth

Reads the obs cache and the regrid cache only; run regrid-obs first.

Usage:
    python analysis/run.py storms/helene_obs_compare.yaml plot-regrid
"""

import csv
import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from hafs_common import QPF_LEVELS
from parent_qpf import qpf_cmap
from plot_units import inches
from best_track import parse_bdeck
import met_regrid
import obs_compare as oc

LABELS = {"mrms": "MRMS", "stage4": "Stage IV", "aorc": "AORC"}
TRUTH = "aorc"
NODATA_COLOR = "#d0d0d0"
# Fixed across hours so a sequence of anomaly maps is directly comparable.
DIFF_LEVELS_IN = np.array([-2, -1, -0.5, -0.25, -0.1, -0.05, -0.01,
                           0.01, 0.05, 0.1, 0.25, 0.5, 1, 2])
# Native grids are subsampled to at most this many cells per axis on
# full-domain maps (display only: ~6x the pixels a panel has anyway).
FULL_DOMAIN_MAX_CELLS = 1500
PANEL_WIDTH_IN = 7.0
DPI = 120


# =============================================================================
# Map figure
# =============================================================================

def _styles():
    qpf, _ = qpf_cmap()
    qpf = qpf.copy()
    qpf.set_bad(NODATA_COLOR)
    qpf_levels = np.asarray(inches(np.asarray(QPF_LEVELS, dtype=float)))
    diff = plt.get_cmap("RdBu", len(DIFF_LEVELS_IN) + 1).copy()
    diff.set_bad(NODATA_COLOR)
    return {
        "qpf": dict(cmap=qpf, norm=mcolors.BoundaryNorm(qpf_levels, qpf.N),
                    ticks=qpf_levels[::2], extend="max", format="%.2g",
                    label="Precipitation (in); grey = no data"),
        "diff": dict(cmap=diff,
                     norm=mcolors.BoundaryNorm(DIFF_LEVELS_IN, diff.N,
                                               extend="both"),
                     ticks=DIFF_LEVELS_IN, extend="both", format="%g",
                     label="AORC minus product (in); blue = AORC wetter"),
    }


def crop_slices(lat, lon, domain, max_cells=None):
    """(row_slice, col_slice) bounding every cell inside `domain` plus a
    one-cell margin, strided down to <= max_cells per axis; None if no cell
    of the grid falls inside. Works on curvilinear grids."""
    lat_min, lat_max, lon_min, lon_max = domain
    inside = ((lat >= lat_min) & (lat <= lat_max)
              & (lon >= lon_min) & (lon <= lon_max))
    rows = np.flatnonzero(inside.any(axis=1))
    cols = np.flatnonzero(inside.any(axis=0))
    if not rows.size:
        return None
    r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 2, lat.shape[0])
    c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 2, lat.shape[1])
    step = 1
    if max_cells:
        step = max(1, math.ceil(max(r1 - r0, c1 - c0) / max_cells))
    return slice(r0, r1, step), slice(c0, c1, step)


def map_figure(panels, domain, track_line, title, out_path):
    """panels: [(panel_title, lat2d, lon2d, data_mm_or_None, style)], style
    "qpf" or "diff". One horizontal colorbar under each run of same-style
    panels; data None draws an "unavailable" panel."""
    styles = _styles()
    lat_min, lat_max, lon_min, lon_max = domain
    map_h = PANEL_WIDTH_IN * (lat_max - lat_min) / (lon_max - lon_min)
    n = len(panels)
    top_in, bottom_in, left_in, right_in, gap_in = 1.0, 1.5, 0.9, 0.3, 1.0
    fig_w = left_in + right_in + n * PANEL_WIDTH_IN + (n - 1) * gap_in
    fig_h = map_h + top_in + bottom_in
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False,
                             subplot_kw={"projection": ccrs.PlateCarree()})
    axes = list(axes[0])
    fig.subplots_adjust(left=left_in / fig_w, right=1 - right_in / fig_w,
                        bottom=bottom_in / fig_h, top=1 - top_in / fig_h,
                        wspace=gap_in / PANEL_WIDTH_IN)

    extra = []
    groups = []   # [style, [axes], mappable]
    for ax, (ptitle, lat, lon, data, style) in zip(axes, panels):
        extra += oc._panel_frame(ax, domain, track_line, ptitle)
        if data is None:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        s = styles[style]
        mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(inches(data)),
                             cmap=s["cmap"], norm=s["norm"], shading="nearest",
                             transform=ccrs.PlateCarree())
        if groups and groups[-1][0] == style:
            groups[-1][1].append(ax)
        else:
            groups.append([style, [ax], mesh])

    for style, group_axes, mesh in groups:
        for ax in group_axes:
            ax.apply_aspect()   # GeoAxes shrink to the map's aspect at draw time
        x0 = group_axes[0].get_position().x0
        x1 = group_axes[-1].get_position().x1
        y0 = min(ax.get_position().y0 for ax in group_axes)
        cax = fig.add_axes([x0, y0 - 0.8 / fig_h, x1 - x0, 0.14 / fig_h])
        s = styles[style]
        fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=s["ticks"],
                     extend=s["extend"], label=s["label"], format=s["format"])
        extra.append(cax)
    extra.append(oc._figure_title(fig, axes, title))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white",
                bbox_extra_artists=extra, pad_inches=0.4)
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================

def missing_regridded(case, sources, timestamps):
    return [case.regrid.output_path(s, t) for t in timestamps for s in sources
            if not case.regrid.output_path(s, t).exists()]


def load_budget(case):
    """{(valid "YYYY-mm-dd HH:MM", source): mean_pct_diff} from the
    regrid-obs conservation CSV, or {} if it hasn't been written."""
    path = case.out_dir / f"regrid_budget_{case.output_slug}.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {(r["valid"], r["source"]): float(r["mean_pct_diff"])
                for r in csv.DictReader(fh)}


def _budget_note(budget, t, source):
    pct = budget.get((f"{t:%Y-%m-%d %H:%M}", source))
    return "" if pct is None else f"   |   area-mean change {pct:+.2f}%"


def _cropped(lat, lon, data, cut):
    if cut is None:
        return lat, lon, None
    return lat[cut], lon[cut], data[cut]


def plot_regrid(case):
    """Native-vs-regridded, product, and anomaly maps for every hour."""
    cfg = case.regrid
    if cfg is None:
        raise SystemExit("ERROR: plot-regrid needs a `regrid:` block in the YAML")
    oc._print_case_header(case, "Plot regridded obs")
    sources = [s for s, skip in (("mrms", case.skip_mrms),
                                 ("stage4", case.skip_stage4),
                                 ("aorc", case.skip_aorc)) if not skip]
    if not sources:
        print("All sources skipped -- nothing to plot.")
        return
    oc._exit_if_cache_incomplete(case, "plot-regrid")
    timestamps = oc.hourly_timestamps(case.valid_start, case.valid_end)
    missing = missing_regridded(case, sources, timestamps)
    if missing:
        print(f"\nERROR: {len(missing)} regridded file(s) missing -- run "
              "regrid-obs first:\n  python analysis/run.py <yaml> regrid-obs\n")
        for path in missing[:25]:
            print(f"  missing: {path}")
        raise SystemExit(1)

    out = case.regrid_plot_dir
    dirs = {k: out / k for k in ("compare-regrid", "compare-products",
                                 "compare-anomaly")}
    domains = {"full": case.domain}
    if case.zoom_domain:
        domains["zoom"] = case.zoom_domain
    grid_desc = f"{cfg.grid_name} grid ({cfg.method})"
    print(f"Regrid cache: {cfg.grid_dir}")
    print(f"Output:       {out}")
    print(f"Domains:      " + "  ".join(f"{k}={v}" for k, v in domains.items()),
          flush=True)

    track = parse_bdeck(case.best_track)
    budget = load_budget(case)
    if not budget:
        print("No regrid-obs conservation CSV found; headers will omit it.")
    native_ll = {}   # source -> (lat2d, lon2d), constant across hours
    cuts = {}        # (grid key, domain name) -> crop slices
    started = time.monotonic()
    for i, t in enumerate(timestamps, 1):
        track_line = oc.track_segment(track, case.valid_start, t)
        stamp = f"{t:%Y%m%d%H}"
        valid = f"valid {t:%Y-%m-%d %HZ}"
        regridded = {}
        for source in sources:
            lat, lon, vals, _ = oc._native_hour(
                case, source, t, want_latlon=source not in native_ll)
            if source not in native_ll:
                native_ll[source] = (lat, lon)
            lat, lon = native_ll[source]
            rlat, rlon, rvals = met_regrid.read_regridded(
                cfg.output_path(source, t))
            regridded[source] = (rlat, rlon, rvals)

            for dname, domain in domains.items():
                max_cells = FULL_DOMAIN_MAX_CELLS if dname == "full" else None
                for key, la, lo in ((source, lat, lon), ("regrid", rlat, rlon)):
                    if (key, dname) not in cuts:
                        cuts[key, dname] = crop_slices(la, lo, domain,
                                                       max_cells)
                label = LABELS[source]
                map_figure(
                    [(f"{label} native",
                      *_cropped(lat, lon, vals, cuts[source, dname]), "qpf"),
                     (f"{label} regridded",
                      *_cropped(rlat, rlon, rvals, cuts["regrid", dname]),
                      "qpf")],
                    domain, track_line,
                    f"{label} 1-h precipitation, {valid}: native vs "
                    f"{grid_desc}{_budget_note(budget, t, source)}",
                    dirs["compare-regrid"] / f"{source}_{dname}_{stamp}.png")

        rlat, rlon, _ = next(iter(regridded.values()))
        cut = cuts["regrid", "full"]
        map_figure(
            [(LABELS[s], *_cropped(*regridded[s], cut), "qpf")
             for s in sources],
            case.domain, track_line,
            f"Regridded 1-h precipitation on the {grid_desc}, {valid}",
            dirs["compare-products"] / f"products_full_{stamp}.png")

        if TRUTH in regridded:
            truth = regridded[TRUTH][2]
            panels = [(f"{LABELS[TRUTH]} (truth)",
                       *_cropped(rlat, rlon, truth, cut), "qpf")]
            for s in sources:
                if s != TRUTH:
                    panels.append((f"{LABELS[TRUTH]} - {LABELS[s]}",
                                   *_cropped(rlat, rlon,
                                             truth - regridded[s][2], cut),
                                   "diff"))
            map_figure(panels, case.domain, track_line,
                       f"1-h precipitation anomaly vs {LABELS[TRUTH]}, "
                       f"{grid_desc}, {valid}",
                       dirs["compare-anomaly"] / f"anomaly_full_{stamp}.png")
        print(f"  [{i:>3}/{len(timestamps)}] {t:%Y-%m-%d %HZ}  plotted "
              f"(+{time.monotonic() - started:.0f}s)", flush=True)
    print(f"\nSaved maps under {out}")
