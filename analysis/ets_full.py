"""
ETS for HAFS QPF (parent domain + moving 2-km nest) verified against
MRMS QPE and NCEP Stage IV QPE over the TC rainfall swath.

Produces one combined ETS-vs-threshold figure (4 curves: parent/nest x
MRMS/StageIV) and one combined CSV. Reuses all GRIB2/MRMS/Stage IV plumbing
from hafs_common.py and parent_qpf.py, and the contingency math + MRMS
plumbing from ets_score.py.

Usage (on Hercules):
    module load miniconda3
    conda activate hafs
    python analysis/run.py storms/helene_hfsa.yaml ets
"""

import sys
import csv
from pathlib import Path

# Make sibling analysis modules importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hafs_common import discover_files, hafs_event_total
from ets_score import (
    contingency_scores, build_mrms_total, tc_swath_mask,
)
from parent_qpf import (
    default_parent_path, read_hafs_tp_records, pick_cumulative_record,
    stage4_total,
)
from hafs_case import from_yaml
from plot_units import inches, miles


def regrid_2d_to_fixed(src_lat, src_lon, data, grid_lat, grid_lon):
    """Interpolate a curvilinear/rectilinear source field onto the fixed mesh.

    src_lat/src_lon may be 1-D axes or 2-D meshes; data is shaped like the
    2-D source mesh. Uses linear griddata; points outside the source hull
    come back NaN (no extrapolation).
    """

    print('inner regrid loop')

    src_lat = np.asarray(src_lat, dtype=float)
    src_lon = np.asarray(src_lon, dtype=float)
    if src_lat.ndim == 1 and src_lon.ndim == 1:
        src_lon, src_lat = np.meshgrid(src_lon, src_lat)

    print('build src lat lon')

    pts = np.column_stack([src_lat.ravel(), src_lon.ravel()])
    vals = np.asarray(data, dtype=float).ravel()
    finite = np.isfinite(vals)

    print('just before griddata')

    out = griddata(
        pts[finite], vals[finite],
        (grid_lat, grid_lon), method="linear",
    )
    return out


def score_pair(fcst_grid, obs_grid, swath, thresholds, contingency_fn):
    """Score one forecast/observation pair over the swath's valid points.

    Valid points are swath & finite(obs) & finite(fcst); kept values are
    zero-filled before thresholding. Returns (rows, n_valid).
    """
    valid = swath & np.isfinite(obs_grid) & np.isfinite(fcst_grid)
    n_valid = int(np.sum(valid))
    fcst = np.nan_to_num(fcst_grid[valid], nan=0.0)
    obs = np.nan_to_num(obs_grid[valid], nan=0.0)
    rows = [contingency_fn(fcst, obs, thr) for thr in thresholds]
    return rows, n_valid


def build_fixed_grid(case):
    """Fixed lat/lon verification mesh from the case configuration."""
    return case.fixed_grid()


def hafs_parent_total(case, grid_lat, grid_lon):
    """HAFS parent cumulative APCP regridded onto the fixed verification mesh.

    Reuses parent_qpf's discovery + cumulative-record selection, then maps the
    parent grid onto the fixed mesh via regrid_2d_to_fixed.
    """
    path = default_parent_path(case)
    if path is None or not path.exists():
        raise RuntimeError("No parent.atm file found for the configured run.")
    records = read_hafs_tp_records(path)
    if not records:
        raise RuntimeError(f"No 'tp' (APCP) records in parent file {path}.")
    rec = pick_cumulative_record(records)
    print(f"  parent 0->{rec['end_step']}h, grid {rec['lats'].shape}, "
          f"max {np.nanmax(rec['data']):.0f} mm")
    return regrid_2d_to_fixed(rec["lats"], rec["lons"], rec["data"],
                              grid_lat, grid_lon)


def stage4_on_fixed(case, max_fhour, grid_lat, grid_lon):
    """Stage IV touched-days total (parent_qpf.stage4_total) on the fixed mesh.

    stage4_total returns its field already masked to parent_qpf's 750 km
    display swath; an unmasked variant is not exposed. We accept that mask
    because it is wider than the 500 km verification swath, so the tighter
    tc_swath_mask applied later governs the scored footprint. Stage IV is
    CONUS-only, so ocean points regrid to NaN and drop out automatically.
    """
    s4_lat, s4_lon, s4_total, s4_label = stage4_total(case, max_fhour)
    if s4_total is None:
        return None, "unavailable"
    grid = regrid_2d_to_fixed(s4_lat, s4_lon, s4_total, grid_lat, grid_lon)
    return grid, s4_label


# obs -> color, forecast -> linestyle/marker, so 4 curves stay legible.
_OBS_COLOR = {"MRMS": "#1f77b4", "Stage IV": "#2ca02c"}
_FCST_STYLE = {"parent": dict(ls="-", marker="o"),
               "nest": dict(ls="--", marker="s")}


def plot_curves(case, results, max_fhour, out_path, caveat=""):
    """results: list of dicts {forecast, observation, rows, n_valid}."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for res in results:
        rows = res["rows"]
        if not rows:
            continue
        thr = [inches(r["threshold"]) for r in rows]
        ets = [r["ets"] for r in rows]
        style = _FCST_STYLE.get(res["forecast"], dict(ls="-", marker="o"))
        ax.plot(thr, ets, color=_OBS_COLOR.get(res["observation"], "gray"),
                lw=2, **style,
                label=f"{res['forecast']} vs {res['observation']} "
                      f"(n={res['n_valid']:,})")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(inches(np.asarray(case.thresholds_mm, dtype=float)))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Rainfall threshold (inches)")
    ax.set_ylabel("Equitable Threat Score (ETS)")
    ax.set_ylim(-0.2, 1.0)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"{case.storm_name} — {case.model_label} QPF ETS vs MRMS & Stage IV\n"
        f"0–{max_fhour}h | init {case.init_dt:%Y-%m-%d %HZ} | "
        f"TC swath ≤{miles(case.mask_radius_km):.0f} miles"
    )
    if caveat:
        fig.text(0.5, -0.02, caveat, ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_verification_fields(case):
    """Build every field the verification products need, once per case.

    Returns a dict: max_fhour, grid_lat, grid_lon, nest_total, apcp_mode,
    parent_total, mrms_total, stage4_grid, s4_label, swath. stage4_grid is
    None when Stage IV is unavailable. Raises RuntimeError when no nest
    files match the case glob or no MRMS hour can be loaded.
    """
    file_pairs = discover_files(case.run_dir, case.storm_glob(),
                                case.fhours_filter)
    if not file_pairs:
        raise RuntimeError(
            f"No files matching {case.storm_glob()} in {case.run_dir}")
    max_fhour = file_pairs[-1][0]
    print(f"Init {case.init_dt:%Y-%m-%d %HZ} | accumulation 0–{max_fhour}h")

    grid_lat, grid_lon = build_fixed_grid(case)
    print(f"Fixed grid: {grid_lat.shape[0]}x{grid_lat.shape[1]} "
          f"@ {case.grid_res}deg")

    print("\nHAFS nest total ...")
    nest_total, apcp_mode = hafs_event_total(file_pairs, grid_lat, grid_lon)
    print(f"  nest APCP mode: {apcp_mode}, max {np.nanmax(nest_total):.0f} mm")

    print("HAFS parent total ...")
    parent_total = hafs_parent_total(case, grid_lat, grid_lon)

    print("MRMS total ...")
    mrms_total = build_mrms_total(case, max_fhour, grid_lat, grid_lon)

    print("Stage IV total ...")
    stage4_grid, s4_label = stage4_on_fixed(case, max_fhour, grid_lat,
                                            grid_lon)

    print("TC verification swath ...")
    swath = tc_swath_mask(case, max_fhour, grid_lat, grid_lon)

    return dict(max_fhour=max_fhour, grid_lat=grid_lat, grid_lon=grid_lon,
                nest_total=nest_total, apcp_mode=apcp_mode,
                parent_total=parent_total, mrms_total=mrms_total,
                stage4_grid=stage4_grid, s4_label=s4_label, swath=swath)


def field_pairs(fields):
    """(forecasts, observations) as (name, grid) lists from a fields dict.

    Stage IV joins the observations only when it was available.
    """
    forecasts = [("parent", fields["parent_total"]),
                 ("nest", fields["nest_total"])]
    observations = [("MRMS", fields["mrms_total"])]
    if fields["stage4_grid"] is not None:
        observations.append(("Stage IV", fields["stage4_grid"]))
    else:
        print("  Stage IV unavailable — scoring MRMS only.")
    return forecasts, observations


def stage4_caveat(fields):
    """Figure-footer caveat describing the Stage IV accumulation window."""
    if fields["stage4_grid"] is None:
        return "Stage IV unavailable — not scored."
    return (f"Stage IV: CONUS-only, 24h 12Z–12Z files summed over touched "
            f"days ({fields['s4_label']}) — window approximates the "
            f"0–{fields['max_fhour']}h forecast accumulation.")


def compute_ets(case, fields=None):
    if fields is None:
        fields = build_verification_fields(case)
    max_fhour = fields["max_fhour"]
    swath = fields["swath"]
    forecasts, observations = field_pairs(fields)

    results = []
    print("\n" + "=" * 84)
    for fname, fgrid in forecasts:
        for oname, ogrid in observations:
            rows, n_valid = score_pair(fgrid, ogrid, swath,
                                       case.thresholds_mm, contingency_scores)
            results.append(dict(forecast=fname, observation=oname,
                                rows=rows, n_valid=n_valid))
            print(f"\n{fname} vs {oname}  (n_valid={n_valid:,})")
            print(f"{'thr':>5} {'a':>7} {'b':>7} {'c':>7} {'d':>7} {'ETS':>7} "
                  f"{'bias':>6} {'POD':>6} {'FAR':>6} {'CSI':>6}")
            for r in rows:
                print(f"{r['threshold']:>5} {r['a']:>7} {r['b']:>7} {r['c']:>7} "
                      f"{r['d']:>7} {r['ets']:>7.3f} {r['bias']:>6.2f} {r['pod']:>6.2f} "
                      f"{r['far']:>6.2f} {r['csi']:>6.2f}")
    print("=" * 84)

    case.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = case.out_dir / f"ets_full_{case.output_slug}.csv"
    out_png = case.out_dir / f"ets_full_{case.output_slug}.png"

    fieldnames = ["forecast", "observation", "threshold", "a", "b", "c", "d",
                  "ets", "bias", "pod", "far", "csi", "hss"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for res in results:
            for r in res["rows"]:
                w.writerow({"forecast": res["forecast"],
                            "observation": res["observation"], **r})
    print(f"\nSaved table: {out_csv}")

    caveat = stage4_caveat(fields)
    print(caveat)
    plot_curves(case, results, max_fhour, out_png, caveat=caveat)
    print(f"Saved plot : {out_png}")


if __name__ == "__main__":
    import sys
    compute_ets(from_yaml(sys.argv[1]))
