"""Distributions and 1:1 comparisons of the regridded obs (stats-regrid).

Everything here runs on the MET-regridded fields written by regrid-obs, so
all three products share one grid and a grid cell can be compared directly
across products. Per hour and for the whole window, into
case.regrid_plot_dir:
  distributions/dist_<variant>_<YYYYMMDDHH>.png   MRMS/Stage IV/AORC overlaid
  one-to-one/oneone_<variant>_<YYYYMMDDHH>.png    the three product pairs
  distributions/dist_<variant>_combined.png, one-to-one/oneone_<variant>_combined.png
  stats_sources_<case>.csv, stats_pairs_<case>.csv

`variant` is "all" (every cell a product reports) or "land" (restricted to
the cells AORC covers, which is land only). Each product's distribution
uses its own valid cells, and each pair uses the cells where both products
report -- so MRMS-vs-Stage IV keeps the ocean while anything against AORC
is land by construction. Dry cells (< 0.1 mm) are left out of the
distributions and reported as a dry fraction instead.

The window-combined figures come from histograms accumulated hour by hour,
never from a pooled point cloud, so memory stays flat over a long window.

Usage:
    python analysis/run.py storms/helene_obs_compare.yaml stats-regrid
"""

import csv
import itertools
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import met_regrid
import obs_compare as oc
from obs_regrid_plots import LABELS, missing_regridded

MIN_RAIN_MM = 0.1        # below this a cell counts as dry, not as light rain
MAX_BIN_MM = 200.0       # heavier hours pile into the last bin
N_BINS = 40
BIN_EDGES = np.logspace(np.log10(MIN_RAIN_MM), np.log10(MAX_BIN_MM), N_BINS + 1)
BIN_CENTERS = np.sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])
# Okabe-Ito: distinguishable under red-green colour blindness.
COLORS = {"mrms": "#e69f00", "stage4": "#0072b2", "aorc": "#009e73"}
DPI = 130


# =============================================================================
# Accumulators -- per hour, then merged into the window total
# =============================================================================

class SourceStats:
    """Wet-cell histogram plus exact moments for one product."""

    def __init__(self):
        self.counts = np.zeros(N_BINS)
        self.n_valid = 0
        self.n_wet = 0
        self.total_mm = 0.0
        self.max_mm = 0.0

    def update(self, vals):
        if not vals.size:
            return
        self.n_valid += vals.size
        self.total_mm += float(vals.sum())
        self.max_mm = max(self.max_mm, float(vals.max()))
        wet = vals[vals >= MIN_RAIN_MM]
        self.n_wet += wet.size
        self.counts += np.histogram(_clipped(wet), bins=BIN_EDGES)[0]

    def merge(self, other):
        self.counts += other.counts
        self.n_valid += other.n_valid
        self.n_wet += other.n_wet
        self.total_mm += other.total_mm
        self.max_mm = max(self.max_mm, other.max_mm)

    @property
    def dry_fraction(self):
        return 1.0 - self.n_wet / self.n_valid if self.n_valid else float("nan")

    def row(self):
        wet_mean = (float(np.sum(self.counts * BIN_CENTERS) / self.n_wet)
                    if self.n_wet else float("nan"))
        return {
            "n_valid": self.n_valid,
            "n_wet": self.n_wet,
            "dry_fraction": round(self.dry_fraction, 4),
            "mean_mm": round(self.total_mm / self.n_valid, 5) if self.n_valid
                       else float("nan"),
            "wet_mean_mm": round(wet_mean, 4),
            "p99_wet_mm": round(hist_percentile(self.counts, 99.0), 3),
            "max_mm": round(self.max_mm, 3),
        }


class PairStats:
    """Joint histogram plus the sums RMSE/bias/r need, for one product pair."""

    def __init__(self):
        self.joint = np.zeros((N_BINS, N_BINS))
        self.n = 0
        self.sa = self.sb = self.saa = self.sbb = self.sab = 0.0

    def update(self, a, b):
        if not a.size:
            return
        self.n += a.size
        self.sa += float(a.sum())
        self.sb += float(b.sum())
        self.saa += float(a @ a)
        self.sbb += float(b @ b)
        self.sab += float(a @ b)
        wet = (a >= MIN_RAIN_MM) | (b >= MIN_RAIN_MM)
        if wet.any():
            self.joint += np.histogram2d(_clipped(a[wet]), _clipped(b[wet]),
                                         bins=(BIN_EDGES, BIN_EDGES))[0]

    def merge(self, other):
        self.joint += other.joint
        self.n += other.n
        for name in ("sa", "sb", "saa", "sbb", "sab"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def metrics(self):
        """Exact bias/RMSE/correlation of a vs b over every pooled cell."""
        if not self.n:
            return {k: float("nan") for k in
                    ("mean_a_mm", "mean_b_mm", "bias_mm", "rmse_mm", "r")}
        n = self.n
        mean_a, mean_b = self.sa / n, self.sb / n
        var_a = max(self.saa / n - mean_a ** 2, 0.0)
        var_b = max(self.sbb / n - mean_b ** 2, 0.0)
        cov = self.sab / n - mean_a * mean_b
        denom = np.sqrt(var_a * var_b)
        return {
            "mean_a_mm": round(mean_a, 5),
            "mean_b_mm": round(mean_b, 5),
            "bias_mm": round(mean_a - mean_b, 5),
            "rmse_mm": round(np.sqrt(max((self.saa - 2 * self.sab + self.sbb)
                                         / n, 0.0)), 5),
            "r": round(cov / denom, 4) if denom > 0 else float("nan"),
        }


def _clipped(vals):
    # Keep the overflow visible in the last bin instead of dropping it.
    return np.clip(vals, MIN_RAIN_MM, BIN_EDGES[-1] * (1 - 1e-9))


def hist_percentile(counts, q):
    """Percentile of the binned wet values, interpolated inside its bin."""
    total = counts.sum()
    if total <= 0:
        return float("nan")
    target = q / 100.0 * total
    cum = np.cumsum(counts)
    i = int(np.searchsorted(cum, target, side="left"))
    i = min(i, N_BINS - 1)
    below = cum[i - 1] if i else 0.0
    frac = (target - below) / counts[i] if counts[i] else 0.0
    lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
    return float(lo * (hi / lo) ** np.clip(frac, 0.0, 1.0))


# =============================================================================
# Figures
# =============================================================================

def distribution_figure(stats, title, out_path):
    """Overlaid wet-cell distributions (left) and exceedance curves (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    drawn = False
    for source, st in stats.items():
        if not st.n_wet:
            continue
        drawn = True
        axes[0].step(BIN_CENTERS, st.counts / st.n_wet, where="mid", lw=1.7,
                     color=COLORS[source],
                     label=f"{LABELS[source]}: {st.n_wet:,} wet cells, "
                           f"dry {100 * st.dry_fraction:.1f}%, "
                           f"p99 {hist_percentile(st.counts, 99):.1f}, "
                           f"max {st.max_mm:.1f} mm")
        exceed = st.counts[::-1].cumsum()[::-1] / st.n_valid
        axes[1].step(BIN_CENTERS, exceed, where="mid", lw=1.7,
                     color=COLORS[source], label=LABELS[source])
    if not drawn:
        plt.close(fig)
        return False
    axes[0].set(xscale="log", yscale="log", xlabel="1-h precipitation (mm)",
                ylabel="fraction of wet cells")
    axes[1].set(xscale="log", yscale="log", xlabel="1-h precipitation (mm)",
                ylabel="fraction of all cells at or above")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def one_to_one_figure(pairs, title, out_path):
    """pairs: [(source_a, source_b, PairStats)] -- one log-log joint
    histogram per pair, a on y against b on x, with the 1:1 line."""
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.4), squeeze=False)
    lo, hi = BIN_EDGES[0], BIN_EDGES[-1]
    for ax, (a, b, st) in zip(axes[0], pairs):
        m = st.metrics()
        if st.joint.sum() <= 0:
            ax.text(0.5, 0.5, "no overlapping wet cells", ha="center",
                    va="center", transform=ax.transAxes)
            continue
        mesh = ax.pcolormesh(BIN_EDGES, BIN_EDGES,
                             np.ma.masked_less_equal(st.joint, 0),
                             norm=LogNorm(), cmap="viridis")
        ax.plot([lo, hi], [lo, hi], color="#d94f4f", lw=1.3, ls="--")
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
               xlabel=f"{LABELS[b]} (mm)", ylabel=f"{LABELS[a]} (mm)",
               title=f"{LABELS[a]} vs {LABELS[b]}")
        ax.set_aspect("equal")
        ax.text(0.03, 0.97,
                f"n={st.n:,}\nbias={m['bias_mm']:+.3f} mm\n"
                f"RMSE={m['rmse_mm']:.3f} mm\nr={m['r']:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        fig.colorbar(mesh, ax=ax, label="cells", shrink=0.85)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# =============================================================================
# Driver
# =============================================================================

def _points(field, mask):
    keep = np.isfinite(field)
    if mask is not None:
        keep &= mask
    return field[keep]


def _pair_points(field_a, field_b, mask):
    keep = np.isfinite(field_a) & np.isfinite(field_b)
    if mask is not None:
        keep &= mask
    return field_a[keep], field_b[keep]


def stats_regrid(case):
    """Distribution and 1:1 diagnostics over the regridded obs."""
    cfg = case.regrid
    if cfg is None:
        raise SystemExit("ERROR: stats-regrid needs a `regrid:` block in the YAML")
    oc._print_case_header(case, "Stats on regridded obs")
    sources = [s for s, skip in (("mrms", case.skip_mrms),
                                 ("stage4", case.skip_stage4),
                                 ("aorc", case.skip_aorc)) if not skip]
    if not sources:
        print("All sources skipped -- nothing to do.")
        return
    timestamps = oc.hourly_timestamps(case.valid_start, case.valid_end)
    missing = missing_regridded(case, sources, timestamps)
    if missing:
        print(f"\nERROR: {len(missing)} regridded file(s) missing -- run "
              "regrid-obs first:\n  python analysis/run.py <yaml> regrid-obs\n")
        for path in missing[:25]:
            print(f"  missing: {path}")
        raise SystemExit(1)

    variants = ["all"]
    if "aorc" in sources:
        variants.append("land")
    else:
        print("AORC skipped, so there is no land mask -- 'all' variant only.")
    pairs = list(itertools.combinations(sources, 2))
    out = case.regrid_plot_dir
    print(f"Regrid cache: {cfg.grid_dir}")
    print(f"Output:       {out}")
    print(f"Variants:     {', '.join(variants)}   pairs: "
          + ", ".join(f"{LABELS[a]}/{LABELS[b]}" for a, b in pairs), flush=True)

    total = {v: ({s: SourceStats() for s in sources},
                 {p: PairStats() for p in pairs}) for v in variants}
    src_rows, pair_rows = [], []
    reported_mask = False
    started = time.monotonic()
    for i, t in enumerate(timestamps, 1):
        fields = {s: met_regrid.read_regridded(cfg.output_path(s, t))[2]
                  for s in sources}
        land = np.isfinite(fields["aorc"]) if "aorc" in fields else None
        if land is not None and not reported_mask:
            reported_mask = True
            print(f"Land mask (AORC coverage): {int(land.sum()):,} of "
                  f"{land.size:,} cells ({100 * land.mean():.1f}%)", flush=True)
        for variant in variants:
            mask = land if variant == "land" else None
            hour_src = {s: SourceStats() for s in sources}
            hour_pair = {p: PairStats() for p in pairs}
            for s in sources:
                hour_src[s].update(_points(fields[s], mask))
                total[variant][0][s].merge(hour_src[s])
                src_rows.append({"scope": "hour", "valid": f"{t:%Y-%m-%d %H:%M}",
                                 "variant": variant, "source": s,
                                 **hour_src[s].row()})
            for a, b in pairs:
                hour_pair[a, b].update(*_pair_points(fields[a], fields[b], mask))
                total[variant][1][a, b].merge(hour_pair[a, b])
                pair_rows.append({"scope": "hour",
                                  "valid": f"{t:%Y-%m-%d %H:%M}",
                                  "variant": variant, "pair": f"{a}_vs_{b}",
                                  "n": hour_pair[a, b].n,
                                  **hour_pair[a, b].metrics()})
            scope = _variant_note(variant)
            distribution_figure(
                hour_src,
                f"1-h precipitation distribution, {t:%Y-%m-%d %HZ} "
                f"({scope}; dry = below {MIN_RAIN_MM} mm)",
                out / "distributions" / f"dist_{variant}_{t:%Y%m%d%H}.png")
            one_to_one_figure(
                [(a, b, hour_pair[a, b]) for a, b in pairs],
                f"Cell-by-cell comparison, {t:%Y-%m-%d %HZ} ({scope})",
                out / "one-to-one" / f"oneone_{variant}_{t:%Y%m%d%H}.png")
        print(f"  [{i:>3}/{len(timestamps)}] {t:%Y-%m-%d %HZ}  done "
              f"(+{time.monotonic() - started:.0f}s)", flush=True)

    window = f"{case.valid_start:%Y-%m-%d %HZ} to {case.valid_end:%Y-%m-%d %HZ}"
    for variant in variants:
        src_total, pair_total = total[variant]
        scope = _variant_note(variant)
        for s in sources:
            src_rows.append({"scope": "combined", "valid": window,
                             "variant": variant, "source": s,
                             **src_total[s].row()})
        for a, b in pairs:
            pair_rows.append({"scope": "combined", "valid": window,
                              "variant": variant, "pair": f"{a}_vs_{b}",
                              "n": pair_total[a, b].n,
                              **pair_total[a, b].metrics()})
        distribution_figure(
            src_total,
            f"1-h precipitation distribution, all hours {window} "
            f"({scope}; dry = below {MIN_RAIN_MM} mm)",
            out / "distributions" / f"dist_{variant}_combined.png")
        one_to_one_figure(
            [(a, b, pair_total[a, b]) for a, b in pairs],
            f"Cell-by-cell comparison, all hours {window} ({scope})",
            out / "one-to-one" / f"oneone_{variant}_combined.png")

    for name, rows in (("sources", src_rows), ("pairs", pair_rows)):
        path = out / f"stats_{name}_{case.output_slug}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {path}")
    print(f"\nFigures under {out}/distributions and {out}/one-to-one")


def _variant_note(variant):
    return ("land only, AORC coverage" if variant == "land"
            else "every cell each product reports, ocean included")
