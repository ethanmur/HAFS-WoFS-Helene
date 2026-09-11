"""Unit tests for obs_regrid_plots' pure logic (cropping, config, budget CSV).

Run directly:   python3 analysis/tests/test_obs_regrid_plots.py
Or via pytest:  pytest analysis/tests/test_obs_regrid_plots.py -v
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from obs_compare import from_yaml
from obs_regrid_plots import crop_slices, load_budget, _budget_note


def _grid():
    lon, lat = np.meshgrid(np.arange(-90.0, -80.0, 0.5),
                           np.arange(25.0, 35.0, 0.5))
    return lat, lon


def test_crop_slices_bounds_domain_with_margin():
    lat, lon = _grid()
    rows, cols = crop_slices(lat, lon, (30.0, 31.0, -85.0, -84.0))
    assert lat[rows][0, 0] == 29.5 and lat[rows][-1, 0] == 31.5
    assert lon[:, cols][0, 0] == -85.5 and lon[:, cols][0, -1] == -83.5


def test_crop_slices_strides_to_max_cells():
    lat, lon = _grid()
    rows, cols = crop_slices(lat, lon, (25.0, 35.0, -90.0, -80.0), max_cells=5)
    assert lat[rows, cols].shape[0] <= 5 and lat[rows, cols].shape[1] <= 5


def test_crop_slices_none_outside_grid():
    lat, lon = _grid()
    assert crop_slices(lat, lon, (40.0, 41.0, -70.0, -69.0)) is None


def _write_yaml(tmp, **extra):
    cfg = {"best_track": "bt.dat", "valid_start": 2024092401,
           "valid_end": 2024092404, "domain": [15, 42, -100, -60],
           "out_dir": str(tmp / "out"), **extra}
    path = tmp / "case.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_from_yaml_regrid_plots_block():
    tmp = Path(tempfile.mkdtemp())
    case = from_yaml(_write_yaml(tmp, regrid_plots={
        "out_dir": str(tmp / "maps"), "zoom_domain": [34, 35, -83, -81]}))
    assert case.regrid_plot_dir == tmp / "maps"
    assert case.zoom_domain == (34.0, 35.0, -83.0, -81.0)
    default = from_yaml(_write_yaml(tmp))
    assert default.regrid_plot_dir == tmp / "out" / "regrid"
    assert default.zoom_domain is None


def test_from_yaml_rejects_bad_zoom_domain():
    tmp = Path(tempfile.mkdtemp())
    try:
        from_yaml(_write_yaml(tmp, regrid_plots={"zoom_domain": [34, 35]}))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_load_budget_reads_conservation_csv():
    tmp = Path(tempfile.mkdtemp())
    case = from_yaml(_write_yaml(tmp))
    assert load_budget(case) == {}
    case.out_dir.mkdir(parents=True)
    (case.out_dir / f"regrid_budget_{case.output_slug}.csv").write_text(
        "valid,source,mean_pct_diff\n2024-09-24 02:00,aorc,-0.198\n")
    budget = load_budget(case)
    t = datetime(2024, 9, 24, 2)
    assert "-0.20%" in _budget_note(budget, t, "aorc")
    assert _budget_note(budget, t, "mrms") == ""


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
