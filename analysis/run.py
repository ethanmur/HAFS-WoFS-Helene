"""Single entry point for the HAFS QPF/ETS framework.

    python analysis/run.py <case.yaml> [parent|ets|rmse|cycles|cycles-compare|all|compare|replot|ml|download-obs|obs-compare]

Loads a StormCase from the YAML case file and runs the requested product(s):
  parent  nest + parent QPF vs MRMS + Stage IV 4-panel figure
  ets     the combined parent+nest ETS-vs-threshold figure + CSV
  rmse    storm-total RMSE/MAE/bias/r scatter panels + CSV
  cycles  per-initialization comparison on a common valid window (takes a cycles YAML)
  cycles-compare compare HAFS-A/B/M cycle CSVs; missing models are allowed
  all     parent + ets + rmse (fields built once; default)
  compare HFSA-vs-HFSB rainfall comparison (takes a comparison YAML)
  replot  redraw the comparison figures from existing CSVs (no recompute)
  ml      pooled ML regime diagnostics over a feature CSV
  download-obs  fetch/cache MRMS, Stage IV, and AORC obs only -- no
                regridding or plotting; run this on a login node
                (takes an obs-compare YAML)
  obs-compare   MRMS/Stage IV/AORC observation-vs-observation comparison,
                no HAFS forecast involved; reads the cache only and never
                downloads -- errors immediately if something download-obs
                should have fetched is missing (takes an obs-compare YAML)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = ("parent", "ets", "rmse", "cycles", "cycles-compare", "all",
            "compare", "replot", "ml", "download-obs", "obs-compare")


def parse_args(argv):
    """(yaml_path, command) from argv; command defaults to 'all'."""
    if not argv:
        print("usage: run.py <case.yaml> "
              "[parent|ets|rmse|cycles|cycles-compare|all|compare|replot|ml|download-obs|obs-compare]")
        raise SystemExit(2)
    yaml_path = argv[0]
    command = argv[1] if len(argv) > 1 else "all"
    if command not in COMMANDS:
        print(f"unknown command '{command}'; choose from {COMMANDS}")
        raise SystemExit(2)
    return yaml_path, command


def dispatch(case, command):
    """Run the requested product(s) for a loaded StormCase."""
    from parent_qpf import generate_parent_figure
    from ets_full import compute_ets, build_verification_fields
    from rmse_scatter import compute_rmse
    if command in ("parent", "all"):
        generate_parent_figure(case)
    if command == "ets":
        compute_ets(case)
    if command == "rmse":
        compute_rmse(case)
    if command == "all":
        # Build the expensive verification fields once, share across products.
        fields = build_verification_fields(case)
        compute_ets(case, fields=fields)
        compute_rmse(case, fields=fields)


def main(argv):
    yaml_path, command = parse_args(argv)
    if command in ("download-obs", "obs-compare"):
        from obs_compare import from_yaml as obs_from_yaml
        obs_case = obs_from_yaml(yaml_path)
        if command == "download-obs":
            from obs_compare import download_obs
            download_obs(obs_case)
        else:
            from obs_compare import run_obs_compare
            run_obs_compare(obs_case)
        return
    if command == "ml":
        from ml_regime import load_ml_config, run_ml
        run_ml(load_ml_config(yaml_path))
        return
    if command == "cycles-compare":
        from cycles_compare import (load_cycles_comparison,
                                    generate_cycles_comparison)
        generate_cycles_comparison(load_cycles_comparison(yaml_path))
        return
    if command in ("compare", "replot"):
        from compare import (load_comparison, generate_comparison,
                             replot_from_csv)
        cfg = load_comparison(yaml_path)
        (generate_comparison if command == "compare" else replot_from_csv)(cfg)
        return
    if command == "cycles":
        from hafs_case import cycles_from_yaml
        from cycles import compute_cycles
        ccase = cycles_from_yaml(yaml_path)
        print(f"Case   : {ccase.storm_name} ({ccase.model_label})")
        print(f"Window : {ccase.valid_start:%Y-%m-%d %HZ} -> "
              f"{ccase.valid_end:%Y-%m-%d %HZ}  | run_root: {ccase.run_root}")
        print(f"Output : {ccase.out_dir}  | command: {command}")
        compute_cycles(ccase)
        return
    from hafs_case import from_yaml
    case = from_yaml(yaml_path)
    print(f"Case   : {case.storm_name} ({case.model_label})")
    print(f"Init   : {case.init_dt:%Y-%m-%d %HZ}  | run_dir: {case.run_dir}")
    print(f"Domain : {case.domain}  | track points: {len(case.track)}")
    print(f"Output : {case.out_dir}  | command: {command}")
    dispatch(case, command)


if __name__ == "__main__":
    main(sys.argv[1:])
