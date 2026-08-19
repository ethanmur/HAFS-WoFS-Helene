"""Generate HAFS graphics and browse every storm from one local viewer.

Typical use, from the repository root::

    python analysis/viewer.py
    python analysis/viewer.py --generate missing
    python analysis/viewer.py --generate always --case helene_hfsa

The viewer uses only the Python standard library plus PyYAML.  While it is
running it rescans ``analysis/output`` every few seconds, so plots produced by
another terminal appear automatically.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORMS_DIR = REPO_ROOT / "storms"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "output"
PRODUCTS = (
    ("parent_qpf", "Side-by-side QPF", "plot"),
    ("ets_full", "ETS", "plot"),
    ("rmse_scatter", "RMSE", "plot"),
)


def _path_from_repo(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def read_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def config_kind(cfg: dict) -> str:
    if "run_dir" in cfg:
        return "case"
    if "run_root" in cfg:
        return "cycles"
    if "cases" in cfg:
        return "compare"
    if isinstance(cfg.get("models"), list) and all(
            isinstance(item, dict) and "cycles_yaml" in item
            for item in cfg["models"]):
        return "cycles-compare"
    return "unknown"


def discover_configs(storms_dir: Path, selectors=()) -> list[Path]:
    configs = sorted(storms_dir.glob("*.yaml")) + sorted(storms_dir.glob("*.yml"))
    if not selectors:
        return configs
    wanted = {Path(item).stem for item in selectors}
    selected = [path for path in configs if path.stem in wanted or path.name in selectors]
    missing = wanted - {path.stem for path in selected}
    if missing:
        raise ValueError(f"Unknown case selector(s): {', '.join(sorted(missing))}")
    return selected


def output_dir_for(path: Path, cfg: dict) -> Path:
    return _path_from_repo(cfg.get("out_dir", f"analysis/output/{path.stem}"))


def model_label(cfg: dict) -> str:
    if cfg.get("model_label"):
        return str(cfg["model_label"])
    if isinstance(cfg.get("models"), dict):
        return " / ".join(str(name) for name in cfg["models"])
    if isinstance(cfg.get("models"), list):
        return " / ".join(str(item.get("name", "model")) for item in cfg["models"])
    source = str(cfg.get("run_dir", cfg.get("run_root", ""))).upper()
    if "HFSA" in source:
        return "HAFS-A"
    if "HFSB" in source:
        return "HAFS-B"
    return "Comparison" if "cases" in cfg else "HAFS"


def expected_case_files(path: Path, cfg: dict) -> list[Path]:
    init = str(cfg.get("init", ""))
    slug = path.stem if init and init in path.stem else f"{path.stem}_{init}"
    if not init:
        return []  # The track file supplies init; only the full loader knows it.
    out = output_dir_for(path, cfg)
    return [out / f"{prefix}_{slug}.png" for prefix, _, _ in PRODUCTS]


def needs_generation(path: Path, cfg: dict) -> bool:
    kind = config_kind(cfg)
    expected = expected_case_files(path, cfg) if kind == "case" else []
    if expected:
        return any(not item.exists() for item in expected)
    out = output_dir_for(path, cfg)
    return not out.exists() or not any(out.glob("*.png"))


def generate(configs: list[Path], mode: str) -> list[tuple[Path, int]]:
    """Run the existing scientific entry point; return failed configs/codes."""
    failures = []
    for path in configs:
        cfg = read_config(path)
        kind = config_kind(cfg)
        if cfg.get("data_available") is False:
            print(f"Skipping {path.stem}: data_available is false")
            continue
        command = {"case": "all", "cycles": "cycles",
                   "compare": "compare",
                   "cycles-compare": "cycles-compare"}.get(kind)
        if command is None or (mode == "missing" and not needs_generation(path, cfg)):
            continue
        print(f"\n--- Generating {path.stem} ({command}) ---", flush=True)
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "analysis" / "run.py"), str(path), command],
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode:
            failures.append((path, result.returncode))
            print(f"WARNING: {path.name} failed (exit {result.returncode}); continuing")
    return failures


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _file_record(path: Path, output_root: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "url": path.resolve().relative_to(output_root.resolve()).as_posix(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "size": stat.st_size,
    }


def build_manifest(configs: list[Path], output_root: Path) -> dict:
    """Describe configured cases plus unassigned output files for the browser."""
    output_root.mkdir(parents=True, exist_ok=True)
    cases = []
    assigned = set()
    for path in configs:
        cfg = read_config(path)
        kind = config_kind(cfg)
        out = output_dir_for(path, cfg)
        files = []
        if out.exists() and _inside(out, output_root):
            candidates = sorted(p for p in out.iterdir()
                                if p.suffix.lower() in {".png", ".gif", ".csv"})
            if kind == "case" and cfg.get("init"):
                init = str(cfg["init"])
                slug = path.stem if init in path.stem else f"{path.stem}_{init}"
                candidates = [p for p in candidates if slug in p.name]
            for item in candidates:
                files.append(_file_record(item, output_root))
                assigned.add(item.resolve())
        default_init = {"cycles": "all cycles"}.get(kind, "comparison")
        cases.append({
            "id": path.stem,
            "storm": str(cfg.get("storm_name", cfg.get("label", path.stem))),
            "model": model_label(cfg),
            "init": str(cfg.get("init", default_init)),
            "kind": kind,
            "config": path.resolve().relative_to(REPO_ROOT).as_posix(),
            "files": files,
        })

    orphan_files = []
    for item in sorted(output_root.rglob("*")):
        if (item.is_file()
                and item.suffix.lower() in {".png", ".gif", ".csv"}
                and item.resolve() not in assigned):
            orphan_files.append(_file_record(item, output_root))
    if orphan_files:
        cases.append({
            "id": "other_graphics", "storm": "Other graphics", "model": "", "init": "",
            "kind": "other", "config": "", "files": orphan_files,
        })
    return {"updated": datetime.now().isoformat(timespec="seconds"), "cases": cases}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HAFS analysis viewer</title>
<style>
:root{color-scheme:light;--ink:#172234;--muted:#64748b;--line:#dbe3ec;--blue:#0b66c3;--bg:#f4f7fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}
header{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid var(--line);padding:18px clamp(18px,4vw,56px)}
h1{font-size:23px;margin:0 0 12px}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:end}
label{font-size:12px;color:var(--muted);display:grid;gap:4px}select,input{min-width:180px;padding:8px 10px;border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink)}
#status{margin-left:auto;color:var(--muted);font-size:12px}main{padding:24px clamp(18px,4vw,56px) 60px}.case{margin-bottom:35px}
.case h2{font-size:19px;margin:0}.meta{color:var(--muted);font-size:13px;margin:2px 0 12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:16px}
.card{background:white;border:1px solid var(--line);border-radius:10px;overflow:hidden;box-shadow:0 2px 9px #23354d0b}.card img{display:block;width:100%;height:350px;object-fit:contain;background:#fff}
.caption{padding:9px 12px;border-top:1px solid var(--line);display:flex;gap:10px;justify-content:space-between}.caption a{color:var(--blue);text-decoration:none}.stamp{color:var(--muted);font-size:11px}
.tables{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.tables a{background:#e9f2fc;color:var(--blue);padding:6px 10px;border-radius:6px;text-decoration:none}.empty{padding:20px;background:#fff;border:1px dashed #b9c5d2;border-radius:8px;color:var(--muted)}
@media(max-width:600px){.card img{height:auto}#status{width:100%;margin-left:0}}
</style></head><body>
<header><h1>HAFS analysis viewer</h1><div class="controls">
<label>Storm<select id="storm"><option value="">All storms</option></select></label>
<label>Model<select id="model"><option value="">All models</option></select></label>
<label>Initialization<select id="init"><option value="">All initializations</option></select></label>
<label>Find<input id="search" placeholder="ETS, RMSE, filename…"></label><span id="status">Loading…</span>
</div></header><main id="content"></main>
<script>
let manifest={cases:[]}; const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function options(id,values,label){const e=document.getElementById(id),old=e.value;e.innerHTML=`<option value="">All ${label}</option>`+[...new Set(values.filter(Boolean))].sort().map(v=>`<option>${esc(v)}</option>`).join('');e.value=old}
function title(name){if(name.startsWith('parent_qpf_'))return'Side-by-side QPF';if(name.startsWith('ets_'))return'ETS';if(name.startsWith('rmse_scatter_'))return'RMSE';if(name.startsWith('cycles_compare_ets_'))return'HAFS-A/B/M ETS comparison';if(name.startsWith('cycles_compare_fss'))return'HAFS-A/B/M FSS comparison';if(name.startsWith('cycles_metrics_'))return'Cycle metrics';if(name.startsWith('cycles_ets_heatmap_'))return'Cycle ETS heatmap';if(name.startsWith('cycles_ets_bars_'))return'Model ETS by rainfall threshold';if(name.startsWith('cycles_fss_heatmap_'))return'Cycle FSS heatmap';if(name.startsWith('cycles_difference_'))return'Cycle forecast difference animation';if(name.startsWith('cycles_observed_'))return'Cycle observed QPF animation';if(name.startsWith('cycles_qpf_'))return'Cycle parent QPF animation';if(name.startsWith('compare_'))return'Model comparison';return name.replace(/\.(png|gif)$/,'').replaceAll('_',' ')}
const fileUrl=f=>f.data||('/'+encodeURI(f.url)),imageUrl=f=>f.data||(fileUrl(f)+'?v='+encodeURIComponent(f.modified));
function render(){options('storm',manifest.cases.map(c=>c.storm),'storms');options('model',manifest.cases.map(c=>c.model),'models');options('init',manifest.cases.map(c=>c.init),'initializations');
 const filters=['storm','model','init'].map(id=>document.getElementById(id).value),q=document.getElementById('search').value.toLowerCase();let shown=0;
 const html=manifest.cases.filter(c=>(!filters[0]||c.storm===filters[0])&&(!filters[1]||c.model===filters[1])&&(!filters[2]||c.init===filters[2])).map(c=>{const pics=c.files.filter(f=>(f.name.endsWith('.png')||f.name.endsWith('.gif'))&&(!q||(c.storm+' '+c.model+' '+f.name).toLowerCase().includes(q))),tables=c.files.filter(f=>f.name.endsWith('.csv')&&(!q||f.name.toLowerCase().includes(q)));if(q&&!pics.length&&!tables.length)return'';shown++;
 return `<section class="case"><h2>${esc(c.storm)}</h2><div class="meta">${esc([c.model,c.init,c.kind,c.config].filter(Boolean).join(' · '))}</div>${pics.length?`<div class="grid">${pics.map(f=>`<article class="card"><a href="${fileUrl(f)}" target="_blank"><img src="${imageUrl(f)}" loading="lazy" alt="${esc(title(f.name))}"></a><div class="caption"><a href="${fileUrl(f)}" target="_blank">${esc(title(f.name))}</a><span class="stamp">${esc(f.modified.replace('T',' '))}</span></div></article>`).join('')}</div>`:'<div class="empty">No plots generated yet. Run with <code>--generate missing</code>.</div>'}${tables.length?`<div class="tables">${tables.map(f=>`<a href="${fileUrl(f)}" target="_blank" download="${esc(f.name)}">CSV · ${esc(f.name)}</a>`).join('')}</div>`:''}</section>`}).join('');
 document.getElementById('content').innerHTML=html||'<div class="empty">No matching graphics.</div>';document.getElementById('status').textContent=`${shown} group${shown===1?'':'s'} · updated ${manifest.updated.replace('T',' ')}`}
async function refresh(){try{const r=await fetch('/api/manifest',{cache:'no-store'});manifest=await r.json();render()}catch(e){document.getElementById('status').textContent='Viewer disconnected'}}
['storm','model','init','search'].forEach(id=>document.getElementById(id).addEventListener(id==='search'?'input':'change',render));refresh();setInterval(refresh,5000);
</script></body></html>'''


def export_offline_html(manifest: dict, output_root: Path, destination: Path) -> Path:
    """Write a self-contained gallery that needs neither server nor tunnel."""
    exported = json.loads(json.dumps(manifest))
    for case in exported["cases"]:
        for record in case["files"]:
            source = output_root / record["url"]
            mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
            payload = base64.b64encode(source.read_bytes()).decode("ascii")
            record["data"] = f"data:{mime};base64,{payload}"
    html = HTML.replace(
        "let manifest={cases:[]};",
        "let manifest=" + json.dumps(exported, separators=(",", ":")) + ";",
        1,
    ).replace("refresh();setInterval(refresh,5000);", "render();", 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html)
    return destination


def make_handler(configs: list[Path], output_root: Path):
    class ViewerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output_root), **kwargs)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                body = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/manifest":
                body = json.dumps(build_manifest(configs, output_root)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, fmt, *args):
            if os.environ.get("HAFS_VIEWER_HTTP_LOG"):
                super().log_message(fmt, *args)

    return ViewerHandler


def access_instructions(host: str, port: int, ssh_host: str | None = None,
                        environ=None) -> list[str]:
    """User-facing URLs, including an SSH tunnel hint on remote systems."""
    environ = os.environ if environ is None else environ
    loopback = host in {"127.0.0.1", "localhost", "::1"}
    if environ.get("SSH_CONNECTION") and loopback:
        destination = ssh_host or environ.get("HAFS_VIEWER_SSH_HOST") or "<your Hercules SSH host>"
        return [
            "Viewer is running on the remote login node.",
            "On your laptop, open a SECOND terminal and run:",
            f"  ssh -N -L {port}:127.0.0.1:{port} {destination}",
            "Keep both terminals open, then browse to:",
            f"  http://127.0.0.1:{port}",
            "If Hercules rejects forwarding as 'administratively prohibited',",
            "stop this server and run: python analysis/viewer.py --export",
        ]
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return [f"Viewer: http://{display_host}:{port}  (Ctrl-C to stop)"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], metavar="NAME",
                        help="YAML stem to include; repeat for more than one")
    parser.add_argument("--generate", choices=("never", "missing", "always"), default="never",
                        help="generate products before opening the viewer (default: never)")
    parser.add_argument("--storms-dir", type=Path, default=DEFAULT_STORMS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ssh-host",
                        help="your laptop's SSH alias for this HPC (instructions only)")
    parser.add_argument("--export", nargs="?", const="analysis/output/hafs-viewer.html",
                        metavar="HTML",
                        help="write one offline HTML gallery and exit (default: analysis/output/hafs-viewer.html)")
    parser.add_argument("--no-serve", action="store_true",
                        help="only generate products and write viewer-manifest.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configs = discover_configs(args.storms_dir.resolve(), args.case)
    if not configs:
        raise SystemExit(f"No storm YAML files found in {args.storms_dir}")
    failures = generate(configs, args.generate) if args.generate != "never" else []
    output_root = args.output_dir.resolve()
    manifest = build_manifest(configs, output_root)
    manifest_path = output_root / "viewer-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Indexed {len(manifest['cases'])} graphics groups in {manifest_path}")
    if failures:
        print(f"Generation finished with {len(failures)} failed config(s); existing plots remain viewable.")
    if args.export:
        destination = _path_from_repo(args.export).resolve()
        export_offline_html(manifest, output_root, destination)
        size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"Offline viewer: {destination} ({size_mb:.1f} MB)")
        print("Download this one file and open it in your laptop browser.")
        return 1 if failures else 0
    if args.no_serve:
        return 1 if failures else 0
    try:
        server = ThreadingHTTPServer((args.host, args.port),
                                     make_handler(configs, output_root))
    except OSError as exc:
        raise SystemExit(
            f"Could not start viewer on {args.host}:{args.port}: {exc}\n"
            f"Try another port, for example: --port {args.port + 1}"
        ) from exc
    for line in access_instructions(args.host, args.port, args.ssh_host):
        print(line)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
