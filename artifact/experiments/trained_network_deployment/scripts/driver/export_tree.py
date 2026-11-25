""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

RECORD = "E2E_EXPORT.json"
TREES = ("src", "tests", "scripts", "nevresim", "docs", "templates")
PLUGIN_DIR = "build/mimarsinan_sanafe_plugins"
PLUGIN_GLOB = "libmimarsinan_*.so"


HUNK_MOVED = re.compile(r"^Hunk #\d+ succeeded at \d+ (with fuzz \d+|\(offset )", re.M)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root: Path, sub: str) -> dict:
    base = Path(root) / sub
    return {str(p.relative_to(base)): _sha256(p) for p in sorted(base.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def manifest_sha256(listing: dict) -> str:
    return hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()


def _git(worktree: str, *args) -> str:
    return subprocess.run(["git", "-C", worktree, *args], capture_output=True, text=True, check=True).stdout.strip()


def _patch(args: list, cwd: Path, patch: Path) -> subprocess.CompletedProcess:
    with open(patch, "rb") as f:
        return subprocess.run(["patch", "-p1", "--batch", "--fuzz=0", *args], cwd=str(cwd), stdin=f,
                              capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"})


def apply_patch(out: Path, patch: Path) -> str:
    ""
    reverse = _patch(["--reverse", "--dry-run", "--force"], out, patch)
    forward = _patch(["--forward", "--dry-run"], out, patch)
    if reverse.returncode == 0 and forward.returncode != 0:
        if HUNK_MOVED.search(reverse.stdout):
            raise SystemExit(f"REFUSED: the e2e patch is in the head only at an offset or fuzz: {reverse.stdout[-800:]}")
        return "in head"
    if forward.returncode != 0:
        raise SystemExit(f"REFUSED: the e2e patch does not apply at fuzz 0: {forward.stdout[-1500:]}")
    if HUNK_MOVED.search(forward.stdout):
        raise SystemExit(f"REFUSED: the e2e patch applies only at an offset or fuzz: {forward.stdout[-1500:]}")
    done = _patch(["--forward"], out, patch)
    if done.returncode != 0:
        raise SystemExit(f"REFUSED: the e2e patch failed: {done.stdout[-1500:]}")
    return "applied"


def copy_sanafe_plugins(worktree, out) -> int:
    ""
    src = Path(worktree) / PLUGIN_DIR
    libs = sorted(src.glob(PLUGIN_GLOB)) if src.is_dir() else []
    if not libs:
        raise SystemExit(f"REFUSED: no SANA-FE plugin build in {src} (scripts/build_sanafe_plugins.py builds it)")
    dst = Path(out) / PLUGIN_DIR
    dst.mkdir(parents=True, exist_ok=True)
    for lib in libs:
        shutil.copy2(lib, dst / lib.name)
    return len(libs)


def plugin_manifest(root) -> dict | None:
    base = Path(root) / PLUGIN_DIR
    if not base.is_dir():
        return None
    return {p.name: _sha256(p) for p in sorted(base.glob(PLUGIN_GLOB))}


def make(worktree: str, commit: str, out: Path, patch: Path, nevresim_sha256: str | None = None,
         with_sanafe_plugins: bool = False) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"REFUSED: {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(["git", "-C", worktree, "archive", commit], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(out)], stdin=archive.stdout, check=True)
    if archive.wait() != 0:
        raise SystemExit("REFUSED: git archive failed")
    nev_src, nev_dst = Path(worktree) / "nevresim", out / "nevresim"
    if nev_dst.exists():
        shutil.rmtree(nev_dst)
    nev_dst.mkdir()
    for item in sorted(nev_src.iterdir()):
        if item.name == ".git":
            continue
        subprocess.run(["cp", "-a", str(item), str(nev_dst)], check=True)
    datasets = out / "datasets"
    if not datasets.exists():
        os.symlink(Path(worktree) / "datasets", datasets)
    state = apply_patch(out, patch)
    if with_sanafe_plugins:
        copy_sanafe_plugins(worktree, out)
    nevresim_now = manifest_sha256(manifest(out, "nevresim"))
    if nevresim_sha256 is not None and nevresim_now != nevresim_sha256:
        raise SystemExit(f"REFUSED: the export's nevresim manifest {nevresim_now} is not the one "
                         f"expected at the head of record ({nevresim_sha256})")
    return record(worktree, commit, out, patch, patch_state=state)


def record(worktree: str, commit: str, out: Path, patch: Path, patch_state: str | None = None) -> dict:
    out = Path(out)
    full = _git(worktree, "rev-parse", f"{commit}^{{commit}}")
    rec = {"format": "e2e-export/1", "commit": full, "tree": _git(worktree, "rev-parse", f"{full}^{{tree}}"),
           "worktree": str(worktree), "patch": str(patch), "patch_sha256": _sha256(Path(patch)),
           "patch_state": patch_state,
           "datasets_link": os.readlink(out / "datasets") if (out / "datasets").is_symlink() else None,
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for sub in TREES:
        listing = manifest(out, sub)
        rec[f"{sub}_files"], rec[f"{sub}_manifest_sha256"] = len(listing), manifest_sha256(listing)
    plugins = plugin_manifest(out)
    if plugins:
        rec["sanafe_plugins_files"], rec["sanafe_plugins_manifest_sha256"] = len(plugins), manifest_sha256(plugins)
    (out / RECORD).write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec


def verify(root: Path) -> dict:
    ""
    root = Path(root)
    rec = json.loads((root / RECORD).read_text())
    held = [sub for sub in TREES if f"{sub}_manifest_sha256" in rec]
    now = {sub: manifest_sha256(manifest(root, sub)) for sub in held}
    changed = [sub for sub in held if now[sub] != rec[f"{sub}_manifest_sha256"]]
    ok = not changed
    out = {"commit": rec["commit"], "tree": rec["tree"], "export": str(root), "export_record_sha256":
           _sha256(root / RECORD), "patch_sha256": rec["patch_sha256"], "src_manifest_sha256": now.get("src"),
           "nevresim_manifest_sha256": now.get("nevresim"), "export_verified": ok, "dirty": not ok,
           "status_sha256": None, "diff_sha256": rec["patch_sha256"], "untracked_sha256": {},
           "trees_verified": held, "trees_not_recorded": [sub for sub in TREES if sub not in held],
           "trees_changed": changed}
    if "sanafe_plugins_manifest_sha256" in rec:
        plugins = plugin_manifest(root) or {}
        out["sanafe_plugins_manifest_sha256"] = manifest_sha256(plugins)
        if out["sanafe_plugins_manifest_sha256"] != rec["sanafe_plugins_manifest_sha256"]:
            out.update(export_verified=False, dirty=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("make", "record", "verify"))
    ap.add_argument("--worktree")
    ap.add_argument("--commit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch")
    ap.add_argument("--nevresim-sha256", default=None)
    ap.add_argument("--with-sanafe-plugins", action="store_true", help="copy and record the SANA-FE plugin build")
    a = ap.parse_args()
    if a.cmd == "make":
        rec = make(a.worktree, a.commit, Path(a.out), Path(a.patch), a.nevresim_sha256, a.with_sanafe_plugins)
    elif a.cmd == "record":
        rec = record(a.worktree, a.commit, Path(a.out), Path(a.patch))
    else:
        rec = verify(Path(a.out))
        if rec["trees_not_recorded"]:
            print(f"NOT VERIFIED: the record holds no manifest of {', '.join(rec['trees_not_recorded'])} (an export "
                  "made by an earlier tool); the trees it holds are verified", file=sys.stderr)
        if not rec["export_verified"]:
            print(json.dumps(rec, indent=1))
            return 1
    print(json.dumps(rec, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
