#!/usr/bin/env python3
"""Assemble the power overlay: what one card power session needs, and nothing else.

The full package (``make_package.py``) freezes its B1 fixtures by running the
committed RTL cosimulation, which needs Icarus Verilog. A power session runs
none of those fixtures: it loads the two bitstreams and the two sealed bundles
the September campaigns already staged on ``/data``, and drives them with the
host code. So this overlay carries only

  * ``host/``: the board driver, its fake pyxrt, the deployment executor, the
    two bundle modules copied VERBATIM from ``src/`` (hash-checked, as the full
    packager does), and the three power modules (sampler, estimator, protocol);
  * ``scripts/hacc/odin_board.sbatch``: the board job, with ``ODIN_MODE=power``;
  * ``deployment/``: the committed witness bundle and its replay, which the
    no-hardware selftest drives;
  * ``power/``: the session plan;
  * ``MANIFEST.json``: the generating commit, whether the tree was dirty, and
    every file's sha256.

``--build`` (route D) adds what a compile-partition rebuild reads, and nothing
else: the RTL of every fabric (``hw/fpga``, the vendored ODIN tree, the
committed generated fabrics), the build scripts (``build_xclbn.sh``,
``toolchain.sh``, ``cards.sh``, ``chips.sh``, every card's v++ config,
``gen_link_cfg.py``), ``odin_build.sbatch``, and the ``kernel.xml`` emitted from
the host-side register SSOT with the package's verifying stand-in for
``gen_kernel_xml.py`` (its digest injected, as ``make_package.py`` does). The
build needs no cosimulation, so neither does this overlay.

    python3 scripts/hacc/make_power_overlay.py            # dist/odin_power_overlay.zip
    python3 scripts/hacc/make_power_overlay.py --stage-only --out <dir>
    python3 scripts/hacc/make_power_overlay.py --build    # plus the rebuild payload

On the cluster: unzip under ``${ODIN_DATA_ROOT:-/data}/$USER/``, then
``python3 odin_power_overlay/host/odin_power_protocol.py selftest --results
odin_power_overlay/results/power_selftest`` before any board job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
PACKAGE_SRC = REPO / "scripts" / "hacc" / "package"
CHIP_SIM = REPO / "src" / "mimarsinan" / "chip_simulation"
DIST = REPO / "dist"
NAME = "odin_power_overlay"
ZIP_EPOCH = (2026, 1, 1, 0, 0, 0)

HOST_FILES = (
    "odin_board_driver.py", "fake_pyxrt_for_selftest.py",
    "odin_deployment_executor.py", "odin_power_sampler.py",
    "odin_power_estimate.py", "odin_power_protocol.py",
)
VERBATIM_MODULES = ("odin_deployment_bundle.py", "odin_deployment_encoding.py")
DEPLOYMENT_FILES = ("nc1_two_core_passes.json", "nc1_two_core_passes_capture.json")
PLANS = ("plan_hacc_u55c_tcad.json",)

HACC = REPO / "scripts" / "hacc"
#: What ``build_xclbn.sh`` and ``odin_build.sbatch`` read under scripts/hacc/
#: (the full package ships the same files, ``make_package.py`` build_tree).
BUILD_SCRIPTS = ("build_xclbn.sh", "toolchain.sh", "cards.sh", "chips.sh",
                 "odin_u55c.cfg", "odin_u250.cfg", "gen_link_cfg.py")
KERNEL_NAME = "odin_fpga_kernel_top"
STAND_IN_TOKEN = "__ODIN_KERNEL_XML_SHA256__"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload() -> List[Tuple[Path, str]]:
    files: List[Tuple[Path, str]] = []
    files += [(PACKAGE_SRC / "host" / name, f"host/{name}") for name in HOST_FILES]
    files += [(CHIP_SIM / name, f"host/{name}") for name in VERBATIM_MODULES]
    files.append((PACKAGE_SRC / "sbatch" / "odin_board.sbatch",
                  "scripts/hacc/odin_board.sbatch"))
    files += [(PACKAGE_SRC / "deployment" / name, f"deployment/{name}")
              for name in DEPLOYMENT_FILES]
    files += [(PACKAGE_SRC / "power" / name, f"power/{name}") for name in PLANS]
    return files


def build_payload() -> List[Tuple[Path, str]]:
    """The RTL and the scripts a rebuild compiles, at their repository paths."""
    files: List[Tuple[Path, str]] = []
    for pattern in ("hw/fpga/kernel/*.v", "hw/fpga/mem/*.v", "hw/gen/chips/*/*.v"):
        files += [(path, str(path.relative_to(REPO))) for path in sorted(REPO.glob(pattern))]
    vendor = REPO / "hw" / "vendor" / "odin"
    for path in sorted(vendor.rglob("*")):
        if path.is_file() and (path.suffix in (".v", ".md", ".sha256")
                               or path.name == "LICENSE"):
            files.append((path, str(path.relative_to(REPO))))
    files += [(HACC / name, f"scripts/hacc/{name}") for name in BUILD_SCRIPTS]
    files.append((PACKAGE_SRC / "sbatch" / "odin_build.sbatch",
                  "scripts/hacc/odin_build.sbatch"))
    return files


def frozen_kernel_xml(work: Path) -> str:
    """kernel.xml from the register SSOT (``scripts/hacc/gen_kernel_xml.py``)."""
    output = work / "_kernel.xml"
    subprocess.run([sys.executable, str(HACC / "gen_kernel_xml.py"), "--output",
                    str(output), "--kernel", KERNEL_NAME], check=True,
                   capture_output=True, text=True, cwd=str(REPO))
    text = output.read_text(encoding="utf-8")
    output.unlink()
    return text


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def stage(root: Path, *, build: bool = False) -> Dict[str, object]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    written: Dict[str, str] = {}
    files = payload() + (build_payload() if build else [])
    for source, relative in files:
        if not source.is_file():
            raise SystemExit(f"REFUSING: {source} is missing")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != sha256_file(source):
            raise SystemExit(f"REFUSING: {relative} is not byte-identical to {source}")
        written[relative] = sha256_file(target)
    if build:
        xml = frozen_kernel_xml(root)
        (root / "scripts" / "hacc" / "kernel.xml").write_text(xml, encoding="utf-8")
        stand_in = (PACKAGE_SRC / "gen_kernel_xml.py").read_text(encoding="utf-8")
        if stand_in.count(STAND_IN_TOKEN) != 1:
            raise SystemExit("REFUSING: the gen_kernel_xml.py stand-in carries no single "
                             "digest placeholder")
        digest = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        (root / "scripts" / "hacc" / "gen_kernel_xml.py").write_text(
            stand_in.replace(STAND_IN_TOKEN, digest), encoding="utf-8")
        for relative in ("scripts/hacc/kernel.xml", "scripts/hacc/gen_kernel_xml.py"):
            written[relative] = sha256_file(root / relative)
    manifest = {
        "package": NAME,
        "generating_commit": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain", "--", "scripts/hacc", "src", "hw")
                               if build else
                               git("status", "--porcelain", "--", "scripts/hacc", "src")),
        "files": dict(sorted(written.items())),
    }
    if build:
        manifest["build"] = True
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def write_zip(root: Path, zip_path: Path) -> str:
    if zip_path.exists():
        zip_path.unlink()
    names = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            info = zipfile.ZipInfo(f"{NAME}/{name}", date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = 0o755 if name.endswith((".py", ".sh", ".sbatch")) else 0o644
            info.external_attr = (mode << 16) | 0o600
            archive.writestr(info, (root / name).read_bytes())
    return sha256_file(zip_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DIST), help="the directory to write into")
    parser.add_argument("--stage-only", action="store_true",
                        help="lay the overlay out under <out>/odin_power_overlay, no zip")
    parser.add_argument("--build", action="store_true",
                        help="also stage the rebuild payload (route D): RTL, build "
                             "scripts, odin_build.sbatch, the frozen kernel.xml")
    options = parser.parse_args(argv)
    out = Path(options.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = stage(out / NAME, build=options.build)
    print(f"[overlay] staged {len(manifest['files'])} files under {out / NAME} "
          f"(commit {manifest['generating_commit'][:12] or '?'}, "
          f"dirty={manifest['worktree_dirty']})")
    if not options.stage_only:
        digest = write_zip(out / NAME, out / f"{NAME}.zip")
        print(f"[overlay] wrote {out / (NAME + '.zip')} sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
