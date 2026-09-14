#!/usr/bin/env python3
"""Apply ``third_party/patches/<dist>/NNN-slug.patch`` to installed distributions.

The hook exists so the next third-party fix is a patch file against a declared
dependency, not a vendored source tree. It is dormant today: no third-party
source needs patching (see third_party/patches/README.md), so a run is a no-op.

Idempotent: each distribution's applied set is recorded beside it as
``<dist>.mimarsinan-patches.json`` (``patch file -> sha256``). Re-running with
the same hashes does nothing; a changed or new patch is applied and recorded.
A patch whose content changed after it was applied cannot be re-applied onto
the already-patched tree -- reinstall the distribution first
(``uv sync --reinstall-package <dist>``), which the error says.

``--check`` applies nothing and exits non-zero when any patch on disk is not
recorded as applied; that is what the third-party contract test runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = REPO_ROOT / "third_party" / "patches"
RECORD_SUFFIX = ".mimarsinan-patches.json"


def patch_sets() -> dict[str, list[Path]]:
    """``distribution name -> its patch files``, ordered by file name (NNN- prefix)."""
    if not PATCH_ROOT.is_dir():
        return {}
    sets: dict[str, list[Path]] = {}
    for dist_dir in sorted(p for p in PATCH_ROOT.iterdir() if p.is_dir()):
        patches = sorted(dist_dir.glob("*.patch"))
        if patches:
            sets[dist_dir.name] = patches
    return sets


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_root(dist_name: str) -> Path:
    """Directory the patches' ``-p1`` paths are relative to (the site-packages root)."""
    try:
        dist = md.distribution(dist_name)
    except md.PackageNotFoundError as exc:
        raise LookupError(
            f"{dist_name} has a patch directory but is not installed; "
            f"install it (an extra?) or remove third_party/patches/{dist_name}/"
        ) from exc
    located = dist.locate_file("")
    return Path(str(located)).resolve()


def _record_path(dist_name: str, root: Path) -> Path:
    return root / f"{dist_name}{RECORD_SUFFIX}"


def _read_record(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _patch(root: Path, patch_file: Path, dry_run: bool) -> subprocess.CompletedProcess[str]:
    cmd = ["patch", "-p1", "--forward"]
    if dry_run:
        cmd.append("--dry-run")
    cmd += ["-i", str(patch_file)]
    return subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)


def check() -> list[str]:
    """Names every patch on disk that is not recorded as applied."""
    problems: list[str] = []
    for dist_name, patches in patch_sets().items():
        try:
            root = _install_root(dist_name)
        except LookupError as exc:
            problems.append(str(exc))
            continue
        record = _read_record(_record_path(dist_name, root))
        for patch_file in patches:
            digest = _sha256(patch_file)
            if record.get(patch_file.name) != digest:
                problems.append(
                    f"{dist_name}: {patch_file.name} is not applied to {root} "
                    f"(run `python scripts/apply_patches.py`)"
                )
    return problems


def apply_all() -> int:
    sets = patch_sets()
    if not sets:
        print(f"no third-party patches to apply ({PATCH_ROOT} is empty by design)")
        return 0
    if shutil.which("patch") is None:
        print("the `patch` utility is required to apply third-party patches", file=sys.stderr)
        return 1

    failed = False
    for dist_name, patches in sets.items():
        root = _install_root(dist_name)
        record_path = _record_path(dist_name, root)
        record = _read_record(record_path)
        for patch_file in patches:
            digest = _sha256(patch_file)
            if record.get(patch_file.name) == digest:
                print(f"{dist_name}: {patch_file.name} already applied")
                continue
            dry = _patch(root, patch_file, dry_run=True)
            if dry.returncode != 0:
                print(
                    f"{dist_name}: {patch_file.name} does not apply to {root}:\n"
                    f"{dry.stdout}{dry.stderr}"
                    f"  reinstall the distribution first: "
                    f"`uv sync --reinstall-package {dist_name}`",
                    file=sys.stderr,
                )
                failed = True
                continue
            real = _patch(root, patch_file, dry_run=False)
            if real.returncode != 0:
                print(
                    f"{dist_name}: {patch_file.name} failed after a clean dry run:\n"
                    f"{real.stdout}{real.stderr}",
                    file=sys.stderr,
                )
                failed = True
                continue
            record[patch_file.name] = digest
            print(f"{dist_name}: applied {patch_file.name}")
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report unapplied patches without touching anything",
    )
    args = parser.parse_args(argv)

    if args.check:
        problems = check()
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    return apply_all()


if __name__ == "__main__":
    sys.exit(main())
