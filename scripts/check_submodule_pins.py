#!/usr/bin/env python3
"""Submodule-pin gate: the declared set is what the tree carries, and every pin is fetchable.

Two modes:

``--offline`` (no network)
    The declaration check alone: ``.gitmodules`` and the tree's gitlinks must
    name exactly the same paths, and that set must equal the expected set
    (``nevresim`` -- every other third-party inclusion is a declared dependency
    in ``pyproject.toml``). This is what the unit suite runs.

default (network)
    The offline checks, then for each submodule: the recorded gitlink must be
    reachable from one of the remote's refs, so a fresh clone can check it out.
    An unpushed pin is the failure this catches (nevresim carried one until
    2026-09-15).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The only inclusion that stays a submodule: co-owned C++ source consumed by
# path, with no Python package and nothing on PyPI. Everything else is declared
# in pyproject.toml (see docs and third_party/patches/README.md).
EXPECTED_SUBMODULES = frozenset({"nevresim"})


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd or REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def declared_submodules() -> set[str]:
    """Paths declared in ``.gitmodules`` (empty when the file is absent)."""
    gitmodules = REPO_ROOT / ".gitmodules"
    if not gitmodules.exists():
        return set()
    out = subprocess.run(
        ["git", "config", "-f", str(gitmodules), "--get-regexp", r"^submodule\..*\.path$"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    ).stdout
    return {line.split(" ", 1)[1].strip() for line in out.splitlines() if " " in line}


def gitlinks() -> dict[str, str]:
    """``path -> pinned commit`` for every mode-160000 entry in the index."""
    out = _git("ls-files", "--stage")
    pins: dict[str, str] = {}
    for line in out.splitlines():
        if not line.startswith("160000"):
            continue
        meta, path = line.split("\t", 1)
        pins[path.strip()] = meta.split()[1]
    return pins


def check_declarations() -> list[str]:
    declared = declared_submodules()
    pins = gitlinks()
    linked = set(pins)
    problems: list[str] = []
    for path in sorted(linked - declared):
        problems.append(
            f"gitlink {path!r} has no .gitmodules entry "
            f"(a stray gitlink; `git rm --cached {path}` removes it)"
        )
    for path in sorted(declared - linked):
        problems.append(f".gitmodules declares {path!r} but the index carries no gitlink for it")
    unexpected = sorted(linked & declared - EXPECTED_SUBMODULES)
    for path in unexpected:
        problems.append(
            f"submodule {path!r} is not in the expected set "
            f"({', '.join(sorted(EXPECTED_SUBMODULES))}); third-party code is a "
            f"declared dependency in pyproject.toml, not a checked-in tree"
        )
    for path in sorted(EXPECTED_SUBMODULES - (linked & declared)):
        problems.append(f"expected submodule {path!r} is missing from .gitmodules or the index")
    return problems


def check_pin_is_fetchable(path: str, pin: str) -> list[str]:
    """The pin must be reachable from some ``origin`` ref, or no fresh clone can use it."""
    sub = REPO_ROOT / path
    if not (sub / ".git").exists():
        return [f"{path}: not initialised locally; cannot verify the pin (run `git submodule update --init {path}`)"]
    try:
        refs = _git("ls-remote", "origin", cwd=sub)
    except subprocess.CalledProcessError as exc:
        return [f"{path}: `git ls-remote origin` failed: {exc.stderr.strip()}"]
    remote_shas = {line.split("\t", 1)[0] for line in refs.splitlines() if line}
    if pin in remote_shas:
        return []
    subprocess.run(
        ["git", "fetch", "--quiet", "origin"], cwd=str(sub), capture_output=True, text=True
    )
    for sha in remote_shas:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", pin, sha],
            cwd=str(sub),
            capture_output=True,
            text=True,
        )
        if ancestry.returncode == 0:
            return []
    return [
        f"{path}: pinned commit {pin} is not reachable from any ref on origin -- "
        f"a fresh clone fails at `git submodule update --init {path}`. Push the "
        f"branch that carries it."
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="declaration checks only; never touches the network",
    )
    args = parser.parse_args(argv)

    problems = check_declarations()
    if not args.offline:
        for path, pin in sorted(gitlinks().items()):
            problems.extend(check_pin_is_fetchable(path, pin))

    if problems:
        print("submodule pin check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    scope = "declarations" if args.offline else "declarations and remote reachability"
    print(f"submodule pin check OK ({scope}): {', '.join(sorted(gitlinks())) or 'no submodules'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
