#!/usr/bin/env python3
"""Print annotated mimarsinan source tree with per-file and per-directory Python stats."""

from __future__ import annotations

import argparse
from pathlib import Path

MAX_LOC = 300
MAX_PY_PER_DIR = 10


def loc(path: Path) -> int:
    try:
        return sum(1 for _ in path.open(encoding="utf-8"))
    except OSError:
        return 0


def collect_py_stats(root: Path) -> tuple[dict[Path, int], dict[Path, list[tuple[str, int]]]]:
    """Return dir -> total_loc, dir -> [(filename, loc), ...] for direct .py children."""
    file_locs: dict[Path, int] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        file_locs[path] = loc(path)

    dir_children: dict[Path, list[tuple[str, int]]] = {}
    dir_totals: dict[Path, int] = {}

    all_dirs = {root}
    for p in file_locs:
        all_dirs.add(p.parent)
        d = p.parent
        while d != root.parent and str(d).startswith(str(root)):
            all_dirs.add(d)
            d = d.parent

    for directory in sorted(all_dirs, key=lambda p: (len(p.parts), str(p))):
        children: list[tuple[str, int]] = []
        total = 0
        for path, n in file_locs.items():
            if path.parent == directory:
                children.append((path.name, n))
                total += n
        dir_children[directory] = sorted(children, key=lambda x: x[0])
        dir_totals[directory] = total

    return dir_totals, dir_children


def subtree_py_count(directory: Path, file_locs: dict[Path, int]) -> int:
    return sum(1 for p in file_locs if p.is_relative_to(directory))


def subtree_total_loc(directory: Path, file_locs: dict[Path, int]) -> int:
    return sum(n for p, n in file_locs.items() if p.is_relative_to(directory))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "mimarsinan",
    )
    parser.add_argument("--max-loc", type=int, default=MAX_LOC)
    parser.add_argument("--max-py", type=int, default=MAX_PY_PER_DIR)
    args = parser.parse_args()

    root = args.root.resolve()
    file_locs = {
        p: n
        for p, n in (
            (path, loc(path))
            for path in sorted(root.rglob("*.py"))
            if "__pycache__" not in path.parts
        )
    }

    def immediate_subdirs(parent: Path) -> list[Path]:
        if not parent.is_dir():
            return []
        return sorted(
            [
                d
                for d in parent.iterdir()
                if d.is_dir() and not d.name.startswith((".", "__pycache__"))
            ],
            key=lambda p: p.name,
        )

    def print_tree(directory: Path, prefix: str = "", is_last: bool = True) -> None:
        if directory == root:
            name = f"{directory.name}/"
        else:
            branch = "└── " if is_last else "├── "
            name = f"{branch}{directory.name}/"
        py_count = subtree_py_count(directory, file_locs)
        total = subtree_total_loc(directory, file_locs)
        direct_py = sorted(
            [p for p in file_locs if p.parent == directory],
            key=lambda p: p.name,
        )
        direct_count = len(direct_py)
        stats = f"[{direct_count} direct .py, {py_count} subtree .py, {total} subtree LOC]"
        print(f"{prefix}{name}  {stats}")

        child_prefix = prefix + ("    " if is_last else "│   ")
        entries: list[tuple[str, Path | None, int | None]] = []
        for p in direct_py:
            entries.append(("file", p, file_locs[p]))
        for d in immediate_subdirs(directory):
            if any(p.is_relative_to(d) for p in file_locs) or (d / "__init__.py").exists():
                entries.append(("dir", d, None))

        for i, (kind, item, nloc) in enumerate(entries):
            last = i == len(entries) - 1
            branch = "└── " if last else "├── "
            if kind == "file":
                assert isinstance(item, Path) and nloc is not None
                flag = " !" if nloc > args.max_loc else ""
                print(f"{child_prefix}{branch}{item.name}  ({nloc} LOC){flag}")
            else:
                assert isinstance(item, Path)
                print_tree(item, child_prefix, is_last=last)

    print("=" * 72)
    print(f"Annotated source tree: {root}")
    print("=" * 72)
    print_tree(root)

    # Project-level stats
    over_loc = sorted(
        ((n, p.relative_to(root).as_posix()) for p, n in file_locs.items() if n > args.max_loc),
        reverse=True,
    )
    over_py_dirs = []
    seen = set()
    for directory in sorted({p.parent for p in file_locs} | {root}, key=str):
        d = directory
        while d not in seen and str(d).startswith(str(root)):
            seen.add(d)
            direct = [p for p in file_locs if p.parent == d]
            if len(direct) > args.max_py:
                over_py_dirs.append(
                    (len(direct), subtree_py_count(d, file_locs), d.relative_to(root).as_posix() or ".")
                )
            if d == root:
                break
            d = d.parent

    over_py_dirs.sort(reverse=True)

    print()
    print("=" * 72)
    print(f"Project stats (all .py under {root.name}/)")
    print("=" * 72)
    print(f"Total Python files: {len(file_locs)}")
    print(f"Total LOC:          {sum(file_locs.values())}")
    print()
    print(f"Files > {args.max_loc} LOC ({len(over_loc)}):")
    if over_loc:
        for n, rel in over_loc:
            print(f"  {n:5d}  {rel}")
    else:
        print("  (none)")

    print()
    print(f"Directories with > {args.max_py} direct .py files ({len(over_py_dirs)}):")
    if over_py_dirs:
        for direct_n, subtree_n, rel in over_py_dirs:
            print(f"  {direct_n:3d} direct / {subtree_n:4d} subtree .py  {rel}/")
    else:
        print("  (none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
