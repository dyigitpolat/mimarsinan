#!/usr/bin/env python3
"""Render a routed checkpoint's placement CSV as a die map — PNG, or SVG anywhere.

    python3 host/render_die_map.py --csv placement.csv --out die_map.png

WHY TWO PATHS. The map is drawn on whatever node the post-build MINING ran on,
and a HACC compile node is promised Vitis and python3 — not matplotlib. So this
tries matplotlib first and falls back to writing the SVG by hand out of the
standard library. Both paths draw the SAME rectangles from the SAME bins; only
the ink differs, and the output names which path drew it.

THE CSV is what scripts/hacc/mine_checkpoint.sh's Tcl writes, one row per placed
leaf cell: ``name,class,site,x,y`` on an integer site grid. ``class`` is the
cell's allegiance — the ODIN kernel's own hierarchy, split by the sub-block it
belongs to, or ``shell`` for everything the platform brought. Kernel classes get
ink; the shell stays grey, because the point of the picture is where OUR logic
landed on the die.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

REQUIRED_COLUMNS = ("name", "class", "x", "y")
SHELL_CLASS = "shell"
SHELL_INK = "#9aa0a6"

#: Colour-blind-safe categorical ink for the kernel's own classes, in the order
#: the classes are first seen, so a re-render of the same CSV is stable.
KERNEL_INK = (
    "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9",
    "#f0e442", "#7f3b08",
)


class DieMapRefusal(RuntimeError):
    """The placement evidence cannot be drawn; the message says what is missing."""


def read_placement(path: str) -> List[Dict[str, Any]]:
    """The placed cells, refusing loud on a CSV that is not one."""
    if not os.path.isfile(path):
        raise DieMapRefusal(
            f"{path} does not exist. It is written by the Tcl in "
            f"scripts/hacc/mine_checkpoint.sh against a ROUTED checkpoint; "
            f"there is no way to draw a die map without one")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DieMapRefusal(f"{path} carries a header and no cells")
    missing = [name for name in REQUIRED_COLUMNS if name not in rows[0]]
    if missing:
        raise DieMapRefusal(
            f"{path} has no {missing} column(s); it declares "
            f"{sorted(rows[0])}. A die map drawn from the wrong columns would "
            f"be a picture of nothing")
    cells: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            cells.append({
                "name": str(row["name"]),
                "class": str(row["class"]) or SHELL_CLASS,
                "x": int(float(row["x"])), "y": int(float(row["y"])),
            })
        except (TypeError, ValueError) as exc:
            raise DieMapRefusal(
                f"{path} row {index + 2}: x/y are not numbers ({exc})") from exc
    return cells


def bin_cells(cells: Sequence[Dict[str, Any]], *, bins: int
              ) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[str, Any]]:
    """Fold cells onto a ``bins`` x ``bins`` grid; each bin keeps its majority class.

    A bin that holds any kernel cell is a KERNEL bin: the shell is everywhere and
    would otherwise swallow the very logic the map exists to locate.
    """
    xs = [cell["x"] for cell in cells]
    ys = [cell["y"] for cell in cells]
    extent = {"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys)}
    span_x = max(extent["x1"] - extent["x0"], 1)
    span_y = max(extent["y1"] - extent["y0"], 1)
    grid: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for cell in cells:
        column = min(bins - 1, (cell["x"] - extent["x0"]) * bins // (span_x + 1))
        row = min(bins - 1, (cell["y"] - extent["y0"]) * bins // (span_y + 1))
        slot = grid.setdefault(
            (column, row), {"count": 0, "classes": {}})
        slot["count"] += 1
        slot["classes"][cell["class"]] = slot["classes"].get(cell["class"], 0) + 1
    for slot in grid.values():
        kernel = {name: n for name, n in slot["classes"].items()
                  if name != SHELL_CLASS}
        pool = kernel or slot["classes"]
        slot["class"] = max(sorted(pool), key=lambda name: pool[name])
    extent["bins"] = bins
    return grid, extent


def ink_for(classes: Sequence[str]) -> Dict[str, str]:
    """A stable colour per class: the shell is grey, the kernel's blocks are not."""
    palette: Dict[str, str] = {SHELL_CLASS: SHELL_INK}
    index = 0
    for name in classes:
        if name in palette:
            continue
        palette[name] = KERNEL_INK[index % len(KERNEL_INK)]
        index += 1
    return palette


def _ordered_classes(cells: Sequence[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    for cell in cells:
        if cell["class"] not in seen:
            seen.append(cell["class"])
    return seen


def render_svg(grid, extent, palette, *, title: str, out: str,
               size: int = 900) -> str:
    """The stdlib path: one rect per occupied bin, no dependency at all."""
    bins = int(extent["bins"])
    step = size / bins
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
        f'height="{size + 70}" viewBox="0 0 {size} {size + 70}">',
        f'<rect width="{size}" height="{size + 70}" fill="#ffffff"/>',
        f'<text x="10" y="24" font-family="sans-serif" font-size="16">'
        f'{_escape(title)}</text>',
    ]
    for (column, row), slot in sorted(grid.items()):
        x = column * step
        # SVG's y grows downward and a die's row index grows upward.
        y = 40 + (bins - 1 - row) * step
        opacity = min(1.0, 0.35 + slot["count"] / 40.0)
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{step:.2f}" '
            f'height="{step:.2f}" fill="{palette[slot["class"]]}" '
            f'fill-opacity="{opacity:.3f}"/>')
    for index, (name, colour) in enumerate(sorted(palette.items())):
        x = 10 + index * 150
        parts.append(
            f'<rect x="{x}" y="{size + 44}" width="14" height="14" '
            f'fill="{colour}"/>'
            f'<text x="{x + 20}" y="{size + 56}" font-family="sans-serif" '
            f'font-size="12">{_escape(name)}</text>')
    parts.append("</svg>")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts) + "\n")
    return "svg(stdlib)"


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def render_png(grid, extent, palette, *, title: str, out: str) -> str:
    """The matplotlib path, when the node happens to have it."""
    import matplotlib  # noqa: PLC0415 - optional, probed by render()
    matplotlib.use("Agg")
    import matplotlib.patches as patches  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    bins = int(extent["bins"])
    figure, axes = plt.subplots(figsize=(8, 8))
    for (column, row), slot in sorted(grid.items()):
        axes.add_patch(patches.Rectangle(
            (column, row), 1, 1, facecolor=palette[slot["class"]],
            alpha=min(1.0, 0.35 + slot["count"] / 40.0), edgecolor="none"))
    axes.set_xlim(0, bins)
    axes.set_ylim(0, bins)
    axes.set_aspect("equal")
    axes.set_title(title)
    axes.set_xlabel("site column (binned)")
    axes.set_ylabel("site row (binned)")
    axes.legend(handles=[
        patches.Patch(facecolor=colour, label=name)
        for name, colour in sorted(palette.items())
    ], loc="upper right", fontsize=8)
    figure.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return "png(matplotlib)"


def render(csv_path: str, out: str, *, title: str, bins: int,
           want: str = "auto") -> Tuple[str, str]:
    """``(which path drew it, where it landed)`` — the fallback renames the file."""
    cells = read_placement(csv_path)
    grid, extent = bin_cells(cells, bins=bins)
    palette = ink_for(_ordered_classes(cells))
    if want in ("auto", "png"):
        try:
            return render_png(grid, extent, palette, title=title, out=out), out
        except ImportError as exc:
            if want == "png":
                raise DieMapRefusal(
                    f"--format png was asked for and matplotlib is not "
                    f"importable here ({exc}); the SVG path needs nothing at "
                    f"all") from exc
            out = os.path.splitext(out)[0] + ".svg"
    return render_svg(grid, extent, palette, title=title, out=out), out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="ODIN kernel on the die")
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--format", choices=("auto", "png", "svg"),
                        default="auto", dest="want")
    options = parser.parse_args(argv)
    try:
        drew, path = render(options.csv, options.out, title=options.title,
                            bins=options.bins, want=options.want)
    except DieMapRefusal as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2
    print(f"[die-map] {drew} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
