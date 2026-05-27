"""Shared Graphviz helpers."""

from __future__ import annotations

import base64
import html
import mimetypes
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore
from mimarsinan.mapping.packing.softcore_mapping import HardCoreMapping
from mimarsinan.common.layer_key import layer_key_from_node_name
from mimarsinan.common.safe_numeric import safe_float

def _percent(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{(100.0 * float(n) / float(d)):.1f}%"


def _compress_ranges(indices: Sequence[int], *, max_ranges: int = 10) -> str:
    if not indices:
        return "-"

    xs = sorted(int(i) for i in indices)
    ranges: list[tuple[int, int]] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append((start, prev))
        start = prev = x
    ranges.append((start, prev))

    def _fmt(a: int, b: int) -> str:
        return str(a) if a == b else f"{a}-{b}"

    out = [_fmt(a, b) for a, b in ranges[: max_ranges]]
    if len(ranges) > max_ranges:
        out.append(f"...(+{len(ranges) - max_ranges} ranges)")
    return ",".join(out)


def _truncate(s: str, *, max_chars: int = 260) -> str:
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _dot_html_label(rows: Sequence[tuple[str, str]], *, title: str, color: str) -> str:
    """
    Build a Graphviz HTML-like label (safe-escaped).
    """
    t = html.escape(str(title))
    c = html.escape(str(color))
    body = []
    body.append(f'<TR><TD BGCOLOR="{c}" COLSPAN="2"><B>{t}</B></TD></TR>')
    for k, v in rows:
        body.append(
            "<TR>"
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="9">{html.escape(str(k))}</FONT></TD>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="9">{html.escape(str(v))}</FONT></TD>'
            "</TR>"
        )
    return "<\n<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n" + "\n".join(body) + "\n</TABLE>\n>"


def _dot_html_label_mixed(
    rows: Sequence[tuple[str, str, bool]],
    *,
    title: str,
    color: str,
) -> str:
    """
    Like _dot_html_label, but allows marking row values as raw HTML.
    rows: (key, value, value_is_html)
    """
    t = html.escape(str(title))
    c = html.escape(str(color))
    body = []
    body.append(f'<TR><TD BGCOLOR="{c}" COLSPAN="2"><B>{t}</B></TD></TR>')
    for k, v, v_is_html in rows:
        k_html = html.escape(str(k))
        if v_is_html:
            v_html = str(v)
        else:
            v_html = html.escape(str(v))
        body.append(
            "<TR>"
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="9">{k_html}</FONT></TD>'
            f'<TD ALIGN="LEFT"><FONT POINT-SIZE="9">{v_html}</FONT></TD>'
            "</TR>"
        )
    return "<\n<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n" + "\n".join(body) + "\n</TABLE>\n>"


def _stack_sample_lines(names: list[str], ids: list[int]) -> str:
    """Produce a vertical stack of [id] name lines for large node groups."""
    n = len(ids)
    if n == 0:
        return "-"

    # Sort by id
    order = sorted(range(n), key=lambda i: ids[i])
    ids_s = [ids[i] for i in order]
    names_s = [names[i] for i in order]

    def _short(full: str) -> str:
        return _truncate(str(full), max_chars=70)

    if n <= 3:
        lines = [f"[{ids_s[i]}] {_short(names_s[i])}" for i in range(n)]
    else:
        mid_i = n // 2
        lines = [
            f"[{ids_s[0]}] {_short(names_s[0])}",
            "...",
            f"[{ids_s[mid_i]}] {_short(names_s[mid_i])}",
            "...",
            f"[{ids_s[-1]}] {_short(names_s[-1])}",
        ]

    # Escape each line and join with HTML breaks.
    return "<BR/>".join(html.escape(l) for l in lines)


def _embed_svg_images(svg_path: str) -> None:
    """Embed referenced raster images in an SVG as base64 data URIs."""
    svg_dir = os.path.dirname(os.path.abspath(svg_path))

    with open(svg_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    def _resolve(raw_path: str) -> str:
        if os.path.isabs(raw_path):
            return raw_path
        return os.path.join(svg_dir, raw_path)

    def _to_data_uri(abs_path: str) -> str | None:
        try:
            with open(abs_path, "rb") as fh:
                data = fh.read()
            mime = mimetypes.guess_type(abs_path)[0] or "image/png"
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{b64}"
        except OSError:
            return None

    def _replace_image_href(m: "re.Match[str]") -> str:
        full_tag = m.group(0)
        href_m = re.search(r'(xlink:href|href)="([^"]*)"', full_tag)
        if href_m is None:
            return full_tag
        attr, raw_path = href_m.group(1), href_m.group(2)
        if raw_path.startswith("data:"):
            return full_tag
        abs_path = _resolve(raw_path)
        data_uri = _to_data_uri(abs_path)
        if data_uri is None:
            return full_tag
        return full_tag.replace(
            f'{attr}="{raw_path}"',
            f'{attr}="{data_uri}"',
            1,
        )

    content = re.sub(r"<image\b[^>]*>", _replace_image_href, content, flags=re.DOTALL)

    with open(svg_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def try_render_dot(dot_path: str, *, formats: Iterable[str] = ("svg",)) -> list[str]:
    """Best-effort rendering of a .dot file using the system dot binary."""
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        return []

    out_paths: list[str] = []
    base, _ = os.path.splitext(dot_path)
    for fmt in formats:
        fmt = str(fmt).lower().strip(".")
        out_path = base + "." + fmt
        try:
            subprocess.run(
                [dot_bin, f"-T{fmt}", dot_path, "-o", out_path],
                check=True,
                capture_output=True,
                text=True,
            )
            if fmt == "svg":
                _embed_svg_images(out_path)
            out_paths.append(out_path)
        except Exception:
            continue
    return out_paths


