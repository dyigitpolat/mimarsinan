from __future__ import annotations

import html
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore
from mimarsinan.mapping.softcore_mapping import HardCoreMapping

import re


def _as_float(x: Any) -> float | None:
    try:
        # torch scalar / numpy scalar / python numeric
        return float(x)
    except Exception:
        return None


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


_RE_CONV_POS = re.compile(r"^(.*)_pos\d+_\d+_g\d+$")
_RE_FC_TILE = re.compile(r"^(.*)_tile_\d+_\d+$")


def _layer_key_from_node_name(name: str) -> str:
    """
    Best-effort grouping key that collapses per-position/per-tile neural cores into a "layer stack".
    """
    s = str(name)
    m = _RE_CONV_POS.match(s)
    if m:
        return m.group(1)
    m = _RE_FC_TILE.match(s)
    if m:
        return m.group(1)
    if "_psum_" in s:
        return s.split("_psum_", 1)[0]
    return s


def _stack_sample_lines(names: list[str], ids: list[int]) -> str:
    """
    Produce a vertical stack like:
      [first] ...
      ...
      [middle] ...
      ...
      [last] ...
    """
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


def write_ir_graph_summary_dot(
    ir_graph: IRGraph,
    out_dot: str,
    *,
    title: str | None = None,
    max_edges: int = 64,
) -> None:
    """
    Summarized IRGraph visualization:
    - groups many NeuralCore nodes into a single "layer stack" node by name pattern
    - shows only [first] ... [middle] ... [last] for large stacks
    - aggregates edges at the group level to keep the graph readable
    """
    os.makedirs(os.path.dirname(out_dot) or ".", exist_ok=True)

    # Build ordered groups
    group_order: list[tuple[str, str]] = []  # (group_key, kind)
    group_nodes: dict[str, list[IRNode]] = {}
    node_id_to_group: dict[int, str] = {}

    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            gk = f"neural::{_layer_key_from_node_name(node.name)}"
            kind = "neural"
        elif isinstance(node, ComputeOp):
            gk = f"compute::{node.id}::{node.name}"
            kind = "compute"
        else:
            gk = f"other::{node.id}::{getattr(node, 'name', type(node).__name__)}"
            kind = "other"

        if gk not in group_nodes:
            group_order.append((gk, kind))
            group_nodes[gk] = []
        group_nodes[gk].append(node)
        try:
            node_id_to_group[int(node.id)] = gk
        except Exception:
            pass

    # Assign DOT node ids for groups
    group_key_to_dot: dict[str, str] = {gk: f"g{i}" for i, (gk, _) in enumerate(group_order)}

    lines: list[str] = []
    lines.append("digraph IRGraphSummary {")
    lines.append("  rankdir=LR;")
    lines.append("  graph [fontname=\"Helvetica\", fontsize=10, labelloc=t];")
    lines.append("  node [shape=plaintext, fontname=\"Helvetica\"];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")

    if title:
        lines.append(f"  label=\"{_truncate(title, max_chars=140)} (summary)\";")

    lines.append("  input [label=< <B>INPUT</B> >, shape=plaintext];")
    lines.append("  const1 [label=< <B>CONST(1)</B> >, shape=plaintext];")
    lines.append("  output [label=< <B>OUTPUT</B> >, shape=plaintext];")

    # Emit group nodes
    for gk, kind in group_order:
        dot_id = group_key_to_dot[gk]
        nodes = group_nodes[gk]

        if kind == "compute":
            op = nodes[0]
            assert isinstance(op, ComputeOp)
            rows = [
                ("kind", "ComputeOp"),
                ("op_type", str(op.op_type)),
                ("input_shape", str(op.input_shape)),
                ("output_shape", str(op.output_shape)),
                ("params", _truncate(str(op.params), max_chars=220)),
            ]
            label = _dot_html_label(rows, title=str(op.name), color="#FAD7A0")
            lines.append(f"  {dot_id} [label={label}];")
            continue

        if kind == "neural":
            # Stack summary
            core_ids = [int(n.id) for n in nodes if hasattr(n, "id")]
            core_names = [str(getattr(n, "name", f"n{getattr(n, 'id', '?')}")) for n in nodes]

            shapes: dict[tuple[int, int], int] = {}
            roles: dict[str, int] = {}
            thresholds: list[float] = []
            latencies: list[int] = []
            for n in nodes:
                if not isinstance(n, NeuralCore):
                    continue
                ax = int(n.core_matrix.shape[0])
                neu = int(n.core_matrix.shape[1])
                shapes[(ax, neu)] = shapes.get((ax, neu), 0) + 1
                if n.psum_role is not None:
                    roles[str(n.psum_role)] = roles.get(str(n.psum_role), 0) + 1
                thr = _as_float(n.threshold)
                if thr is not None:
                    thresholds.append(thr)
                if n.latency is not None:
                    latencies.append(int(n.latency))

            shapes_txt = ", ".join(
                f"{ax}×{neu}×{cnt}" for (ax, neu), cnt in sorted(shapes.items(), key=lambda x: (-x[1], x[0]))
            )
            ids_txt = _compress_ranges(sorted(core_ids), max_ranges=8)
            thr_txt = "-"
            if thresholds:
                thr_txt = f"{min(thresholds):.3g}..{max(thresholds):.3g}" if len(thresholds) > 1 else f"{thresholds[0]:.3g}"
            lat_txt = "-"
            if latencies:
                lat_txt = f"{min(latencies)}..{max(latencies)}" if len(latencies) > 1 else str(latencies[0])

            # Best-effort: infer conv grid and group count from names
            pos_coords: list[tuple[int, int]] = []
            g_idxs: set[int] = set()
            for nm in core_names:
                m = re.search(r"_pos(\d+)_(\d+)_g(\d+)$", nm)
                if m:
                    pos_coords.append((int(m.group(1)), int(m.group(2))))
                    g_idxs.add(int(m.group(3)))
            conv_txt = "-"
            if pos_coords:
                hs = [p[0] for p in pos_coords]
                ws = [p[1] for p in pos_coords]
                conv_txt = f"pos_grid={max(hs)+1}×{max(ws)+1}, g={len(g_idxs) or 1}"

            role_txt = "-"
            if roles:
                role_txt = ", ".join(f"{k}:{v}" for k, v in sorted(roles.items()))

            base_name = gk.split("neural::", 1)[1]
            stack_html = _stack_sample_lines(core_names, core_ids)

            rows_mixed: list[tuple[str, str, bool]] = [
                ("kind", "NeuralCore stack", False),
                ("cores", f"{len(nodes)}", False),
                ("ids", ids_txt, False),
                ("shapes", shapes_txt or "-", False),
                ("conv_hint", conv_txt, False),
                ("psum_roles", role_txt, False),
                ("threshold", thr_txt, False),
                ("latency", lat_txt, False),
                ("stack", stack_html, True),
            ]
            label = _dot_html_label_mixed(rows_mixed, title=base_name, color="#D6EAF8")
            lines.append(f"  {dot_id} [label={label}];")
            continue

        # Fallback
        label = _dot_html_label([("kind", kind), ("count", str(len(nodes)))], title=gk, color="#E5E7E9")
        lines.append(f"  {dot_id} [label={label}];")

    # Aggregate edges between groups
    edge_stats: dict[tuple[str, str], dict[str, Any]] = {}
    # (src_dot, tgt_dot) -> {links:int, src_idx:set[int], src_cores:set[int]}

    for gk, kind in group_order:
        tgt_dot = group_key_to_dot[gk]
        nodes = group_nodes[gk]
        for node in nodes:
            flat = list(getattr(node, "input_sources", np.array([])).flatten())
            for src in flat:
                if not isinstance(src, IRSource):
                    continue
                if src.is_off():
                    continue

                if src.is_input():
                    src_dot = "input"
                    src_core_id = -2
                elif src.is_always_on():
                    src_dot = "const1"
                    src_core_id = -3
                else:
                    src_group = node_id_to_group.get(int(src.node_id))
                    if src_group is None:
                        continue
                    src_dot = group_key_to_dot[src_group]
                    src_core_id = int(src.node_id)

                key = (src_dot, tgt_dot)
                st = edge_stats.setdefault(key, {"links": 0, "src_idx": set(), "src_cores": set()})
                st["links"] += 1
                st["src_idx"].add(int(src.index))
                if src_core_id >= 0:
                    st["src_cores"].add(src_core_id)

    # Emit edges (cap)
    emitted = 0
    for (src_dot, tgt_dot), st in edge_stats.items():
        if emitted >= max_edges:
            break
        links = int(st["links"])
        idx_ranges = _compress_ranges(sorted(st["src_idx"]), max_ranges=8)
        src_core_count = len(st["src_cores"])
        label = f"links={links}\\nsrc_idx={idx_ranges}"
        if src_core_count:
            label += f"\\nsrc_cores={src_core_count}"
        lines.append(f"  {src_dot} -> {tgt_dot} [label=\"{_truncate(label, max_chars=200)}\"];")
        emitted += 1

    # Outputs (grouped)
    out_flat = list(ir_graph.output_sources.flatten())
    by_src: dict[str, set[int]] = {}
    for src in out_flat:
        if not isinstance(src, IRSource) or src.is_off():
            continue
        if src.is_input():
            src_dot = "input"
        elif src.is_always_on():
            src_dot = "const1"
        else:
            src_group = node_id_to_group.get(int(src.node_id))
            if src_group is None:
                continue
            src_dot = group_key_to_dot[src_group]
        by_src.setdefault(src_dot, set()).add(int(src.index))

    for src_dot, idxs in by_src.items():
        label = f"idx={_compress_ranges(sorted(idxs), max_ranges=8)}"
        lines.append(f"  {src_dot} -> output [label=\"{_truncate(label, max_chars=200)}\"];")

    lines.append("}")

    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _embed_svg_images(svg_path: str) -> None:
    """
    Post-process a rendered SVG to embed referenced raster images (PNG/JPG/…)
    as base64 data URIs.  This makes the SVG fully self-contained so it renders
    correctly in browsers, Cursor's preview pane, and any other SVG viewer
    without needing access to the local filesystem.

    Non-image ``xlink:href`` / ``href`` values (e.g. links to other SVGs) are
    left unchanged — only ``<image …>`` elements have their sources inlined.
    """
    import re as _re
    import base64
    import mimetypes

    svg_dir = os.path.dirname(os.path.abspath(svg_path))

    with open(svg_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    def _resolve(raw_path: str) -> str:
        """Return the absolute path for *raw_path* (which may be abs or relative)."""
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
        """Replace the href inside an <image …> tag with a data URI."""
        full_tag = m.group(0)
        href_m = _re.search(r'(xlink:href|href)="([^"]*)"', full_tag)
        if href_m is None:
            return full_tag
        attr, raw_path = href_m.group(1), href_m.group(2)
        if raw_path.startswith("data:"):
            return full_tag  # already embedded
        abs_path = _resolve(raw_path)
        data_uri = _to_data_uri(abs_path)
        if data_uri is None:
            return full_tag  # file not found – leave as-is
        return full_tag.replace(
            f'{attr}="{raw_path}"',
            f'{attr}="{data_uri}"',
            1,
        )

    # Only touch <image …> elements; leave <a xlink:href="…"> links alone.
    content = _re.sub(r"<image\b[^>]*>", _replace_image_href, content, flags=_re.DOTALL)

    with open(svg_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def try_render_dot(dot_path: str, *, formats: Iterable[str] = ("svg",)) -> list[str]:
    """
    Best-effort rendering of a .dot file to images using the system `dot` binary.
    Returns the written output paths (may be empty).
    SVG outputs are post-processed to embed any referenced raster images as
    base64 data URIs, making the SVG fully self-contained for browsers and
    SVG viewers (including Cursor's preview pane).
    """
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
            # Non-fatal: keep .dot as the source of truth.
            continue
    return out_paths


# ---------------------------------------------------------------------------
# IRGraph visualization (NeuralCore + ComputeOp)
# ---------------------------------------------------------------------------


def write_ir_graph_dot(
    ir_graph: IRGraph,
    out_dot: str,
    *,
    title: str | None = None,
    max_edges_per_node: int = 64,
) -> None:
    os.makedirs(os.path.dirname(out_dot) or ".", exist_ok=True)

    lines: list[str] = []
    lines.append("digraph IRGraph {")
    lines.append("  rankdir=LR;")
    lines.append("  graph [fontname=\"Helvetica\", fontsize=10, labelloc=t];")
    lines.append("  node [shape=plaintext, fontname=\"Helvetica\"];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")

    if title:
        lines.append(f"  label=\"{_truncate(title, max_chars=140)}\";")

    # Special nodes
    lines.append("  input [label=< <B>INPUT</B> >, shape=plaintext];")
    lines.append("  const1 [label=< <B>CONST(1)</B> >, shape=plaintext];")
    lines.append("  output [label=< <B>OUTPUT</B> >, shape=plaintext];")

    # Nodes
    for node in ir_graph.nodes:
        nid = f"n{int(node.id)}"
        if isinstance(node, NeuralCore):
            ax = int(node.core_matrix.shape[0])
            neu = int(node.core_matrix.shape[1])
            nnz = int(np.count_nonzero(node.core_matrix))
            total = int(ax * neu)
            rows = [
                ("id", str(node.id)),
                ("type", "NeuralCore"),
                ("shape(axons×neurons)", f"{ax}×{neu}"),
                ("weights nnz", f"{nnz}/{total} ({_percent(nnz, total)})"),
                ("threshold", str(_as_float(node.threshold) if node.threshold is not None else "n/a")),
                ("latency", str(node.latency if node.latency is not None else "n/a")),
            ]
            if node.psum_group_id is not None or node.psum_role is not None:
                rows.append(("psum", f"group={node.psum_group_id}, role={node.psum_role}"))
            label = _dot_html_label(rows, title=str(node.name), color="#D6EAF8")
        elif isinstance(node, ComputeOp):
            rows = [
                ("id", str(node.id)),
                ("type", "ComputeOp"),
                ("op_type", str(node.op_type)),
                ("input_shape", str(node.input_shape)),
                ("output_shape", str(node.output_shape)),
                ("params", _truncate(str(node.params), max_chars=220)),
            ]
            label = _dot_html_label(rows, title=str(node.name), color="#FAD7A0")
        else:
            rows = [("id", str(getattr(node, "id", "?"))), ("type", type(node).__name__)]
            label = _dot_html_label(rows, title=str(getattr(node, "name", "node")), color="#E5E7E9")

        lines.append(f"  {nid} [label={label}];")

    # Edges (compressed by source node_id)
    for node in ir_graph.nodes:
        tgt = f"n{int(node.id)}"
        flat = list(node.input_sources.flatten())

        by_src: dict[int, dict[str, list[int]]] = {}
        # by_src[src_node_id] = {"tgt_ax": [...], "src_idx": [...]}
        for ax_idx, src in enumerate(flat):
            if not isinstance(src, IRSource):
                continue
            if src.is_off():
                continue
            key = int(src.node_id)
            entry = by_src.setdefault(key, {"tgt_ax": [], "src_idx": []})
            entry["tgt_ax"].append(int(ax_idx))
            entry["src_idx"].append(int(src.index))

        # Cap edges per node to avoid unreadable graphs
        src_ids = list(by_src.keys())
        if len(src_ids) > max_edges_per_node:
            src_ids = src_ids[:max_edges_per_node]

        for src_node_id in src_ids:
            entry = by_src[src_node_id]
            if src_node_id == -2:
                src_name = "input"
            elif src_node_id == -3:
                src_name = "const1"
            else:
                src_name = f"n{src_node_id}"

            label = (
                f"axons:{_compress_ranges(entry['tgt_ax'])}\\n"
                f"idx:{_compress_ranges(entry['src_idx'])}\\n"
                f"n={len(entry['tgt_ax'])}"
            )
            lines.append(f"  {src_name} -> {tgt} [label=\"{_truncate(label, max_chars=220)}\"];")

    # Outputs
    out_flat = list(ir_graph.output_sources.flatten())
    by_src_out: dict[int, dict[str, list[int]]] = {}
    for out_i, src in enumerate(out_flat):
        if not isinstance(src, IRSource):
            continue
        if src.is_off():
            continue
        key = int(src.node_id)
        entry = by_src_out.setdefault(key, {"out_i": [], "src_idx": []})
        entry["out_i"].append(int(out_i))
        entry["src_idx"].append(int(src.index))

    for src_node_id, entry in by_src_out.items():
        if src_node_id == -2:
            src_name = "input"
        elif src_node_id == -3:
            src_name = "const1"
        else:
            src_name = f"n{src_node_id}"
        label = f"out:{_compress_ranges(entry['out_i'])}\\nidx:{_compress_ranges(entry['src_idx'])}"
        lines.append(f"  {src_name} -> output [label=\"{_truncate(label, max_chars=220)}\"];")

    lines.append("}")

    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# SoftCoreMapping visualization (actual mapped cores)
# ---------------------------------------------------------------------------


def write_softcore_mapping_dot(
    soft_core_mapping: Any,
    out_dot: str,
    *,
    title: str | None = None,
    cluster_by_psum_group: bool = True,
    max_edges_per_node: int = 64,
) -> None:
    """
    Visualize a mapped SoftCoreMapping (legacy mapping representation).
    Expects:
      - soft_core_mapping.cores: list[SoftCore]
      - soft_core_mapping.output_sources: list[SpikeSource]
    """
    os.makedirs(os.path.dirname(out_dot) or ".", exist_ok=True)

    cores = list(getattr(soft_core_mapping, "cores", []))
    out_sources = list(getattr(soft_core_mapping, "output_sources", []))

    lines: list[str] = []
    lines.append("digraph SoftCoreMapping {")
    lines.append("  rankdir=LR;")
    lines.append("  graph [fontname=\"Helvetica\", fontsize=10, labelloc=t];")
    lines.append("  node [shape=plaintext, fontname=\"Helvetica\"];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")

    if title:
        lines.append(f"  label=\"{_truncate(title, max_chars=140)}\";")

    lines.append("  input [label=< <B>INPUT</B> >, shape=plaintext];")
    lines.append("  const1 [label=< <B>CONST(1)</B> >, shape=plaintext];")
    lines.append("  output [label=< <B>OUTPUT</B> >, shape=plaintext];")

    # Optional cluster by psum_group_id
    by_group: dict[int | None, list[Any]] = {}
    for c in cores:
        by_group.setdefault(getattr(c, "psum_group_id", None), []).append(c)

    def _emit_core_node(c: Any):
        cid = int(getattr(c, "id"))
        nid = f"c{cid}"

        mat = getattr(c, "core_matrix")
        ax = int(mat.shape[0])
        neu = int(mat.shape[1])
        nnz = int(np.count_nonzero(mat))
        total = int(ax * neu)

        name = getattr(c, "name", None) or f"core_{cid}"
        psum = ""
        if getattr(c, "psum_group_id", None) is not None or getattr(c, "psum_role", None) is not None:
            psum = f"group={getattr(c,'psum_group_id',None)}, role={getattr(c,'psum_role',None)}"

        rows = [
            ("id", str(cid)),
            ("type", "SoftCore"),
            ("shape(axons×neurons)", f"{ax}×{neu}"),
            ("weights nnz", f"{nnz}/{total} ({_percent(nnz, total)})"),
            ("threshold", str(_as_float(getattr(c, "threshold", None)) or "n/a")),
            ("latency", str(getattr(c, "latency", None) if getattr(c, "latency", None) is not None else "n/a")),
        ]
        if psum:
            rows.append(("psum", psum))

        # Color by psum_role (if present)
        role = getattr(c, "psum_role", None)
        if role == "partial_pos":
            color = "#D5F5E3"
        elif role == "partial_neg":
            color = "#FADBD8"
        elif role == "accum":
            color = "#FCF3CF"
        else:
            color = "#EBF5FB"

        label = _dot_html_label(rows, title=str(name), color=color)
        lines.append(f"  {nid} [label={label}];")

    if cluster_by_psum_group and any(g is not None for g in by_group):
        for gid, group_cores in by_group.items():
            if gid is None:
                for c in group_cores:
                    _emit_core_node(c)
                continue
            lines.append(f"  subgraph cluster_psum_{int(gid)} {{")
            lines.append("    style=rounded; color=\"#AAB7B8\";")
            lines.append(f"    label=\"psum_group {int(gid)}\";")
            for c in group_cores:
                _emit_core_node(c)
            lines.append("  }")
    else:
        for c in cores:
            _emit_core_node(c)

    # Edges from axon_sources (compressed by source core_)
    for c in cores:
        cid = int(getattr(c, "id"))
        tgt = f"c{cid}"
        axon_sources: list[SpikeSource] = list(getattr(c, "axon_sources", []))

        by_src: dict[tuple[str, int], dict[str, list[int]]] = {}
        for ax_idx, src in enumerate(axon_sources):
            if src.is_off_:
                continue
            if src.is_input_:
                key = ("input", -2)
            elif src.is_always_on_:
                key = ("const1", -3)
            else:
                key = ("core", int(src.core_))

            entry = by_src.setdefault(key, {"tgt_ax": [], "src_idx": []})
            entry["tgt_ax"].append(int(ax_idx))
            entry["src_idx"].append(int(src.neuron_))

        keys = list(by_src.keys())
        if len(keys) > max_edges_per_node:
            keys = keys[:max_edges_per_node]

        for key in keys:
            kind, src_core = key
            entry = by_src[key]
            if kind == "input":
                src_node = "input"
            elif kind == "const1":
                src_node = "const1"
            else:
                src_node = f"c{src_core}"

            label = (
                f"axons:{_compress_ranges(entry['tgt_ax'])}\\n"
                f"idx:{_compress_ranges(entry['src_idx'])}\\n"
                f"n={len(entry['tgt_ax'])}"
            )
            lines.append(f"  {src_node} -> {tgt} [label=\"{_truncate(label, max_chars=220)}\"];")

    # Outputs
    by_src_out: dict[tuple[str, int], dict[str, list[int]]] = {}
    for out_i, src in enumerate(out_sources):
        if src.is_off_:
            continue
        if src.is_input_:
            key = ("input", -2)
        elif src.is_always_on_:
            key = ("const1", -3)
        else:
            key = ("core", int(src.core_))

        entry = by_src_out.setdefault(key, {"out_i": [], "src_idx": []})
        entry["out_i"].append(int(out_i))
        entry["src_idx"].append(int(src.neuron_))

    for key, entry in by_src_out.items():
        kind, src_core = key
        if kind == "input":
            src_node = "input"
        elif kind == "const1":
            src_node = "const1"
        else:
            src_node = f"c{src_core}"

        label = f"out:{_compress_ranges(entry['out_i'])}\\nidx:{_compress_ranges(entry['src_idx'])}"
        lines.append(f"  {src_node} -> output [label=\"{_truncate(label, max_chars=220)}\"];")

    lines.append("}")

    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# HardCoreMapping visualization (physical cores)
# ---------------------------------------------------------------------------


def write_hardcore_mapping_dot(
    hard_core_mapping: HardCoreMapping,
    out_dot: str,
    *,
    title: str | None = None,
    max_edges_per_node: int = 128,
) -> None:
    os.makedirs(os.path.dirname(out_dot) or ".", exist_ok=True)

    cores = list(hard_core_mapping.cores)
    out_sources = list(hard_core_mapping.output_sources.flatten())

    # Invert neuron_mapping for "packed softcore ids" summary
    packed: dict[int, dict[int, list[int]]] = {}
    # packed[hard_core_idx][soft_core_id] = [hard_neuron_idxs...]
    for (soft_core_id, soft_neuron), (hard_core_idx, hard_neuron) in hard_core_mapping.neuron_mapping.items():
        packed.setdefault(int(hard_core_idx), {}).setdefault(int(soft_core_id), []).append(int(hard_neuron))

    lines: list[str] = []
    lines.append("digraph HardCoreMapping {")
    lines.append("  rankdir=LR;")
    lines.append("  graph [fontname=\"Helvetica\", fontsize=10, labelloc=t];")
    lines.append("  node [shape=plaintext, fontname=\"Helvetica\"];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")

    if title:
        lines.append(f"  label=\"{_truncate(title, max_chars=140)}\";")

    lines.append("  input [label=< <B>INPUT</B> >, shape=plaintext];")
    lines.append("  const1 [label=< <B>CONST(1)</B> >, shape=plaintext];")
    lines.append("  output [label=< <B>OUTPUT</B> >, shape=plaintext];")

    for i, core in enumerate(cores):
        nid = f"h{i}"
        ax = int(core.axons_per_core)
        neu = int(core.neurons_per_core)
        used_ax = int(ax - core.available_axons)
        used_neu = int(neu - core.available_neurons)
        nnz = int(np.count_nonzero(core.core_matrix))
        total = int(ax * neu)

        rows = [
            ("id", str(i)),
            ("type", "HardCore"),
            ("capacity(axons×neurons)", f"{ax}×{neu}"),
            ("used(axons×neurons)", f"{used_ax}×{used_neu}"),
            ("weights nnz", f"{nnz}/{total} ({_percent(nnz, total)})"),
            ("threshold", str(_as_float(core.threshold) if core.threshold is not None else "n/a")),
            ("latency", str(core.latency if core.latency is not None else "n/a")),
            ("unusable_space", str(int(getattr(core, "unusable_space", 0)))),
        ]

        # Packed softcores summary (by soft core id -> neuron ranges)
        soft_ids = packed.get(i, {})
        if soft_ids:
            # Keep it readable.
            items = []
            for sid, hard_neus in sorted(soft_ids.items())[:10]:
                items.append(f"{sid}:{_compress_ranges(hard_neus, max_ranges=6)}")
            if len(soft_ids) > 10:
                items.append(f"...(+{len(soft_ids)-10})")
            rows.append(("packed_softcores", _truncate(" ".join(items), max_chars=220)))

        label = _dot_html_label(rows, title=f"HardCore {i}", color="#E8DAEF")
        lines.append(f"  {nid} [label={label}];")

    # Edges from axon_sources (compressed)
    for i, core in enumerate(cores):
        tgt = f"h{i}"
        axon_sources: list[SpikeSource] = list(core.axon_sources)

        by_src: dict[tuple[str, int], dict[str, list[int]]] = {}
        for ax_idx, src in enumerate(axon_sources):
            if src.is_off_:
                continue
            if src.is_input_:
                key = ("input", -2)
            elif src.is_always_on_:
                key = ("const1", -3)
            else:
                key = ("core", int(src.core_))

            entry = by_src.setdefault(key, {"tgt_ax": [], "src_idx": []})
            entry["tgt_ax"].append(int(ax_idx))
            entry["src_idx"].append(int(src.neuron_))

        keys = list(by_src.keys())
        if len(keys) > max_edges_per_node:
            keys = keys[:max_edges_per_node]

        for key in keys:
            kind, src_core = key
            entry = by_src[key]
            if kind == "input":
                src_node = "input"
            elif kind == "const1":
                src_node = "const1"
            else:
                src_node = f"h{src_core}"

            label = (
                f"axons:{_compress_ranges(entry['tgt_ax'])}\\n"
                f"idx:{_compress_ranges(entry['src_idx'])}\\n"
                f"n={len(entry['tgt_ax'])}"
            )
            lines.append(f"  {src_node} -> {tgt} [label=\"{_truncate(label, max_chars=220)}\"];")

    # Outputs
    by_src_out: dict[tuple[str, int], dict[str, list[int]]] = {}
    for out_i, src in enumerate(out_sources):
        if src.is_off_:
            continue
        if src.is_input_:
            key = ("input", -2)
        elif src.is_always_on_:
            key = ("const1", -3)
        else:
            key = ("core", int(src.core_))

        entry = by_src_out.setdefault(key, {"out_i": [], "src_idx": []})
        entry["out_i"].append(int(out_i))
        entry["src_idx"].append(int(src.neuron_))

    for key, entry in by_src_out.items():
        kind, src_core = key
        if kind == "input":
            src_node = "input"
        elif kind == "const1":
            src_node = "const1"
        else:
            src_node = f"h{src_core}"

        label = f"out:{_compress_ranges(entry['out_i'])}\\nidx:{_compress_ranges(entry['src_idx'])}"
        lines.append(f"  {src_node} -> output [label=\"{_truncate(label, max_chars=220)}\"];")

    lines.append("}")

    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# HybridHardCoreMapping visualization (multi-stage)
# ---------------------------------------------------------------------------


@dataclass
class HybridVizArtifacts:
    program_dot: str
    segment_dots: list[str]


def write_hybrid_hardcore_mapping_dots(
    hybrid: HybridHardCoreMapping,
    out_dir: str,
    *,
    basename: str = "hybrid_mapping",
) -> HybridVizArtifacts:
    """
    Writes:
      - <basename>.dot: stage-level program view
      - <basename>_segment{i}.dot: detailed HardCoreMapping view for each neural stage
    """
    os.makedirs(out_dir, exist_ok=True)

    program_dot = os.path.join(out_dir, f"{basename}.dot")
    segment_dots: list[str] = []

    # Stage-level program graph
    lines: list[str] = []
    lines.append("digraph HybridProgram {")
    lines.append("  rankdir=LR;")
    lines.append("  graph [fontname=\"Helvetica\", fontsize=10, labelloc=t];")
    lines.append("  node [shape=plaintext, fontname=\"Helvetica\"];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")
    lines.append("  label=\"HybridHardCoreMapping (neural segments + ComputeOp sync barriers)\";")

    prev = "input"
    lines.append("  input [label=< <B>INPUT</B> >];")

    neural_seg_idx = 0
    for stage_idx, stage in enumerate(hybrid.stages):
        sid = f"s{stage_idx}"
        if stage.kind == "neural":
            mapping = stage.hard_core_mapping
            core_count = len(mapping.cores) if mapping is not None else 0
            rows = [
                ("stage", str(stage_idx)),
                ("kind", "neural"),
                ("name", stage.name),
                ("hard_cores", str(core_count)),
            ]
            label = _dot_html_label(rows, title=f"Neural Segment {neural_seg_idx}", color="#D6EAF8")
            lines.append(f"  {sid} [label={label}];")

            seg_dot = os.path.join(out_dir, f"{basename}_segment{neural_seg_idx}.dot")
            if mapping is not None:
                write_hardcore_mapping_dot(
                    mapping,
                    seg_dot,
                    title=f"Hybrid segment {neural_seg_idx}: {stage.name}",
                )
            segment_dots.append(seg_dot)
            neural_seg_idx += 1
        else:
            op = stage.compute_op
            op_type = getattr(op, "op_type", "compute")
            rows = [
                ("stage", str(stage_idx)),
                ("kind", "compute"),
                ("name", stage.name),
                ("op_type", str(op_type)),
                ("params", _truncate(str(getattr(op, "params", {})), max_chars=200)),
            ]
            label = _dot_html_label(rows, title="SYNC BARRIER", color="#FAD7A0")
            lines.append(f"  {sid} [label={label}];")

        lines.append(f"  {prev} -> {sid};")
        prev = sid

    lines.append("  output [label=< <B>OUTPUT</B> >];")
    lines.append(f"  {prev} -> output;")
    lines.append("}")

    with open(program_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return HybridVizArtifacts(program_dot=program_dot, segment_dots=segment_dots)


def write_hybrid_hardcore_mapping_combined_dot(
    hybrid: HybridHardCoreMapping,
    out_dot: str,
    *,
    segment_graph_pngs: Sequence[str] | None = None,
    segment_heatmap_pngs: Sequence[str] | None = None,
    title: str | None = None,
) -> None:
    """
    A single "combined" hybrid visualization that includes:
    - stage connectivity (program flow)
    - ComputeOp metadata
    - per-neural-segment heatmaps (and optionally a connectivity thumbnail per segment)

    This is meant to be a readable overview, not a replacement for:
    - the detailed segment hardcore mapping graphs
    - the detailed heatmap-only views
    """

    os.makedirs(os.path.dirname(out_dot) or ".", exist_ok=True)

    def _shape_prod(shape) -> int | None:
        if shape is None:
            return None
        try:
            n = 1
            for d in shape:
                n *= int(d)
            return int(n)
        except Exception:
            return None

    def _input_size_from_mapping(mapping: HardCoreMapping) -> int | None:
        m = -1
        for core in mapping.cores:
            for src in core.axon_sources:
                if getattr(src, "is_input_", False):
                    m = max(m, int(src.neuron_))
        return (m + 1) if m >= 0 else None

    def _mapping_stats(mapping: HardCoreMapping) -> dict[str, str]:
        total_ax = sum(int(c.axons_per_core) for c in mapping.cores)
        total_neu = sum(int(c.neurons_per_core) for c in mapping.cores)
        used_ax = sum(int(c.axons_per_core - c.available_axons) for c in mapping.cores)
        used_neu = sum(int(c.neurons_per_core - c.available_neurons) for c in mapping.cores)
        nnz = sum(int(np.count_nonzero(c.core_matrix)) for c in mapping.cores)
        cap = sum(int(c.axons_per_core * c.neurons_per_core) for c in mapping.cores)
        unusable = sum(int(getattr(c, "unusable_space", 0)) for c in mapping.cores)

        return {
            "hard_cores": str(len(mapping.cores)),
            "axons_used": f"{used_ax}/{total_ax} ({_percent(used_ax, total_ax)})",
            "neurons_used": f"{used_neu}/{total_neu} ({_percent(used_neu, total_neu)})",
            "weights_nnz": f"{nnz}/{cap} ({_percent(nnz, cap)})",
            "unusable_space": str(unusable),
            "segment_in": str(_input_size_from_mapping(mapping) or "n/a"),
            "segment_out": str(len(mapping.output_sources.flatten())),
        }

    def _img(src_path: str, *, w: int, h: int) -> str:
        p = os.path.abspath(src_path)
        return f'<IMG SRC="{html.escape(p)}" WIDTH="{int(w)}" HEIGHT="{int(h)}" SCALE="TRUE"/>'

    # Graph
    lines: list[str] = []
    lines.append("digraph HybridCombined {")
    lines.append("  rankdir=LR;")
    lines.append("  graph [fontname=\"Helvetica\", fontsize=10, labelloc=t];")
    lines.append("  node [shape=plaintext, fontname=\"Helvetica\"];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")
    lines.append(f"  label=\"{_truncate(title or 'Hybrid runtime overview', max_chars=140)}\";")

    lines.append("  input [label=< <B>INPUT</B> >];")
    lines.append("  output [label=< <B>OUTPUT</B> >];")

    # Walk stages; build nodes and sequential edges.
    neural_idx = 0
    prev = "input"
    last_compute: ComputeOp | None = None

    for stage_idx, stage in enumerate(hybrid.stages):
        sid = f"s{stage_idx}"

        if stage.kind == "neural":
            mapping = stage.hard_core_mapping
            if mapping is None:
                stats = {}
            else:
                stats = _mapping_stats(mapping)

            # Find thumbnails (optional)
            heat = None
            if segment_heatmap_pngs is not None and neural_idx < len(segment_heatmap_pngs):
                heat = segment_heatmap_pngs[neural_idx]
            graph_png = None
            if segment_graph_pngs is not None and neural_idx < len(segment_graph_pngs):
                graph_png = segment_graph_pngs[neural_idx]

            # Edge size label depends on previous stage
            edge_label = ""
            if last_compute is not None:
                n_out = _shape_prod(last_compute.output_shape)
                if n_out is not None and last_compute.output_shape is not None:
                    edge_label = f"{n_out} ({last_compute.output_shape})"
                elif n_out is not None:
                    edge_label = f"{n_out}"
                last_compute = None

            # Node label: stats + images
            title_txt = f"Neural Segment {neural_idx}: {stage.name}"

            rows = []
            for k in [
                "hard_cores",
                "segment_in",
                "segment_out",
                "axons_used",
                "neurons_used",
                "weights_nnz",
                "unusable_space",
            ]:
                if k in stats:
                    rows.append((k, stats[k]))

            # Build HTML label with an image row
            img_cells = []
            if graph_png is not None and os.path.exists(graph_png):
                img_cells.append(_img(graph_png, w=260, h=160))
            if heat is not None and os.path.exists(heat):
                img_cells.append(_img(heat, w=260, h=160))

            img_row_html = "-"
            if img_cells:
                img_row_html = (
                    "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"2\">"
                    "<TR>"
                    + "".join(f"<TD>{c}</TD>" for c in img_cells)
                    + "</TR></TABLE>"
                )

            rows_mixed: list[tuple[str, str, bool]] = [(k, v, False) for (k, v) in rows]
            rows_mixed.append(("thumbnails", img_row_html, True))

            label = _dot_html_label_mixed(rows_mixed, title=title_txt, color="#D6EAF8")

            # Click-through to the detailed segment SVG if it exists (same directory as out_dot)
            seg_svg_guess = f"hybrid_hardcore_mapping_segment{neural_idx}.svg"
            attrs = [f"label={label}"]
            if os.path.exists(os.path.join(os.path.dirname(out_dot) or ".", seg_svg_guess)):
                attrs.append(f'URL="{seg_svg_guess}"')
                attrs.append(f'tooltip="Open detailed connectivity for segment {neural_idx}"')
            lines.append(f"  {sid} [{', '.join(attrs)}];")

            # Edge from prev to this stage
            if edge_label:
                lines.append(f"  {prev} -> {sid} [label=\"{_truncate(edge_label, max_chars=120)}\"];")
            else:
                lines.append(f"  {prev} -> {sid};")

            prev = sid
            neural_idx += 1
            continue

        if stage.kind == "compute":
            op = stage.compute_op
            assert op is not None
            last_compute = op

            n_in = _shape_prod(op.input_shape)
            n_out = _shape_prod(op.output_shape)

            rows = [
                ("kind", "ComputeOp (sync barrier)"),
                ("op_type", str(op.op_type)),
                ("in", f"{n_in} {op.input_shape}" if n_in is not None else str(op.input_shape)),
                ("out", f"{n_out} {op.output_shape}" if n_out is not None else str(op.output_shape)),
                ("params", _truncate(str(op.params), max_chars=220)),
                ("spiking", "spike-counts -> rates -> op -> respike"),
            ]
            label = _dot_html_label(rows, title=f"SYNC: {op.name}", color="#FAD7A0")
            lines.append(f"  {sid} [label={label}];")

            # Edge from prev to compute: label with compute input size
            edge_label = ""
            if n_in is not None and op.input_shape is not None:
                edge_label = f"{n_in} ({op.input_shape})"
            elif n_in is not None:
                edge_label = f"{n_in}"
            if edge_label:
                lines.append(f"  {prev} -> {sid} [label=\"{_truncate(edge_label, max_chars=120)}\"];")
            else:
                lines.append(f"  {prev} -> {sid};")

            prev = sid
            continue

        raise ValueError(f"Unknown hybrid stage kind: {stage.kind}")

    # Final edge to output (label with last segment output size if possible)
    final_label = ""
    # Find last neural mapping
    last_neural = None
    for st in reversed(hybrid.stages):
        if st.kind == "neural" and st.hard_core_mapping is not None:
            last_neural = st.hard_core_mapping
            break
    if last_neural is not None:
        final_label = str(len(last_neural.output_sources.flatten()))
    if final_label:
        lines.append(f"  {prev} -> output [label=\"{final_label}\"];")
    else:
        lines.append(f"  {prev} -> output;")

    lines.append("}")

    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


