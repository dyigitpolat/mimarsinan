"""IR graph DOT writers."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore
from mimarsinan.common.presentation import layer_key_from_node_name
from mimarsinan.common.presentation import safe_float

import re


from mimarsinan.visualization.graphviz.common import (
    _compress_ranges,
    _dot_html_label,
    _dot_html_label_mixed,
    _stack_sample_lines,
    _truncate,
)

def write_ir_graph_summary_dot(
    ir_graph: IRGraph,
    out_dot: str,
    *,
    title: str | None = None,
    max_edges: int = 64,
) -> None:
    """Summarized IRGraph visualization with grouped nodes and capped edges."""
    os.makedirs(os.path.dirname(out_dot) or ".", exist_ok=True)

    group_order: list[tuple[str, str]] = []
    group_nodes: dict[str, list[IRNode]] = {}
    node_id_to_group: dict[int, str] = {}

    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            gk = f"neural::{layer_key_from_node_name(node.name)}"
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
            core_ids = [int(n.id) for n in nodes if hasattr(n, "id")]
            core_names = [str(getattr(n, "name", f"n{getattr(n, 'id', '?')}")) for n in nodes]

            shapes: dict[tuple[int, int], int] = {}
            roles: dict[str, int] = {}
            thresholds: list[float] = []
            latencies: list[int] = []
            for n in nodes:
                if not isinstance(n, NeuralCore):
                    continue
                _mat = n.get_core_matrix(ir_graph)
                ax = int(_mat.shape[0])
                neu = int(_mat.shape[1])
                shapes[(ax, neu)] = shapes.get((ax, neu), 0) + 1
                if n.psum_role is not None:
                    roles[str(n.psum_role)] = roles.get(str(n.psum_role), 0) + 1
                thr = safe_float(n.threshold)
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

        label = _dot_html_label([("kind", kind), ("count", str(len(nodes)))], title=gk, color="#E5E7E9")
        lines.append(f"  {dot_id} [label={label}];")

    edge_stats: dict[tuple[str, str], dict[str, Any]] = {}

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
