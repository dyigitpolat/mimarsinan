"""Graphviz writer."""

from __future__ import annotations

import html
import os
from typing import Sequence

import numpy as np

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.packing.softcore import HardCoreMapping

from mimarsinan.visualization.graphviz.common import _percent, _truncate, _dot_html_label, _dot_html_label_mixed


def write_hybrid_hardcore_mapping_combined_dot(
    hybrid: HybridHardCoreMapping,
    out_dot: str,
    *,
    segment_graph_pngs: Sequence[str] | None = None,
    segment_heatmap_pngs: Sequence[str] | None = None,
    title: str | None = None,
) -> None:
    """Combined hybrid visualization with stage flow and segment thumbnails."""

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

    lines: list[str] = []
    lines.append("digraph HybridCombined {")
    lines.append("  rankdir=LR;")
    lines.append("  graph [fontname=\"Helvetica\", fontsize=10, labelloc=t];")
    lines.append("  node [shape=plaintext, fontname=\"Helvetica\"];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9];")
    lines.append(f"  label=\"{_truncate(title or 'Hybrid runtime overview', max_chars=140)}\";")

    lines.append("  input [label=< <B>INPUT</B> >];")
    lines.append("  output [label=< <B>OUTPUT</B> >];")

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

            heat = None
            if segment_heatmap_pngs is not None and neural_idx < len(segment_heatmap_pngs):
                heat = segment_heatmap_pngs[neural_idx]
            graph_png = None
            if segment_graph_pngs is not None and neural_idx < len(segment_graph_pngs):
                graph_png = segment_graph_pngs[neural_idx]

            edge_label = ""
            if last_compute is not None:
                n_out = _shape_prod(last_compute.output_shape)
                if n_out is not None and last_compute.output_shape is not None:
                    edge_label = f"{n_out} ({last_compute.output_shape})"
                elif n_out is not None:
                    edge_label = f"{n_out}"
                last_compute = None

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

            img_cells = []
            if graph_png is not None and os.path.exists(graph_png):
                img_cells.append(_img(graph_png, w=260, h=160))
            if heat is not None and os.path.exists(heat) and mapping is not None and mapping.cores:
                max_ax = max(c.axons_per_core for c in mapping.cores)
                max_nu = max(c.neurons_per_core for c in mapping.cores)
                max_side = 260
                if max_ax >= max_nu:
                    thumb_w = max(1, round(max_side * max_nu / max_ax))
                    thumb_h = max_side
                else:
                    thumb_w = max_side
                    thumb_h = max(1, round(max_side * max_ax / max_nu))
                img_cells.append(_img(heat, w=thumb_w, h=thumb_h))
            elif heat is not None and os.path.exists(heat):
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

            seg_svg_guess = f"hybrid_hardcore_mapping_segment{neural_idx}.svg"
            attrs = [f"label={label}"]
            if os.path.exists(os.path.join(os.path.dirname(out_dot) or ".", seg_svg_guess)):
                attrs.append(f'URL="{seg_svg_guess}"')
                attrs.append(f'tooltip="Open detailed connectivity for segment {neural_idx}"')
            lines.append(f"  {sid} [{', '.join(attrs)}];")

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

    final_label = ""
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


