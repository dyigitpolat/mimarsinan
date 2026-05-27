"""Graphviz writer."""

from __future__ import annotations

import html
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, IRSource, NeuralCore
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.common.layer_key import layer_key_from_node_name
from mimarsinan.common.safe_numeric import safe_float

import re


from mimarsinan.visualization.graphviz.common import try_render_dot, _embed_svg_images, _percent, _compress_ranges, _truncate, _dot_html_label, _dot_html_label_mixed, _stack_sample_lines

from mimarsinan.visualization.graphviz.hybrid_types import HybridVizArtifacts

def write_hybrid_hardcore_mapping_dots(
    hybrid: HybridHardCoreMapping,
    out_dir: str,
    *,
    basename: str = "hybrid_mapping",
) -> HybridVizArtifacts:
    """Write stage-level and per-segment hybrid mapping DOT files."""
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
