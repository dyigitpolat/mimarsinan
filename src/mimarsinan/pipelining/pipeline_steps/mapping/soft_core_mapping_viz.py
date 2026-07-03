"""Optional IR graph visualization for :class:`SoftCoreMappingStep`."""

from __future__ import annotations

import os

from mimarsinan.common.best_effort import best_effort


def write_ir_graph_visualizations(step, model, ir_graph) -> None:
    if not step.pipeline.config.get("generate_visualizations", False):
        return
    with best_effort("IRGraph visualization"):
        from mimarsinan.visualization.graphviz import (
            try_render_dot,
            write_ir_graph_dot,
            write_ir_graph_summary_dot,
        )

        node_count = len(ir_graph.nodes)
        large_graph = node_count > 500

        out_dot = os.path.join(step.pipeline.working_directory, "ir_graph.dot")
        write_ir_graph_dot(
            ir_graph,
            out_dot,
            title=f"IRGraph: {getattr(model, 'name', type(model).__name__)}",
        )
        if large_graph:
            print(f"[SoftCoreMappingStep] Wrote IRGraph DOT: {out_dot} (render skipped: {node_count} nodes)")
        else:
            rendered = try_render_dot(out_dot, formats=("svg", "png"))
            if rendered:
                print(f"[SoftCoreMappingStep] Wrote IRGraph visualization: {out_dot} (+ {', '.join(rendered)})")
            else:
                print(f"[SoftCoreMappingStep] Wrote IRGraph visualization: {out_dot} (render skipped: graphviz 'dot' not found)")

        out_sum = os.path.join(step.pipeline.working_directory, "ir_graph_summary.dot")
        write_ir_graph_summary_dot(
            ir_graph,
            out_sum,
            title=f"IRGraph: {getattr(model, 'name', type(model).__name__)}",
        )
        rendered_sum = try_render_dot(out_sum, formats=("svg", "png"))
        if rendered_sum:
            print(f"[SoftCoreMappingStep] Wrote IRGraph summary: {out_sum} (+ {', '.join(rendered_sum)})")
        else:
            print(f"[SoftCoreMappingStep] Wrote IRGraph summary: {out_sum} (render skipped: graphviz 'dot' not found)")
