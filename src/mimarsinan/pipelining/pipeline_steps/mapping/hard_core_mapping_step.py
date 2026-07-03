from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep

from mimarsinan.common.best_effort import best_effort
from mimarsinan.common.env import vram_probe_enabled
from mimarsinan.pipelining.core.hybrid_mapping_consumer import load_hybrid_mapping_for_step
from mimarsinan.pipelining.core.engine.pipeline_helpers import run_optional_viz
from mimarsinan.pipelining.core.simulation_factory import run_hcm_mapping_metric

import torch
import os


def _vram_probe(tag: str) -> None:
    """Opt-in VRAM/RSS probe when ``MIMARSINAN_VRAM_PROBE=1``."""
    if not vram_probe_enabled():
        return
    rss = 0
    with best_effort("read process RSS via psutil"):
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alc = torch.cuda.memory_allocated()
        rsv = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
    else:
        alc = rsv = peak = 0
    print(
        f"[VRAM::HCM] {tag:<52} "
        f"RSS={rss/1e6:8.1f} MB  "
        f"alc={alc/1e6:8.1f} MB  "
        f"rsv={rsv/1e6:8.1f} MB  "
        f"peak={peak/1e6:8.1f} MB",
        flush=True,
    )

class HardCoreMappingStep(PipelineStep):
    REQUIRES = ("model", "ir_graph", "platform_constraints_resolved")
    PROMISES = ("hard_core_mapping",)

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def validate(self):
        """Return hard-core spiking simulation accuracy from the step metric run."""
        m = getattr(self, "_last_metric", None)
        if m is None:
            raise RuntimeError(
                "Hard-core spiking simulation did not produce a metric; "
                "the step must run run_hcm_mapping_metric successfully."
            )
        return m

    def process(self):
        self._last_metric = None
        _vram_probe("process_entry")
        model = self.get_entry("model")
        ir_graph = self.get_entry('ir_graph')
        sim_len = int(self.pipeline.config["simulation_steps"])
        platform_constraints = self.get_entry("platform_constraints_resolved")
        _vram_probe("after_load_entries")

        hybrid_mapping = load_hybrid_mapping_for_step(self.pipeline, self)

        neural_segs = hybrid_mapping.get_neural_segments()
        compute_ops = hybrid_mapping.get_compute_ops()
        scheduled_stages = [s for s in hybrid_mapping.stages if s.schedule_pass_index is not None]
        if scheduled_stages:
            seg_pass_counts: dict[int, int] = {}
            for s in scheduled_stages:
                si = s.schedule_segment_index or 0
                seg_pass_counts[si] = max(seg_pass_counts.get(si, 0), (s.schedule_pass_index or 0) + 1)
            detail = ", ".join(
                f"seg {si}: {pc} pass{'es' if pc > 1 else ''}"
                for si, pc in sorted(seg_pass_counts.items())
            )
            print(
                f"[HardCoreMappingStep] Hybrid program (scheduled): {len(seg_pass_counts)} neural segment(s) "
                f"({detail}), {len(compute_ops)} compute op(s)"
            )
        else:
            print(
                f"[HardCoreMappingStep] Hybrid program: {len(neural_segs)} neural segments, "
                f"{len(compute_ops)} compute ops"
            )

        _vram_probe("after_build_hybrid")
        self.add_entry("hard_core_mapping", hybrid_mapping, "pickle")
        _vram_probe("after_pickle_save")

        _vram_probe("before_test")
        acc = run_hcm_mapping_metric(
            self.pipeline,
            ir_graph,
            platform_constraints,
            hybrid_mapping=hybrid_mapping,
            model=model,
            cache_key="hybrid_mapping",
        )
        _vram_probe("after_test")
        self._last_metric = float(acc)
        print(f"[HardCoreMappingStep] Hard-core Spiking Simulation Test: {acc}")

        if self.pipeline.config.get("generate_visualizations", False):
            def _viz():
              from mimarsinan.visualization.hardcore_visualization import HardCoreMappingVisualizer
              from mimarsinan.visualization.graphviz import (
                  try_render_dot,
                  write_hybrid_hardcore_mapping_dots,
                  write_hybrid_hardcore_mapping_combined_dot,
              )

              artifacts = write_hybrid_hardcore_mapping_dots(
                  hybrid_mapping,
                  self.pipeline.working_directory,
                  basename="hybrid_hardcore_mapping",
              )

              heatmaps = []
              for i, seg in enumerate(hybrid_mapping.get_neural_segments()):
                  heat_path = self.pipeline.working_directory + f"/hybrid_segment{i}_hardcore_heatmap.png"
                  HardCoreMappingVisualizer(seg).visualize(heat_path)
                  heatmaps.append(heat_path)

              rendered = try_render_dot(artifacts.program_dot, formats=("svg", "png"))
              if rendered:
                  print(f"[HardCoreMappingStep] Wrote hybrid program visualization: {artifacts.program_dot} (+ {', '.join(rendered)})")
              else:
                  print(f"[HardCoreMappingStep] Wrote hybrid program visualization: {artifacts.program_dot} (render skipped: graphviz 'dot' not found)")

              segment_pngs = []
              for i, seg_dot in enumerate(artifacts.segment_dots):
                  rendered_seg = try_render_dot(seg_dot, formats=("svg", "png"))
                  if rendered_seg:
                      print(f"[HardCoreMappingStep] Wrote hybrid segment {i} visualization: {seg_dot} (+ {', '.join(rendered_seg)})")
                  else:
                      print(f"[HardCoreMappingStep] Wrote hybrid segment {i} visualization: {seg_dot} (render skipped: graphviz 'dot' not found)")
                  segment_pngs.append(os.path.splitext(seg_dot)[0] + ".png")

              combined_dot = self.pipeline.working_directory + "/hybrid_hardcore_mapping_combined.dot"
              write_hybrid_hardcore_mapping_combined_dot(
                  hybrid_mapping,
                  combined_dot,
                  segment_graph_pngs=segment_pngs,
                  segment_heatmap_pngs=heatmaps,
                  title=f"Hybrid mapping: {getattr(model, 'name', type(model).__name__)}",
              )
              rendered_combined = try_render_dot(combined_dot, formats=("svg", "png"))
              if rendered_combined:
                  print(f"[HardCoreMappingStep] Wrote hybrid combined overview: {combined_dot} (+ {', '.join(rendered_combined)})")
              else:
                  print(f"[HardCoreMappingStep] Wrote hybrid combined overview: {combined_dot} (render skipped: graphviz 'dot' not found)")

            run_optional_viz("HardCoreMappingStep", _viz)