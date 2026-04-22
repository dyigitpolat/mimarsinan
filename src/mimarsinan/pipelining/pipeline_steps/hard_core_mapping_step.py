from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.mapping.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow

import torch.nn as nn
import traceback
import os

class HardCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        # Unified-only: always compile the tuned IRGraph into a HybridHardCoreMapping.
        # For neural-only graphs this will simply be a single neural segment.
        requires = ["model", "ir_graph", "scaled_simulation_length", "platform_constraints_resolved"]
        promises = ["hard_core_mapping"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        """Return the hard-core simulation accuracy.

        When the simulation failed (OOM, CUDA error, etc.) we deliberately
        surface 0.0 rather than falling back to the pipeline's target
        metric.  Falling back to the target silently masked mapping
        failures: the step reported the pre-mapping reference accuracy as
        if the sim had succeeded.  Returning 0.0 lets the pipeline's
        cross-step accuracy budget see the real drop and react.
        """
        if getattr(self, "_last_metric_is_failure", False):
            return 0.0
        m = getattr(self, "_last_metric", None)
        if m is not None:
            return m
        # No sim ran AND no explicit failure flag — conservative behavior
        # is to treat the step as untested and defer to the target.  This
        # branch is reached only when the try/except block was skipped
        # entirely (e.g. a future refactor that bypasses the sim), which
        # we don't currently do.
        return self.pipeline.get_target_metric()

    def process(self):
        self._last_metric = None
        self._last_metric_is_failure = False
        model = self.get_entry("model")
        ir_graph = self.get_entry('ir_graph')
        sim_len = int(self.get_entry("scaled_simulation_length"))
        platform_constraints = self.get_entry("platform_constraints_resolved")

        hybrid_mapping = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph,
            cores_config=platform_constraints["cores"],
            allow_neuron_splitting=bool(platform_constraints.get("allow_neuron_splitting", False)),
            allow_scheduling=bool(platform_constraints.get("allow_scheduling", False)),
            allow_coalescing=bool(platform_constraints.get("allow_coalescing", False)),
        )

        # Report structure — distinguish scheduled passes from segments.
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

        self.add_entry("hard_core_mapping", hybrid_mapping, "pickle")
        
        # Run a spiking simulation test to verify the hard-core mapping
        try:
            device = self.pipeline.config["device"]
            flow = SpikingHybridCoreFlow(
                self.pipeline.config["input_shape"],
                hybrid_mapping,
                sim_len,
                None,
                self.pipeline.config["firing_mode"],
                self.pipeline.config["spike_generation_mode"],
                self.pipeline.config["thresholding_mode"],
                spiking_mode=self.pipeline.config.get("spiking_mode", "rate"),
            )
            flow = flow.to(device)
            # ``simulation_batch_count`` caps the per-core test pass to
            # keep the verification sim inside the run budget on large
            # models; the SoftCoreMapping pass already uses the same
            # override. ``None`` preserves the legacy full-test-set pass.
            sim_batches = self.pipeline.config.get("simulation_batch_count", None)
            trainer = BasicTrainer(
                flow,
                device,
                DataLoaderFactory(self.pipeline.data_provider_factory),
                None,
            )
            # Honour ``max_simulation_samples`` with the same seeded
            # subsampling as ``SimulationRunner`` and the SCM
            # verification — keeps the three metrics comparable.
            max_samples = int(self.pipeline.config.get("max_simulation_samples", 0) or 0)
            if max_samples > 0:
                acc = trainer.test_on_subsample(
                    max_samples=max_samples,
                    seed=int(self.pipeline.config.get("seed", 0)),
                )
            else:
                acc = trainer.test(max_batches=sim_batches)
            self._last_metric = float(acc)
            print(f"[HardCoreMappingStep] Hard-core Spiking Simulation Test: {acc}")
        except Exception as e:
            # The sim is the only check that this step produced a working
            # hardcore program.  If it fails, ``validate()`` must surface
            # the failure — previously we left ``_last_metric = None`` and
            # ``validate`` silently fell back to the pre-mapping target
            # accuracy, so a crashed sim looked like a clean pass.  Mark
            # the step explicitly as failed so ``validate`` returns 0.0
            # and the accuracy-budget gate fires.
            print(f"[HardCoreMappingStep] Hard-core simulation test FAILED: {e}")
            traceback.print_exc()
            self._last_metric = None
            self._last_metric_is_failure = True

        # Visualize the hybrid program (stage-level) + each neural segment's HardCoreMapping.
        if self.pipeline.config.get("generate_visualizations", False):
          try:
              from mimarsinan.visualization.hardcore_visualization import HardCoreMappingVisualizer
              from mimarsinan.visualization.mapping_graphviz import (
                  try_render_dot,
                  write_hybrid_hardcore_mapping_dots,
                  write_hybrid_hardcore_mapping_combined_dot,
              )

              artifacts = write_hybrid_hardcore_mapping_dots(
                  hybrid_mapping,
                  self.pipeline.working_directory,
                  basename="hybrid_hardcore_mapping",
              )

              # Also emit per-segment weight heatmaps for quick inspection.
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

              # Combined overview (program connectivity + ComputeOps + thumbnails: connectivity + heatmaps).
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
          except Exception as e:
              print(f"[HardCoreMappingStep] Hybrid mapping visualization failed (non-fatal): {e}")