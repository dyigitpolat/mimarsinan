from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.visualization.hardcore_visualization import HardCoreMappingVisualizer
from mimarsinan.visualization.mapping_graphviz import (
    try_render_dot,
    write_hybrid_hardcore_mapping_dots,
    write_hybrid_hardcore_mapping_combined_dot,
)

from mimarsinan.mapping.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow

import torch.nn as nn
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
        return self.pipeline.get_target_metric()

    def process(self):
        model = self.get_entry("model")
        ir_graph = self.get_entry('ir_graph')
        sim_len = int(self.get_entry("scaled_simulation_length"))
        platform_constraints = self.get_entry("platform_constraints_resolved")

        hybrid_mapping = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph,
            cores_config=platform_constraints["cores"],
        )
        print(
            f"[HardCoreMappingStep] Hybrid program: {len(hybrid_mapping.get_neural_segments())} neural segments, "
            f"{len(hybrid_mapping.get_compute_ops())} compute ops"
        )

        self.add_entry("hard_core_mapping", hybrid_mapping, "pickle")
        
        # Run a spiking simulation test to verify the hard-core mapping
        try:
            preprocessor = nn.Sequential(model.get_preprocessor(), model.in_act)
            flow = SpikingHybridCoreFlow(
                self.pipeline.config["input_shape"],
                hybrid_mapping,
                sim_len,
                preprocessor,
                self.pipeline.config["firing_mode"],
                self.pipeline.config["spike_generation_mode"],
                self.pipeline.config["thresholding_mode"],
                spiking_mode=self.pipeline.config.get("spiking_mode", "rate"),
            )
            acc = BasicTrainer(
                flow,
                self.pipeline.config["device"],
                DataLoaderFactory(self.pipeline.data_provider_factory),
                None,
            ).test()
            print(f"[HardCoreMappingStep] Hard-core Spiking Simulation Test: {acc}")
        except Exception as e:
            print(f"[HardCoreMappingStep] Hard-core simulation test failed (non-fatal): {e}")

        # Visualize the hybrid program (stage-level) + each neural segment's HardCoreMapping.
        try:
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