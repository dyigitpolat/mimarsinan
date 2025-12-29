from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.mapping.softcore_mapping import HardCoreMapping, HardCore
from mimarsinan.visualization.hardcore_visualization import HardCoreMappingVisualizer
from mimarsinan.visualization.mapping_graphviz import (
    try_render_dot,
    write_hardcore_mapping_dot,
    write_hybrid_hardcore_mapping_dots,
    write_hybrid_hardcore_mapping_combined_dot,
)

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.spiking_core_flow import SpikingCoreFlow
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow

from mimarsinan.mapping.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    build_hybrid_hard_core_mapping,
)

import torch.nn as nn

class HardCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        requires = ["tuned_soft_core_mapping", "model", "ir_graph", "scaled_simulation_length"]
        promises = ["hard_core_mapping"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.preprocessor = None

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        model = self.get_entry("model")
        self.preprocessor = model.get_preprocessor()
        
        soft_core_mapping = self.get_entry('tuned_soft_core_mapping')
        ir_graph = self.get_entry('ir_graph')
        sim_len = int(self.get_entry("scaled_simulation_length"))
        
        # Check if we have a unified IR with ComputeOps
        has_compute_ops = len(ir_graph.get_compute_ops()) > 0 if ir_graph else False
        
        if has_compute_ops:
            # Hybrid runtime path:
            # - pack each neural-only segment into a HardCoreMapping
            # - keep ComputeOps as explicit sync-barrier stages (rate -> op -> respike)
            print("[HardCoreMappingStep] Model contains ComputeOps - building HybridHardCoreMapping (multi-stage runtime)")

            hybrid_mapping = build_hybrid_hard_core_mapping(
                ir_graph=ir_graph,
                cores_config=self.pipeline.config["cores"],
            )
            print(
                f"[HardCoreMappingStep] Hybrid program: {len(hybrid_mapping.get_neural_segments())} neural segments, "
                f"{len(hybrid_mapping.get_compute_ops())} compute ops"
            )

            self.add_entry("hard_core_mapping", hybrid_mapping, "pickle")

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

                # New: Combined overview (program connectivity + ComputeOps + thumbnails: connectivity + heatmaps).
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

            # Quick correctness/accuracy smoke test (python simulator with correct sync semantics).
            preprocessor = nn.Sequential(model.get_preprocessor(), model.in_act)
            flow = SpikingHybridCoreFlow(
                self.pipeline.config["input_shape"],
                hybrid_mapping,
                sim_len,
                preprocessor,
                self.pipeline.config["firing_mode"],
                self.pipeline.config["spike_generation_mode"],
                self.pipeline.config["thresholding_mode"],
            )
            print(
                "Hybrid Runtime Simulation Test:",
                BasicTrainer(
                    flow,
                    self.pipeline.config["device"],
                    DataLoaderFactory(self.pipeline.data_provider_factory),
                    None,
                ).test(),
            )
            return

        # Neural-only path: use existing HardCoreMapping
        # support heterogeneous hardware cores
        available_hardware_cores = []
        for core_type in self.pipeline.config['cores']:
            for _ in range(core_type['count']):
                hard_core = HardCore(core_type['max_axons'], core_type['max_neurons'])
                available_hardware_cores.append(hard_core)
                

        hard_core_mapping = HardCoreMapping(available_hardware_cores)
        
        hard_core_mapping.map(soft_core_mapping)
        print("Hard Core Mapping done.")

        # Graphviz connectivity + utilization view (more informative than a matrix heatmap alone).
        try:
            # Keep legacy heatmap output name ("hardcore_mapping.png") intact by
            # writing the graphviz artifacts under a different basename.
            out_dot = self.pipeline.working_directory + "/hardcore_mapping_graph.dot"
            write_hardcore_mapping_dot(
                hard_core_mapping,
                out_dot,
                title=f"HardCoreMapping: {getattr(model, 'name', type(model).__name__)}",
            )
            rendered = try_render_dot(out_dot, formats=("svg", "png"))
            if rendered:
                print(f"[HardCoreMappingStep] Wrote HardCoreMapping visualization: {out_dot} (+ {', '.join(rendered)})")
            else:
                print(f"[HardCoreMappingStep] Wrote HardCoreMapping visualization: {out_dot} (render skipped: graphviz 'dot' not found)")
        except Exception as e:
            print(f"[HardCoreMappingStep] HardCoreMapping graph visualization failed (non-fatal): {e}")

        HardCoreMappingVisualizer(hard_core_mapping).visualize(
            self.pipeline.working_directory + "/hardcore_mapping.png"
        )
        print("Hard Core Mapping visualized.")

        print("Hard Core Mapping Test:", BasicTrainer(
            SpikingCoreFlow(
                self.pipeline.config["input_shape"], 
                hard_core_mapping, 
                self.pipeline.config["simulation_steps"], self.preprocessor,
                self.pipeline.config["firing_mode"],
                self.pipeline.config["spike_generation_mode"],
                self.pipeline.config["thresholding_mode"]), 
            self.pipeline.config["device"], 
            DataLoaderFactory(self.pipeline.data_provider_factory), None).test())

        self.add_entry("hard_core_mapping", hard_core_mapping, 'pickle')