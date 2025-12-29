from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.pipelining.pipeline_steps.perceptron_fusion_step import FusedLinear
from mimarsinan.mapping.mapping_utils import SoftCoreMapping
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import ir_graph_to_soft_core_mapping
from mimarsinan.mapping.chip_latency import ChipLatency
from mimarsinan.models.layers import SavedTensorDecorator
from mimarsinan.models.layers import TransformedActivation

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow


import torch.nn as nn
import torch

import numpy as np
import os

class SoftCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        requires = ["model"]
        promises = ["soft_core_mapping", "ir_graph"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        model = self.get_entry('model')

        for perceptron in model.get_perceptrons():
            if isinstance(perceptron.layer, FusedLinear):
                perceptron.layer = self.bring_back_bias(perceptron.layer)

        # Always emit a mapper/hardware flowchart for debugging, even if mapping will fail
        # later due to unsupported non-spiking ops (e.g., pooling).
        try:
            from mimarsinan.visualization.softcore_flowchart import write_softcore_flowchart_dot

            # Use the model's actual parameter device for the dummy forward trace.
            # Pipeline config may be CUDA even if the cached model is currently on CPU.
            try:
                flowchart_device = next(model.parameters()).device
            except StopIteration:
                flowchart_device = self.pipeline.config["device"]

            out_dot = os.path.join(self.pipeline.working_directory, "softcore_flowchart.dot")
            write_softcore_flowchart_dot(
                model.get_mapper_repr(),
                out_dot,
                input_shape=tuple(self.pipeline.config["input_shape"]),
                max_axons=int(self.pipeline.config["max_axons"]),
                max_neurons=int(self.pipeline.config["max_neurons"]),
                allow_axon_tiling=bool(self.pipeline.config.get("allow_axon_tiling", False)),
                device=flowchart_device,
            )
            print(f"[SoftCoreMappingStep] Wrote flowchart DOT: {out_dot}")
        except Exception as e:
            print(f"[SoftCoreMappingStep] Flowchart generation failed (non-fatal): {e}")
        
        validator = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)

        self._calculate_input_activation_scales(model, validator, 1.0)

        bits = self.pipeline.config['weight_bits']
        q_max = (2 ** (bits - 1)) - 1
        
        # Use the new IRMapping which supports both neural cores and compute ops
        ir_mapping = IRMapping(
            q_max=q_max,
            firing_mode=self.pipeline.config["firing_mode"],
            max_axons=self.pipeline.config.get("max_axons"),
            max_neurons=self.pipeline.config.get("max_neurons"),
            allow_axon_tiling=self.pipeline.config.get("allow_axon_tiling", False),
        )
        
        ir_graph = ir_mapping.map(model.get_mapper_repr())
        self.add_entry("ir_graph", ir_graph, 'pickle')

        # Write a detailed IRGraph visualization (NeuralCore + ComputeOp),
        # including compressed connectivity ranges and op metadata.
        try:
            from mimarsinan.visualization.mapping_graphviz import (
                try_render_dot,
                write_ir_graph_dot,
                write_ir_graph_summary_dot,
            )

            out_dot = os.path.join(self.pipeline.working_directory, "ir_graph.dot")
            write_ir_graph_dot(
                ir_graph,
                out_dot,
                title=f"IRGraph: {getattr(model, 'name', type(model).__name__)}",
            )
            rendered = try_render_dot(out_dot, formats=("svg", "png"))
            if rendered:
                print(f"[SoftCoreMappingStep] Wrote IRGraph visualization: {out_dot} (+ {', '.join(rendered)})")
            else:
                print(f"[SoftCoreMappingStep] Wrote IRGraph visualization: {out_dot} (render skipped: graphviz 'dot' not found)")

            # Also write a summarized IR graph visualization (layer stacks).
            out_sum = os.path.join(self.pipeline.working_directory, "ir_graph_summary.dot")
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
        except Exception as e:
            print(f"[SoftCoreMappingStep] IRGraph visualization failed (non-fatal): {e}")
        
        # Check if graph has ComputeOps (non-neural operations)
        compute_ops = ir_graph.get_compute_ops()
        neural_cores = ir_graph.get_neural_cores()
        
        print(f"[SoftCoreMappingStep] IR Graph: {len(neural_cores)} neural cores, {len(compute_ops)} compute ops")
        
        if compute_ops:
            # Graph contains non-neural ops - use unified simulation path
            print(f"[SoftCoreMappingStep] Model contains {len(compute_ops)} non-neural operations:")
            for op in compute_ops:
                print(f"  - {op.name}: {op.op_type}")
            print("[SoftCoreMappingStep] Will use UnifiedCoreFlow for simulation.")

            # Also report a *soft-core* spiking simulation result at this stage (pre-tuning),
            # so it's easy to compare before/after CoreFlow Tuning.
            try:
                preprocessor = nn.Sequential(model.get_preprocessor(), model.in_act)
                flow = SpikingUnifiedCoreFlow(
                    self.pipeline.config["input_shape"],
                    ir_graph,
                    int(self.pipeline.config["simulation_steps"]),
                    preprocessor,
                    self.pipeline.config["firing_mode"],
                    self.pipeline.config["spike_generation_mode"],
                    self.pipeline.config["thresholding_mode"],
                )
                acc = BasicTrainer(
                    flow,
                    self.pipeline.config["device"],
                    DataLoaderFactory(self.pipeline.data_provider_factory),
                    None,
                ).test()
                print(f"[SoftCoreMappingStep] Soft-core (Unified IR) Spiking Simulation Test: {acc}")
            except Exception as e:
                print(f"[SoftCoreMappingStep] Soft-core (Unified IR) simulation failed (non-fatal): {e}")
            
            # Store a marker that this is a unified IR graph
            # For HardCoreMapping compatibility, we can only map the neural cores
            # The compute ops will be handled separately during simulation
            soft_core_mapping = None  # Cannot produce SoftCoreMapping with ComputeOps
        else:
            # Neural-only graph - convert to SoftCoreMapping for backward compatibility
            soft_core_mapping = ir_graph_to_soft_core_mapping(ir_graph)
            ChipLatency(soft_core_mapping).calculate()

            for core in soft_core_mapping.cores:
                scale = core.parameter_scale.cpu().numpy()
                assert np.allclose(
                    core.core_matrix * scale, np.round(core.core_matrix * scale),
                    atol=1e-3, rtol=1e-3), f"{core.core_matrix * scale}"

            # Write a detailed SoftCoreMapping visualization (actual mapped cores + connectivity).
            try:
                from mimarsinan.visualization.mapping_graphviz import (
                    try_render_dot,
                    write_softcore_mapping_dot,
                )

                out_dot = os.path.join(self.pipeline.working_directory, "softcore_mapping.dot")
                write_softcore_mapping_dot(
                    soft_core_mapping,
                    out_dot,
                    title=f"SoftCoreMapping: {getattr(model, 'name', type(model).__name__)}",
                    cluster_by_psum_group=True,
                )
                rendered = try_render_dot(out_dot, formats=("svg", "png"))
                if rendered:
                    print(f"[SoftCoreMappingStep] Wrote SoftCoreMapping visualization: {out_dot} (+ {', '.join(rendered)})")
                else:
                    print(f"[SoftCoreMappingStep] Wrote SoftCoreMapping visualization: {out_dot} (render skipped: graphviz 'dot' not found)")
            except Exception as e:
                print(f"[SoftCoreMappingStep] SoftCoreMapping visualization failed (non-fatal): {e}")

        self.add_entry("soft_core_mapping", soft_core_mapping, 'pickle')
    
    def _calculate_input_activation_scales(self, model, validator, rate):
        for perceptron in model.get_perceptrons():
            if not isinstance(perceptron.input_activation, TransformedActivation):
                perceptron.input_activation = TransformedActivation(perceptron.input_activation, [])
                
            perceptron.input_activation.decorate(SavedTensorDecorator())

        validator.validate()

        max_target_scale = 0.0
        for perceptron in model.get_perceptrons():
            saved_tensor_dec = perceptron.input_activation.pop_decorator()
            if saved_tensor_dec.latest_input is None:
                raise RuntimeError(
                    "Failed to capture input activations for input scaling. "
                    f"Perceptron '{getattr(perceptron, 'name', '<unnamed>')}' did not record any inputs. "
                    "This typically happens when a Mapper's forward bypasses `perceptron.input_activation`."
                )
            in_min = saved_tensor_dec.latest_input.min()
            in_max = saved_tensor_dec.latest_input.max()
            x = saved_tensor_dec.latest_input
            
            bins = 1000
            activation_hist = torch.histc(x.flatten(), bins=bins, min=in_min.item(), max=in_max.item())
            bin_edges = torch.linspace(in_min.item(), in_max.item(), steps=bins+1).to(self.pipeline.config['device'])

            activation_hist *= bin_edges[1:].to(self.pipeline.config['device'])
            activation_hist[activation_hist < 0] = 0
            hist_sum = activation_hist.sum()
            cumulative_hist = activation_hist.cumsum(0)
            cumulative_hist /= hist_sum

            clip_rate = 0.999
            
            # # find the index of the bin which first exceeds the rate
            index = (cumulative_hist > clip_rate).flatten().nonzero()[0].to(self.pipeline.config['device'])
            clipped_act_scale = bin_edges[index].item()

            target_act_scale = (in_max * (1.0 - rate) + rate * clipped_act_scale) 

            perceptron.set_input_activation_scale(target_act_scale)
            max_target_scale = max(max_target_scale, target_act_scale)

    def bring_back_bias(self, fused_linear_layer):
        assert isinstance(fused_linear_layer, FusedLinear), 'Input layer must be an instance of LinearWithoutBias'
        
        # Get the weights from the existing layer
        weights = fused_linear_layer.linear.weight.data
        
        # Split the weights back into the main weights and the bias
        main_weights, bias = weights[:, :-1], weights[:, -1]

        # Create a new layer with the main weights and bias
        out_features, in_features = main_weights.shape
        new_layer = nn.Linear(in_features, out_features)
        new_layer.weight.data = main_weights
        new_layer.bias.data = bias

        return new_layer