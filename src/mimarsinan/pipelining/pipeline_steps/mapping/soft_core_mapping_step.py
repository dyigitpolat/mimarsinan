from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency

from mimarsinan.pipelining.core.engine.pipeline_helpers import run_optional_viz
from mimarsinan.pipelining.core.simulation_factory import run_hcm_mapping_metric
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.common.diagnostics import phase_profiler
from mimarsinan.pipelining.pipeline_steps.mapping.fused_linear import FusedLinear
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_ir_pruning import (
    apply_ir_pruning_if_enabled,
)
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_viz import (
    write_ir_graph_visualizations,
)

import torch.nn as nn
import torch

import os


class SoftCoreMappingStep(PipelineStep):

    def __init__(self, pipeline):
        requires = ["fused_model", "platform_constraints_resolved"]
        promises = ["ir_graph"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)
        self.trainer = None
        self._soft_core_spiking_metric = None

    def validate(self):
        if self._soft_core_spiking_metric is None:
            raise RuntimeError(
                "Soft-core spiking simulation did not produce a metric; "
                "the step must run run_hcm_mapping_metric successfully."
            )
        return self._soft_core_spiking_metric

    def pipeline_metric(self):
        if self._soft_core_spiking_metric is None:
            raise RuntimeError(
                "Soft-core spiking simulation did not produce a metric; "
                "the step must run run_hcm_mapping_metric successfully."
            )
        return self._soft_core_spiking_metric

    def process(self):
        model = self.get_entry("fused_model")
        platform_constraints = self.get_entry("platform_constraints_resolved")

        cores = platform_constraints.get("cores", [])
        from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params

        mapping_params = resolve_platform_mapping_params(
            cores,
            allow_coalescing=bool(platform_constraints.get("allow_coalescing", False)),
        )
        resolved_hardware_bias = mapping_params.hardware_bias
        effective_max_axons = mapping_params.effective_max_axons
        resolved_max_neurons = mapping_params.effective_max_neurons
        resolved_allow_coalescing = mapping_params.allow_coalescing

        for perceptron in model.get_perceptrons():
            if isinstance(perceptron.layer, FusedLinear):
                perceptron.layer = self.bring_back_bias(perceptron.layer)

        if self.pipeline.config.get("generate_visualizations", False):
            def _flowchart():
                from mimarsinan.visualization.softcore_flowchart_dot import write_softcore_flowchart_dot

                try:
                    flowchart_device = next(model.parameters()).device
                except StopIteration:
                    flowchart_device = self.pipeline.config["device"]
                out_dot = os.path.join(
                    self.pipeline.working_directory, "softcore_flowchart.dot"
                )
                write_softcore_flowchart_dot(
                    model.get_mapper_repr(),
                    out_dot,
                    input_shape=tuple(self.pipeline.config["input_shape"]),
                    max_axons=int(effective_max_axons),
                    max_neurons=int(resolved_max_neurons),
                    device=flowchart_device,
                )
                print(f"[SoftCoreMappingStep] Wrote flowchart DOT: {out_dot}")

            run_optional_viz("SoftCoreMappingStep", _flowchart)
        
        _PHASE_TAG = "SoftCoreMappingStep"
        def _phase(name):
            return phase_profiler(_PHASE_TAG, name)

        with _phase("basic_trainer_ctor"):
            self.trainer = make_basic_trainer(self.pipeline, model)

        act_q = bool(self.pipeline.config.get("activation_quantization", False))

        spiking_mode = self.pipeline.config.get("spiking_mode", "lif")
        if spiking_mode == "ttfs_quantized" and not act_q:
            print(
                "[SoftCoreMappingStep] Warning: ttfs_quantized is on but activation_quantization is off; "
                "deployment accuracy may drop compared to training."
            )

        self._apply_ttfs_quantization_bias_compensation(model, act_q)
        self._apply_negative_value_shift_compensation(model)

        from mimarsinan.transformations.quantization_bounds import quantization_bounds

        bits = self.pipeline.config['weight_bits']
        _, q_max = quantization_bounds(bits)

        ir_mapping = IRMapping(
            q_max=q_max,
            firing_mode=self.pipeline.config["firing_mode"],
            max_axons=effective_max_axons,
            max_neurons=resolved_max_neurons,
            allow_coalescing=resolved_allow_coalescing,
            hardware_bias=resolved_hardware_bias,
        )
        
        mapper_repr = model.get_mapper_repr()
        if hasattr(mapper_repr, "assign_perceptron_indices"):
            mapper_repr.assign_perceptron_indices()
        with _phase("ir_mapping.map"):
            ir_graph = ir_mapping.map(mapper_repr)

        if bool(self.pipeline.config.get("negative_value_shift", False)):
            from mimarsinan.mapping.support.neg_shift_bias import (
                transfer_negative_shifts_to_ir,
            )
            transfer_negative_shifts_to_ir(model, ir_graph)

        wt_q = bool(self.pipeline.config.get("weight_quantization", False))
        from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph

        with _phase("weight_quantization"):
            quantize_ir_graph(ir_graph, bits, weight_quantization=wt_q)

        with _phase("ir_latency"):
            max_latency = IRLatency(ir_graph).calculate()
        print(f"[SoftCoreMappingStep] IR Graph max latency: {max_latency}")

        ir_graph = apply_ir_pruning_if_enabled(self, model, ir_graph, _PHASE_TAG)

        with _phase("pickle_save"):
            self.add_entry("ir_graph", ir_graph, 'pickle')

        write_ir_graph_visualizations(self, model, ir_graph)

        compute_ops = ir_graph.get_compute_ops()
        neural_cores = ir_graph.get_neural_cores()
        print(f"[SoftCoreMappingStep] IR Graph: {len(neural_cores)} neural cores, {len(compute_ops)} compute ops")
        if compute_ops:
            print(f"[SoftCoreMappingStep] Model contains {len(compute_ops)} non-neural operations:")
            for op in compute_ops:
                print(f"  - {op.name}: {op.op_type}")

        device = self.pipeline.config["device"]
        try:
            model.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with _phase("sim_hcm_metric"):
            acc = run_hcm_mapping_metric(
                self.pipeline,
                ir_graph,
                platform_constraints,
                model=model,
                device=device,
                outer_oom_retry=True,
            )
        self._soft_core_spiking_metric = float(acc)
        print(f"[SoftCoreMappingStep] Soft-core Spiking Simulation Test: {acc}")

    def _apply_ttfs_quantization_bias_compensation(self, model, act_q: bool) -> None:
        spiking = str(self.pipeline.config.get("spiking_mode", "lif"))
        if spiking == "ttfs" and act_q:
            print(
                "[SoftCoreMappingStep] WARNING: spiking_mode='ttfs' with "
                "activation_quantization=True is unsupported for SCM parity; "
                "use ttfs_quantized or disable activation_quantization.",
            )
        if spiking != "ttfs_quantized" or not act_q:
            return
        from mimarsinan.mapping.support.ttfs_bias import (
            apply_ttfs_quantization_bias_compensation,
        )

        apply_ttfs_quantization_bias_compensation(
            model, self.pipeline.config["target_tq"],
        )

    def _apply_negative_value_shift_compensation(self, model) -> None:
        """Opt-in (``negative_value_shift``): shift negative-producing ComputeOp
        boundaries into the encodable domain and pre-correct the consuming
        perceptron's bias, so NF and HCM recover negatives losslessly. Pre-mapping so
        the mapped core inherits the baked bias; the shift travels on the IR ComputeOps
        (``transfer_negative_shifts_to_ir``) to the hybrid build."""
        if not bool(self.pipeline.config.get("negative_value_shift", False)):
            return
        spiking_mode = str(self.pipeline.config.get("spiking_mode", "lif"))
        from mimarsinan.mapping.support.neg_shift_bias import (
            apply_negative_value_shifts,
            calibration_forward_for_mode,
        )

        # Fails loud (NotImplementedError) for unsupported spiking modes.
        forward_fn = calibration_forward_for_mode(spiking_mode)

        T = int(self.pipeline.config["simulation_steps"])
        device = self.pipeline.config["device"]
        batches = [x for x, _ in self.trainer.iter_validation_batches(2)]
        if not batches:
            return
        calibration_x = torch.cat(batches, dim=0).to(device)
        apply_negative_value_shifts(model, calibration_x, T, forward_fn=forward_fn)

    def bring_back_bias(self, fused_linear_layer):
        assert isinstance(fused_linear_layer, FusedLinear), 'Input layer must be an instance of LinearWithoutBias'

        weights = fused_linear_layer.linear.weight.data
        main_weights, bias = weights[:, :-1], weights[:, -1]

        out_features, in_features = main_weights.shape
        new_layer = nn.Linear(in_features, out_features)
        new_layer.weight.data = main_weights
        new_layer.bias.data = bias

        for src in (fused_linear_layer, getattr(fused_linear_layer, "linear", None)):
            if src is None:
                continue
            for buf_name, buf_val in src.named_buffers():
                if not hasattr(new_layer, buf_name):
                    new_layer.register_buffer(buf_name, buf_val.clone())

        return new_layer
