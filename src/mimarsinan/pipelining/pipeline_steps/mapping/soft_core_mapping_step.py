from typing import Iterable, cast

import mimarsinan.pipelining.core.nf_scm_parity as nf_scm_parity
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.support.bias_compensation import (
    apply_negative_value_shifts,
    apply_ttfs_quantization_bias_compensation,
    calibration_forward_for_mode,
    transfer_negative_shifts_to_ir,
)
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.verification.capacity import estimate_cores_needed
from mimarsinan.mapping.verification.onchip_majority import assert_onchip_majority_or_raise
from mimarsinan.mapping.weight_reuse import (
    format_weight_reuse_summary,
    weight_reuse_plan_from_graph,
)
from mimarsinan.models.nn.activations.ttfs_spiking import refresh_perceptron_bias_references
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.transformations.pruning.committed_masks import (
    commit_perceptron_pruning,
    verify_committed_pruning,
)
from mimarsinan.tuning.orchestration.adaptation_manager import (
    model_trained_sync_exact,
)
from mimarsinan.transformations.quantization_bounds import quantization_bounds

from mimarsinan.pipelining.core.engine.pipeline_helpers import run_optional_viz
from mimarsinan.pipelining.core.simulation_factory import (
    build_deployment_contract,
    build_identity_mapping_for_pipeline,
    build_spiking_hybrid_flow,
    run_scm_identity_metric,
)
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.common.best_effort import best_effort
from mimarsinan.common.diagnostics import phase_profiler
from mimarsinan.pipelining.pipeline_steps.mapping.fused_linear import FusedLinear
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_ir_pruning import (
    apply_ir_pruning_if_enabled,
)
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_structured_pruning import (
    apply_structured_pruning_if_enabled,
)
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_viz import (
    write_ir_graph_visualizations,
)

import torch.nn as nn
import torch

import os


class SoftCoreMappingStep(PipelineStep):
    REQUIRES = ("fused_model", "platform_constraints_resolved")
    PROMISES = ("ir_graph",)

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)
        self.trainer: BasicTrainer | None = None
        self._soft_core_spiking_metric = None

    def _validation_sample_batches(self, n_batches: int) -> list:
        """Inputs-only validation batches for calibration/parity sampling; process() must construct the trainer first."""
        assert self.trainer is not None, "trainer is not constructed yet"
        # cast: the validation cache yields (input, target) tensor pairs; _gpu_val_cache is untyped upstream.
        batches = cast(
            "Iterable[tuple[torch.Tensor, torch.Tensor]]",
            self.trainer.iter_validation_batches(n_batches),
        )
        return [x for x, _ in batches]

    def validate(self):
        if self._soft_core_spiking_metric is None:
            raise RuntimeError(
                "Soft-core spiking simulation did not produce a metric; "
                "the step must run run_scm_identity_metric successfully."
            )
        return self._soft_core_spiking_metric

    def pipeline_metric(self):
        if self._soft_core_spiking_metric is None:
            raise RuntimeError(
                "Soft-core spiking simulation did not produce a metric; "
                "the step must run run_scm_identity_metric successfully."
            )
        return self._soft_core_spiking_metric

    def process(self):
        plan = DeploymentPlan.of(self.pipeline)
        model = self.get_entry("fused_model")
        platform_constraints = self.get_entry("platform_constraints_resolved")

        cores = platform_constraints.get("cores", [])
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
                refresh_perceptron_bias_references(perceptron)

        apply_structured_pruning_if_enabled(self, model, "SoftCoreMappingStep")

        # W-CAL-2: pruning must hold in the committed raw parameters at mapping
        # time — the deployed executor never fires the enforcement hooks.
        self._commit_pruning_to_raw_params(model)

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

        act_q = plan.activation_quantization

        if plan.uses_ttfs_floor_ceil_convention and not act_q:
            print(
                f"[SoftCoreMappingStep] Warning: spiking_mode={plan.spiking_mode!r} "
                "trains the TTFS floor+half-step-bias convention but "
                "activation_quantization is off; deployment accuracy may drop "
                "compared to training."
            )

        self._apply_ttfs_quantization_bias_compensation(model, act_q)
        self._apply_negative_value_shift_compensation(model)

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
        # Recompute per-source input scales here so mapping is self-contained; idempotent in activation_scales, so byte-identical when WeightQuantizationStep already populated them.
        compute_per_source_scales(mapper_repr)
        # Re-propagate boundary input scales here so a retuned upstream theta cannot leave the segment-entry grid-snap normalizing by a stale scale; idempotent in activation_scales.
        propagate_boundary_input_scales(model, input_data_scale=1.0)
        # Fail loud if any bias/weight write since the commit above broke the
        # committed-pruning contract (mask * param == param) about to be mapped.
        self._verify_pruning_committed(model)
        with _phase("ir_mapping.map"):
            ir_graph = ir_mapping.map(mapper_repr)

        if bool(self.pipeline.config.get("negative_value_shift", False)):
            transfer_negative_shifts_to_ir(model, ir_graph)

        wt_q = plan.weight_quantization
        with _phase("weight_quantization"):
            quantize_ir_graph(ir_graph, bits, weight_quantization=wt_q)

        with _phase("ir_latency"):
            max_latency = IRLatency(ir_graph).calculate()
        print(f"[SoftCoreMappingStep] IR Graph max latency: {max_latency}")

        ir_graph = apply_ir_pruning_if_enabled(self, model, ir_graph, _PHASE_TAG)

        with _phase("onchip_majority_gate"):
            self._run_onchip_majority_gate(model, ir_graph)

        with _phase("capacity_gate"):
            self._run_capacity_gate(ir_graph, platform_constraints)

        with _phase("pickle_save"):
            self.add_entry("ir_graph", ir_graph, 'pickle')

        write_ir_graph_visualizations(self, model, ir_graph)

        compute_ops = ir_graph.get_compute_ops()
        neural_cores = ir_graph.get_neural_cores()
        print(f"[SoftCoreMappingStep] IR Graph: {len(neural_cores)} neural cores, {len(compute_ops)} compute ops")
        if bool(platform_constraints.get("allow_weight_reuse", False)):
            reuse_plan = weight_reuse_plan_from_graph(ir_graph)
            print(
                "[SoftCoreMappingStep] Weight-reuse schedule: "
                + format_weight_reuse_summary(reuse_plan)
            )
        if compute_ops:
            print(f"[SoftCoreMappingStep] Model contains {len(compute_ops)} non-neural operations:")
            for op in compute_ops:
                print(f"  - {op.name}: {op.op_type}")

        # Run the NF↔SCM gate before model.to("cpu") below, which does not move the mapper-graph compute modules, so the whole model must still be on one device.
        with _phase("nf_scm_parity_gate"):
            self._run_nf_scm_parity_gate(model, ir_graph)
            self._run_torch_sim_parity_check(model, ir_graph)

        device = self.pipeline.config["device"]
        with best_effort("move model to cpu before identity-metric run"):
            model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with _phase("sim_identity_metric"):
            acc = run_scm_identity_metric(
                self.pipeline,
                ir_graph,
                platform_constraints,
                model=model,
                device=device,
                outer_oom_retry=True,
            )
        self._soft_core_spiking_metric = float(acc)
        print(f"[SoftCoreMappingStep] Soft-core (identity-mapped) Spiking Simulation Test: {acc}")

    def _commit_pruning_to_raw_params(self, model) -> None:
        """Commit every perceptron's prune masks into its raw parameters."""
        for perceptron in model.get_perceptrons():
            commit_perceptron_pruning(perceptron)

    def _verify_pruning_committed(self, model) -> None:
        """Fail-loud committed-pruning contract check at soft-core-mapping time."""
        verify_committed_pruning(
            model.get_perceptrons(), where="SoftCoreMappingStep pre-IR-mapping",
        )

    def _run_onchip_majority_gate(self, model, ir_graph) -> None:
        """Params-based FLOOR gate: raise only BELOW the on-chip floor; VALID_FLAGGED mappings between floor and 50% majority still deploy."""
        if not bool(self.pipeline.config.get("onchip_majority_gate", True)):
            return
        total_params = int(sum(p.numel() for p in model.parameters()))
        breakdown = assert_onchip_majority_or_raise(
            ir_graph,
            total_params=total_params,
            min_fraction=float(
                self.pipeline.config.get("onchip_majority_min_fraction", 0.2)
            ),
        )
        print(
            f"[SoftCoreMappingStep] on-chip parameter majority: "
            f"{breakdown.fraction:.2%} on chip "
            f"(on-chip={breakdown.onchip_params}, host={breakdown.host_params}, "
            f"total={breakdown.total_params})"
        )

    def _run_capacity_gate(self, ir_graph, platform_constraints):
        """Static placement-capacity gate: raise ``CapacityExceededError`` early when the sound core-count lower bound exceeds the budget (peak-phase-aware when scheduling is allowed)."""
        if not bool(self.pipeline.config.get("capacity_gate", True)):
            return None
        estimate = estimate_cores_needed(ir_graph, platform_constraints)
        if estimate.scheduled:
            print(
                f"[SoftCoreMappingStep] placement capacity (SCHEDULED): "
                f"peak phase {estimate.peak_phase_cores} cores over "
                f"{estimate.phase_count} reprogram phases, budget "
                f"{estimate.cores_available} (feasible={estimate.feasible})"
            )
        else:
            print(
                f"[SoftCoreMappingStep] placement capacity: needs "
                f">= {estimate.cores_needed} hard cores, budget "
                f"{estimate.cores_available} (feasible={estimate.feasible})"
            )
        estimate.raise_if_infeasible()
        return estimate

    def _run_torch_sim_parity_check(self, model, ir_graph) -> None:
        """Per-run torch↔deployed-sim parity: the NF torch forward must agree with the exact spiking sim ``run_scm_identity_metric`` deploys, so a deployment divergence cannot hide behind the metric's subsample."""
        if not bool(self.pipeline.config.get("scm_torch_sim_parity_check", True)):
            return
        contract = build_deployment_contract(self.pipeline)
        if not (
            nf_scm_parity.nf_scm_parity_enabled(contract) or contract.is_synchronized()
        ):
            return
        n = int(self.pipeline.config.get("scm_torch_sim_parity_samples", 256))
        if n <= 0:
            return
        batches = self._validation_sample_batches(8)
        if not batches:
            return
        samples = torch.cat(batches)[:n]
        identity_mapping = build_identity_mapping_for_pipeline(
            ir_graph, pipeline_config=self.pipeline.config,
        )
        flow = build_spiking_hybrid_flow(self.pipeline, identity_mapping, model=model)
        # The torch reference is corrected for the floor-convention double shift
        # (comp baked above while the trained ShiftDecorator is still installed);
        # the deployed flow keeps the real artifact and the threshold is unchanged.
        reference = nf_scm_parity.torch_parity_reference(model)
        agreement = nf_scm_parity.assert_torch_vs_deployed_sim_parity_or_raise(
            reference, flow, samples,
            min_agreement=float(
                self.pipeline.config.get("scm_torch_sim_parity_min_agreement", 0.98)
            ),
        )
        print(
            f"[SoftCoreMappingStep] torch↔deployed-sim parity: {agreement:.4f} "
            f"over {int(samples.shape[0])} samples"
        )

    def _run_nf_scm_parity_gate(self, model, ir_graph) -> None:
        """Rung-1↔rung-2 per-neuron lock for analytic schedules: compare NF activations against the identity-mapped contract run neuron-by-neuron and fail loud (accuracy tolerance alone is too coarse)."""
        contract = build_deployment_contract(self.pipeline)
        if not nf_scm_parity.nf_scm_parity_enabled(contract):
            return
        if contract.is_cascaded():
            n_samples = int(
                self.pipeline.config.get("nf_scm_parity_samples_cascaded", 64)
            )
            if n_samples <= 0:
                return
            batches = self._validation_sample_batches(1)
            if not batches:
                return
            samples = batches[0][:n_samples]
            agreement = nf_scm_parity.assert_cascaded_nf_scm_agreement_or_raise(
                self.pipeline,
                model,
                ir_graph,
                samples,
                min_agreement=float(
                    self.pipeline.config.get("nf_scm_parity_min_agreement", 0.98)
                ),
            )
            print(
                f"[SoftCoreMappingStep] NF↔SCM cascaded decision agreement: "
                f"{agreement:.4f} over {int(samples.shape[0])} samples"
            )
            return
        n_samples = int(self.pipeline.config.get("nf_scm_parity_samples", 2))
        if n_samples <= 0:
            return
        batches = self._validation_sample_batches(1)
        if not batches:
            return
        samples = batches[0][:n_samples]
        # Loose default while continuous ttfs (the only mode reaching this per-neuron path) has an uncalibrated residual; its wrong-dynamics signature is ~40%.
        default_budget = 0.25
        fraction = nf_scm_parity.assert_nf_scm_parity_or_raise(
            self.pipeline,
            model,
            ir_graph,
            samples,
            atol=float(self.pipeline.config.get("nf_scm_parity_atol", 1e-6)),
            max_mismatch_fraction=float(
                self.pipeline.config.get(
                    "nf_scm_parity_max_mismatch_fraction", default_budget
                )
            ),
        )
        print(
            f"[SoftCoreMappingStep] NF↔SCM per-neuron parity: "
            f"{fraction:.4%} mismatch fraction over {int(samples.shape[0])} samples"
        )

    def _apply_ttfs_quantization_bias_compensation(self, model, act_q: bool) -> None:
        plan = DeploymentPlan.of(self.pipeline)
        spiking = str(plan.spiking_mode)
        if spiking == "ttfs" and act_q:
            print(
                "[SoftCoreMappingStep] WARNING: spiking_mode='ttfs' with "
                "activation_quantization=True is unsupported for SCM parity; "
                "use ttfs_quantized or disable activation_quantization.",
            )
        # Bakes the half-step shift aligning the floor-trained decode to the deployed ceil kernel; idempotent per perceptron.
        if not plan.uses_ttfs_floor_ceil_convention or not act_q:
            return
        if model_trained_sync_exact(model):
            # [MBH T6] The exact-kernel QAT endpoint already trains the deployed
            # ceil convention; the half-step compensation exists solely to
            # reconcile the floor proxy and would double-shift here.
            assert plan.is_synchronized_ttfs, (
                "sync-exact QAT marker on a non-synchronized plan: the exact-kernel "
                "endpoint is only defined for the synchronized schedule."
            )
            print(
                "[SoftCoreMappingStep] sync-exact QAT endpoint detected: "
                "skipping TTFS half-step bias compensation."
            )
            return
        apply_ttfs_quantization_bias_compensation(
            model, self.pipeline.config["target_tq"],
        )

    def _apply_negative_value_shift_compensation(self, model) -> None:
        """Opt-in (``negative_value_shift``): shift negative-producing ComputeOp boundaries into the encodable domain and pre-correct the consumer's bias, so NF and HCM recover negatives losslessly."""
        if not bool(self.pipeline.config.get("negative_value_shift", False)):
            return
        spiking_mode = str(DeploymentPlan.of(self.pipeline).spiking_mode)
        forward_fn = calibration_forward_for_mode(spiking_mode)

        T = int(self.pipeline.config["simulation_steps"])
        device = self.pipeline.config["device"]
        batches = self._validation_sample_batches(2)
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
