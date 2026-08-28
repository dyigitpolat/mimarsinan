from mimarsinan.common.reporter import emit_reporter_event
import warnings
from typing import Iterable, cast

import mimarsinan.pipelining.core.nf_scm_parity as nf_scm_parity
from mimarsinan.config_schema.registry import effective_value as _effective
from mimarsinan.pipelining.core.steps.pipeline_step import METRIC_CARRIED, PipelineStep
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.chip_simulation.spiking_semantics import is_lif, requires_ttfs_firing
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.depth_balancing import (
    assert_relays_alive,
    insert_depth_balancing_relays,
)
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.support.bias_compensation import (
    apply_ttfs_quantization_bias_compensation,
    transfer_negative_shifts_to_ir,
)
from mimarsinan.mapping.support.negative_boundary import (
    ensure_negative_boundary_policy,
)
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.mapping.verification.capacity import (
    PACKER_DIVERGENCE_MARGIN,
    estimate_cores_needed,
)
from mimarsinan.mapping.verification.onchip_fraction import (
    assert_onchip_validity_or_raise,
)
from mimarsinan.mapping.weight_reuse import (
    format_weight_reuse_summary,
    weight_reuse_plan_from_graph,
)
from mimarsinan.models.nn.activations.ttfs_spiking import refresh_perceptron_bias_references
from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
from mimarsinan.mapping.support.boundary_grids import install_boundary_grids
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
from mimarsinan.pipelining.core.spike_count_gate import certificate_gate_armed
from mimarsinan.pipelining.core.gates.value_gates import (
    run_model_value_parity_gate,
    run_value_identity_metric,
    value_certificate_gate_armed,
)
from mimarsinan.pipelining.core.simulation_factory import (
    build_deployment_contract,
    build_identity_mapping_for_pipeline,
    build_spiking_hybrid_flow,
    run_membrane_readout_diagnostic,
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


def print_weight_reuse_report(ir_graph) -> None:
    """The always-on reuse-phase report: a pure, total IR read (a bankless
    graph honestly reads all-reprogram — weight reuse is never configuration)."""
    print(
        "[SoftCoreMappingStep] Weight-reuse schedule: "
        + format_weight_reuse_summary(weight_reuse_plan_from_graph(ir_graph))
    )


class SoftCoreMappingStep(PipelineStep):
    REQUIRES = ("fused_model", "platform_constraints_resolved")
    PROMISES = ("ir_graph", "deployment_record_scm")

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
        if getattr(self, "_identity_metric_derived", False):
            return self.pipeline.get_target_metric()
        if self._soft_core_spiking_metric is None:
            raise RuntimeError(
                "Soft-core spiking simulation did not produce a metric; "
                "the step must run run_scm_identity_metric successfully."
            )
        return self._soft_core_spiking_metric

    def validate_metric_kind(self) -> str:
        if getattr(self, "_identity_metric_derived", False):
            return METRIC_CARRIED
        return super().validate_metric_kind()

    def pipeline_metric(self):
        if getattr(self, "_identity_metric_derived", False):
            return self.pipeline.get_target_metric()
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
        # A SIGNED boundary carries negatives verbatim; this policy exists to
        # guard the unsigned [0,1] spike-encode clamp, so the contract's own
        # signedness decides — not the domain name.
        if not packaging_contract_for(plan).boundary.signed:
            self._apply_negative_boundary_policy(model)

        bits = self.pipeline.config['weight_bits']
        _, q_max = quantization_bounds(bits)

        ir_mapping = IRMapping(
            q_max=q_max,
            firing_mode=self.pipeline.config["firing_mode"],
            max_axons=effective_max_axons,
            max_neurons=resolved_max_neurons,
            allow_coalescing=resolved_allow_coalescing,
            hardware_bias=resolved_hardware_bias,
            bias_row_splitting=plan.bias_row_splitting.active,
        )
        
        mapper_repr = model.get_mapper_repr()
        if hasattr(mapper_repr, "assign_perceptron_indices"):
            mapper_repr.assign_perceptron_indices()
        # Recompute per-source input scales here so mapping is self-contained; idempotent in activation_scales, so byte-identical when WeightQuantizationStep already populated them.
        compute_per_source_scales(
            mapper_repr,
            arm_wire_value_ops=not requires_ttfs_firing(
                str(DeploymentPlan.of(self.pipeline).spiking_mode)
            ),
        )
        # Re-propagate boundary input scales here so a retuned upstream theta cannot leave the segment-entry grid-snap normalizing by a stale scale; idempotent in activation_scales.
        propagate_boundary_input_scales(
            model, input_data_scale=plan.workload.input_data_scale
        )
        # Fail loud if any bias/weight write since the commit above broke the
        # committed-pruning contract (mask * param == param) about to be mapped.
        self._verify_pruning_committed(model)
        with _phase("ir_mapping.map"):
            ir_graph = ir_mapping.map(mapper_repr)
        # [mvm AQ] realize the model's boundary grids onto the mapped cores.
        install_boundary_grids(ir_graph, model)

        if bool(self.pipeline.config.get("negative_value_shift", True)):
            transfer_negative_shifts_to_ir(model, ir_graph)

        relays_enabled, relay_cores_inserted = self._apply_depth_balancing_relays(
            ir_graph, plan,
        )

        wt_q = plan.weight_quantization
        with _phase("weight_quantization"):
            quantize_ir_graph(ir_graph, bits, weight_quantization=wt_q)
        if relays_enabled:
            # [C5/V9] the strict-'<' exact-theta dead-relay lattice guard, on
            # the QUANTIZED graph (threshold = scale after integerization).
            assert_relays_alive(
                ir_graph,
                thresholding_mode=build_deployment_contract(
                    self.pipeline
                ).thresholding_mode,
            )

        with _phase("ir_latency"):
            max_latency = IRLatency(ir_graph).calculate()
        print(f"[SoftCoreMappingStep] IR Graph max latency: {max_latency}")

        ir_graph = apply_ir_pruning_if_enabled(self, model, ir_graph, _PHASE_TAG)

        with _phase("onchip_validity_gate"):
            self._run_onchip_validity_gate(model, ir_graph)

        with _phase("capacity_gate"):
            self._run_capacity_gate(ir_graph, platform_constraints)

        with _phase("pickle_save"):
            self.add_entry("ir_graph", ir_graph, 'pickle')

        write_ir_graph_visualizations(self, model, ir_graph)

        compute_ops = ir_graph.get_compute_ops()
        neural_cores = ir_graph.get_neural_cores()
        print(f"[SoftCoreMappingStep] IR Graph: {len(neural_cores)} neural cores, {len(compute_ops)} compute ops")
        print_weight_reuse_report(ir_graph)
        self._emit_deployment_record_scm(
            ir_graph, relay_cores_inserted, int(max_latency),
        )
        if compute_ops:
            print(f"[SoftCoreMappingStep] Model contains {len(compute_ops)} non-neural operations:")
            for op in compute_ops:
                print(f"  - {op.name}: {op.op_type}")

        if plan.mode_policy().observes_values():
            # [mvm R-edge] the value analogue of NF↔SCM: model ≡ identity
            # program, exact by construction in fp64 (FATAL).
            with _phase("value_parity_gate"):
                run_model_value_parity_gate(self.pipeline, model, ir_graph)
        else:
            with _phase("nf_scm_parity_gate"):
                self._run_nf_scm_parity_gate(model, ir_graph)
                self._run_torch_sim_parity_check(model, ir_graph)
                self._run_membrane_readout_diagnostic(model, ir_graph)

        device = self.pipeline.config["device"]
        with best_effort("move model to cpu before identity-metric run"):
            model.to("cpu")
        # empty_cache alone cannot release cycle-held parity-flow tensors
        # (nn.Module graphs are cyclic); collect first [calculus 16.13].
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if certificate_gate_armed(self.pipeline) or value_certificate_gate_armed(
            self.pipeline
        ):
            # [§17] identity ≡ packed is CERTIFIED at Hard Core Mapping
            # (counts twin or value twin), so identity accuracy equals the
            # packed read there: derived, not re-measured.
            self._identity_metric_derived = True
            print(
                "[SoftCoreMappingStep] rung-2 identity metric DERIVED via the "
                "twin certificate (identity ≡ packed); the deployed read "
                "lands at Hard Core Mapping."
            )
        elif plan.mode_policy().observes_values():
            with _phase("value_identity_metric"):
                acc = run_value_identity_metric(
                    self.pipeline, ir_graph, device=device,
                )
            self._soft_core_spiking_metric = float(acc)
            print(f"[SoftCoreMappingStep] Soft-core (identity-mapped) Value Program Test: {acc}")
        else:
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

    def _emit_deployment_record_scm(
        self, ir_graph, relay_cores_inserted: int, ir_max_latency: int
    ) -> None:
        """Persist the SCM deployment-record fragment: the weight-reuse plan
        (an unconditional pure IR read, provenance "planned@scm"), the
        formerly-dropped relay count, and the IR latency census."""
        reuse_plan = weight_reuse_plan_from_graph(ir_graph)
        self.add_entry("deployment_record_scm", {
            "reuse_plan": {
                "reprogram_passes": int(reuse_plan.reprogram_passes),
                "reuse_passes": int(reuse_plan.reuse_passes),
                "params_reloaded": int(reuse_plan.params_reloaded),
            },
            "relay_cores_inserted": int(relay_cores_inserted),
            "ir_max_latency": int(ir_max_latency),
        }, "basic")

    def _commit_pruning_to_raw_params(self, model) -> None:
        """Commit every perceptron's prune masks into its raw parameters."""
        for perceptron in model.get_perceptrons():
            commit_perceptron_pruning(perceptron)

    def _verify_pruning_committed(self, model) -> None:
        """Fail-loud committed-pruning contract check at soft-core-mapping time."""
        verify_committed_pruning(
            model.get_perceptrons(), where="SoftCoreMappingStep pre-IR-mapping",
        )

    def _apply_depth_balancing_relays(self, ir_graph, plan) -> tuple[bool, int]:
        """[C5] pre-quantize relay insertion for unequal-depth intra-segment
        fan-in (gated OFF by default; the pass is a no-op on gap-free graphs).
        Returns ``(enabled, relays_inserted)`` — enabled arms the post-quantize
        guard; the count feeds the deployment-record SCM fragment."""
        enabled = (
            is_lif(plan.spiking_mode)
            and bool(self.pipeline.config.get("lif_depth_balancing_relays", False))
        )
        if not enabled:
            return False, 0
        from mimarsinan.transformations.quantization_bounds import (
            quantization_bounds,
        )

        _, q_max = quantization_bounds(int(self.pipeline.config["weight_bits"]))
        inserted = insert_depth_balancing_relays(
            ir_graph,
            thresholding_mode=build_deployment_contract(
                self.pipeline
            ).thresholding_mode,
            q_max=q_max,
        )
        if inserted:
            print(
                f"[SoftCoreMappingStep] depth-balancing relays: {inserted} "
                f"identity relay core(s) inserted (unequal-depth fan-in, V6)."
            )
        return True, int(inserted)

    def _run_onchip_validity_gate(self, model, ir_graph) -> None:
        """Authoritative tiered validity gate on the mapped IR graph, over BOTH the
        on-chip PARAMS fraction (counted from the graph) and the on-chip OPS/MAC
        fraction (the model's forward-MAC share). Raises only when EITHER metric is
        below the floor (an erroneous host-majority deployment); a mapping between
        floor and the 50% majority is VALID_FLAGGED and still deploys. The floor is
        ``onchip_min_fraction`` and the majority ``onchip_majority_fraction``;
        ``onchip_majority_gate=false`` (or thresholds of 0) is the intentional-offload
        escape."""
        config = self.pipeline.config
        if not bool(_effective(config, "onchip_majority_gate")):
            return
        report = assert_onchip_validity_or_raise(
            ir_graph,
            model,
            config["input_shape"],
            int(config["num_classes"]),
            encoding_placement=str(config.get("encoding_layer_placement", "subsume")),
            floor=float(_effective(config, "onchip_min_fraction")),
            majority=float(_effective(config, "onchip_majority_fraction")),
        )
        pb = report.param_breakdown
        flagged = " [FLAGGED: below the on-chip majority]" if report.is_flagged else ""
        print(
            f"[SoftCoreMappingStep] on-chip validity {report.tier}{flagged}: "
            f"params {report.param_frac:.2%} "
            f"(on-chip={pb.onchip_params}, host={pb.host_params}, total={pb.total_params}), "
            f"ops {report.mac_frac:.2%} "
            f"(on-chip={report.mac_estimate.onchip}, host={report.mac_estimate.host}, "
            f"total={report.mac_estimate.total})"
        )

    def _run_capacity_gate(self, ir_graph, platform_constraints):
        """Static placement-capacity gate: raise ``CapacityExceededError`` early when the sound core-count lower bound exceeds the budget (peak-phase-aware when scheduling is allowed)."""
        if not bool(_effective(self.pipeline.config, "capacity_gate")):
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
        if estimate.within_packer_divergence_band():
            warnings.warn(
                f"[SoftCoreMappingStep] static capacity bound "
                f"{estimate.cores_needed} is within the packer divergence band "
                f"of the {estimate.cores_available}-core budget (the estimator "
                f"has measured >= {PACKER_DIVERGENCE_MARGIN:.0%} optimism on "
                "conv vehicles) — expect possible packing failure; enable "
                "allow_scheduling or grow the platform.",
                stacklevel=2,
            )
        return estimate

    def _run_torch_sim_parity_check(self, model, ir_graph) -> None:
        """Per-run torch↔deployed-sim parity: the NF torch forward must agree with the exact spiking sim ``run_scm_identity_metric`` deploys, so a deployment divergence cannot hide behind the metric's subsample."""
        if not bool(_effective(self.pipeline.config, "scm_torch_sim_parity_check")):
            return
        contract = build_deployment_contract(self.pipeline)
        if not nf_scm_parity.torch_sim_parity_enabled(contract):
            return
        n = int(_effective(self.pipeline.config, "scm_torch_sim_parity_samples"))
        if n <= 0:
            return
        assert self.trainer is not None, "trainer is not constructed yet"
        pairs = list(cast(
            "Iterable[tuple[torch.Tensor, torch.Tensor]]",
            self.trainer.iter_validation_batches(8),
        ))
        if not pairs:
            return
        samples = torch.cat([x for x, _ in pairs])[:n]
        ys = [y for _, y in pairs]
        labels = (
            torch.cat(ys)[:n]
            if all(isinstance(y, torch.Tensor) for y in ys) else None
        )
        identity_mapping = build_identity_mapping_for_pipeline(
            ir_graph, pipeline_config=self.pipeline.config,
        )
        flow = build_spiking_hybrid_flow(self.pipeline, identity_mapping, model=model)
        # The torch reference is corrected for the floor-convention double shift
        # (comp baked above while the trained ShiftDecorator is still installed);
        # the deployed flow keeps the real artifact and the threshold is unchanged.
        reference = nf_scm_parity.torch_parity_reference(model)
        # [calculus §17] the original model and the deployed IR are DIFFERENT
        # float programs: their argmax agreement is an analytic DRIFT report,
        # not the faithfulness gate (that is the spike-count certificate).
        # Only a catastrophic collapse (Type-B class) fails the step.
        configured = float(
            _effective(self.pipeline.config, "scm_torch_sim_parity_min_agreement")
        )
        agreement = nf_scm_parity.assert_torch_vs_deployed_sim_parity_or_raise(
            reference, flow, samples,
            min_agreement=min(0.90, configured),
            labels=labels,
        )
        status = "ok" if agreement >= configured else "DRIFT (non-fatal)"
        print(
            f"[SoftCoreMappingStep] torch-model↔deployed-IR analytic drift: "
            f"{agreement:.4f} over {int(samples.shape[0])} samples [{status}]"
        )
        emit_reporter_event(self.pipeline.reporter, "parity", {
            "kind": "scm_torch_sim",
            "agreement": float(agreement),
            "samples": int(samples.shape[0]),
        })

    def _run_membrane_readout_diagnostic(self, model, ir_graph) -> None:
        """[C2] Engagement report for the armed membrane readout, including
        whether the honesty gate lets the deployed metric read consume the
        membrane decode (all enabled chip backends must export final
        membranes; otherwise the read stays counts)."""
        if not (
            is_lif(DeploymentPlan.of(self.pipeline).spiking_mode)
            and bool(self.pipeline.config.get("lif_membrane_readout", False))
        ):
            return
        batches = self._validation_sample_batches(1)
        if not batches:
            return
        identity_mapping = build_identity_mapping_for_pipeline(
            ir_graph, pipeline_config=self.pipeline.config,
        )
        run_membrane_readout_diagnostic(
            self.pipeline, identity_mapping, batches[0], model=model,
        )

    def _run_nf_scm_parity_gate(self, model, ir_graph) -> None:
        """Rung-1↔rung-2 per-neuron lock for analytic schedules: compare NF activations against the identity-mapped contract run neuron-by-neuron and fail loud (accuracy tolerance alone is too coarse)."""
        contract = build_deployment_contract(self.pipeline)
        if not nf_scm_parity.nf_scm_parity_enabled(contract):
            return
        # ONE sample-count key for both statistics; 0 disables the gate on either
        # branch (the deployment-faithfulness flag names this key for both).
        n_samples = int(_effective(self.pipeline.config, "nf_scm_parity_samples"))
        if n_samples <= 0:
            return
        batches = self._validation_sample_batches(1)
        if not batches:
            return
        samples = batches[0][:n_samples]
        if contract.is_streamed_lif():
            nf_scm_parity.assert_streamed_nf_scm_exact_or_raise(
                self.pipeline, model, ir_graph, samples,
            )
            print(
                f"[SoftCoreMappingStep] NF↔SCM streamed EXACT: outputs "
                f"torch.equal over {int(samples.shape[0])} samples (atol=0)"
            )
            return
        if contract.is_cascaded():
            agreement = nf_scm_parity.assert_cascaded_nf_scm_agreement_or_raise(
                self.pipeline,
                model,
                ir_graph,
                samples,
                min_agreement=float(
                    _effective(self.pipeline.config, "nf_scm_parity_min_agreement")
                ),
            )
            print(
                f"[SoftCoreMappingStep] NF↔SCM cascaded decision agreement: "
                f"{agreement:.4f} over {int(samples.shape[0])} samples"
            )
            return
        fraction = nf_scm_parity.assert_nf_scm_parity_or_raise(
            self.pipeline,
            model,
            ir_graph,
            samples,
            atol=float(_effective(self.pipeline.config, "nf_scm_parity_atol")),
            max_mismatch_fraction=float(
                _effective(self.pipeline.config, "nf_scm_parity_max_mismatch_fraction")
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
            # The exact-kernel QAT endpoint already trains the deployed
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

    def _apply_negative_boundary_policy(self, model) -> None:
        """Make every negative ComputeOp→neural boundary lossless, both ways.

        ``negative_value_shift=true`` shifts the boundary into the encodable
        domain and pre-corrects the consuming perceptron's bias; ``false``
        subsumes the consumers forward onto the host until a non-negative
        activation absorbs the range. Either way the policy re-checks that no
        negative boundary is left on-chip-encoded, so the [0,1] spike-encode
        clamp can never silently drop a value.
        """
        ensure_negative_boundary_policy(
            model,
            self.trainer,
            spiking_mode=str(DeploymentPlan.of(self.pipeline).spiking_mode),
            simulation_steps=int(self.pipeline.config["simulation_steps"]),
            device=self.pipeline.config["device"],
            shift_enabled=bool(self.pipeline.config.get("negative_value_shift", True)),
            soma_law=SomaLaw.resolve(self.pipeline.config),
        )

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
