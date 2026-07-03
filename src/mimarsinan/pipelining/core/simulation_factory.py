"""Shared construction of hybrid mappings and HCM spiking flows for pipeline steps."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan


def build_neural_behavior_config(pipeline) -> NeuralBehaviorConfig:
    return NeuralBehaviorConfig.from_deployment_config(pipeline.config)


def build_deployment_contract(pipeline) -> SpikingDeploymentContract:
    return SpikingDeploymentContract.from_pipeline_config(pipeline.config)


def build_hybrid_mapping_for_pipeline(
    ir_graph: IRGraph,
    platform_constraints: dict[str, Any],
    *,
    pipeline_config: dict[str, Any] | None = None,
) -> Any:
    strategy = MappingStrategy.resolve(
        ChipCapabilities.from_platform_constraints(platform_constraints)
    )
    hybrid_mapping = build_hybrid_hard_core_mapping(
        ir_graph=ir_graph,
        cores_config=platform_constraints["cores"],
        strategy=strategy,
    )
    # Install negative-value shifts carried on the IR ComputeOps (no-op when none).
    from mimarsinan.mapping.support.neg_shift_bias import propagate_negative_shifts_to_hybrid

    propagate_negative_shifts_to_hybrid(ir_graph, hybrid_mapping)
    # Provenance stamp: lets cached copies be detected as stale when the
    # ir_graph is regenerated (e.g. a resumed run).
    hybrid_mapping.source_ir_build_token = getattr(ir_graph, "build_token", None)
    return hybrid_mapping


def build_spiking_hybrid_flow(
    pipeline,
    hybrid_mapping,
    *,
    preprocessor=None,
    model=None,
) -> SpikingHybridCoreFlow:
    cfg = pipeline.config
    plan = DeploymentPlan.of(pipeline)
    contract = build_deployment_contract(pipeline)
    if preprocessor is None and model is not None:
        preprocessor = getattr(model, "preprocessor", None)
    flow = SpikingHybridCoreFlow(
        cfg["input_shape"],
        hybrid_mapping,
        contract.simulation_steps,
        preprocessor,
        contract.firing_mode,
        contract.spike_generation_mode,
        contract.thresholding_mode,
        spiking_mode=contract.spiking_mode,
        cycle_accurate_lif_forward=plan.cycle_accurate_lif_forward,
        ttfs_cycle_schedule=contract.ttfs_cycle_schedule,
    )
    if plan.cycle_accurate_lif_forward and model is not None:
        from mimarsinan.spiking.lif_utils import apply_cycle_accurate_trains_to_model

        apply_cycle_accurate_trains_to_model(model, True)
    return flow.to(cfg["device"])


def build_identity_mapping_for_pipeline(
    ir_graph: IRGraph,
    *,
    pipeline_config: dict[str, Any] | None = None,
) -> Any:
    """1:1 NeuralCore→HardCore mapping with the same wire effects as the packed
    build (negative-value shifts propagated)."""
    identity_mapping = build_identity_hybrid_mapping(ir_graph=ir_graph)
    from mimarsinan.mapping.support.neg_shift_bias import propagate_negative_shifts_to_hybrid

    propagate_negative_shifts_to_hybrid(ir_graph, identity_mapping)
    return identity_mapping


def run_trainer_metric(
    pipeline,
    model,
    *,
    device: str | None = None,
    max_batch_cap: int | None = None,
) -> float:
    """Run the same subsample/full test policy as HCM spiking simulation."""
    device = device or pipeline.config["device"]
    trainer = BasicTrainer(
        model,
        device,
        DataLoaderFactory.for_pipeline(pipeline),
        None,
    )
    try:
        if max_batch_cap is not None:
            trainer.set_test_batch_size(
                min(int(trainer.test_batch_size), int(max_batch_cap))
            )
        # The deployment accuracy metric must be evaluated on the SAME test set as
        # its torch reference (the NF step's full-set ``trainer.test()``); otherwise
        # comparing a full-set NF number to a small ``max_simulation_samples``
        # subsample manufactures a spurious "NF↔SCM drop" that is pure sampling
        # variance (the sim is bit-exact to the torch cascade per-sample — see the
        # torch↔deployed-sim parity check). Default to the full test set so the
        # reported deployed accuracy is honest and comparable; ``max_simulation_samples``
        # only subsamples when ``deployment_metric_full_eval`` is explicitly off.
        plan = DeploymentPlan.of(pipeline)
        max_samples = 0 if plan.deployment_metric_full_eval else plan.max_simulation_samples
        if max_samples > 0:
            return float(
                trainer.test_on_subsample(
                    max_samples=max_samples,
                    seed=plan.seed,
                )
            )
        return float(trainer.test(max_batches=plan.simulation_batch_count))
    finally:
        trainer.close()


def run_hcm_spiking_test(
    pipeline,
    flow: SpikingHybridCoreFlow,
    *,
    device: str | None = None,
    max_batch_cap: int | None = None,
    retry_on_oom: bool = False,
) -> float:
    """Run soft-core / HCM metric test with subsample or batch limit from config."""
    device = device or pipeline.config["device"]
    attempt_cap = max_batch_cap
    last_error: Exception | None = None

    for attempt in range(2 if retry_on_oom else 1):
        try:
            return run_trainer_metric(
                pipeline,
                flow,
                device=device,
                max_batch_cap=attempt_cap,
            )
        except RuntimeError as exc:
            last_error = exc
            oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
            is_oom = isinstance(exc, oom_cls) or "out of memory" in str(exc).lower()
            if not retry_on_oom or not is_oom or attempt >= 1:
                raise
            attempt_cap = DeploymentPlan.of(pipeline).simulation_batch_size

    if last_error is not None:
        raise last_error
    raise RuntimeError("run_hcm_spiking_test failed without raising")


def preprocess_hybrid_sample(
    pipeline,
    hybrid_mapping,
    sample: torch.Tensor,
    *,
    device: str | None = None,
) -> np.ndarray:
    """Apply the hybrid flow preprocessor; return float64 ``(1, D)`` numpy input."""
    device = device or pipeline.config["device"]
    flow = build_spiking_hybrid_flow(pipeline, hybrid_mapping).to(device).eval()
    x = sample.to(device)
    with torch.no_grad():
        x = flow.preprocessor(x).view(1, -1).to(torch.float64)
    return x.detach().cpu().numpy()


def record_ttfs_hcm_reference(
    pipeline,
    hybrid_mapping,
    sample: torch.Tensor,
    *,
    sample_index: int = 0,
):
    """Build HCM TTFS reference ``TtfsRunRecord`` for one sample (B=1)."""
    from mimarsinan.chip_simulation.ttfs.ttfs_executor import run_ttfs_hybrid_contract

    if sample.shape[0] != 1:
        raise ValueError("record_ttfs_hcm_reference requires batch_size == 1")

    contract = build_deployment_contract(pipeline)
    device = pipeline.config["device"]

    flow = build_spiking_hybrid_flow(pipeline, hybrid_mapping).to(device).eval()
    x_np = preprocess_hybrid_sample(
        pipeline, hybrid_mapping, sample, device=device,
    )
    run = run_ttfs_hybrid_contract(
        hybrid_mapping,
        x_np,
        sample_index=sample_index,
        contract=contract,
    )
    return flow, run.record


def record_hcm_reference(
    pipeline,
    hybrid_mapping,
    sample: torch.Tensor,
    *,
    sample_index: int = 0,
    device: str | None = None,
):
    """Build HCM flow and return ``(flow, record)`` from ``forward_with_recording``."""
    device = device or pipeline.config["device"]
    flow = build_spiking_hybrid_flow(pipeline, hybrid_mapping).to(device).eval()
    with torch.no_grad():
        _out, ref = flow.forward_with_recording(
            sample.to(device), sample_index=sample_index
        )
    return flow, ref


def run_hcm_mapping_metric(
    pipeline,
    ir_graph: IRGraph,
    platform_constraints: dict[str, Any],
    *,
    hybrid_mapping: Any | None = None,
    model=None,
    cache_key: str = "hybrid_mapping",
    device: str | None = None,
    max_batch_cap: int | None = None,
    retry_on_oom: bool = False,
    outer_oom_retry: bool = False,
) -> float:
    """Build hybrid mapping, run HCM spiking test, optionally cache mapping.

    Returns test accuracy. Raises on simulation failure (including after OOM retry).
    """
    device = device or pipeline.config["device"]
    attempt_cap = max_batch_cap
    last_exc: Exception | None = None

    for attempt in range(2 if outer_oom_retry else 1):
        try:
            if hybrid_mapping is None:
                hybrid_mapping = build_hybrid_mapping_for_pipeline(
                    ir_graph,
                    platform_constraints,
                    pipeline_config=pipeline.config,
                )
                pipeline.cache.add(cache_key, hybrid_mapping, "pickle")
            flow = build_spiking_hybrid_flow(
                pipeline, hybrid_mapping, model=model,
            )
            return float(
                run_hcm_spiking_test(
                    pipeline,
                    flow,
                    device=device,
                    max_batch_cap=attempt_cap,
                    retry_on_oom=retry_on_oom,
                )
            )
        except RuntimeError as exc:
            last_exc = exc
            oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
            is_oom = isinstance(exc, oom_cls) or "out of memory" in str(exc).lower()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not outer_oom_retry or not is_oom or attempt >= 1:
                raise
            attempt_cap = DeploymentPlan.of(pipeline).simulation_batch_size
        except Exception:
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("run_hcm_mapping_metric failed without raising")


def run_scm_identity_metric(
    pipeline,
    ir_graph: IRGraph,
    platform_constraints: dict[str, Any],
    *,
    model=None,
    device: str | None = None,
    max_batch_cap: int | None = None,
    retry_on_oom: bool = False,
    outer_oom_retry: bool = False,
) -> float:
    """Rung-2 gate: run the spiking test on the identity-mapped IR graph.

    Isolates IR semantics (weights, shifts, banks, segment partition, wire
    effects) from packing; the packed mapping is rung 3's concern
    (``run_hcm_mapping_metric``) and is NOT built or cached here.
    """
    del platform_constraints  # identity mapping ignores hardware core shapes
    device = device or pipeline.config["device"]
    attempt_cap = max_batch_cap
    last_exc: Exception | None = None

    identity_mapping = build_identity_mapping_for_pipeline(
        ir_graph, pipeline_config=pipeline.config,
    )

    for attempt in range(2 if outer_oom_retry else 1):
        try:
            flow = build_spiking_hybrid_flow(
                pipeline, identity_mapping, model=model,
            )
            return float(
                run_hcm_spiking_test(
                    pipeline,
                    flow,
                    device=device,
                    max_batch_cap=attempt_cap,
                    retry_on_oom=retry_on_oom,
                )
            )
        except RuntimeError as exc:
            last_exc = exc
            oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
            is_oom = isinstance(exc, oom_cls) or "out of memory" in str(exc).lower()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not outer_oom_retry or not is_oom or attempt >= 1:
                raise
            attempt_cap = DeploymentPlan.of(pipeline).simulation_batch_size

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("run_scm_identity_metric failed without raising")


def assert_spike_parity_or_raise(ref, actual) -> None:
    """Raise ``AssertionError`` with ``format_first_diff`` when records diverge."""
    from mimarsinan.chip_simulation.recording.spike_recorder import compare_records, format_first_diff

    diffs = compare_records(ref, actual)
    if diffs:
        raise AssertionError(format_first_diff(diffs))
