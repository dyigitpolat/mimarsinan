"""[mvm] value-domain deployment gates and metric reads (R-edge, C-edge, census)."""

from __future__ import annotations

import copy

import torch

from mimarsinan.certification.value_certificate import (
    VALUE_TWIN_FP64_ATOL,
    certify_twin_flow_values,
)
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.config_schema.registry import effective_value as _effective
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.simulation_factory import (
    build_hybrid_mapping_for_pipeline,
    run_hcm_spiking_test,
)
from mimarsinan.pipelining.core.spike_count_gate import _certificate_samples


def value_certificate_gate_armed(pipeline) -> bool:
    """Whether the value twin certificate runs: a values-observable plan with
    a nonzero sample budget. Shared by the gate and the SCM rung-2 derivation."""
    plan = DeploymentPlan.of(pipeline)
    observable, _reason = plan.mode_policy().certification_observable()
    if observable != "values":
        return False
    return int(_effective(pipeline.config, "value_parity_samples")) > 0


def _fp64_identity_flow(ir_graph) -> ValueHybridCoreFlow:
    identity = build_identity_hybrid_mapping(ir_graph=ir_graph)
    return ValueHybridCoreFlow(identity, device="cpu", dtype=torch.float64)


def run_model_value_parity_gate(pipeline, model, ir_graph) -> None:
    """[mvm R-edge, FATAL] model forward ≡ identity value program (fp64).

    Identical effective weights make this edge exact by construction; any
    mismatch is a mapping defect, never an honest residual.
    """
    n = int(_effective(pipeline.config, "value_parity_samples"))
    if n <= 0:
        print("[ValueParityGate] SKIP (value_parity_samples=0)")
        return
    samples = _certificate_samples(pipeline, model, n)
    if samples is None:
        print("[ValueParityGate] SKIP (no validation batch available)")
        return
    samples = samples.detach().to("cpu", torch.float64)
    reference = copy.deepcopy(model).to("cpu").double().eval()
    identity_flow = _fp64_identity_flow(ir_graph)
    with torch.no_grad():
        want = reference(samples)
        got = identity_flow(samples)
    max_abs_delta = float((got - want).abs().max().item()) if want.numel() else 0.0
    if max_abs_delta > VALUE_TWIN_FP64_ATOL:
        raise RuntimeError(
            f"[mvm R-edge] model↔identity value parity FAILED: "
            f"max|delta|={max_abs_delta:.3e} > atol={VALUE_TWIN_FP64_ATOL:.1e} "
            f"over {int(samples.shape[0])} samples — the identity program must "
            f"reproduce the model's affine math exactly."
        )
    print(
        f"[ValueParityGate] PASS model≡identity "
        f"(max|delta|={max_abs_delta:.3e}, n={int(samples.shape[0])})"
    )


def run_value_twin_certificate_gate(pipeline, model, ir_graph, hybrid_mapping):
    """[mvm C-edge, FATAL] identity ≡ packed value program per neuron-window."""
    if not value_certificate_gate_armed(pipeline):
        print("[ValueTwinCertificate] SKIP (unarmed)")
        return None
    n = int(_effective(pipeline.config, "value_parity_samples"))
    samples = _certificate_samples(pipeline, model, n)
    if samples is None:
        print("[ValueTwinCertificate] SKIP (no validation batch available)")
        return None
    samples = samples.detach().to("cpu", torch.float64)
    reference_flow = _fp64_identity_flow(ir_graph)
    backend_flow = ValueHybridCoreFlow(
        hybrid_mapping, device="cpu", dtype=torch.float64
    )
    certificate, detail = certify_twin_flow_values(
        reference_flow, backend_flow, samples
    )
    if not certificate.passed:
        raise RuntimeError(
            f"value twin certificate FAILED (identity↔packed): "
            f"{certificate.summary()} | {detail}"
        )
    print(f"[ValueTwinCertificate] PASS {certificate.summary()}")
    return certificate


def _value_metric_flow(pipeline, hybrid_mapping) -> ValueHybridCoreFlow:
    device = pipeline.config["device"]
    return ValueHybridCoreFlow(hybrid_mapping, device=device, dtype=torch.float32)


def run_value_identity_metric(pipeline, ir_graph, *, device=None) -> float:
    """Rung-2 census through the identity value program (unarmed configs only)."""
    identity = build_identity_hybrid_mapping(ir_graph=ir_graph)
    flow = _value_metric_flow(pipeline, identity)
    return float(run_hcm_spiking_test(pipeline, flow, device=device, retry_on_oom=True))


def run_value_mapping_metric(
    pipeline,
    ir_graph,
    platform_constraints,
    *,
    hybrid_mapping=None,
    cache_key: str = "hybrid_mapping",
    device=None,
) -> float:
    """Deployed census through the packed value program (the mvm HCM metric)."""
    if hybrid_mapping is None:
        hybrid_mapping = build_hybrid_mapping_for_pipeline(
            ir_graph, platform_constraints, pipeline_config=pipeline.config,
        )
        pipeline.cache.add(cache_key, hybrid_mapping, "pickle")
    flow = _value_metric_flow(pipeline, hybrid_mapping)
    return float(run_hcm_spiking_test(pipeline, flow, device=device, retry_on_oom=True))
