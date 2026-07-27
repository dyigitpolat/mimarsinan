"""[mvm] value-domain deployment gates and metric reads (R-edge, C-edge, census)."""

from __future__ import annotations

import copy

import torch

from mimarsinan.certification.value_certificate import (
    VALUE_R_EDGE_AQ_LSB_BOUND,
    VALUE_R_EDGE_WQ_ATOL,
    VALUE_TWIN_FP64_ATOL,
    certify_twin_flow_values,
)
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.nn.activations.value_quantizer import value_grid_levels
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


def _activation_bits(pipeline) -> "int | None":
    bits = _effective(pipeline.config, "activation_bits")
    return int(bits) if bits else None


def boundary_grid_lsb(activation_bits, ir_graph) -> "float | None":
    """One step of the widest armed boundary grid, or None when AQ is off."""
    if not activation_bits:
        return None
    levels = value_grid_levels(int(activation_bits))
    if levels <= 0:
        return None
    scales = [
        float(torch.as_tensor(node.input_activation_scale).max())
        for node in ir_graph.nodes if isinstance(node, NeuralCore)
    ]
    armed = [s for s in scales if s > 0.0]
    return (max(armed) / levels) if armed else None


def _assert_aq_grid_parity(got, want, max_abs_delta, lsb, n_samples) -> None:
    """[mvm AQ R-edge] Judge in GRID units + exact decisions (both FATAL).

    An AQ program is piecewise constant, so the fp seed between the model's
    fp32 weights and the chip's integer-accumulate-then-divide crosses grid
    edges and amplifies; a scalar atol measures that chaos, not the mapping.
    """
    bound = VALUE_R_EDGE_AQ_LSB_BOUND * lsb
    decisions = (
        float((got.argmax(-1) == want.argmax(-1)).double().mean())
        if want.dim() >= 2 and want.shape[-1] > 1 else 1.0
    )
    if decisions < 1.0:
        raise RuntimeError(
            f"[mvm AQ R-edge] DECISION parity FAILED: {decisions:.4f} over "
            f"{n_samples} samples — the deployed program must reproduce the "
            f"model's decisions exactly (max|delta|={max_abs_delta:.3e}, "
            f"one boundary LSB={lsb:.3e})."
        )
    if max_abs_delta > bound:
        raise RuntimeError(
            f"[mvm AQ R-edge] grid parity FAILED: max|delta|="
            f"{max_abs_delta:.3e} exceeds {VALUE_R_EDGE_AQ_LSB_BOUND:g} "
            f"boundary LSB ({bound:.3e}) over {n_samples} samples — a "
            f"divergence beyond one grid step is a mapping defect, not "
            f"grid-edge chaos."
        )
    print(
        f"[ValueParityGate] PASS model≡identity [AQ grid] "
        f"(max|delta|={max_abs_delta:.3e} = {max_abs_delta / lsb:.2f} LSB, "
        f"bound={VALUE_R_EDGE_AQ_LSB_BOUND:g} LSB, decisions=1.0000, "
        f"n={n_samples})"
    )


def _fp64_identity_flow(pipeline, ir_graph) -> ValueHybridCoreFlow:
    identity = build_identity_hybrid_mapping(ir_graph=ir_graph)
    return ValueHybridCoreFlow(
        identity, device="cpu", dtype=torch.float64,
        activation_bits=_activation_bits(pipeline),
    )


def run_model_value_parity_gate(pipeline, model, ir_graph) -> None:
    """[mvm R-edge, FATAL] model forward ≡ identity value program (fp64).

    Float programs share every bit: exact by construction. Quantized
    programs carry the honest fp32-projection residual (the model holds
    round(w*s)/s in fp32; the chip program computes int/s in fp64), so the
    edge is judged at the measured ``VALUE_R_EDGE_WQ_ATOL``.
    """
    n = int(_effective(pipeline.config, "value_parity_samples"))
    if n <= 0:
        print("[ValueParityGate] SKIP (value_parity_samples=0)")
        return
    samples = _certificate_samples(pipeline, model, n)
    if samples is None:
        print("[ValueParityGate] SKIP (no validation batch available)")
        return
    quantized = bool(DeploymentPlan.of(pipeline).weight_quantization)
    atol = VALUE_R_EDGE_WQ_ATOL if quantized else VALUE_TWIN_FP64_ATOL
    samples = samples.detach().to("cpu", torch.float64)
    reference = copy.deepcopy(model).to("cpu").double().eval()
    identity_flow = _fp64_identity_flow(pipeline, ir_graph)
    with torch.no_grad():
        want = reference(samples)
        got = identity_flow(samples)
    max_abs_delta = float((got - want).abs().max().item()) if want.numel() else 0.0

    lsb = boundary_grid_lsb(_activation_bits(pipeline), ir_graph)
    if lsb is not None:
        _assert_aq_grid_parity(got, want, max_abs_delta, lsb, int(samples.shape[0]))
        return

    if max_abs_delta > atol:
        raise RuntimeError(
            f"[mvm R-edge] model↔identity value parity FAILED: "
            f"max|delta|={max_abs_delta:.3e} > atol={atol:.1e} "
            f"(weight_quantization={quantized}) over {int(samples.shape[0])} "
            f"samples — the identity program must reproduce the model's "
            f"affine math at this edge's exactness class."
        )
    print(
        f"[ValueParityGate] PASS model≡identity "
        f"(max|delta|={max_abs_delta:.3e}, atol={atol:.1e}, "
        f"n={int(samples.shape[0])})"
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
    reference_flow = _fp64_identity_flow(pipeline, ir_graph)
    backend_flow = ValueHybridCoreFlow(
        hybrid_mapping, device="cpu", dtype=torch.float64,
        activation_bits=_activation_bits(pipeline),
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
    return ValueHybridCoreFlow(
        hybrid_mapping, device=device, dtype=torch.float32,
        activation_bits=_activation_bits(pipeline),
    )


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
