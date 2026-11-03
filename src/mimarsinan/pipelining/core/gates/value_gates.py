"""[mvm] value-domain deployment gates and metric reads (R-edge, C-edge, census)."""

from __future__ import annotations

import copy

import torch

from mimarsinan.certification.value_certificate import (
    VALUE_LAW_ATOL,
    VALUE_R_EDGE_AQ_LSB_BOUND,
    VALUE_R_EDGE_WQ_ATOL,
    VALUE_TWIN_FP64_ATOL,
    certify_twin_flow_values,
    exposed_nodes,
    value_edge_atol,
)
from mimarsinan.certification.operand_certificate import certify_operand_program
from mimarsinan.chip_simulation.deployed_host import (
    DEPLOYED_HOST_DEVICE,
    announce as announce_host,
    value_host_arithmetic,
)
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir.deployment_dtype import VALUE_ARITHMETIC
from mimarsinan.pipelining.core.census.arithmetic import (
    announce,
    value_census_arithmetic,
)
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir.core_law import graph_carries_laws
from mimarsinan.mapping.lanes.ir.fusion import LANE_REPORT
from mimarsinan.mapping.support.boundary_grids import widest_boundary_step
from mimarsinan.mapping.support.host_kernel_evaluation import forward_of, twin_as_program
from mimarsinan.config_schema.registry import effective_value as _effective
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.simulation_factory import (
    build_hybrid_mapping_for_pipeline,
    run_hcm_spiking_test,
)
from mimarsinan.pipelining.core.spike_count_gate import _certificate_samples
from mimarsinan.torch_mapping.value_parity.value_program import fp64_identity_program


def _value_host(pipeline, evaluation: str):
    """[U-HDA] A value certificate's arithmetic on the deployed host (``chip_simulation/deployed_host.py``), printed."""
    host = value_host_arithmetic(evaluation, pipeline.config.get("device"))
    announce_host(host, getattr(pipeline, "reporter", None))
    return host


def value_certificate_gate_armed(pipeline) -> bool:
    """Whether the value twin certificate runs: a values-observable plan with
    a nonzero sample budget. Shared by the gate and the SCM rung-2 derivation."""
    plan = DeploymentPlan.of(pipeline)
    observable, _reason = plan.mode_policy().certification_observable()
    if observable != "values":
        return False
    return int(_effective(pipeline.config, "value_parity_samples")) > 0


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


class LaneParityDisarmedError(ToolchainRefusal, RuntimeError):
    """A lane program whose R-edge is disarmed (``value_parity_samples`` 0)."""


def run_model_value_parity_gate(pipeline, model, ir_graph) -> None:
    """[mvm R-edge, FATAL] model forward ≡ identity value program (fp64).

    Float programs share every bit: exact by construction. Quantized
    programs carry the honest fp32-projection residual (the model holds
    round(w*s)/s in fp32; the chip program computes int/s in fp64), so the
    edge is judged at the measured ``VALUE_R_EDGE_WQ_ATOL``.
    """
    n = int(_effective(pipeline.config, "value_parity_samples"))
    if n <= 0:
        if getattr(ir_graph, LANE_REPORT, None) is not None:
            raise LaneParityDisarmedError(
                "value_parity_samples=0 disarms the R-edge, the one pipeline certificate that compares the model "
                "with the program; a lane program (fused adds, clamps and pools on chip) keeps it armed: declare "
                "value_parity_samples >= 1")
        print("[ValueParityGate] SKIP (value_parity_samples=0)")
        return
    samples = _certificate_samples(pipeline, model, n)
    if samples is None:
        print("[ValueParityGate] SKIP (no validation batch available)")
        return
    quantized = bool(DeploymentPlan.of(pipeline).weight_quantization)
    atol = VALUE_R_EDGE_WQ_ATOL if quantized else VALUE_TWIN_FP64_ATOL
    host = _value_host(pipeline, "Soft Core Mapping value R-edge (K4: the model copy and the identity program)")
    samples = samples.detach().to(host.device, VALUE_ARITHMETIC.core_dtype)
    reference = copy.deepcopy(model).to(host.device).to(VALUE_ARITHMETIC.core_dtype).eval()
    identity_flow = fp64_identity_program(ir_graph)
    with torch.no_grad():
        want = twin_as_program(ir_graph, forward_of, reference, samples)  # [U-PS] the model as the program evaluates
        got = identity_flow(samples)
    max_abs_delta = float((got - want).abs().max().item()) if want.numel() else 0.0

    # The CONTRACT names the certificate class; the gate never re-derives it.
    # [E284] NF and the identity program of a law-carrying program read one integer law: exact.
    if graph_carries_laws(ir_graph):
        atol = VALUE_LAW_ATOL
    elif packaging_contract_for(DeploymentPlan.of(pipeline)).boundary_is_gridded:
        lsb = widest_boundary_step(ir_graph)
        if lsb is None:
            raise RuntimeError(
                "[mvm AQ R-edge] the packaging contract declares a gridded "
                "boundary but no core carries an armed input_activation_scale "
                "— Boundary Quantization did not reach this program."
            )
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
    host = _value_host(pipeline, "Hard Core Mapping value twin certificate (K5: the identity and packed programs)")
    samples = samples.detach().to(host.device, VALUE_ARITHMETIC.core_dtype)
    if getattr(ir_graph, "dynamic_bank_bindings", None):
        return _operand_program_gate(model, ir_graph, hybrid_mapping, samples)
    # The identity reference writes out every node the backend's stages write (its pass carries included).
    reference_flow = fp64_identity_program(ir_graph, exposed=exposed_nodes(hybrid_mapping))
    backend_flow = ValueHybridCoreFlow(
        hybrid_mapping, device=host.device, dtype=VALUE_ARITHMETIC.core_dtype
    )
    certificate, detail = certify_twin_flow_values(
        reference_flow, backend_flow, samples,
        atol=value_edge_atol(graph_carries_laws(ir_graph)),
    )
    if not certificate.passed:
        raise RuntimeError(
            f"value twin certificate FAILED (identity↔packed): "
            f"{certificate.summary()} | {detail}"
        )
    print(f"[ValueTwinCertificate] PASS {certificate.summary()}")
    return certificate


def _operand_program_gate(model, ir_graph, hybrid_mapping, samples):
    """[N2] a program with dynamic operand banks: every edge of design §3.2 at atol 0, the payload edge included."""
    reference = copy.deepcopy(model).to(DEPLOYED_HOST_DEVICE).to(VALUE_ARITHMETIC.core_dtype).eval()
    certificate = certify_operand_program(reference, build_identity_hybrid_mapping(ir_graph=ir_graph),
                                          hybrid_mapping, samples)
    if not certificate.passed:
        raise RuntimeError(f"operand program certificate FAILED: {certificate.summary()}")
    print(f"[ValueTwinCertificate] PASS operand program {certificate.summary()}")
    return certificate


def _run_value_census(pipeline, hybrid_mapping, census: str, device) -> float:
    """[U-CEN] A value census in the deployed read's arithmetic (``pipelining/core/census/arithmetic.py``): float64
    on the CPU, or on the pipeline's device where that is the CPU; ``device`` is the device the pipeline asks for (its
    own when None)."""
    arithmetic = value_census_arithmetic(pipeline, census, device=device)
    announce(pipeline, arithmetic)
    flow = ValueHybridCoreFlow(hybrid_mapping, device=arithmetic.device, dtype=VALUE_ARITHMETIC.core_dtype)
    return float(run_hcm_spiking_test(pipeline, flow, device=arithmetic.device, retry_on_oom=True))


def run_value_identity_metric(pipeline, ir_graph, *, device=None) -> float:
    """Rung-2 census through the identity value program (unarmed configs only)."""
    identity = build_identity_hybrid_mapping(ir_graph=ir_graph)
    return _run_value_census(pipeline, identity, "Soft Core Mapping identity value program census", device)


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
    return _run_value_census(pipeline, hybrid_mapping, "Hard Core Mapping packed value program census", device)
