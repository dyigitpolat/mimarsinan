"""D3: end-to-end SCHEDULED-BUILD probe — the Scheduled path REALIZES weight-reuse
phases bit-exact, not just the capacity ESTIMATE.

A small ``deep_cnn`` (d4 w6 @ 8x8) is the genuine vehicle: it maps fully on-chip,
its two same-shape conv stages share weight banks (so the weight-reuse classifier
sees real reprogram-vs-reuse passes), and it GENUINELY overflows a deliberately
tight 6-core budget (UNSCHED infeasible) so scheduling actually triggers. The probe
locks three distinct facts on the SAME real IR graph:

(a) ESTIMATE — ``estimate_cores_needed(allow_scheduling=True)`` reports
    ``scheduled=True``, ``phase_count > 1``, ``peak_phase_cores <= budget``, and the
    UNSCHEDULED verdict on the same IR/budget is INFEASIBLE (the SUM overflows).

(b) BUILD vs ESTIMATE STRUCTURE — the Scheduled builder
    (``build_hybrid_hard_core_mapping`` with ``allow_scheduling``) realizes exactly
    ``phase_count`` neural stages, each occupying ``<= budget`` hard cores (the
    estimate's per-segment ``ceil(bound/budget)`` split), AND the
    ``weight_reuse_plan_from_graph`` time-domain decomposition over the same IR is
    ``N reprogram`` (= #distinct weight banks) + ``M reuse`` (= positions sharing a
    resident bank), with ``total_passes == #IR neural cores``.

(c) BIT-EXACT — the scheduled-built deployment is value-identical (float64
    ``atol=0``) to BOTH the non-scheduled single-pool reference (built on a generous
    budget where the SUM fits) AND the torch neuromorphic forward.

The estimate's ``phase_count`` (capacity time-multiplex: ``Σ ceil(seg_bound/budget)``)
and the reuse plan's ``reprogram/reuse`` split (weight-bank residency) are TWO
distinct decompositions of the same schedule; the probe asserts both and how they
relate, so a regression in either the capacity estimate, the scheduled builder, or
the reuse classifier trips a loud, diagnosable failure.

This probe confirms the roadmap's "VGG@224 -> ~16 reprogram + ~142 reuse" claim
**by mechanism** (the classifier + scheduled builder demonstrably work on a real,
weight-bank-sharing conv graph and stay bit-exact) — it does NOT build VGG@224 (a
heavy ~40s mapping path); the VGG number itself remains a per-conv-block analytical
projection, stated honestly in ``docs/research/findings/D3_scheduled_build_probe.md``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.mapping.verification.capacity import estimate_cores_needed
from mimarsinan.mapping.weight_reuse import weight_reuse_plan_from_graph
from mimarsinan.models.builders.deep_cnn_builder import DeepCNNBuilder
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward


T = 4
INPUT_SHAPE = (1, 8, 8)
NUM_CLASSES = 10

# A small genuine conv stack: two same-shape conv segments share weight banks (so the
# reuse classifier sees real reuse passes) and it overflows the tight budget below.
_DEEP_CNN_CFG = {"depth": 4, "width": 6, "base_activation": "ReLU"}

# Wide enough hard cores that no fan-in/width is itself the constraint — the budget bites
# only through the *count* of cores, so scheduling (not splitting) is what we exercise.
_CORE_WIDTH = {"max_axons": 256, "max_neurons": 256}

# Deliberately tight: the SUM of the per-segment lower bounds (13) exceeds 6, but the
# PEAK per-segment phase fits, so the scheduled path is feasible and the single pool is not.
_TIGHT_BUDGET = 6
_GENEROUS_BUDGET = 4000  # the non-scheduled single-pool reference fits comfortably here


def _build_deep_cnn_ir():
    """Convert the small deep_cnn to a LIF IR graph (the shared real graph under test)."""
    builder = DeepCNNBuilder("cpu", INPUT_SHAPE, NUM_CLASSES, {})
    model = builder.build(_DEEP_CNN_CFG)
    torch.manual_seed(0)
    flow = convert_torch_model(model, INPUT_SHAPE, NUM_CLASSES, device="cpu").eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    for p in flow.get_perceptrons():
        lif = LIFActivation(
            T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<="
        )
        p.base_activation = lif
        p.activation = lif
    repr_.assign_perceptron_indices()
    ir = IRMapping(
        q_max=127.0,
        firing_mode="Default",
        max_axons=_CORE_WIDTH["max_axons"],
        max_neurons=_CORE_WIDTH["max_neurons"],
        allow_coalescing=False,
    ).map(repr_)
    return flow, ir


def _cores_config(count: int):
    return [dict(_CORE_WIDTH, count=count)]


def _build_hcm(ir, *, count: int, allow_scheduling: bool):
    """Pack ``ir`` into a runnable HCM under the scheduled or single-pool path."""
    strategy = MappingStrategy.from_permissions(
        allow_neuron_splitting=True,
        allow_coalescing=False,
        allow_scheduling=allow_scheduling,
    )
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=_cores_config(count), strategy=strategy
    )
    hcm = SpikingHybridCoreFlow(
        INPUT_SHAPE,
        hybrid,
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="Default",
        spike_mode="Uniform",
        thresholding_mode="<=",
        spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    )
    return hybrid, hcm


def _neural_stages(hybrid):
    return [s for s in hybrid.stages if s.kind == "neural"]


def _stage_hard_core_count(stage):
    return len(stage.hard_core_mapping.soft_core_placements_per_hard_core)


# --------------------------------------------------------------------------- #
# (a) ESTIMATE — scheduling triggers on a genuine overflow.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_a_scheduled_estimate_triggers_on_genuine_overflow():
    """The small deep_cnn genuinely overflows the tight budget as one pool, but the
    PEAK per-segment phase fits — so ``allow_scheduling=True`` is feasible with
    ``phase_count > 1`` and ``peak_phase_cores <= budget``, while the SAME IR/budget
    is INFEASIBLE without scheduling."""
    _, ir = _build_deep_cnn_ir()
    cores = _cores_config(_TIGHT_BUDGET)

    sched = estimate_cores_needed(
        ir, {"cores": cores, "allow_coalescing": False, "allow_scheduling": True}
    )
    assert sched.scheduled is True
    assert sched.feasible is True
    assert sched.cores_available == _TIGHT_BUDGET
    # scheduling is GENUINELY needed: the single-pool SUM overflows the budget.
    assert sched.cores_needed > _TIGHT_BUDGET
    # more than one reprogramming phase actually happens.
    assert sched.phase_count > 1
    # the peak instantaneous chip occupancy fits the budget (the whole point).
    assert 0 < sched.peak_phase_cores <= _TIGHT_BUDGET

    unsched = estimate_cores_needed(
        ir, {"cores": cores, "allow_coalescing": False, "allow_scheduling": False}
    )
    assert unsched.scheduled is False
    assert unsched.feasible is False  # the single pool provably cannot fit
    assert unsched.cores_needed == sched.cores_needed  # same intrinsic softcore count


# --------------------------------------------------------------------------- #
# (b) BUILD vs ESTIMATE STRUCTURE — the realized schedule matches the estimate.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_b_scheduled_build_realizes_estimate_phase_structure():
    """The Scheduled BUILD realizes exactly ``phase_count`` neural stages, each
    occupying ``<= budget`` hard cores (the estimate's per-segment
    ``ceil(bound/budget)`` split) — not just the capacity ESTIMATE."""
    _, ir = _build_deep_cnn_ir()
    cores = _cores_config(_TIGHT_BUDGET)
    sched = estimate_cores_needed(
        ir, {"cores": cores, "allow_coalescing": False, "allow_scheduling": True}
    )

    hybrid, _ = _build_hcm(ir, count=_TIGHT_BUDGET, allow_scheduling=True)
    stages = _neural_stages(hybrid)

    # The builder realizes exactly the estimate's reprogramming-pass count.
    assert len(stages) == sched.phase_count
    assert len(stages) > 1  # scheduling actually split the graph into >1 pass

    # Every realized phase fits the budget, and the largest equals the estimate's peak.
    per_stage = [_stage_hard_core_count(s) for s in stages]
    assert all(0 < c <= _TIGHT_BUDGET for c in per_stage)
    assert max(per_stage) == sched.peak_phase_cores

    # The per-segment phase split matches ``ceil(segment_bound / budget)``.
    expected_phases_per_segment = sum(
        math.ceil(b / sched.cores_available) for b in sched.per_segment.values()
    )
    assert len(stages) == expected_phases_per_segment


@pytest.mark.integration
def test_b_weight_reuse_plan_matches_graph_structure():
    """The realized weight-reuse phase decomposition (from
    ``weight_reuse_plan_from_graph``) is ``N reprogram`` (= #distinct weight banks)
    + ``M reuse`` (= positions sharing a resident bank), with ``total_passes`` equal
    to the number of IR neural cores — the conv-weight-reuse mechanism the roadmap's
    VGG@224 "16 reprogram + 142 reuse" claim rests on, on a REAL graph."""
    _, ir = _build_deep_cnn_ir()

    plan = weight_reuse_plan_from_graph(ir)
    neural_cores = ir.get_neural_cores()
    distinct_banks = {
        c.weight_bank_id for c in neural_cores if c.weight_bank_id is not None
    }
    owned_cores = [c for c in neural_cores if c.weight_bank_id is None]

    # N reprogram = #distinct shared banks + #owned (non-shared) cores.
    assert plan.reprogram_passes == len(distinct_banks) + len(owned_cores)
    # this small deep_cnn DOES share banks (so the reuse mechanism is genuinely exercised).
    assert len(distinct_banks) >= 1
    assert plan.reuse_passes > 0, (
        "the probe model must exhibit real weight reuse, else the conv-reuse "
        "mechanism is not being exercised"
    )
    # every IR neural core is exactly one pass (reprogram or reuse).
    assert plan.total_passes == len(neural_cores)
    assert plan.reuse_passes == plan.total_passes - plan.reprogram_passes
    # one barrier between consecutive passes (the design's (M + N - 1)).
    assert plan.sync_barrier_count == plan.total_passes - 1
    # most passes reuse (the conv-position collapse the roadmap quantifies).
    assert plan.reuse_fraction > 0.5


# --------------------------------------------------------------------------- #
# (c) BIT-EXACT — the scheduled build is value-identical where both fit.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_c_scheduled_build_is_bit_exact_vs_reference_and_torch():
    """The scheduled-built deployment is value-identical (float64 ``atol=0``) to BOTH
    the non-scheduled single-pool reference (generous budget) AND the torch
    neuromorphic forward — scheduling time-multiplexes the SAME softcores, it does
    not perturb a single value."""
    flow, ir = _build_deep_cnn_ir()

    _, hcm_sched = _build_hcm(ir, count=_TIGHT_BUDGET, allow_scheduling=True)
    _, hcm_ref = _build_hcm(ir, count=_GENEROUS_BUDGET, allow_scheduling=False)

    torch.manual_seed(3)
    x = torch.rand(6, *INPUT_SHAPE)
    with torch.no_grad():
        nf = chip_aligned_segment_forward(flow, x, T).double()
        out_sched = hcm_sched(x).double() / float(T)
        out_ref = hcm_ref(x).double() / float(T)

    sched_vs_ref = float((out_sched - out_ref).abs().max())
    sched_vs_torch = float((out_sched - nf).abs().max())
    ref_vs_torch = float((out_ref - nf).abs().max())

    assert sched_vs_ref == 0.0, (
        f"scheduled build diverged from the single-pool reference: max|Δ|={sched_vs_ref}"
    )
    assert sched_vs_torch == 0.0, (
        f"scheduled build diverged from torch NF: max|Δ|={sched_vs_torch}"
    )
    # the reference itself must be exact too (else the comparison is vacuous).
    assert ref_vs_torch == 0.0, (
        f"non-scheduled reference diverged from torch NF: max|Δ|={ref_vs_torch}"
    )


@pytest.mark.integration
def test_c_scheduled_build_has_more_stages_than_single_pool():
    """Sanity that the scheduled path is REALLY doing something different: it emits
    strictly more neural stages than the single-pool path on the same IR (the
    capacity-split reprogramming passes), and yet stays bit-exact (asserted above)."""
    _, ir = _build_deep_cnn_ir()
    sched_hybrid, _ = _build_hcm(ir, count=_TIGHT_BUDGET, allow_scheduling=True)
    pool_hybrid, _ = _build_hcm(ir, count=_GENEROUS_BUDGET, allow_scheduling=False)
    assert len(_neural_stages(sched_hybrid)) > len(_neural_stages(pool_hybrid))
