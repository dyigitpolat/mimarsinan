"""Code-level witnesses: each off arm changes outputs where its paired control agrees; base violations are named.

A base violation is a hypothesis this tree's toolchain itself breaks (no assertion site yet). Its witness pins
the counterexample on this base, attributed to the arm's mechanism, so the test fails whenever the violation
disappears or changes class: the owner's fix (plan section 21) flips it, and the closure item that integrates
the fix re-pins it (plan section 13 rule 6), where the owner's own test becomes the necessity witness.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import mimarsinan.certification.randomized.executors as ex
from mimarsinan.certification.randomized.arms import ARMS, ON_LONG_WINDOW, ON_VALUES
from mimarsinan.certification.randomized.attribution import (
    _first_restoring_layer,
    _producer_at_or_after_consumer,
    attribute,
    counterexample_tags,
)
from mimarsinan.certification.randomized.cost_draws import platform_constraints
from mimarsinan.certification.randomized.generator import DEFAULT_SPACE
from mimarsinan.certification.randomized.instance import (
    CASCADED_TTFS,
    NEURAL,
    STREAMED_LIF,
    SYNCHRONIZED_LIF,
    SYNCHRONIZED_TTFS,
    WINDOWED_LIF,
    Instance,
    Layer,
    Network,
)
from mimarsinan.certification.randomized.runner import (
    COUNTEREXAMPLE,
    EQUAL,
    RUN_ERROR,
    Verdict,
    _run,
    compare,
    deployment,
    instance_for,
    reference,
    reference_nodes,
    run_instance,
    structure,
)

#: Off arm -> a seed where it changes outputs (found by the screen; pinned so the witness is reproducible).
OFF_WITNESSES = {"H1-theta": 0, "H1-latency": 262, "H5-collapse": 32, "H5-collapse/windowed": 32, "H8": 1}
#: The R4 arm's witness: a windowed-LIF deployment whose dithered encode rotates a slice by its channel index in
#: the deployed stage input, not its fused one (class position_dependent_encode).
R4_SEED = 6
#: Base-violation arm -> a seed where the base toolchain runs a counterexample of the arm's mechanism.
BASE_WITNESSES = {"H5-skew": 1, "R6": 16, "H0": 0, "R4": R4_SEED}
#: A synchronized-LIF hypothesis-on counterexample the cut ablation alone would call ``u1_cut``: its
#: deployment walks in dependency order, its identity program does not (re-pin at CL1 with R6's owner).
WALK_WITNESS = ("on_scheduled", 67)


class TestCodeLevelWitnesses:
    @pytest.mark.parametrize("arm", sorted(OFF_WITNESSES))
    def test_off_arm_differs_and_control_agrees(self, arm):
        seed = OFF_WITNESSES[arm]
        off = run_instance(seed, arm)
        assert off.outcome == COUNTEREXAMPLE and off.mismatches > 0, off.to_dict()
        control = run_instance(seed, arm, control=True)
        assert control.outcome == EQUAL and control.windows == off.windows, control.to_dict()

    def test_every_switching_arm_has_a_witness(self):
        switching = {a for a, arm in ARMS.items() if arm.hypothesis and arm.exercisable and arm.changes_a_field}
        assert switching == set(OFF_WITNESSES)


class TestBaseViolations:
    @pytest.mark.parametrize("arm", sorted(BASE_WITNESSES))
    def test_the_base_runs_a_counterexample_of_the_arms_mechanism(self, arm):
        verdict = run_instance(BASE_WITNESSES[arm], arm)
        assert verdict.outcome == COUNTEREXAMPLE, verdict.to_dict()
        assert ARMS[arm].mechanism in attribute(verdict), verdict.to_dict()

    def test_every_mechanism_arm_has_a_witness(self):
        assert {a for a, arm in ARMS.items() if arm.mechanism} == set(BASE_WITNESSES)

    def test_the_r4_violation_needs_the_dithered_encode_on_both_programs(self, monkeypatch):
        """The R4 arm holds a function-bearing axis (phase-dithered Uniform) on the identity program and the
        deployment alike, so it switches nothing off; the same deployment without the dither is equal."""
        arm = ARMS["R4"]
        assert arm.phase_dither and not arm.changes_a_field and arm.control() == arm
        monkeypatch.setitem(ARMS, "R4/undithered", replace(arm, id="R4/undithered", phase_dither=False))
        assert run_instance(R4_SEED, "R4/undithered").outcome == EQUAL

    @pytest.mark.parametrize("semantics, tag", [(STREAMED_LIF, "mixed_input_pass"),
                                                (WINDOWED_LIF, "windowed_cached_trains"),
                                                (SYNCHRONIZED_LIF, "run_error"), (CASCADED_TTFS, "run_error"),
                                                (SYNCHRONIZED_TTFS, "run_error")])
    def test_a_cached_trains_crash_is_tagged_by_its_rule(self, semantics, tag):
        """u1_design.md section 3: under streamed LIF a pass that mixes cached trains with an external slice
        breaks R4 (per-slice input kinds); under windowed LIF a single-level chunk without level stages breaks
        R5 (level units). One message, two rules: the semantics decides, and under any other semantics the
        message names no rule, so the crash stays an unattributed run error (never filed under R4)."""
        verdict = Verdict(seed=0, arm="on", semantics=semantics, motifs=(), declaration={}, drawn=(),
                          outcome=RUN_ERROR, detail="ValueError: encode_segment_input: stage 's' has cached spike "
                                                    "trains for some inputs but is missing spike train(s)")
        assert attribute(verdict) == (tag,)

    def test_a_walk_counterexample_is_the_walks_whichever_side_walks_out_of_order(self):
        """The identity program walks a consumer before its producer (ChipLatency's ALAP raise, then the
        stored-latency count walk); the multi-pass deployment happens to walk in dependency order. Undoing
        the cut would copy the reference's walk and 'restore' equality; the walk ablation names the cause."""
        arm_id, seed = WALK_WITNESS
        verdict = run_instance(seed, arm_id)
        assert verdict.outcome == COUNTEREXAMPLE and verdict.semantics == "synchronized_lif", verdict.to_dict()
        assert verdict.multipass and verdict.network_mismatches == verdict.mismatches, verdict.to_dict()
        tags = attribute(verdict)
        assert "latency_order_walk" in tags and "u1_cut" not in tags, tags
        assert "reference_walk" in tags and "deployment_walk" not in tags, tags
        # The cut ablation alone (round 0's attribution) names the cut: undoing it copies the identity's walk.
        arm = ARMS[arm_id]
        instance = instance_for(seed, arm)
        ref_ir, _ref_map, ref_nodes = reference_nodes(instance, arm)
        with arm.engaged():
            layer = _first_restoring_layer(instance, platform_constraints(verdict.declaration, hardware_bias=
                                                                          instance.hardware_bias),
                                           seed, arm, ref_ir, ref_nodes)
        assert layer == "pass_cut"


#: The cost-only deployment that cuts a two-layer chain between its layers: one core of one neuron, scheduling on
#: (two passes, the L1 -> L2 wire carried).
TWO_PASSES = {"cores": [{"max_axons": 4, "max_neurons": 1, "count": 1}], "allow_coalescing": False,
              "allow_neuron_splitting": False, "allow_scheduling": True, "max_schedule_passes": 8,
              "core_value_granularity": {}}
#: A long-window hypothesis-on seed whose counterexample the integer-count ablation alone restores (found by the
#: screen's long-window stratum; pinned so the class stays visible to the confirmatory run).
LONG_WINDOW_WITNESS = (ON_LONG_WINDOW, 25)


def _rebuild_chain(T: int, theta: int = 3) -> Instance:
    """input -> L1 (weight theta, threshold theta) -> L2 (1, 1) under synchronized LIF, every input count 0..T."""
    l1 = Layer("L1", NEURAL, ("input",), 1, weights=((theta,),), bias=(0,), theta=theta)
    l2 = Layer("L2", NEURAL, ("L1",), 1, weights=((1,),), bias=(0,), theta=1)
    return Instance(seed=0, semantics=SYNCHRONIZED_LIF, timesteps=T, hardware_bias=False,
                    network=Network(1, (l1, l2), ("L1", "L2")), inputs=tuple((k,) for k in range(T + 1)))


class TestTheCountRebuild:
    """H8's base violation (``models/spiking/hybrid/executors/sync_counts.py:56-59``, no Wave 1 group's file): the
    synchronized LIF count executor rebuilds each core's count as ``stair * T / theta`` in float64, which leaves the
    integer lattice for (T, theta) beyond the screen's declared window range. A consumer in the same segment then
    reads ``4.999999999999999`` where a deployment that cuts the wire re-encodes the count and reads 5. Re-pin when
    the owner rebuilds the count on the integer lattice: the counterexample becomes equal."""

    def test_a_rebuilt_count_off_the_lattice_makes_a_pass_cut_change_outputs(self):
        instance, arm = _rebuild_chain(13), ARMS["on"]
        constraints = platform_constraints(TWO_PASSES, hardware_bias=False)
        ref = reference(instance, arm)
        assert not _producer_at_or_after_consumer(ref.mapping)  # R6 holds: L1 walks before L2
        l1 = ex.per_neuron(ref.ir, ref.nodes)[ref.perceptron_of["L1"]][:, 0].tolist()
        assert (l1[5], l1[10]) == (4.999999999999999, 9.999999999999998), l1
        ir, mapping = deployment(instance, constraints, 0, arm)
        assert structure(mapping)["multipass"] and structure(mapping)["carried"]
        windows, mismatches, first = compare(ref.ir, ref.nodes, ir, _run(mapping, instance, arm))
        assert (windows, mismatches) == (28, 2), first  # COUNTEREXAMPLE: L2 at counts 5 and 10
        assert counterexample_tags(instance, constraints, 0, arm) == ("layer:counts", "count_rebuild")

    def test_the_rebuild_is_exact_on_the_declared_window_range(self):
        """Why the T <= 8 screen cannot see the class: on X3a's declared range (T in 1-8, theta in 1-16) every
        rebuilt count is its integer, end to end through the executor; the long-window stratum draws beyond it."""
        assert (DEFAULT_SPACE.timesteps, DEFAULT_SPACE.theta) == ((1, 8), (1, 16))
        for T in range(1, 9):
            for theta in range(1, 17):
                ref = reference(_rebuild_chain(T, theta), ARMS["on"])
                l1 = ex.per_neuron(ref.ir, ref.nodes)[ref.perceptron_of["L1"]]
                assert bool((l1 == torch.round(l1)).all()), (T, theta, l1.flatten().tolist())
        long_window = ARMS[ON_LONG_WINDOW].timesteps
        assert long_window is not None and long_window[0] == DEFAULT_SPACE.timesteps[1] + 1

    def test_the_long_window_stratum_runs_a_counterexample_of_the_rebuild(self):
        arm_id, seed = LONG_WINDOW_WITNESS
        verdict = run_instance(seed, arm_id)
        assert verdict.outcome == COUNTEREXAMPLE and verdict.semantics == SYNCHRONIZED_LIF, verdict.to_dict()
        assert "count_rebuild" in attribute(verdict), verdict.to_dict()


def _coresident_alap(T: int = 4) -> Instance:
    """A = n(input) -> B = n(A) -> Z = n(B, X); X = n(input); Y = n(input) -> V = n(Y). ASAP: A, X, Y at 0;
    B, V at 1; Z at 2. ALAP: X at 1, Y at 0. X and Y share one threshold and one IR (ASAP) depth."""
    def ones(n: int):
        return (tuple(1 for _ in range(n)),)

    layers = (
        Layer("A", NEURAL, ("input",), 1, weights=ones(4), bias=(0,), theta=2),
        Layer("B", NEURAL, ("A",), 1, weights=ones(1), bias=(0,), theta=3),
        Layer("X", NEURAL, ("input",), 1, weights=ones(4), bias=(0,), theta=1),
        Layer("Z", NEURAL, ("B", "X"), 1, weights=ones(2), bias=(0,), theta=4),
        Layer("Y", NEURAL, ("input",), 1, weights=ones(4), bias=(0,), theta=1),
        Layer("V", NEURAL, ("Y",), 1, weights=ones(1), bias=(0,), theta=1),
    )
    return Instance(seed=0, semantics=STREAMED_LIF, timesteps=T, hardware_bias=True,
                    network=Network(4, layers, tuple(layer.name for layer in layers)),
                    inputs=((T, T, T, T), (1, 2, 3, 4)))


class TestLatentDefects:
    def test_coresident_soft_cores_of_one_ir_depth_and_two_alap_latencies(self):
        """H1-latency's general form (every neuron at its identity-program latency): the packing predicate
        compares IR (ASAP) depth, so X and Y share a hard core, which ChipLatency raises to X's ALAP
        latency 1 while the identity program runs Y at 0; V then reads Y one cycle late (3 spikes, not 4).
        Latent as shipped (the runtime classes never co-locate layers); live under value-keyed classes.
        Re-pin at CL1 with G5's executed-latency assertion."""
        instance, arm = _coresident_alap(), ARMS[ON_VALUES]
        declaration = {"cores": [{"max_axons": 8, "max_neurons": 2, "count": 5}], "allow_coalescing": False,
                       "allow_neuron_splitting": False, "allow_scheduling": False, "max_schedule_passes": 1,
                       "core_value_granularity": {}}
        ref_ir, _ref_map, ref_nodes = reference_nodes(instance, arm)
        with arm.engaged():
            ir, mapping = deployment(instance, platform_constraints(declaration, hardware_bias=True), 0, arm,
                                     relabel=False)
            windows, mismatches, first = compare(ref_ir, ref_nodes, ir, _run(mapping, instance, arm))
        assert (windows, mismatches) == (12, 2), first
