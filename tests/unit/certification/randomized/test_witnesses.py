"""Code-level witnesses: each off arm changes outputs where its paired control agrees; base violations are named.

A base violation is a hypothesis this tree's toolchain itself breaks (no assertion site yet). Its witness pins
the counterexample on this base, attributed to the arm's mechanism, so the test fails whenever the violation
disappears or changes class: the owner's fix (plan section 21) flips it, and the closure item that integrates
the fix re-pins it (plan section 13 rule 6), where the owner's own test becomes the necessity witness.

CL1 (the integrated tree): G5's pass projection meets H5-skew (R1 pins, R2 delivery delays) and every segment walk
is dependency-ordered (R6), so both arms switch the integrated rule off and pin a counterexample with an equal
paired control; the CL1 fix round (sign-off P1 (a)) levels a coalescing group as one unit, so H0 joins them, and
reads the synchronized LIF count off its window charge (H8's count_rebuild). E1 meets R4, the last base violation:
the dithered comb rotates by each channel's channel at unbounded geometry, so the R4 arm switches that key back to the
channel's stage-input position, with an equal paired control. Two arms are retired with their reason, never
weakened: the
packing predicate's latency clause off is refused by name at the pass projection (H1-latency is asserted at run
time), and a windowed run's transfer discipline changes no output (every carried wire is read by a level unit
that re-encodes its count, G5's R5).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import mimarsinan.certification.randomized.executors as ex
import mimarsinan.chip_simulation.ttfs.ttfs_segment as ttfs_segment
from mimarsinan.certification.randomized.arms import (
    ARMS,
    ON_LONG_WINDOW,
    ON_SCHEDULED,
    ON_VALUES,
    one_latency_unrefused,
)
from mimarsinan.certification.randomized.attribution import (
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
    REFERENCE_ERROR,
    REFUSED,
    RUN_ERROR,
    Verdict,
    _run,
    compare,
    deployment,
    reference,
    reference_nodes,
    run_instance,
    structure,
)
from mimarsinan.mapping.export.law_solver import IntegerLawRangeError
from mimarsinan.models.spiking.hybrid.executors import sync_counts
from mimarsinan.models.spiking.wire_semantics import lif_count_staircase
from mimarsinan.mapping.packing.pass_timing import PassTimingError

#: Off arm -> a seed where it changes outputs (found by the screen; pinned so the witness is reproducible). CL1
#: re-pins: H8 at 210 (seed 1 became a value instance the readout law refuses on the identity program, the H8
#: value-path site below); H5-skew and R6 switch the integrated rules off, at the seeds of their base witnesses; the
#: CL1 fix round's H0 switches the leveling of a coalescing group off, at its base witness's seed. The closure
#: targeted pass (verify R1) re-pins H8 at 90 (windowed LIF, the fix round's screen counterexample for H8): at 210
#: (synchronized TTFS) the TTFS float membrane is the same in every packing past the exact range too, so the
#: deployment equals the identity program there, and the departure H8 names shows on the network relation
#: (``TestTheH8DepartureOnTheTtfsFloatPath``).
#: The R4 arm's witness: a windowed-LIF deployment whose dithered encode, keyed on stage-input positions, rotates a
#: slice by where the relabelled IR's node ids lay it out (class position_dependent_encode). CL1: seed 6's drawn
#: residency plan is refused by name, so the witness moves to 7, the screen's first R4 counterexample; E1 keeps it.
R4_SEED = 7
OFF_WITNESSES = {"H1-theta": 0, "H5-collapse": 32, "H8": 90, "H5-skew": 1, "R6": 16, "H0": 0, "R4": R4_SEED}
#: The base's synchronized-LIF walk witness: the identity program walked a consumer before its producer while the
#: multi-pass deployment walked in dependency order. Retired at CL1 (fixed by the dependency-ordered count walk).
WALK_WITNESS = (ON_SCHEDULED, 67)
#: The R6 off arm's multi-pass counterexample under the hypothesis-on scheduled draw (both programs walk by stored
#: latency there; the walk ablation names the cause).
WALK_OFF_WITNESS = 165


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


class TestTheH8DepartureOnTheTtfsFloatPath:
    """[R1] H8's CL1 pin, seed 210 (synchronized TTFS, coalesced, magnitudes times 2^49 + 1). Past float64's exact
    integer range the lattice hands the neurons to the float path, whose membrane is a fixed function of each
    neuron's own terms, so the deployment equals the identity program; the departure H8 names is the identity
    program's from the network N, absent from the paired control. With the float membrane summed in the layout's own
    order (the step 3 rule) the deployment departs from the identity program as well (5 windows,
    ``closure/targeted/logs/h8_witness_probe.out``)."""

    SEED = 210

    def test_the_deployment_equals_the_identity_program_and_the_network_departs(self):
        off, control = run_instance(self.SEED, "H8").to_dict(), run_instance(self.SEED, "H8", control=True).to_dict()
        assert off["outcome"] == EQUAL and off["network_mismatches"] > 0, off
        assert control["outcome"] == EQUAL and control["network_mismatches"] == 0, control

    def test_in_the_layouts_summation_order_the_deployment_departs(self, monkeypatch):
        def layout_order(x, w, bias):
            v = x @ w.T
            return v if bias is None else v + bias

        monkeypatch.setattr(ttfs_segment, "float_membrane", layout_order)
        off = run_instance(self.SEED, "H8")
        assert off.outcome == COUNTEREXAMPLE and off.mismatches > 0, off.to_dict()


class TestBaseViolations:
    def test_no_arm_is_a_base_violation_on_this_tree(self):
        """E1: R4, the last hypothesis the base violated, is met; every exercisable arm switches its rule off with a
        paired control (``OFF_WITNESSES``), so none attributes its teeth by mechanism."""
        assert not {a for a, arm in ARMS.items() if arm.mechanism}

    def test_the_r4_off_arm_names_the_position_keyed_rotation(self):
        verdict = run_instance(R4_SEED, "R4")
        assert verdict.outcome == COUNTEREXAMPLE and "position_dependent_encode" in attribute(verdict), verdict.to_dict()

    def test_the_r4_off_arm_needs_the_dithered_encode_on_both_programs(self, monkeypatch):
        """The R4 arm holds a function-bearing axis (phase-dithered Uniform) on the identity program, the deployment
        and the paired control alike; the same deployment without the dither is equal with the key switched off."""
        arm = ARMS["R4"]
        assert arm.phase_dither and arm.changes_a_field and arm.control().phase_dither
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

    def test_the_r6_off_arm_is_a_walk_counterexample_on_a_multi_pass_schedule(self, monkeypatch):
        """The base walk witness re-pinned (CL1): with R6 switched off on the hypothesis-on scheduled draw, a
        multi-pass synchronized-LIF deployment and its identity program both walk a consumer at or before its
        producer's stored latency; the walk ablation restores equality and names the walk, never the cut."""
        arm = replace(ARMS[ON_SCHEDULED], id="on_scheduled/R6-off", switch=ARMS["R6"].switch)
        monkeypatch.setitem(ARMS, arm.id, arm)
        verdict = run_instance(WALK_OFF_WITNESS, arm.id)
        assert verdict.outcome == COUNTEREXAMPLE and verdict.semantics == SYNCHRONIZED_LIF, verdict.to_dict()
        assert verdict.multipass, verdict.to_dict()
        tags = attribute(verdict)
        assert "latency_order_walk" in tags and "u1_cut" not in tags, tags
        assert {"reference_walk", "deployment_walk"} <= set(tags), tags
        assert run_instance(WALK_OFF_WITNESS, ON_SCHEDULED).outcome == EQUAL


class TestRetiredWitnesses:
    """Base witnesses the integrated tree fixes, and off arms it refuses by name or makes inert (CL1)."""

    def test_the_base_walk_witness_runs_equal_on_the_integrated_tree(self):
        """Retired: the synchronized LIF count executor walks by ``value_execution.dependency_order`` (merge fix
        75998141), so the identity program no longer reads a buffer not yet written; with R6 off no
        multi-pass counterexample walks out of order on the identity side alone (0 of 13 in seeds 0-399)."""
        arm_id, seed = WALK_WITNESS
        verdict = run_instance(seed, arm_id)
        assert verdict.outcome == EQUAL and verdict.multipass, verdict.to_dict()

    def test_the_latency_off_arm_is_refused_by_name_at_the_pass_projection(self):
        """Retired off arm: H1-latency is asserted at run time. With the packing predicate's latency clause off
        the deployment co-locates soft cores the fused segment runs at two latencies, and G5's pass projection
        refuses it by name (``pass_timing._core_latencies``); the paired control packs them apart, equal."""
        arm = ARMS["H1-latency"]
        assert not arm.exercisable and arm.changes_a_field
        off = run_instance(262, "H1-latency")
        assert off.outcome == REFUSED and off.detail.startswith("PassTimingError"), off.to_dict()
        assert "one hard core starts at one latency" in off.detail
        assert run_instance(262, "H1-latency", control=True).outcome == EQUAL

    def test_the_windowed_transfer_changes_no_output_on_the_integrated_tree(self):
        """Retired off arm: a windowed run's pass transfer forced verbatim is inert, since every carried wire of
        a windowed run is read by a level unit that re-encodes its count (G5's R5) and the TTFS and count
        executors read no transfer (0 teeth in 3,464 admitted on the integrated head). The streamed off arm
        keeps H5-collapse's code-level witness."""
        assert not ARMS["H5-collapse/windowed"].exercisable and ARMS["H5-collapse"].exercisable
        for control in (False, True):
            verdict = run_instance(32, "H5-collapse/windowed", control=control)
            assert verdict.outcome == EQUAL and verdict.multipass and verdict.carried, verdict.to_dict()

    def test_the_h8_value_arm_is_refused_by_name_at_the_law_solver(self):
        """H8's value-path site (G2's E283, from CL1): the off arm's magnitudes put a law past the 2^53 bound,
        which the solver refuses by name on the identity program; its control runs equal to the int64 golden."""
        off = run_instance(7, "H8")
        assert off.semantics == "value" and off.outcome == REFERENCE_ERROR, off.to_dict()
        assert off.detail.startswith(IntegerLawRangeError.__name__) and "2^53" in off.detail
        control = run_instance(7, "H8", control=True)
        assert control.outcome == EQUAL and control.golden_windows > 0 and control.golden_mismatches == 0


#: The cost-only deployment that cuts a two-layer chain between its layers: one core of one neuron, scheduling on
#: (two passes, the L1 -> L2 wire carried).
TWO_PASSES = {"cores": [{"max_axons": 4, "max_neurons": 1, "count": 1}], "allow_coalescing": False,
              "allow_neuron_splitting": False, "allow_scheduling": True, "max_schedule_passes": 8,
              "core_value_granularity": {}}
#: The long-window hypothesis-on seed whose counterexample the integer-count ablation alone restored on the base
#: (found by the screen's long-window stratum); the CL1 fix round's count read off the charge runs it equal.
LONG_WINDOW_WITNESS = (ON_LONG_WINDOW, 25)


def _rebuild_chain(T: int, theta: int = 3) -> Instance:
    """input -> L1 (weight theta, threshold theta) -> L2 (1, 1) under synchronized LIF, every input count 0..T."""
    l1 = Layer("L1", NEURAL, ("input",), 1, weights=((theta,),), bias=(0,), theta=theta)
    l2 = Layer("L2", NEURAL, ("L1",), 1, weights=((1,),), bias=(0,), theta=1)
    return Instance(seed=0, semantics=SYNCHRONIZED_LIF, timesteps=T, hardware_bias=False,
                    network=Network(1, (l1, l2), ("L1", "L2")), inputs=tuple((k,) for k in range(T + 1)))


def _base_count_rebuild(total, theta, T, *, compare_mode="<="):
    """The base tree's synchronized count: the staircase of the window charge rebuilt as ``stair * T / theta``."""
    stair = lif_count_staircase(total / float(T), theta, T, compare_mode=compare_mode)
    return stair * float(T) / torch.clamp(torch.as_tensor(theta, dtype=stair.dtype), min=1e-12)


class TestTheCountRebuild:
    """H8's count_rebuild, fixed in the CL1 fix round (sign-off P1 (a)): the synchronized LIF count executor reads each
    core's count off its window charge (``wire_semantics.lif_window_count``), where the base rebuilt it as
    ``stair * T / theta`` in float64 and left the integer lattice for (T, theta) beyond the declared window range (a
    consumer in the same segment read ``4.999999999999999`` where a deployment that cuts the wire re-encodes 5). The
    base rebuild switched back in is the necessity witness."""

    def _cut_chain(self, T=13):
        instance, arm = _rebuild_chain(T), ARMS["on"]
        constraints = platform_constraints(TWO_PASSES, hardware_bias=False)
        ref = reference(instance, arm)
        assert not _producer_at_or_after_consumer(ref.mapping)  # R6 holds: L1 walks before L2
        ir, mapping = deployment(instance, constraints, 0, arm)
        assert structure(mapping)["multipass"] and structure(mapping)["carried"]
        return instance, arm, constraints, ref, ir, mapping

    def test_the_count_is_its_integer_and_a_pass_cut_changes_no_output(self):
        instance, arm, _constraints, ref, ir, mapping = self._cut_chain()
        l1 = ex.per_neuron(ref.ir, ref.nodes)[ref.perceptron_of["L1"]][:, 0].tolist()
        assert l1 == [float(k) for k in range(14)], l1
        windows, mismatches, first = compare(ref.ir, ref.nodes, ir, _run(mapping, instance, arm))
        assert (windows, mismatches) == (28, 0), first

    def test_the_base_rebuild_makes_the_pass_cut_change_outputs(self, monkeypatch):
        monkeypatch.setattr(sync_counts, "lif_window_count", _base_count_rebuild)
        instance, arm, constraints, ref, ir, mapping = self._cut_chain()
        l1 = ex.per_neuron(ref.ir, ref.nodes)[ref.perceptron_of["L1"]][:, 0].tolist()
        assert (l1[5], l1[10]) == (4.999999999999999, 9.999999999999998), l1
        windows, mismatches, first = compare(ref.ir, ref.nodes, ir, _run(mapping, instance, arm))
        assert (windows, mismatches) == (28, 2), first  # L2 at counts 5 and 10
        assert counterexample_tags(instance, constraints, 0, arm) == ("layer:counts", "count_rebuild")

    @pytest.mark.parametrize("T", [*range(1, 9), 9, 11, 13, 32, 64])
    def test_every_count_is_its_integer_inside_and_beyond_the_declared_window_range(self, T):
        """The declared range (every T in 1-8 and theta in 1-16, exhaustively) and the long-window stratum's
        (T in 9-64) alike."""
        assert (DEFAULT_SPACE.timesteps, DEFAULT_SPACE.theta) == ((1, 8), (1, 16))
        for theta in range(1, 17):
            ref = reference(_rebuild_chain(T, theta), ARMS["on"])
            l1 = ex.per_neuron(ref.ir, ref.nodes)[ref.perceptron_of["L1"]]
            assert bool((l1 == torch.round(l1)).all()), (T, theta, l1.flatten().tolist())
        long_window = ARMS[ON_LONG_WINDOW].timesteps
        assert long_window is not None and long_window[0] == DEFAULT_SPACE.timesteps[1] + 1

    def test_the_long_window_witness_runs_equal(self):
        arm_id, seed = LONG_WINDOW_WITNESS
        verdict = run_instance(seed, arm_id)
        assert verdict.outcome == EQUAL and verdict.semantics == SYNCHRONIZED_LIF, verdict.to_dict()


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
        CL1: G5's executed-latency assertion refuses it by name; with that refusal switched off the core runs
        at the latest resident's latency and the base's counterexample returns (the assertion is load-bearing)."""
        instance, arm = _coresident_alap(), ARMS[ON_VALUES]
        declaration = {"cores": [{"max_axons": 8, "max_neurons": 2, "count": 5}], "allow_coalescing": False,
                       "allow_neuron_splitting": False, "allow_scheduling": False, "max_schedule_passes": 1,
                       "core_value_granularity": {}}
        ref_ir, _ref_map, ref_nodes = reference_nodes(instance, arm)
        constraints = platform_constraints(declaration, hardware_bias=True)
        with arm.engaged():
            with pytest.raises(PassTimingError, match="one hard core starts at one latency"):
                deployment(instance, constraints, 0, arm, relabel=False)
            with one_latency_unrefused():
                ir, mapping = deployment(instance, constraints, 0, arm, relabel=False)
                windows, mismatches, first = compare(ref_ir, ref_nodes, ir, _run(mapping, instance, arm))
        assert (windows, mismatches) == (12, 2), first
