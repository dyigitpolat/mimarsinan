"""The TTFS executor's membrane lattice, decided per neuron from that neuron's own inputs (closure fix X1, X2, P1 2).

Under quantized TTFS and cascaded ``ttfs_cycle_based`` on an integral chip, a neuron whose every nonzero-weight input
is a staircase level ``j/S``, a midpoint ``(2k+1)/(2S)`` or the always-on row has its exact membrane on the
``1/(2S)`` grid, whatever order the dot product sums in; the executor computes that membrane in integers and decides
its staircase exactly. Every other neuron (one reading an input off that grid, such as a raw input of 8-bit codes
k/255 in float64) takes the float membrane of ``float_membrane``, a fixed function of its own nonzero terms (closure
targeted pass, verify R1), so its staircase too is the same in every packing. So a deployment that reorders a sum (the per-axon split's W+/W- rows, a co-resident soft core at an axon offset)
gives every neuron the identity program's output at atol 0, also at a window S that is not a power of two, and an
exact staircase tie is decided as exact arithmetic decides it.

The vehicles are the exactness review's reproducers (``RA/11_wave2/closure/review_exactness/scripts``): B's dense
chains (programs 22 and 51, the pinned exact ties), G's chains on a continuous input (an offload-placed first segment)
and D's forced co-residency on per-synapse cores; and the verify pass's rational raw-input probe
(``RA/11_wave2/closure/verify/scripts/probe_rational_raw_input.py``), whose witnesses hold an exact rational tie.
"""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

import mimarsinan.chip_simulation.ttfs.ttfs_segment as ttfs_segment
from mimarsinan.certification.randomized.arms import ARMS, ON_TTFS_OFFLOAD
from mimarsinan.certification.randomized.instance import CASCADED_TTFS, SYNCHRONIZED_TTFS
from mimarsinan.certification.randomized.runner import COUNTEREXAMPLE, EQUAL, instance_for, run_instance
from mimarsinan.chip_simulation.ttfs.ttfs_executor import (
    _RAW_INPUT_NODE_ID,
    run_ttfs_contract_neural_stage,
    run_ttfs_hybrid_contract,
)
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.platform.platform_constraints import PER_AXON
import mimarsinan.models.spiking.wire_semantics as wire

MODES = (("ttfs_quantized", None), ("ttfs_cycle_based", "cascaded"))


def _chain(rng):
    """Reproducer B's generator: a chain of 2 to 3 dense cores on the 4-bit symmetric grid with the bias row on."""
    dims = [int(rng.integers(64, 128))] + [int(rng.integers(16, 128)) for _ in range(int(rng.integers(1, 3)))]
    dims.append(int(rng.integers(4, 32)))
    nodes, prev = [], -2
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        w = rng.integers(-7, 8, size=(a + 1, b)).astype(np.float64)
        src = np.array([IRSource(node_id=prev, index=j) for j in range(a)] + [IRSource(node_id=-3, index=0)],
                       dtype=object)
        nodes.append(NeuralCore(id=i, name=f"l{i}", input_sources=src, core_matrix=w,
                                threshold=float(rng.integers(8, 64)), latency=i))
        prev = i
    out = np.array([IRSource(node_id=prev, index=j) for j in range(dims[-1])], dtype=object)
    return IRGraph(nodes=nodes, output_sources=out), dims


def _cores(per_axon, *, axons=256, count=8):
    core = {"max_axons": axons, "max_neurons": axons, "count": count, "has_bias": False}
    if per_axon:
        core["weight_sign_granularity"] = PER_AXON
    return [core]


def _layout_order(x, w, bias):
    """The step 3 float membrane: the dot product in the hard core's own axon order (the necessity controls)."""
    v = np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64).T
    return v if bias is None else v + bias


def _segment_outputs(mapping, x, S, mode, schedule):
    run = run_ttfs_hybrid_contract(mapping, x, simulation_length=S, spiking_mode=mode, ttfs_cycle_schedule=schedule)
    return {k: np.asarray(v.seg_output) for k, v in run.record.segments.items()}


def _assert_equal_runs(a, b, x, S):
    for mode, schedule in MODES:
        want, got = _segment_outputs(a, x, S, mode, schedule), _segment_outputs(b, x, S, mode, schedule)
        assert want.keys() == got.keys()
        for k in want:
            np.testing.assert_array_equal(want[k], got[k], err_msg=f"{mode}/{schedule} segment {k}")


def _program_b(index):
    """Program ``index`` of reproducer B (RNG 20261023): its IR, S and midpoint input, the stream consumed in B's order."""
    rng = np.random.default_rng(20261023)
    for p in range(index + 1):
        ir, dims = _chain(rng)
        S = int(rng.choice([4, 5, 7, 8, 12]))
        rng.random((1, dims[0]))
        k = rng.integers(0, S, (1, dims[0]))
        if p == index:
            return ir, S, (k + 0.5) / S, k
    raise AssertionError


def _exact_chain(ir, S, k):
    """Every layer's exact staircase output (Fractions) on the midpoint input ``(k + 1/2) / S``, and the exact ties."""
    x = [Fraction(2 * int(v) + 1, 2 * S) for v in k[0]]
    ties = []
    for node in ir.nodes:
        w = np.asarray(node.core_matrix)
        theta = Fraction(int(node.threshold))
        inputs = x + [Fraction(1)]
        out = []
        for j in range(w.shape[1]):
            v = sum((Fraction(int(w[r, j])) * inputs[r] for r in range(w.shape[0])), Fraction(0))
            ladder = S * (1 - v / theta)
            if ladder.denominator == 1 and 0 < ladder < S:
                ties.append((int(node.id), j))
            k_fire = math.ceil(ladder)
            out.append(Fraction(S - min(max(k_fire, 0), S - 1), S) if k_fire < S else Fraction(0))
        x = out
    return x, ties


class TestThePinnedTies:
    """Reproducer F: programs 22 and 51 of B, each holding an exact staircase tie at S 7 that float64 rounding
    decided one way in the identity program and the other in the per-axon deployment."""

    @pytest.mark.parametrize("program", [22, 51])
    def test_the_identity_program_and_both_deployments_equal_the_exact_staircase(self, program):
        ir, S, x, k = _program_b(program)
        assert S == 7
        exact, ties = _exact_chain(ir, S, k)
        assert ties, "the vehicle holds an exact staircase tie"
        identity = build_identity_hybrid_mapping(ir_graph=ir)
        per_axon = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=_cores(True))
        per_synapse = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=_cores(False))
        _assert_equal_runs(identity, per_axon, x, S)
        _assert_equal_runs(identity, per_synapse, x, S)
        want = np.array([float(v) for v in exact])
        for mode, schedule in MODES:
            (out,) = _segment_outputs(identity, x, S, mode, schedule).values()
            np.testing.assert_array_equal(out, want, err_msg=f"{mode}/{schedule}")


def _program_g(seed, index):
    """Program ``index`` of reproducer G's shard ``seed``: its IR, S and continuous raw input on [0, 1)."""
    rng = np.random.default_rng(seed)
    for p in range(index + 1):
        ir, dims = _chain(rng)
        S = int(rng.choice([4, 5, 7, 8, 12, 16, 32]))
        x = rng.random((1, dims[0]))
        if p == index:
            return ir, S, x
    raise AssertionError


#: Reproducer G's per-axon flips (shards 7000 and 7003): a first segment that reads a continuous raw input, as an
#: offload-placed encoding layer does, at S 5, 7 and 12.
G_FLIPS = [(7000, 9), (7000, 21), (7000, 45), (7000, 74), (7000, 79), (7003, 52), (7003, 58), (7003, 90),
           (7003, 124)]


class TestAnOffloadPlacedFirstSegment:
    @pytest.mark.parametrize("seed,program", G_FLIPS)
    def test_per_axon_and_per_synapse_deployments_equal_the_identity_program(self, seed, program):
        ir, S, x = _program_g(seed, program)
        assert S & (S - 1), "a window that is not a power of two"
        identity = build_identity_hybrid_mapping(ir_graph=ir)
        for per_axon in (True, False):
            _assert_equal_runs(identity, build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=_cores(per_axon)),
                               x, S)


def _siblings(rng):
    """Reproducer D: two sibling output tiles of one perceptron on the raw input (one residency class) and a reader."""
    d0, da, db, dc = (int(rng.integers(40, 120)), int(rng.integers(8, 60)), int(rng.integers(8, 60)),
                      int(rng.integers(4, 24)))
    raw = [IRSource(node_id=-2, index=j) for j in range(d0)] + [IRSource(node_id=-3, index=0)]
    a = NeuralCore(id=0, name="a", input_sources=np.array(raw, dtype=object),
                   core_matrix=rng.integers(-7, 8, (d0 + 1, da)).astype(np.float64),
                   threshold=float(rng.integers(8, 64)), latency=0)
    b = NeuralCore(id=1, name="b", input_sources=np.array(list(raw), dtype=object),
                   core_matrix=rng.integers(-7, 8, (d0 + 1, db)).astype(np.float64),
                   threshold=float(rng.integers(8, 64)), latency=0)
    src = ([IRSource(node_id=0, index=j) for j in range(da)] + [IRSource(node_id=1, index=j) for j in range(db)]
           + [IRSource(node_id=-3, index=0)])
    a.perceptron_index = b.perceptron_index = 0
    b.threshold = a.threshold
    c = NeuralCore(id=2, name="c", input_sources=np.array(src, dtype=object),
                   core_matrix=rng.integers(-7, 8, (da + db + 1, dc)).astype(np.float64),
                   threshold=float(rng.integers(8, 64)), latency=1)
    c.perceptron_index = 1
    out = np.array([IRSource(node_id=2, index=j) for j in range(dc)], dtype=object)
    return IRGraph(nodes=[a, b, c], output_sources=out), d0


def _program_d(index):
    rng = np.random.default_rng(7)
    for p in range(index + 1):
        ir, d0 = _siblings(rng)
        S = int(rng.choice([5, 7]))
        x = (rng.integers(0, S, (1, d0)) + 0.5) / S
        if p == index:
            return ir, S, x
    raise AssertionError


def _coresident_at_offset(mapping):
    return any(len(row) > 1 and any(int(p["axon_offset"]) > 0 for p in row)
               for stage in mapping.stages if stage.kind == "neural"
               for row in stage.hard_core_mapping.soft_core_placements_per_hard_core)


#: Reproducer D at two hard cores (POOL_COUNT 2), on the closure tree before this fix: programs 19 and 68 differed
#: from the identity program on per-synapse and per-axon cores, 55 and 103 on per-axon cores (the review's 2 of 150
#: and 13 of 150; ``closure/fix_round/logs/find_coresident_hits_before.out``).
D_FLIPS = [19, 55, 68, 103]


class TestCoResidencyOnPerSynapseCores:
    @pytest.mark.parametrize("program", D_FLIPS)
    def test_a_soft_core_at_an_axon_offset_runs_as_the_identity_program(self, program):
        ir, S, x = _program_d(program)
        identity = build_identity_hybrid_mapping(ir_graph=ir)
        for per_axon in (False, True):
            deployed = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=_cores(per_axon, axons=512, count=2))
            assert _coresident_at_offset(deployed)
            _assert_equal_runs(identity, deployed, x, S)


class TestTheLatticeStaircase:
    """The exact staircase on the membrane lattice agrees with the float staircase wherever float decides right,
    and decides an exact tie as exact arithmetic does."""

    @pytest.mark.parametrize("half_step", [False, True])
    @pytest.mark.parametrize("S", [1, 2, 3, 5, 7, 8, 12])
    def test_every_lattice_membrane_against_exact_arithmetic(self, S, half_step):
        for theta in (1, 2, 3, 5, 7, 16, 57):
            m = np.arange(-4 * S * theta, 4 * S * theta + 1, dtype=np.float64)
            got = wire.ttfs_quantized_staircase_lattice_np(m, np.float64(theta), S, comparator_half_step=half_step)
            for mi, gi in zip(m.tolist(), got.tolist()):
                ladder = S * (1 - Fraction(int(mi), 2 * S) / theta) - (Fraction(1, 2) if half_step else 0)
                k_fire = math.ceil(ladder)
                want = Fraction(S - min(max(k_fire, 0), S - 1), S) if k_fire < S else Fraction(0)
                assert gi == float(want), (S, theta, mi, half_step)

    def test_it_equals_the_float_staircase_off_the_ties(self):
        rng = np.random.default_rng(5)
        for S in (3, 5, 7, 12, 64):
            theta = float(rng.integers(1, 60))
            m = rng.integers(-2000, 2000, 400).astype(np.float64)
            ladder = S * (1 - (m / (2 * S)) / theta)
            off_tie = np.abs(ladder - np.round(ladder)) > 1e-9
            lattice = wire.ttfs_quantized_staircase_lattice_np(m, np.float64(theta), S)
            floating = wire.ttfs_quantized_staircase_np(m / (2 * S), theta, S)
            np.testing.assert_array_equal(lattice[off_tie], floating[off_tie])


class TestARandomStratumWithoutTheHostEncoder:
    """The class the X3a strata could not see (every X3a TTFS program begins with a host encoder stage): B's chain
    generator on its own stream, the first segment reading the raw input itself (midpoint and continuous), windows
    S that are not powers of two, per-axon and per-synapse deployments, every segment output at atol 0. The
    midpoint draws hold exact staircase ties (counted in exact arithmetic), so the stratum is not vacuous."""

    def test_every_deployment_equals_the_identity_program(self):
        rng = np.random.default_rng(20261024)
        ties = 0
        for _program in range(24):
            ir, dims = _chain(rng)
            S = int(rng.choice([3, 5, 6, 7, 9, 10, 12]))
            k = rng.integers(0, S, (1, dims[0]))
            continuous = rng.random((1, dims[0]))
            ties += len(_exact_chain(ir, S, k)[1])
            identity = build_identity_hybrid_mapping(ir_graph=ir)
            for per_axon in (True, False):
                deployed = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=_cores(per_axon))
                for x in ((k + 0.5) / S, continuous):
                    _assert_equal_runs(identity, deployed, x, S)
        assert ties > 0


class TestTheX3aStratumWithTheEncoderOnChip:
    """[X1] The X3a arm ``on_ttfs_offload``: synchronized and cascaded TTFS with the identity encoder on chip, so the
    first segment reads the raw midpoint input, at T in 3-12, half the draws per-axon. Every admitted deployment
    equals the identity program at atol 0; with the lattice switched off and the float membrane summed in the layout's
    own order (every neuron on the step 3 float path, the class a segment entry off the 1/S grid ran under before
    this fix) the stratum finds it: seeds 0-150 admit 86, 0 counterexamples on, 11 off
    (``closure/fix_round/probes/ttfs_offload_stratum_0_150.out``). Either rule alone keeps the teeth seeds equal
    (``closure/targeted/logs/teeth_controls.out``): the float membrane of R1 is itself the same in every packing."""

    TEETH = (11, 14, 46, 50)

    def test_every_admitted_deployment_equals_the_identity_program(self):
        verdicts = [run_instance(seed, ON_TTFS_OFFLOAD) for seed in range(40)]
        admitted = [v for v in verdicts if v.outcome in (EQUAL, COUNTEREXAMPLE)]
        assert len(admitted) >= 15 and all(v.outcome == EQUAL for v in admitted), [v.seed for v in admitted
                                                                                    if v.outcome != EQUAL]
        assert {v.semantics for v in admitted} == {SYNCHRONIZED_TTFS, CASCADED_TTFS}
        windows = {instance_for(v.seed, ARMS[ON_TTFS_OFFLOAD]).timesteps for v in admitted}
        assert any(t & (t - 1) for t in windows), windows

    @pytest.mark.parametrize("seed", TEETH)
    def test_on_the_step_3_float_path_the_stratum_finds_the_class(self, seed, monkeypatch):
        assert run_instance(seed, ON_TTFS_OFFLOAD).outcome == EQUAL
        monkeypatch.setattr(ttfs_segment, "lattice_membrane", lambda *args, **kwargs: None)
        assert run_instance(seed, ON_TTFS_OFFLOAD).outcome == EQUAL, "the float membrane alone keeps it equal"
        monkeypatch.setattr(ttfs_segment, "float_membrane", _layout_order)
        assert run_instance(seed, ON_TTFS_OFFLOAD).outcome == COUNTEREXAMPLE


def _program_r(seed, index):
    """Program ``index`` of the verify pass's rational raw-input probe on RNG ``seed``: B's chain, S in {3, 5, 15, 17}
    and the 8-bit codes ``k`` of the raw input, the stream consumed in the probe's order."""
    rng = np.random.default_rng(seed)
    for p in range(index + 1):
        ir, dims = _chain(rng)
        S = int(rng.choice([3, 5, 15, 17]))
        k = rng.integers(0, 256, size=(1, dims[0]))
        if p == index:
            return ir, S, k
    raise AssertionError


def _rational_ties(ir, k, S):
    """Neurons of the first IR core whose exact membrane on the intended inputs ``k/255`` (and the bias row) sits on
    a staircase step: ``S (1 - V/theta)`` an integer in ``[0, S]``."""
    core = ir.nodes[0]
    w = np.asarray(core.core_matrix)
    theta = Fraction(int(core.threshold))
    inputs = [Fraction(int(v), 255) for v in k[0]] + [Fraction(1)]
    ties = []
    for n in range(w.shape[1]):
        v = sum((Fraction(int(w[r, n])) * inputs[r] for r in range(w.shape[0]) if w[r, n] != 0), Fraction(0))
        ladder = S * (1 - v / theta)
        if ladder.denominator == 1 and 0 <= ladder <= S:
            ties.append(n)
    return ties


def _first_core_membrane(mapping, x, S, neuron):
    """The float64 membrane of neuron ``neuron`` of IR core 0 in ``mapping``'s first segment under ``ttfs_quantized``."""
    stage = next(s for s in mapping.stages if s.kind == "neural")
    run = run_ttfs_contract_neural_stage(mapping, stage, 0, {_RAW_INPUT_NODE_ID: x}, simulation_length=S,
                                         spiking_mode="ttfs_quantized")
    for hc, row in enumerate(stage.hard_core_mapping.soft_core_placements_per_hard_core):
        for placement in row:
            if int(placement["ir_node_id"]) == 0 and neuron < int(placement["neurons"]):
                return float(run.membrane_voltages[hc][0, int(placement["neuron_offset"]) + neuron])
    raise AssertionError("IR core 0 is not placed whole on one hard core")


#: The verify pass's witnesses (``RA/11_wave2/closure/verify/logs/adversarial/witness_*.json``): (RNG seed, program,
#: S, a first-core neuron whose exact membrane on k/255 is a staircase step). On the fix round's tree the float64
#: inputs put each 1e-16 level units off its step, inside the summation error, and the per-axon deployment emitted
#: another level than the identity program under ttfs_quantized and cascaded ttfs_cycle_based.
R1_WITNESSES = [(8001, 133, 15, 4), (8001, 251, 17, 26), (8001, 251, 17, 70), (8000, 262, 17, 31)]


def _codes_as(k, dtype):
    return k / 255.0 if dtype == "float64" else (k.astype(np.float32) / np.float32(255.0)).astype(np.float64)


class TestTheRationalRawInputWitnesses:
    """[R1] A first segment reading a raw input of 8-bit codes k/255, off the 1/(2S) grid, so its neurons keep the
    float path. Their float membrane is a fixed function of their own terms, so the per-axon split's W+/W- rows give
    the identity program's membrane bit for bit, and every staircase output with it, in float64 and float32."""

    @pytest.mark.parametrize("seed,program,S,neuron", R1_WITNESSES)
    def test_every_deployment_equals_the_identity_program(self, seed, program, S, neuron):
        ir, s, k = _program_r(seed, program)
        assert s == S and neuron in _rational_ties(ir, k, S), "the vehicle holds the exact rational tie"
        identity = build_identity_hybrid_mapping(ir_graph=ir)
        for per_axon in (True, False):
            deployed = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=_cores(per_axon))
            for dtype in ("float64", "float32"):
                _assert_equal_runs(identity, deployed, _codes_as(k, dtype), S)

    @pytest.mark.parametrize("seed,program,S,neuron", R1_WITNESSES)
    def test_the_split_moves_no_bit_of_the_witness_membrane(self, seed, program, S, neuron):
        ir, _, k = _program_r(seed, program)
        x = _codes_as(k, "float64")
        want = _first_core_membrane(build_identity_hybrid_mapping(ir_graph=ir), x, S, neuron)
        for per_axon in (True, False):
            got = _first_core_membrane(build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=_cores(per_axon)),
                                       x, S, neuron)
            assert got.hex() == want.hex(), (per_axon, got.hex(), want.hex())

    @pytest.mark.parametrize("seed,program,S,neuron", R1_WITNESSES[::2])
    def test_with_the_summation_order_of_the_layout_the_witness_differs(self, seed, program, S, neuron, monkeypatch):
        """The necessity witness: the float membrane summed in the hard core's own axon order (the fix round's
        rule) gives the per-axon deployment another output than the identity program."""
        monkeypatch.setattr(ttfs_segment, "float_membrane", _layout_order)
        ir, _, k = _program_r(seed, program)
        x = _codes_as(k, "float64")
        identity = build_identity_hybrid_mapping(ir_graph=ir)
        deployed = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=_cores(True))
        with pytest.raises(AssertionError):
            _assert_equal_runs(identity, deployed, x, S)


class TestARationalRawInputStratum:
    """[R1] The verify probe's class on its own stream: B's chains on raw 8-bit codes k/255 in float64 and float32, S
    in {3, 5, 15, 17}, per-axon and per-synapse deployments, every segment output at atol 0. Exact rational ties
    on the first core are counted, so the stratum is not vacuous."""

    def test_every_deployment_equals_the_identity_program(self):
        rng = np.random.default_rng(20261025)
        ties = 0
        for _program in range(30):
            ir, dims = _chain(rng)
            S = int(rng.choice([3, 5, 15, 17]))
            k = rng.integers(0, 256, size=(1, dims[0]))
            ties += len(_rational_ties(ir, k, S))
            identity = build_identity_hybrid_mapping(ir_graph=ir)
            for per_axon in (True, False):
                deployed = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=_cores(per_axon))
                for dtype in ("float64", "float32"):
                    _assert_equal_runs(identity, deployed, _codes_as(k, dtype), S)
        assert ties > 0
