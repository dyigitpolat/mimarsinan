"""The code-level runner: identity reference against one drawn deployment, per neuron and window."""

from __future__ import annotations

import functools
import math

import pytest
import torch

import mimarsinan.certification.randomized.executors as ex
import mimarsinan.certification.randomized.network as network
import mimarsinan.certification.randomized.program as program
from mimarsinan.certification.randomized.arms import ARMS
from mimarsinan.certification.randomized.cost_draws import PROGRAM_CHANGING
from mimarsinan.certification.randomized.instance import (
    CASCADED_TTFS,
    HOST,
    NEURAL,
    STREAMED_LIF,
    SYNCHRONIZED_LIF,
    SYNCHRONIZED_TTFS,
    VALUE,
    WINDOWED_LIF,
    Instance,
    Layer,
    Network,
)
from mimarsinan.certification.randomized.runner import (
    EQUAL,
    OUTCOMES,
    REFUSED,
    REFUSED_UNNAMED,
    compare,
    deployment,
    instance_for,
    network_relation,
    reference,
    reference_nodes,
    run_instance,
)
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping

ON = ARMS["on"]
SPIKING = (STREAMED_LIF, WINDOWED_LIF, SYNCHRONIZED_LIF, CASCADED_TTFS, SYNCHRONIZED_TTFS)


def _chain(semantics: str = STREAMED_LIF, *, host: bool = False) -> Instance:
    """input(4) -> a(6) [-> host] -> b(3): integer weights, mid-range thresholds, discriminating inputs."""
    a = Layer("a", NEURAL, ("input",), 6, weights=tuple(tuple((r + c) % 5 - 1 for c in range(4)) for r in range(6)),
              bias=(1, 0, -1, 0, 1, 0), theta=2)
    layers = [a]
    if host:
        layers.append(Layer("h", HOST, ("a",), 6, op="relu" if semantics == VALUE else "identity"))
    b = Layer("b", NEURAL, (layers[-1].name,), 3,
              weights=tuple(tuple((r * c) % 4 - 1 for c in range(6)) for r in range(3)), bias=(0, 1, 0), theta=3)
    inputs = ((4, 0, 2, 1), (1, 3, 4, 0), (2, 2, 2, 2), (0, 4, 1, 3))
    return Instance(seed=0, semantics=semantics, timesteps=4, hardware_bias=False,
                    network=Network(4, tuple(layers) + (b,), ("a", "b")), inputs=inputs)


def _constraints(count=8, axons=16, neurons=8, **flags):
    out = {"cores": [{"max_axons": axons, "max_neurons": neurons, "count": count, "has_bias": False}],
           "allow_coalescing": False, "allow_neuron_splitting": False, "allow_scheduling": False,
           "max_schedule_passes": 8}
    out.update(flags)
    return out


class TestTheComparison:
    @pytest.mark.parametrize("semantics", [STREAMED_LIF, VALUE])
    def test_a_packed_chain_equals_its_reference(self, semantics):
        instance = _chain(semantics)
        ref_ir, _m, ref_nodes = reference_nodes(instance, ON)
        ir, mapping = deployment(instance, _constraints(neurons=4, allow_neuron_splitting=True), 3, ON)
        windows, mismatches, first = compare(ref_ir, ref_nodes, ir, ex.run_nodes(mapping, instance))
        assert (windows, mismatches, first) == (4 * 9, 0, None)

    def test_the_reference_discriminates(self):
        """Distinct rows, and counts strictly inside the window somewhere: no saturated or silent vehicle."""
        _ir, _m, nodes = reference_nodes(_chain(), ON)
        out = nodes[-1]
        assert len({tuple(r.tolist()) for r in out}) == 4
        assert bool(((out > 0) & (out < 4)).any()) and bool((out == 0).any())

    def test_one_altered_neuron_is_found(self):
        instance = _chain()
        ref_ir, _m, ref_nodes = reference_nodes(instance, ON)
        ir, mapping = deployment(instance, _constraints(), 3, ON)
        nodes = ex.run_nodes(mapping, instance)
        b_node = next(n for n in ir.get_neural_cores() if n.perceptron_index == 2)
        nodes[int(b_node.id)] = nodes[int(b_node.id)].clone()
        nodes[int(b_node.id)][2, 1] += 1
        assert compare(ref_ir, ref_nodes, ir, nodes)[1:] == (1, (2, 2, 1))

    def test_an_unreported_neuron_is_a_coverage_error_never_a_pass(self):
        instance = _chain()
        ref_ir, _m, ref_nodes = reference_nodes(instance, ON)
        ir, mapping = deployment(instance, _constraints(neurons=4, allow_neuron_splitting=True), 3, ON)
        nodes = ex.run_nodes(mapping, instance)
        tile = next(n for n in ir.get_neural_cores() if n.perceptron_index == 1)
        del nodes[int(tile.id)]
        with pytest.raises(ex.CoverageError):
            compare(ref_ir, ref_nodes, ir, nodes)


class TestRelabelling:
    @pytest.mark.parametrize("seed", range(5))
    def test_a_relabelled_ir_is_the_same_program(self, seed):
        instance = _chain()
        mapped = program.mapper_graph(instance)
        ir = program.integer_ir(mapped, max_axons=16, max_neurons=4, allow_coalescing=False, hardware_bias=False)
        before = ex.per_neuron(ir, ex.run_nodes(build_identity_hybrid_mapping(ir_graph=ir), instance))
        id_map = program.relabel(ir, seed)
        assert sorted(id_map) == sorted(id_map.values())
        after = ex.per_neuron(ir, ex.run_nodes(build_identity_hybrid_mapping(ir_graph=ir), instance))
        assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)


class TestVerdicts:
    def test_an_infeasible_wide_fan_in_is_a_refusal(self):
        instance = _chain()
        with pytest.raises(ValueError, match="fan-in"):
            deployment(instance, _constraints(axons=3), 0, ON)

    @pytest.mark.parametrize("seed", range(12))
    def test_every_verdict_is_classified_and_draws_only_program_changing_paths(self, seed):
        verdict = run_instance(seed, "on")
        assert verdict.outcome in OUTCOMES
        assert set(verdict.drawn) <= set(PROGRAM_CHANGING)
        if verdict.outcome in (REFUSED, REFUSED_UNNAMED):
            assert verdict.detail
        if verdict.outcome == EQUAL:
            assert verdict.windows > 0 and verdict.active > 0

    def test_a_verdict_is_a_function_of_its_seed(self):
        first, again = run_instance(4, "on_scheduled").to_dict(), run_instance(4, "on_scheduled").to_dict()
        first.pop("wall_s"), again.pop("wall_s")
        assert first == again


def _eye(semantics: str, T: int, counts, *, theta: int = 1, weights=None) -> Instance:
    """One neural layer over ``len(counts)`` input wires (the identity at theta 1 unless ``weights``)."""
    n = len(counts)
    rows = weights if weights is not None else tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
    layer = Layer("n0", NEURAL, ("input",), len(rows), weights=tuple(tuple(r) for r in rows),
                  bias=tuple(0 for _ in rows), theta=theta)
    return Instance(seed=0, semantics=semantics, timesteps=T, hardware_bias=True,
                    network=Network(n, (layer,), ("n0",)), inputs=(tuple(counts),))


def _float_staircase_departs(semantics: str, T: int, k: int) -> bool:
    """Whether the base's floating-point staircase departs from N on an identity cell (theta 1, one input count
    ``k``): the synchronized LIF count staircase ``floor(T ((Q / T) / theta))`` and the synchronized TTFS ladder
    ``ceil(S (1 - V / theta))`` on the 1/(2S)-snapped level (``models/spiking/wire_semantics.py``). The other
    spiking semantics count spikes and take no quotient."""
    if semantics == SYNCHRONIZED_LIF:
        c = math.floor(T * ((k / T) / 1.0))
        return (1.0 * c / T) * T / 1.0 != k
    if semantics == SYNCHRONIZED_TTFS:
        v = round(k / T * (2 * T)) / (2 * T)
        fire = math.ceil(T * (1.0 - v / 1.0))
        level = (T - min(max(fire, 0), T - 1)) / T if fire < T else 0.0
        return level != k / T
    return False


_IDENTITY_CELLS = [
    pytest.param(semantics, T, k, marks=pytest.mark.xfail(
        strict=True, reason="H8-staircase: the base's floating-point staircase loses this exact tie "
                            "(wire_semantics.py, owner to be named by the conductor)"))
    if _float_staircase_departs(semantics, T, k) else (semantics, T, k)
    for semantics in SPIKING for T in range(1, 9) for k in range(T + 1)]


@functools.lru_cache(maxsize=None)
def _identity_layer(semantics: str, T: int):
    """``(reported, N)`` per declared count ``0..T`` for the identity layer at theta 1, one run per window."""
    instance = _eye(semantics, T, range(T + 1))
    ref = reference(instance, ON)
    got = ex.per_neuron(ref.ir, ref.nodes)[ref.perceptron_of["n0"]].tolist()[0]
    want = network.observable(instance, network.evaluate(instance)["n0"]).tolist()[0]
    return tuple(got), tuple(want)


class TestTheInputPath:
    """The executors run the inputs the document declares: a count k reaches the program as k spikes."""

    @pytest.mark.parametrize("semantics", SPIKING)
    @pytest.mark.parametrize("T", range(1, 9))
    def test_the_identity_programs_encoder_emits_every_declared_count(self, semantics, T):
        instance = _eye(semantics, T, range(T + 1))
        ref = reference(instance, ON)
        assert ex.encoder_counts(ref.mapping, instance).tolist() == [list(range(T + 1))]

    @pytest.mark.parametrize("semantics, T, k", _IDENTITY_CELLS)
    def test_every_spiking_identity_layer_reports_every_declared_count(self, semantics, T, k):
        """End to end at theta 1 (one hop, the identity weights): the executor that defines y reports N's count
        (or level) for every count k <= T <= 8. The cells where the base's floating-point staircase departs from
        N are strict xfails (H8-staircase's base violation; an owner's integer staircase flips them)."""
        got, want = _identity_layer(semantics, T)
        assert got[k] == want[k], (semantics, T, k, got, want)


class TestTheNetworkRelation:
    """Proposition 2's relation, measured per instance: the identity program against the integer network N."""

    def test_n_is_the_integer_staircase_and_the_affine_law(self):
        lif = network.evaluate(_eye(STREAMED_LIF, 5, (3, 5), theta=3, weights=((1, 0), (2, 1))))
        assert lif["n0"].tolist() == [[1, 3]]  # floor(3 / 3), floor(11 / 3)
        value = network.evaluate(_eye(VALUE, 1, (3, -2), weights=((1, 0), (2, 1))))
        assert value["n0"].tolist() == [[3, 4]]
        summed = network.evaluate(Instance(
            seed=0, semantics=STREAMED_LIF, timesteps=4, hardware_bias=True,
            network=Network(2, (Layer("h", HOST, ("input", "input"), 2, op="sum"),
                                Layer("n0", NEURAL, ("h",), 2, weights=((1, 0), (0, 1)), bias=(0, 0))),
                            ("n0",)),
            inputs=((3, 1),)))
        assert summed["h"].tolist() == [[4, 2]]  # min(3 + 3, 4), 1 + 1

    @pytest.mark.parametrize("semantics, mismatches", [
        (STREAMED_LIF, 0), (WINDOWED_LIF, 0), (SYNCHRONIZED_LIF, 2), (SYNCHRONIZED_TTFS, 1)])
    def test_an_exact_tie_witnesses_the_synchronized_staircase_defect(self, semantics, mismatches):
        """One input count 3 at T = 5 into two neurons of theta 3 with weights 1 and 2: Q = 3 and 6, and N
        gives 1 and 2. The synchronized LIF count staircase floor(T ((Q / T) / theta)) loses both exact ties
        (floor(0.999...)), the TTFS ladder ceil(S (1 - V / theta)) loses Q = 6 (level 1 for 2); streamed and
        windowed LIF count spikes and keep both (wire_semantics.py, a base defect stated as Proposition 2's
        boundary). Re-pin when an owner computes the staircase on the integer charge."""
        instance = _eye(semantics, 5, (3,), theta=3, weights=((1,), (2,)))
        claim, windows, found, _first = network_relation(instance, reference(instance, ON))
        assert (windows, found) == (2, mismatches)
        assert claim == (network.CLAIMED if semantics in (SYNCHRONIZED_LIF, SYNCHRONIZED_TTFS)
                         else network.CONDITIONAL)

    @pytest.mark.parametrize("seed", range(8))
    def test_every_verdict_records_the_relation_and_its_claim(self, seed):
        verdict = run_instance(seed, "on")
        assert verdict.network_claim == network.claim(verdict.semantics)
        compared = verdict.network_claim != network.NOT_CLAIMED
        assert (verdict.network_windows > 0) == compared
        assert 0 <= verdict.network_mismatches <= verdict.network_windows
        assert len(verdict.weight_ranges) >= 2 and 0.0 <= verdict.input_dependent <= 1.0

    def test_cascaded_ttfs_is_outside_the_relation(self):
        instance = _eye(CASCADED_TTFS, 5, (5,), theta=3, weights=((1,),))
        assert network_relation(instance, reference(instance, ON)) == (network.NOT_CLAIMED, 0, 0, None)


class TestStrata:
    def test_the_hypothesis_on_strata_draw_disjoint_networks(self):
        """on, on_scheduled and on_values_classes share a seed range but not a network (independent trials)."""
        for seed in range(20):
            nets = [instance_for(seed, ARMS[a]).network for a in ("on", "on_scheduled", "on_values_classes")]
            assert len({repr(n) for n in nets}) == 3, seed

    def test_an_unstreamed_arm_keeps_the_plain_seed(self):
        assert instance_for(7, ARMS["on"]).seed == 7 and instance_for(7, ARMS["on_scheduled"]).seed != 7

    def test_the_long_window_stratum_draws_windows_beyond_the_declared_range(self):
        """Targeted pass: a hypothesis-on stratum of scheduled synchronized deployments at T in 9-64, its own
        generator stream, so a defect that binds only beyond the declared T in 1-8 (the synchronized LIF count
        rebuild) is visible to the screen; every other arm keeps the declared range."""
        arm = ARMS["on_long_window"]
        assert arm.timesteps == (9, 64) and not arm.hypothesis and arm.stream == "on_long_window"
        windows = set()
        for seed in range(40):
            instance = instance_for(seed, arm)
            assert instance.semantics in (SYNCHRONIZED_LIF, SYNCHRONIZED_TTFS), seed
            assert 9 <= instance.timesteps <= 64 and all(0 <= k <= instance.timesteps
                                                        for row in instance.inputs for k in row), seed
            windows.add(instance.timesteps)
        assert len(windows) > 10
        assert all(instance_for(seed, ARMS[a]).timesteps <= 8 for seed in range(40)
                   for a in ("on", "on_scheduled", "on_values_classes", "R6"))
