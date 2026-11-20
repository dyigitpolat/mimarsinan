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
    COUNTEREXAMPLE,
    EQUAL,
    OUTCOMES,
    REFERENCE_ERROR,
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


_IDENTITY_CELLS = [(semantics, T, k) for semantics in SPIKING for T in range(1, 9) for k in range(T + 1)]
#: The cells where the base's floating-point staircase departed from N (the strict xfails before the CL1 fix round:
#: H8-staircase's base violation, fixed by the staircases on the integer charge, sign-off P1 (a)).
_FLOAT_DEPARTED = [cell for cell in _IDENTITY_CELLS if _float_staircase_departs(*cell)]


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
        (or level) for every count k <= T <= 8, the cells where the base's floating-point staircase departed from
        N included (``_FLOAT_DEPARTED``: strict xfails until the CL1 fix round's integer staircases)."""
        got, want = _identity_layer(semantics, T)
        assert got[k] == want[k], (semantics, T, k, got, want)

    def test_the_float_staircase_departed_on_a_cell_the_executor_now_reports(self):
        assert (SYNCHRONIZED_TTFS, 7, 6) in _FLOAT_DEPARTED


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
        (STREAMED_LIF, 0), (WINDOWED_LIF, 0), (SYNCHRONIZED_LIF, 0), (SYNCHRONIZED_TTFS, 0)])
    def test_an_exact_tie_keeps_the_synchronized_staircases_on_n(self, semantics, mismatches):
        """One input count 3 at T = 5 into two neurons of theta 3 with weights 1 and 2: Q = 3 and 6, and N
        gives 1 and 2. On the base the synchronized LIF count staircase floor(T ((Q / T) / theta)) lost both exact
        ties (floor(0.999...)) and the TTFS ladder ceil(S (1 - V / theta)) lost Q = 6 (level 1 for 2), 2 and 1
        mismatches; re-pinned in the CL1 fix round (sign-off P1 (a)), where both staircases decide on the integer
        charge (``lif_window_count``, ``ttfs_quantized_staircase_lattice_np``) and every semantics keeps both."""
        instance = _eye(semantics, 5, (3,), theta=3, weights=((1,), (2,)))
        claim, windows, found, _first = network_relation(instance, reference(instance, ON))
        assert (windows, found) == (2, mismatches)
        assert claim == (network.CLAIMED if semantics in (SYNCHRONIZED_LIF, SYNCHRONIZED_TTFS)
                         else network.CONDITIONAL)

    @pytest.mark.parametrize("seed", range(8))
    def test_every_verdict_records_the_relation_and_its_claim(self, seed, monkeypatch):
        """[U-TGAUGE6] A draw whose run reads the window is built through the mapping's window pass, as Soft Core
        Mapping builds it (``x3a_window``), so every verdict records the relation and its claim. Relay-free, seed 6's
        identity program holds a live edge at a latency gap other than one and is refused by name before any relation
        (``test_a_relay_free_gapped_draw_is_refused_at_its_identity_program``)."""
        from unit.certification.x3a_window import map_through_the_window, reads_the_window

        if reads_the_window(instance_for(seed, ON).semantics):
            map_through_the_window(monkeypatch)
        verdict = run_instance(seed, "on")
        assert verdict.outcome != REFERENCE_ERROR, verdict.to_dict()
        assert verdict.network_claim == network.claim(verdict.semantics)
        compared = verdict.network_claim != network.NOT_CLAIMED
        assert (verdict.network_windows > 0) == compared
        assert 0 <= verdict.network_mismatches <= verdict.network_windows
        assert len(verdict.weight_ranges) >= 2 and 0.0 <= verdict.input_dependent <= 1.0

    def test_a_relay_free_gapped_draw_is_refused_at_its_identity_program(self):
        """[U-TGAUGE5] The screen builds its identity program without relays where no window pass is armed: seed 6 (a
        per-cycle cascade holding a live edge at a latency gap other than one) is refused by name there."""
        verdict = run_instance(6, "on")
        assert verdict.outcome == REFERENCE_ERROR and verdict.detail.startswith("UnbalancedFanInError"), \
            verdict.to_dict()

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


def _value_net(layers, outputs, inputs) -> Instance:
    return Instance(seed=0, semantics=VALUE, timesteps=1, hardware_bias=False,
                    network=Network(4, tuple(layers), tuple(outputs)), inputs=tuple(inputs))


def _deep_value_chain() -> Instance:
    """input(4) -> a(4) -> b(4) -> c(2), every weight 8 at theta 1: a reaches 256, b 8,192, c 262,144 > 32,767."""
    def dense(name, source, width, fan_in):
        return Layer(name, NEURAL, (source,), width, weights=tuple((8,) * fan_in for _ in range(width)),
                     bias=(0,) * width, theta=1)

    return _value_net((dense("a", "input", 4, 4), dense("b", "a", 4, 4), dense("c", "b", 2, 4)), ("a", "b", "c"),
                      ((8, 8, 8, 8), (1, 0, 0, 0), (0, 2, 0, 1), (3, 0, 1, 0)))


class TestTheIntegerValueRegime:
    """CL1 (plan section 10.4 acceptance 7): the value semantics runs G2's integer readout law (E283) on every core,
    its node outputs compared at atol 0, and every value run against the int64 golden (``int64_golden``)."""

    def test_every_value_core_carries_the_law_and_every_host_fed_core_its_grid(self):
        ref = reference(_chain(VALUE, host=True), ON)
        cores = list(ref.ir.get_neural_cores())
        by_id = {int(n.id): n for n in ref.ir.nodes}
        assert cores and all(getattr(c, "readout_law", None) is not None for c in cores)
        for core in cores:
            fed = any(not isinstance(by_id.get(int(s.node_id)), type(core)) for s in core.input_sources.flatten()
                      if not (s.is_off() or s.is_always_on()))
            grid = getattr(core, "boundary_grid", None)
            assert (grid is not None and grid.armed and grid.step == 1.0) == fed, core.name

    def test_the_value_runs_equal_the_int64_golden_and_n_at_atol_0(self):
        instance = _deep_value_chain()
        ref = reference(instance, ON)
        x = ex.input_tensor(instance, torch.float64).numpy()
        assert program.golden_relation(ref.mapping, x, ref.nodes) == (4 * 10, 0)
        assert network_relation(instance, ref)[1:3] == (4 * 10, 0)
        ir, mapping = deployment(instance, _constraints(neurons=2, allow_neuron_splitting=True), 3, ON)
        nodes = ex.run_nodes(mapping, instance)
        assert program.golden_relation(mapping, x, nodes) == (4 * 10, 0)
        assert compare(ref.ir, ref.nodes, ir, nodes)[1] == 0

    def test_the_law_saturates_where_the_affine_core_would_not(self):
        """The grid binds: c's charge 262,144 clamps at 32,767 on the program and in N; the affine N departs."""
        instance = _deep_value_chain()
        levels = program.law_levels(program.VALUE_LAW)
        assert ex.RUNS[VALUE].law is program.VALUE_LAW
        ref = reference(instance, ON)
        got = ex.per_neuron(ref.ir, ref.nodes)[program.mapper_graph(instance).perceptron_of["c"]]
        assert got[0].tolist() == [float(levels)] * 2
        assert network.evaluate(instance, levels=levels)["c"][0].tolist() == [levels] * 2
        assert network.evaluate(instance)["c"][0].tolist() == [262144] * 2

    def test_a_core_reading_host_and_core_values_is_refused_by_name(self):
        """One DAC grid per core (G2's law grids): a core over concat(host ReLU, core) is refused, never run."""
        from mimarsinan.common.refusal import ToolchainRefusal
        from mimarsinan.mapping.export.law_grids import ReadoutLawGridError

        a = Layer("a", NEURAL, ("input",), 2, weights=((1, 0, 0, 0), (0, 1, 0, 0)), bias=(0, 0), theta=1)
        h = Layer("h", HOST, ("a",), 2, op="relu")
        b = Layer("b", NEURAL, ("h", "a"), 1, weights=((1, 1, 1, 1),), bias=(0,), theta=1)
        with pytest.raises(ReadoutLawGridError) as caught:
            reference(_value_net((a, h, b), ("a", "b"), ((1, 2, 3, 4),)), ON)
        assert isinstance(caught.value, ToolchainRefusal)

    def test_the_golden_edge_finds_an_executor_off_its_law(self, monkeypatch):
        """An executor whose law output is one code off on every core: identity and deployment still agree (both
        run it), and the int64 golden, which evaluates the stamped law itself, finds every output."""
        from mimarsinan.chip_simulation.value_run import value_execution

        instance = _deep_value_chain()
        law = value_execution.run_core_law
        monkeypatch.setattr(value_execution, "run_core_law",
                            lambda signal, prepared, weight=None: law(signal, prepared, weight) + 1.0)
        ref = reference(instance, ON)
        x = ex.input_tensor(instance, torch.float64).numpy()
        windows, mismatches = program.golden_relation(ref.mapping, x, ref.nodes)
        assert windows == 4 * 10 and mismatches == windows
        ir, mapping = deployment(instance, _constraints(neurons=2, allow_neuron_splitting=True), 3, ON)
        assert compare(ref.ir, ref.nodes, ir, ex.run_nodes(mapping, instance))[1] == 0

    def test_a_golden_mismatch_is_a_counterexample(self, monkeypatch):
        """[MN8] A defect the identity program and the deployment share (an executor one code off its law) makes no
        mismatch between them, and the int64 golden finds it: the verdict counts it as a counterexample of the
        line it runs on, attributed to the shared departure (never left to a line of its own)."""
        from mimarsinan.certification.randomized.attribution import attribute
        from mimarsinan.chip_simulation.value_run import value_execution

        clean = run_instance(7, "H8", control=True)
        assert clean.semantics == VALUE and clean.outcome == EQUAL and clean.golden_windows > 0
        law = value_execution.run_core_law
        monkeypatch.setattr(value_execution, "run_core_law",
                            lambda signal, prepared, weight=None: law(signal, prepared, weight) + 1.0)
        shared = run_instance(7, "H8", control=True)
        assert shared.mismatches == 0 and shared.golden_mismatches > 0
        assert shared.outcome == COUNTEREXAMPLE, shared.to_dict()
        assert attribute(shared) == ("golden_departure",)
