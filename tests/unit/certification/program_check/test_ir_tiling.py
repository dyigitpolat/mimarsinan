"""[E1 F1] The deployment's IR at the declared geometry against the IR at unbounded geometry, per IR neuron (condition
``ir-tiling``): established on the X3a R4 deployments (tiled, relabelled, dithered), where the verdict then reads
against the IR at unbounded geometry and assumes no IR tiling; assumed where no IR at unbounded geometry is given; and
one seeded tiling fault per named rule, built into the deployment's IR before its program is built (so the program
deploys its own IR faithfully and only this condition can see the fault), each failing Theorem 1 by name."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from mimarsinan.certification.program_check import CheckDeclaration, check_deployment
from mimarsinan.certification.program_check.report import ASSUMED_CONDITION, ESTABLISHED, FAILED
from mimarsinan.certification.randomized import executors as ex
from mimarsinan.certification.randomized.arms import ARMS
from mimarsinan.certification.randomized.cost_draws import platform_constraints
from mimarsinan.certification.randomized.instance import VALUE
from mimarsinan.certification.randomized.runner import (
    _run, compare, deployment, instance_for, reference, targeted_declaration)
from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

#: X3a R4 seeds whose deployment tiles a perceptron at the declared geometry (seed 7: perceptrons 3 to 5) and seeds
#: whose deployment holds a host op (the X3a host motifs), each relabelled.
TILED_SEEDS = (7, 25, 28)


def _arm():
    return replace(ARMS["R4"], id="R4/keyed", switch=None)


def _windowed(dithered: bool = True) -> CheckDeclaration:
    return CheckDeclaration(semantics="windowed_lif", window=8, pass_transfer="collapse", dithered_encode=dithered)


def _setting(seed: int, arm=None):
    arm = arm or _arm()
    instance = instance_for(seed, arm)
    constraints = platform_constraints(targeted_declaration(seed, arm), hardware_bias=instance.hardware_bias)
    return instance, constraints, arm


def _tiled_ir(instance, constraints, seed, arm):
    ir, _mapping = deployment(instance, constraints, seed, arm)
    return ir


def _program_of(ir, instance, constraints):
    run = ex.RUNS[instance.semantics]
    return build_hybrid_mapping_for_pipeline(ir, dict(constraints), pipeline_config=dict(run.config))


def _check(seed: int, mutate=None, *, dithered: bool = True, given: bool = True):
    instance, constraints, arm = _setting(seed)
    ref = reference(instance, arm)
    ir = _tiled_ir(instance, constraints, seed, arm)
    if mutate is not None:
        mutate(ir)
    program = _program_of(ir, instance, constraints)
    decl = replace(_windowed(dithered), window=int(instance.timesteps),
                   core_types=tuple(dict(c) for c in constraints["cores"]))
    return check_deployment(program, ir_graph=ir, declaration=decl, unbounded_ir=ref.ir if given else None)


@pytest.mark.parametrize("seed", TILED_SEEDS)
def test_a_tiled_relabelled_dithered_deployment_is_established_against_the_ir_at_unbounded_geometry(seed):
    report = _check(seed)
    assert report.results["ir-tiling"].status == ESTABLISHED and report.results["ir-tiling"].checked
    assert report.theorem1.startswith("established on the program against the IR at unbounded geometry")
    assert "IR tiling" not in report.theorem1 and "identity builder" in report.theorem1
    assert report.reference == "the IR at unbounded geometry"
    detail = report.certificate_records()[0]["detail"]
    assert "IR tiling" not in detail and "the IR at unbounded geometry" in detail


def test_a_one_neuron_tile_reads_its_unbounded_neurons_bias():
    """X3a R4 seed 182: a perceptron of 25 neurons tiled 24 + 1 on hardware-bias cores; the one-neuron tile's bias is
    a vector of one entry, the unbounded node's a vector of 25 (a width-1 vector once read as a scalar, a false
    ``bias`` failure in the first E1 run, ``e1/runs/w15/x3a``)."""
    report = _check(182)
    assert report.results["ir-tiling"].status == ESTABLISHED, report.results["ir-tiling"].violations


def test_without_the_ir_at_unbounded_geometry_the_tiling_stays_assumed():
    report = _check(7, given=False)
    assert report.results["ir-tiling"].status == ASSUMED_CONDITION
    assert report.theorem1.startswith("established on the program against the IR at the declared geometry")
    assert "IR tiling" in report.theorem1


def test_an_assumed_condition_is_not_counted_as_held():
    """[E1 fix round] ``exact_match_fraction`` is the share established, vacuous or not required; an assumed
    condition is reported beside it, never as held."""
    import json

    record = _check(7, given=False).certificate_records()[0]
    detail = json.loads(record["detail"])
    assert detail["conditions"]["ir-tiling"] == ASSUMED_CONDITION
    held = sum(s in ("established", "vacuous", "not_required") for s in detail["conditions"].values())
    assert record["exact_match_fraction"] == held / len(detail["conditions"]) < 1.0
    assert detail["assumed_conditions"] == ["ir-tiling"]


def test_the_verdict_names_the_neuron_keyed_identity_builder_and_the_dithered_encode():
    """[E1 fix round] Against the IR at unbounded geometry the verdict also rests on the identity program being a
    function of the IR's neurons as ir-tiling keys them; under a dithered encode, on the encoder rotating every encoded
    channel by the program's channel table (executor code the check does not read)."""
    unbounded, undithered, declared = _check(7), _check(7, dithered=False), _check(7, given=False)
    assert "neuron-keyed identity builder" in unbounded.theorem1 and "dithered encode" in unbounded.theorem1
    assert "neuron-keyed identity builder" in undithered.theorem1 and "dithered encode" not in undithered.theorem1
    assert "neuron-keyed identity builder" not in declared.theorem1 and "dithered encode" in declared.theorem1


def _failed(report, rule: str) -> None:
    result = report.results["ir-tiling"]
    assert result.status == FAILED, (result.status, result.note)
    assert rule in {v.rule for v in result.violations}, [v.text() for v in result.violations]
    assert report.theorem1.startswith("failed") and "ir-tiling" in report.theorem1, report.theorem1
    assert report.proposition2.startswith("failed") and "ir-tiling" in report.proposition2, report.proposition2


def _tile(ir) -> NeuralCore:
    """A tile of a perceptron the declared geometry split (its slice starts past 0)."""
    return next(n for n in ir.get_neural_cores() if n.perceptron_output_slice and n.perceptron_output_slice[0] > 0)


def _matrix(node: NeuralCore) -> np.ndarray:
    node.core_matrix = np.array(node.core_matrix, dtype=np.float64)
    return node.core_matrix


def _live(node: NeuralCore):
    matrix = _matrix(node)
    rows, cols = np.nonzero(matrix[:-1] if node.hardware_bias is None else matrix)
    return matrix, int(rows[0]), int(cols[0])


def test_a_tile_whose_weight_the_tiling_changed():
    def mutate(ir):
        matrix, r, c = _live(_tile(ir))
        matrix[r, c] += 1.0
    _failed(_check(7, mutate), "fan_in")


def test_a_tile_reading_another_source():
    def mutate(ir):
        tile = _tile(ir)
        flat = list(np.asarray(tile.input_sources, dtype=object).flatten())
        a, b = 0, 1
        flat[a], flat[b] = flat[b], flat[a]
        tile.input_sources = np.array(flat, dtype=object)
    _failed(_check(7, mutate), "fan_in")


def _tiles_of(ir):
    """Every tile of the perceptron ``_tile`` names: a singleton the tiling changed on one perceptron's tiles packs
    as one residency class, so the program deploys it."""
    p = _tile(ir).perceptron_index
    return [n for n in ir.get_neural_cores() if n.perceptron_index == p]


def test_the_tiles_whose_threshold_the_tiling_changed():
    def mutate(ir):
        for tile in _tiles_of(ir):
            tile.threshold = float(tile.threshold) + 1.0
    _failed(_check(7, mutate), "threshold")


def test_the_tiles_whose_scale_the_tiling_changed():
    def mutate(ir):
        for tile in _tiles_of(ir):
            tile.activation_scale = torch.tensor(2.0)
    _failed(_check(7, mutate), "neuron_scale")


def _hardware_bias_seed() -> int:
    for seed in range(400):
        instance, constraints, arm = _setting(seed)
        if not instance.hardware_bias:
            continue
        try:
            ir = _tiled_ir(instance, constraints, seed, arm)
            _program_of(ir, instance, constraints)
            _tile(ir)
        except Exception:  # noqa: BLE001 -- a refused or untiled draw is no witness
            continue
        return seed
    raise AssertionError("no R4 draw tiles a perceptron on hardware-bias cores")


def test_a_tile_whose_hardware_bias_the_tiling_changed():
    def mutate(ir):
        tile = _tile(ir)
        tile.hardware_bias = np.asarray(tile.hardware_bias, dtype=np.float64) + 1.0
    _failed(_check(_hardware_bias_seed(), mutate), "bias")


def test_a_tile_whose_bias_row_the_tiling_changed():
    def mutate(ir):
        tile = _tile(ir)
        assert tile.hardware_bias is None
        _matrix(tile)[-1, 0] += 1.0
    _failed(_check(7, mutate), "fan_in")


def test_a_tile_whose_ir_latency_attribute_alone_changed_holds():
    """[E1 fix round] The IR ``latency`` attribute (IRLatency's depth) is read by no executor: the executors run a
    core at its executed latency, which ``executed_latency`` compares (``test_ir_tiling_executed``). A change of the
    attribute alone moves no executed latency and no output (``e1/review_soundness/latency_rule_probe.log``), so the
    step-1 ``latency`` rule, which failed it, is retired."""
    def mutate(ir):
        tile = _tile(ir)
        tile.latency = int(tile.latency) + 1
    assert _check(7, mutate).results["ir-tiling"].status == ESTABLISHED


def test_a_tile_whose_activation_kind_the_tiling_changed():
    def mutate(ir):
        _tile(ir).activation_type = "Changed"
    _failed(_check(7, mutate), "activation")


def test_a_tile_keyed_on_another_slice_of_its_perceptron():
    """A tile that names other channels of its perceptron than it computes: its neurons are not the unbounded ones it
    claims, so the neurons it claims hold other fan-ins and its own are missing."""
    def mutate(ir):
        tile = _tile(ir)
        start, end = tile.perceptron_output_slice
        tile.perceptron_output_slice = (start + 1, end + 1)
    report = _check(7, mutate)
    _failed(report, "missing_neuron")
    _failed(report, "extra_neuron")


def test_two_tiles_holding_one_neuron():
    def mutate(ir):
        tile = _tile(ir)
        other = next(n for n in ir.get_neural_cores()
                     if n.perceptron_index == tile.perceptron_index and n is not tile)
        tile.perceptron_output_slice = tuple(other.perceptron_output_slice[:1]) + (
            other.perceptron_output_slice[0] + tile.get_output_count(),)
    _failed(_check(7, mutate), "duplicate_neuron")


def test_a_tile_holding_a_share_of_a_fan_in():
    def mutate(ir):
        _tile(ir).psum_role = "partial_pos"
    _failed(_check(7, mutate), "partial_sum_group")


def test_a_neural_node_with_no_perceptron_identity():
    def mutate(ir):
        _tile(ir).perceptron_index = None
    _failed(_check(7, mutate), "unkeyed_neuron")


def test_the_network_reading_other_neurons():
    def mutate(ir):
        flat = list(np.asarray(ir.output_sources, dtype=object).flatten())
        flat[0], flat[-1] = flat[-1], flat[0]
        ir.output_sources = np.array(flat, dtype=object).reshape(np.shape(ir.output_sources))
    _failed(_check(7, mutate), "program_outputs")


def test_a_channel_table_keyed_on_tile_columns_under_a_dithered_encode(monkeypatch):
    """A channel table that keys a node's outputs on their columns in the node (a tiling-dependent key), as the
    identity builder would build it: it departs from the table the check's identity program of the tiled IR carries
    (the derivation the unbounded side rests on); undithered, the key reads nothing and the condition holds."""
    import mimarsinan.certification.program_check.unbounded.predicate as F
    from mimarsinan.mapping.packing import phase_channels as PCH

    def by_column(ir_graph, reindex_maps=None):
        return {k: 0 for k in PCH.node_phase_channels(ir_graph, reindex_maps)}

    monkeypatch.setattr(F, "node_phase_channels", by_column)
    _failed(_check(7), "phase_channel")
    assert _check(7, dithered=False).results["ir-tiling"].status == ESTABLISHED


def _host_seed() -> int:
    for seed in range(200):
        instance, constraints, arm = _setting(seed)
        if not any(layer.kind == "host" for layer in instance.network.layers):
            continue
        try:
            ir = _tiled_ir(instance, constraints, seed, arm)
            _program_of(ir, instance, constraints)
        except Exception:  # noqa: BLE001 -- a refused draw is no host witness
            continue
        if any(isinstance(n, ComputeOp) and not n.name.startswith("host_encoder") for n in ir.nodes):
            return seed
    raise AssertionError("no R4 draw holds a host op")


def test_a_host_op_running_another_module():
    seed = _host_seed()

    def mutate(ir):
        op = next(n for n in ir.nodes if isinstance(n, ComputeOp) and n.name.startswith("host_")
                  and "encod" not in n.name)
        op.params = {**op.params, "module": torch.nn.Tanh()}
    _failed(_check(seed, mutate), "host_op")


def _value_case(seed_from: int = 0):
    """An X3a value instance whose IR the declared geometry tiles, a tile reading the raw input (so it carries G2's
    boundary grid beside its readout law)."""
    arm = ARMS["on"]
    for seed in range(seed_from, seed_from + 2000):
        instance = instance_for(seed, arm)
        if instance.semantics != VALUE:
            continue
        constraints = platform_constraints(targeted_declaration(seed, arm), hardware_bias=instance.hardware_bias)
        try:
            ir, _m = deployment(instance, constraints, seed, arm)
        except Exception:  # noqa: BLE001 -- a refused draw is no witness
            continue
        if any(n.perceptron_output_slice and n.perceptron_output_slice[0] > 0 and n.boundary_grid is not None
               for n in ir.get_neural_cores()):
            return seed, instance, constraints, arm
    raise AssertionError("no value draw tiles a perceptron that reads the raw input")


def _value_check(mutate=None):
    seed, instance, constraints, arm = _value_case()
    ref = reference(instance, arm)
    ir, _m = deployment(instance, constraints, seed, arm)
    if mutate is not None:
        mutate(ir)
    program = _program_of(ir, instance, constraints)
    decl = CheckDeclaration(semantics="value_integer", window=1, pass_transfer="collapse",
                            accumulation={"value_census": "float64", "int64_golden": "int64"},
                            upload={"value_census": "float32"})
    return check_deployment(program, ir_graph=ir, declaration=decl, unbounded_ir=ref.ir)


def test_a_tiled_integer_value_deployment_is_established_against_the_ir_at_unbounded_geometry():
    report = _value_check()
    assert report.results["ir-tiling"].status == ESTABLISHED, report.results["ir-tiling"].violations


def test_a_tile_whose_readout_law_the_tiling_changed():
    def mutate(ir):
        tile = _tile(ir)
        law = tile.readout_law
        tile.readout_law = type(law)(**{**{f: getattr(law, f) for f in law.__dataclass_fields__},
                                        "m0": np.asarray(law.m0) + 1})
    _failed(_value_check(mutate), "law")


def test_a_tile_whose_identity_adc_clamps_below_its_accumulator_bound():
    """Only ``adc_levels`` differ, but the tile's ADC levels sit below its columns' accumulator bound: its clamp can
    bind, so the laws declare different functions."""
    def mutate(ir):
        tile = _tile(ir)
        law = tile.readout_law
        tile.readout_law = type(law)(**{**{f: getattr(law, f) for f in law.__dataclass_fields__},
                                        "adc_levels": np.ones_like(np.asarray(law.adc_levels))})
    _failed(_value_check(mutate), "law")


def test_a_tile_whose_boundary_grid_the_tiling_changed():
    def mutate(ir):
        tile = next(n for n in ir.get_neural_cores() if n.boundary_grid is not None
                    and n.perceptron_output_slice and n.perceptron_output_slice[0] > 0)
        tile.boundary_grid = replace(tile.boundary_grid, scale=float(tile.boundary_grid.scale) * 2)
    _failed(_value_check(mutate), "boundary_grid")


def test_the_unbounded_reference_runs_what_the_check_establishes():
    """The established seeds' sampled run: the deployment against the identity program at unbounded geometry, equal."""
    for seed in TILED_SEEDS:
        instance, constraints, arm = _setting(seed)
        ref = reference(instance, arm)
        ir, mapping = deployment(instance, constraints, seed, arm)
        with torch.no_grad():
            nodes = _run(mapping, instance, arm)
        assert compare(ref.ir, ref.nodes, ir, nodes)[1] == 0


def test_the_ir_at_unbounded_geometry_compared_with_itself_holds():
    instance, constraints, arm = _setting(7)
    ref = reference(instance, arm)
    program = build_identity_hybrid_mapping(ir_graph=ref.ir, retimed_level_stages=True)
    report = check_deployment(program, ir_graph=ref.ir, declaration=_windowed(), unbounded_ir=copy.deepcopy(ref.ir))
    assert report.results["ir-tiling"].status == ESTABLISHED
