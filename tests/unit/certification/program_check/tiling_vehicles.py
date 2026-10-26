"""[E1 fix round] Hand-built vehicles on which a tiling, a compaction mask or a channel table moves what an identity
program runs, each with its sampled run against the identity program at unbounded geometry (Theorem 1's reference).

``consumer_split``: a perceptron P (4 neurons) whose channels 0-1 feed E and channels 2-3 feed D (an on-chip structural
split, ``ReshapeMapper`` then ``SubscriptMapper``), D also reading a two-deep chain. ChipLatency raises a core to its
deepest consumer's latency minus one, so at unbounded geometry P runs at 1 (for D) and E reads it a cycle late, while
tiled at ``max_neurons`` 2 the tile read only by E runs at 0 (``RA/11_wave2/e1/review_soundness``, witness A2).
``two_depth``: C = [c0 reads a0, c1 reads b0] with b0 one deeper than a0, the rows listed but zero (M1, M2 there).
``host_compacted``: A (3 neurons, one dead) read by a host op and by B in the next segment, so A's outputs are encoded
at a stage input under the dithered encode; the dead column is dropped at soft-core compaction (``pruned_col_mask``)
on both IRs, so the programs' channel tables take the packer's reindexed branch."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from mimarsinan.certification.program_check import CheckDeclaration, check_deployment
from mimarsinan.certification.randomized import executors as ex
from mimarsinan.certification.randomized import program as P
from mimarsinan.certification.randomized.arms import ARMS
from mimarsinan.certification.randomized.instance import HOST, INPUT, NEURAL, Instance, Layer, Network
from mimarsinan.certification.randomized.runner import _run, compare
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import ConcatMapper, InputMapper, ReshapeMapper, SubscriptMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.softcore.compaction import _ir_keep_indices
from mimarsinan.mapping.platform.packaging_contract import SPIKING_PACKAGING
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 4
TRANSFER = {"streamed_lif": "verbatim", "windowed_lif": "collapse", "synchronized_lif": "collapse",
            "cascaded_ttfs": "verbatim", "synchronized_ttfs": "collapse"}


def _inputs(timesteps: int):
    return tuple((a, b) for a in range(timesteps + 1) for b in range(timesteps + 1))


def consumer_split(semantics: str, timesteps: int = T):
    """``(instance, mapped network builder)`` of witness A2: E reads P[0:2], D reads P[2:4] and Q2 (Q1 -> Q2)."""
    def mapped() -> P.MappedNetwork:
        raw = InputMapper((2,))
        enc = P._perceptron(2, 2, timesteps, np.eye(2, dtype=np.float32), np.zeros(2), dtype=torch.float32)
        setattr(enc, "use_cycle_accurate_trains", True)
        x = PerceptronMapper(raw, enc)
        eye2 = np.eye(2)
        p = PerceptronMapper(x, P._perceptron(4, 2, timesteps, np.vstack([eye2, eye2]), np.zeros(4)))
        halves = ReshapeMapper(p, (2, 2))
        s0, s1 = SubscriptMapper(halves, 0), SubscriptMapper(halves, 1)
        q1 = PerceptronMapper(x, P._perceptron(2, 2, timesteps, eye2, np.zeros(2)))
        q2 = PerceptronMapper(q1, P._perceptron(2, 2, timesteps, eye2, np.zeros(2)))
        e = PerceptronMapper(s0, P._perceptron(2, 2, timesteps, eye2, np.zeros(2)))
        d = PerceptronMapper(ConcatMapper([s1, q2], dim=1, rank=2),
                             P._perceptron(2, 4, timesteps, np.hstack([eye2, eye2]), np.zeros(2)))
        rep = ModelRepresentation(ConcatMapper([e, d], dim=1, rank=2))
        mark_encoding_layers(rep, placement="subsume", packaging=SPIKING_PACKAGING)
        rep.assign_perceptron_indices()
        names = {"P": p, "Q1": q1, "Q2": q2, "E": e, "D": d}
        of = {k: int(getattr(m, "perceptron_index")) for k, m in names.items()}
        theta = {"P": 1, "Q1": 1, "Q2": 1, "E": 1, "D": 2}
        return P.MappedNetwork(rep, of, {of[k]: v for k, v in theta.items()})

    instance = Instance(seed=0, semantics=semantics, timesteps=timesteps, hardware_bias=False,
                        network=Network(2, (), ("E", "D")), inputs=_inputs(timesteps))
    return instance, mapped


def two_depth(semantics: str, timesteps: int = T):
    layers = (Layer("A", NEURAL, (INPUT,), 2, weights=((1, 0), (0, 1)), bias=(0, 0), theta=1),
              Layer("B", NEURAL, ("A",), 1, weights=((1, 0),), bias=(0,), theta=1),
              Layer("C", NEURAL, ("A", "B"), 2, weights=((1, 0, 0), (0, 0, 1)), bias=(0, 0), theta=1))
    instance = Instance(seed=0, semantics=semantics, timesteps=timesteps, hardware_bias=False,
                        network=Network(2, layers, ("C",)), inputs=_inputs(timesteps))
    return instance, lambda: P.mapper_graph(instance, "subsume")


def host_compacted(semantics: str, dead: int, timesteps: int = T):
    """A's channel ``dead`` (0 or 1) has no weight and no bias and is read by no nonzero weight."""
    live = [c for c in range(3) if c != dead]
    a = [[0, 0] for _ in range(3)]
    a[live[0]][0], a[live[1]][1] = 1, 1
    b = [[0] * 6 for _ in range(2)]
    b[0][live[0]], b[0][live[1]], b[0][3 + live[0]] = 1, 1, -1
    b[1][live[0]], b[1][live[1]], b[1][3 + live[1]] = 2, -1, 1
    layers = (Layer("A", NEURAL, (INPUT,), 3, weights=tuple(map(tuple, a)), bias=(0, 0, 0), theta=1),
              Layer("H", HOST, ("A",), 3, op="identity"),
              Layer("B", NEURAL, ("A", "H"), 2, weights=tuple(map(tuple, b)), bias=(0, 0), theta=2))
    instance = Instance(seed=0, semantics=semantics, timesteps=timesteps, hardware_bias=False,
                        network=Network(2, layers, ("B",)), inputs=_inputs(timesteps))
    return instance, lambda: P.mapper_graph(instance, "subsume")


def integer_ir(mapped: P.MappedNetwork, semantics: str, max_neurons: Optional[int]):
    return P.integer_ir(mapped, max_axons=None if max_neurons is None else 64, max_neurons=max_neurons,
                        allow_coalescing=False, hardware_bias=False, law=ex.RUNS[semantics].law)


def constraints(max_neurons: int, count: int = 16) -> Dict[str, Any]:
    return {"cores": [{"max_axons": 64, "max_neurons": max_neurons, "count": count, "has_bias": False}],
            "allow_coalescing": False, "allow_neuron_splitting": False, "allow_scheduling": False,
            "max_schedule_passes": 8, "core_value_granularity": {}}


def expanded(ir, nodes):
    """Node outputs in the IR's column space: a column soft-core compaction drops (a dead one) reads 0."""
    out = dict(nodes)
    for node in ir.get_neural_cores():
        if int(node.id) not in nodes:
            continue
        cols = _ir_keep_indices(node)[1]
        if len(cols) != int(node.get_output_count()):
            tensor = nodes[int(node.id)]
            full = torch.zeros(tensor.shape[0], int(node.get_output_count()), dtype=tensor.dtype)
            full[:, cols] = tensor
            out[int(node.id)] = full
    return out


@dataclass
class Case:
    report: Any
    windows: int
    mismatches: int
    ir: Any
    unbounded: Any
    program: Any
    theorem: Any


def case(vehicle, *, max_neurons: int, dithered: bool = False, mutate: Optional[Callable] = None,
         reference: Optional[Callable] = None) -> Case:
    """The vehicle at ``max_neurons`` (a pipeline deployment of its IR there), checked with the IR at unbounded
    geometry given (``reference(ir, unbounded)`` replaces it), and its sampled run against Theorem 1's identity
    program. ``mutate(ir, unbounded)`` edits the two IRs before any program is built."""
    instance, mapped = vehicle
    semantics = instance.semantics
    arm = replace(ARMS["on"], id="tiling", phase_dither=dithered)
    run = ex.RUNS[semantics]
    with contextlib.redirect_stdout(io.StringIO()):
        unbounded = integer_ir(mapped(), semantics, None)
        ir = integer_ir(mapped(), semantics, max_neurons)
        if mutate is not None:
            mutate(ir, unbounded)
        cons = constraints(max_neurons)
        program = build_hybrid_mapping_for_pipeline(ir, dict(cons), pipeline_config=dict(run.config))
        theorem = build_identity_hybrid_mapping(ir_graph=unbounded, retimed_level_stages=run.retimed)
        with torch.no_grad():
            want, got = _run(theorem, instance, arm, identity=True), _run(program, instance, arm)
    windows, mismatches, _ = compare(unbounded, expanded(unbounded, want), ir, expanded(ir, got))
    decl = CheckDeclaration(semantics=semantics, window=int(instance.timesteps), pass_transfer=TRANSFER[semantics],
                            dithered_encode=dithered, core_types=tuple(dict(c) for c in cons["cores"]))
    given = unbounded if reference is None else reference(ir, unbounded)
    report = check_deployment(program, ir_graph=ir, declaration=decl, unbounded_ir=given)
    return Case(report, windows, mismatches, ir, unbounded, program, theorem)


def dead_row_masks(*irs) -> None:
    """M1: every row all zero on its node's columns dropped at compaction (the always-on row kept), no column."""
    for ir in irs:
        for node in ir.get_neural_cores():
            m = np.asarray(node.get_core_matrix(ir), dtype=np.float64)
            rows = [bool(not np.any(m[r] != 0)) for r in range(m.shape[0])]
            rows[-1] = False
            node.pruned_row_mask, node.pruned_col_mask = rows, [False] * m.shape[1]


def perceptron_nodes(ir, perceptron: int):
    return sorted((n for n in ir.get_neural_cores() if n.perceptron_index == perceptron),
                  key=lambda n: n.perceptron_output_slice[0])


def drop_live_row(ir, perceptron: int) -> None:
    """M2: the first tile of ``perceptron`` drops its first row, which carries a nonzero weight."""
    node = perceptron_nodes(ir, perceptron)[0]
    m = np.asarray(node.get_core_matrix(ir), dtype=np.float64)
    assert np.any(m[0] != 0)
    node.pruned_row_mask, node.pruned_col_mask = [r == 0 for r in range(m.shape[0])], [False] * m.shape[1]


def drop_dead_channel(ir, perceptron: int, channel: int) -> None:
    """The node holding ``channel`` of ``perceptron`` drops it at soft-core compaction (it must be dead)."""
    for node in perceptron_nodes(ir, perceptron):
        start, end = node.perceptron_output_slice
        if start <= channel < end:
            m = np.asarray(node.get_core_matrix(ir), dtype=np.float64)
            assert not np.any(m[:, channel - start] != 0)
            node.pruned_col_mask = [start + c == channel for c in range(m.shape[1])]
            node.pruned_row_mask = [False] * m.shape[0]
