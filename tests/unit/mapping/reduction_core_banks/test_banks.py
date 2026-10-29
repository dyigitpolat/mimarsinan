"""[U-L2, B-1] Banked reduction cores in the fused IR, and the shape-only walk stating them alike.

U-L put the pool on the chip's P lane: the lane fusion turns each pool into 0/1 reduction cores, one owned core per
window group (``pool_cores``), so the one-copy arm (W02), which serves a shared bank's instances in turn on one copy,
holds one core set per window group, and a resident program needs every one of them (U-L, L-3 finding: R18 w1/2 needs
242 cores for whole-program residency with the pool lane against 204 with the pool on the host). Declared
(``reduction_core_banks``, under position multiplexing), the window groups of one pool that hold one matrix and read
one producer grid become the instances of one shared bank: one bank per distinct matrix and grid of a lane, each
group an instance (a group of its own matrix is a bank of one instance). The cores' matrices, thresholds and laws are
the unbanked program's, core for core. Each instance's layout record (U-L's ``layout_records``) states its bank. The
shape-only walk (U-E2's ``LaneWalk``) states the same banks, keyed on what both planes hold: the window pattern and the
producers' lane owners. Undeclared or false, the fused IR is U-L's, byte for byte (controls).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from mimarsinan.mapping.ir.lane_law import inserted_record

from unit.mapping.lanes.lane_planes import network_of

from . import vehicles as V

SMALL = ["residual", "shared", "transition"]


def _case(network: str, geometry: str, banks=True):
    platform = V.chip(V.POOL, banks=banks, **V.GEOMETRIES[geometry])
    return platform, V.program(network, V.POOL, platform, banks=banks)


def _owners(core, ir) -> frozenset:
    from mimarsinan.mapping.lanes.law.lane_grids import owner_of

    nodes = {int(n.id): n for n in ir.get_neural_cores()}
    return frozenset(owner_of(nodes[int(s.node_id)]) for s in core.input_sources.flatten() if int(s.node_id) >= 0)


def _distinct(cores, ir) -> int:
    """The distinct (matrix, producer owners, lane) of the unbanked program's window groups."""
    from mimarsinan.mapping.lanes.ir.fusion import LANE_OWNER

    return len({(getattr(c, LANE_OWNER), np.asarray(c.get_core_matrix(ir)).shape,
                 np.asarray(c.get_core_matrix(ir)).tobytes(), _owners(c, ir)) for c in cores})


@pytest.mark.parametrize("geometry", sorted(V.GEOMETRIES))
@pytest.mark.parametrize("network", SMALL)
def test_window_groups_of_one_matrix_and_grid_are_instances_of_one_bank(network, geometry):
    platform, prog = _case(network, geometry)
    before = V.unbanked(network, V.POOL, platform)
    cores, old = V.pool_cores(prog), V.pool_cores(before)
    assert cores and [c.name for c in cores] == [c.name for c in old]
    banks = V.lane_banks(prog)
    assert all(c.weight_bank_id is not None and c.core_matrix is None for c in cores), "every group is an instance"
    assert len(banks) == _distinct(old, before.ir)
    pooled = {int(c.id) for c in cores}
    reduced = [c for c in V.lane_cores(prog) if int(c.id) not in pooled]  # a reduction's reduced cores stay owned
    assert all(c.weight_bank_id is None for c in reduced)
    for bank_id, instances in banks.items():
        bank = prog.ir.weight_banks[bank_id]
        for core in instances:
            twin = next(c for c in old if c.name == core.name)
            assert np.array_equal(np.asarray(core.get_core_matrix(prog.ir)), np.asarray(twin.core_matrix))
            assert core.weight_row_slice == (0, bank.core_matrix.shape[1]) and bank.perceptron_index is None
            assert float(core.threshold) == float(twin.threshold)
            assert _owners(core, prog.ir) == _owners(instances[0], prog.ir)


def test_a_pool_of_many_window_groups_of_one_shape_is_one_bank():
    """The residual network's global average pool on 256-axon cores: every window group holds the same 0/1 matrix
    over the one producer, so the pool is one bank of several instances (the shape of ResNet-18's head)."""
    _platform, prog = _case("residual", "together")
    banks = V.lane_banks(prog)
    assert len(banks) == 1 and len(next(iter(banks.values()))) >= 2


@pytest.mark.parametrize("network", SMALL)
def test_each_instance_s_layout_record_states_its_bank_and_its_level(network):
    _platform, prog = _case(network, "together")
    for core in V.pool_cores(prog):
        record = prog.ir.layout_softcores[int(core.layout_softcore_index)]
        assert inserted_record(core)
        assert record.bank_id == int(core.weight_bank_id)
        assert record.latency_tag == int(core.latency)
        assert (record.input_count, record.output_count) == (core.get_input_count(), core.get_output_count())


@pytest.mark.parametrize("network", SMALL)
def test_every_instance_holds_the_unbanked_core_s_law_and_one_law_per_bank(network):
    platform, prog = _case(network, "together")
    before = {c.name: c for c in V.unbanked(network, V.POOL, platform).ir.get_neural_cores()}
    for core in prog.ir.get_neural_cores():
        assert core.readout_law.digest() == before[core.name].readout_law.digest(), core.name
    assert V.lane_banks(prog), "the vehicle banks its pool lane"
    for instances in V.lane_banks(prog).values():
        assert len({c.readout_law.digest() for c in instances}) == 1


@pytest.mark.parametrize("network", SMALL)
@pytest.mark.parametrize("banks", [None, False], ids=["absent", "false"])
def test_control_absent_or_false_the_fused_ir_is_u_l_s(network, banks):
    """Control (green before the change): undeclared or false, no pool core is banked and the IR pickles as U-L's."""
    platform, prog = _case(network, "together", banks=banks)
    assert not V.lane_banks(prog)
    reference = V.unbanked(network, V.POOL, platform)
    assert pickle.dumps(_canonical(prog.ir)) == pickle.dumps(_canonical(reference.ir))


def _canonical(ir) -> list:
    return [(int(n.id), n.name, None if getattr(n, "core_matrix", None) is None else np.asarray(n.core_matrix).tobytes(),
             getattr(n, "weight_bank_id", None), int(getattr(n, "latency", 0) or 0)) for n in ir.nodes] + [
        (int(k), np.asarray(b.core_matrix).tobytes()) for k, b in sorted(ir.weight_banks.items())]


@pytest.mark.parametrize("geometry", sorted(V.GEOMETRIES))
@pytest.mark.parametrize("network", SMALL)
def test_the_shape_only_walk_states_the_same_banks(network, geometry):
    """The candidate plane's lane-aware walk gives each inserted pool core the bank the builder's fusion gave it (the
    same id: banks are numbered after every bank the walk records, in the order the pools are met)."""
    from mimarsinan.deployment_record.introspection.scaling_ladder_lanes import layout_walk
    from mimarsinan.mapping.lanes.shape_walk import LaneWalk

    platform, prog = _case(network, geometry)
    net = network_of(prog, V.NETWORKS[network][2])
    lanes = LaneWalk.of(net.mapper_repr, platform)
    walk = layout_walk(net, platform, wire_census=False, lanes=lanes)
    walked = {spec.name: spec.bank_id for spec in (walk.layout_softcores[walk._node_id_to_softcore_idx[n]]
                                                   for n in lanes.inserted)}
    built = {c.name: int(c.weight_bank_id) for c in V.pool_cores(prog)}
    assert built and {k: walked[k] for k in built} == built


def test_instances_of_one_bank_holding_different_laws_are_refused_by_name():
    """The guard behind the key: a bank's instances must hold one solved law (one copy states one); a program whose
    planes' key missed a grid is refused by name, never served on one copy."""
    from types import SimpleNamespace

    from mimarsinan.mapping.lanes.ir.reduction_banks import ReductionBankLawError, refuse_split_bank_laws

    law = lambda digest: SimpleNamespace(digest=lambda: digest)  # noqa: E731
    nodes = [SimpleNamespace(id=i, weight_bank_id=7) for i in (1, 2)] + [SimpleNamespace(id=3, weight_bank_id=None)]
    ir = SimpleNamespace(nodes=nodes)
    refuse_split_bank_laws(ir, {1: law("a"), 2: law("a"), 3: law("b")}, [7])
    refuse_split_bank_laws(ir, {1: law("a"), 2: law("b"), 3: law("b")}, [])  # a bank the decision did not make
    with pytest.raises(ReductionBankLawError, match="reduction_core_banks.*reduction banks \\[7\\]"):
        refuse_split_bank_laws(ir, {1: law("a"), 2: law("b"), 3: law("b")}, [7])
