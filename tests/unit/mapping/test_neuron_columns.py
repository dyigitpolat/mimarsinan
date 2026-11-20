"""[U-LIVE r2] A value a core holds per neuron follows its neuron through every site that drops, selects or reorders
neurons (U-LIVE's check, M-2; the conductor's rulings of 2026-11-13, (a)).

Soft-core column compaction and IR pruning's compaction cut a core's matrix and its hardware bias at the kept columns
and left a threshold held per neuron (U-PCH3's per-neuron registers) whole, and the hard core read the threshold's
first entries: a kept neuron behind a dropped one fired at another neuron's threshold. U-LIVE's marks of unread neurons
reach soft-core compaction on the composed head (the check's pinned case tree481 at seed 13; its fold driver's
per-neuron lane 24 of 48 count reads apart). The one rule: ``compaction.take_neuron_columns`` takes every value a
core's class declares per neuron (``VALUE_LAYOUTS``) at the kept columns, both column compactions and a fragment's
window call it, and a per-neuron threshold read for another number of neurons is refused by name, never taken at its
first entries. The tests below hold each site on its smallest form and, end to end, in both families.
"""

from __future__ import annotations

from types import SimpleNamespace

from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point
from spiking_points import declaring_fire_while_integrating, point_of_mode

import numpy as np
import pytest
import torch

import unit.spiking.host_gather_forms as hg
import unit.spiking.position_fold_forms as pf

import mimarsinan.transformations.normalization_aware_perceptron_quantization as napq_module
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.mapping.packing.softcore import SoftCore
from mimarsinan.mapping.packing.softcore.compaction import compact_soft_core_mapping
from mimarsinan.mapping.packing.softcore.fragments import make_real_fragments
from mimarsinan.mapping.packing.softcore.hard_core import HardCore
from mimarsinan.mapping.pruning.ir_pruning_compact import _compact_node
from mimarsinan.mapping.support.neuron_thresholds import neuron_thresholds
from mimarsinan.mapping.support.value_layouts import ValueLayoutError

THETA = np.array([3.0, 5.0, 7.0, 11.0, 13.0])
BIAS = np.array([0.5, -1.0, 2.0, -3.0, 4.0])
MAGNITUDE = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
#: A lane core's records per column (pool output, reduction instances, operand row), each offset to tell them apart.
LANE_RECORDS = {"lane_outputs": 100, "lane_reduced_instances": 200, "lane_operand_rows": 300}


def _law(width: int) -> CoreLaw:
    c = np.arange(width)
    return CoreLaw(input_step=1.0 + c, m0=10 + c, shift=c, bias=-c, lo=-100 - c, hi=100 + c, output_step=0.5 + c,
                   adc_shift=0 * c, adc_levels=64 + c)


def _soft(width: int = 5, *, marked=(0, 2)) -> SoftCore:
    """A soft core of ``width`` neurons over 3 input axons holding a register, a bias entry and a law column per neuron,
    its ``marked`` neurons (before the kept ones) in its column mask."""
    core = SoftCore(core_matrix=np.arange(3.0 * width).reshape(3, width) + 1.0,
                    axon_sources=[SpikeSource(-2, i, is_input=True) for i in range(3)], id=0)
    core.threshold = THETA[:width].copy()
    core.hardware_bias = BIAS[:width].copy()
    core.readout_law = _law(width)
    core.pruned_row_mask = [False] * 3
    core.pruned_col_mask = [c in marked for c in range(width)]
    return core


def _node(width: int = 5) -> NeuralCore:
    node = NeuralCore(id=0, name="n", input_sources=np.array([IRSource(-2, i) for i in range(3)], dtype=object),
                      core_matrix=np.arange(3.0 * width).reshape(3, width) + 1.0)
    node.threshold = THETA[:width].copy()
    node.hardware_bias = BIAS[:width].copy()
    node.readout_law = _law(width)  # pyright: ignore[reportAttributeAccessIssue] -- declared-only attribute
    node.pruned_row_magnitude = MAGNITUDE[:width].copy()  # pyright: ignore[reportAttributeAccessIssue]
    for name, offset in LANE_RECORDS.items():  # a lane core's records per column
        setattr(node, name, np.arange(width, dtype=np.int64) + offset)
    return node


# ---------------------------------------------------------------------------------------------------------------------
# each site on its smallest form
# ---------------------------------------------------------------------------------------------------------------------
def test_soft_core_compaction_takes_every_per_neuron_value_at_the_kept_columns():
    """Neurons 0 and 2 marked before the kept 1, 3 and 4: each kept neuron keeps its own register, bias and law column
    (the composed head kept the five registers, and the hard core read the first three)."""
    core = _soft()
    outputs = [SpikeSource(0, n) for n in (1, 3, 4)]
    compact_soft_core_mapping([core], outputs)
    keep = [1, 3, 4]
    assert core.get_output_count() == 3
    assert np.asarray(core.threshold).tolist() == THETA[keep].tolist()
    assert np.asarray(core.hardware_bias).tolist() == BIAS[keep].tolist()
    assert core.readout_law.m0.tolist() == (10 + np.array(keep)).tolist()
    assert [(s.core_, s.neuron_) for s in outputs] == [(0, 0), (0, 1), (0, 2)]


def test_a_soft_core_bias_only_compaction_takes_every_per_neuron_value_at_the_kept_columns():
    """Every row pruned (the bias-only branch): the kept neurons keep their own values as well."""
    core = _soft()
    core.pruned_row_mask = [True] * 3
    compact_soft_core_mapping([core], [SpikeSource(0, 4)])
    assert np.asarray(core.threshold).tolist() == THETA[[1, 3, 4]].tolist()
    assert np.asarray(core.hardware_bias).tolist() == BIAS[[1, 3, 4]].tolist()
    assert core.readout_law.m0.tolist() == [11, 13, 14]


def test_ir_pruning_compaction_takes_every_per_neuron_value_at_the_kept_columns():
    """IR pruning's column compaction of a node: its registers, bias, law and dropped rows' record follow the kept
    neurons (the composed head kept the registers and the law whole), and a later row drop adds to the carried
    record."""
    node = _node()
    _compact_node(node, pruned_rows={1}, pruned_cols={0, 2})
    keep = [1, 3, 4]
    assert node.core_matrix.shape == (2, 3)
    assert np.asarray(node.threshold).tolist() == THETA[keep].tolist()
    assert np.asarray(node.hardware_bias).tolist() == BIAS[keep].tolist()
    assert node.readout_law.m0.tolist() == (10 + np.array(keep)).tolist()
    dropped = np.abs(np.arange(15.0).reshape(3, 5)[1] + 1.0)[keep]
    assert np.asarray(node.pruned_row_magnitude).tolist() == (MAGNITUDE[keep] + dropped).tolist()
    for name, offset in LANE_RECORDS.items():
        assert np.asarray(getattr(node, name)).tolist() == (np.array(keep) + offset).tolist(), name


def test_a_lane_core_s_records_per_column_are_declared_per_neuron():
    """The lane fusion's records per column are the names its constants give, each declared per neuron by the node."""
    from mimarsinan.mapping.lanes.ir.fusion import OPERAND_ROWS
    from mimarsinan.mapping.lanes.ir.pool_cores import LANE_OUTPUTS
    from mimarsinan.mapping.lanes.ir.reduce_cores import REDUCED_INSTANCES
    from mimarsinan.mapping.support.value_layouts import ValueLayout, declared_names

    assert {OPERAND_ROWS, LANE_OUTPUTS, REDUCED_INSTANCES} == set(LANE_RECORDS)
    assert set(LANE_RECORDS) <= set(declared_names(NeuralCore, ValueLayout.PER_NEURON))


def test_a_fragment_takes_every_per_neuron_value_at_its_window():
    """A neuron split of a soft core (a layer over two cores): each fragment holds its window's registers, bias and
    law, as it did by its own three slices before the one function."""
    core = _soft(marked=())
    first, second = make_real_fragments(softcore=core, first_neurons=2, remaining_neurons=3, group_id=0)
    for frag, window in ((first, [0, 1]), (second, [2, 3, 4])):
        assert np.asarray(frag.threshold).tolist() == THETA[window].tolist()
        assert np.asarray(frag.hardware_bias).tolist() == BIAS[window].tolist()
        assert frag.readout_law.m0.tolist() == (10 + np.array(window)).tolist()


@pytest.mark.parametrize("holder", ["soft core", "ir node"])
def test_the_one_function_carries_every_name_its_holder_declares_per_neuron(holder):
    """Every name the holder's class declares per neuron is taken at the kept columns; one held entry stays as it is
    held; a value of another size is refused by name."""
    from mimarsinan.mapping.packing.softcore.compaction import take_neuron_columns
    from mimarsinan.mapping.support.value_layouts import ValueLayout, declared_names

    core = _soft(marked=()) if holder == "soft core" else _node()
    names = declared_names(type(core), ValueLayout.PER_NEURON)
    assert {"threshold", "hardware_bias", "readout_law"} <= set(names)
    for k, name in enumerate(names):
        if name != "readout_law":
            setattr(core, name, np.arange(5.0) + 10.0 * k)
    take_neuron_columns(core, [4, 1])
    for k, name in enumerate(names):
        value = getattr(core, name)
        got = value.m0.tolist() if name == "readout_law" else np.asarray(value).tolist()
        assert got == ([14, 11] if name == "readout_law" else [4.0 + 10.0 * k, 1.0 + 10.0 * k]), name
    one = _soft(marked=())
    one.threshold, one.hardware_bias = 2.0, np.array([0.5])  # one held entry: a register, a bias every neuron reads
    take_neuron_columns(one, [0, 3])
    assert one.threshold == 2.0 and np.asarray(one.hardware_bias).tolist() == [0.5]
    bad = _soft(marked=())
    bad.threshold = np.arange(4.0)
    with pytest.raises(ValueLayoutError, match="threshold"):
        take_neuron_columns(bad, [0, 3])


# ---------------------------------------------------------------------------------------------------------------------
# the named refusal: a per-neuron threshold read for another number of neurons
# ---------------------------------------------------------------------------------------------------------------------
def test_a_per_neuron_threshold_read_for_another_number_of_neurons_is_refused_by_name():
    with pytest.raises(ValueLayoutError, match="5 registers read for 3 of 3 neurons"):
        neuron_thresholds(np.arange(5.0), 3)
    with pytest.raises(ValueLayoutError):
        neuron_thresholds(np.arange(2.0), 3)
    assert neuron_thresholds(np.arange(5.0), 3, 5).tolist() == [0.0, 1.0, 2.0]  # a core's used neurons of its column
    assert neuron_thresholds(4.0, 3).tolist() == [4.0, 4.0, 4.0]


def test_a_soft_core_whose_registers_are_not_its_neurons_is_refused_by_name_where_it_is_placed():
    """A site that cut a soft core's neurons and kept its registers whole is a named refusal at placement, never a
    hard core reading the first registers."""
    core = SoftCore(core_matrix=np.ones((2, 3)), axon_sources=[SpikeSource(-2, i, is_input=True) for i in range(2)],
                    id=0)
    core.threshold = THETA.copy()  # five registers for three neurons
    hard = HardCore(axons_per_core=4, neurons_per_core=4)
    with pytest.raises(ValueLayoutError, match="5 registers read for 3 of 3 neurons"):
        hard.add_softcore(core)


# ---------------------------------------------------------------------------------------------------------------------
# end to end: the marks of U-LIVE's dead-read pass in Soft Core Mapping's order, per-neuron grids (U-PCH3)
# ---------------------------------------------------------------------------------------------------------------------
class _PerNeuron(napq_module.NormalizationAwarePerceptronQuantization):
    """The row's projection with ``weight_scale_granularity: per_neuron`` (U-PCH3's option)."""

    def __init__(self, *args, **kwargs):
        kwargs["per_neuron"] = True
        super().__init__(*args, **kwargs)


@pytest.fixture
def per_neuron_grids(monkeypatch):
    monkeypatch.setattr(napq_module, "NormalizationAwarePerceptronQuantization", _PerNeuron)
    monkeypatch.setattr(hg, "NormalizationAwarePerceptronQuantization", _PerNeuron)


def _soft_core_mapping_order(monkeypatch, retimed: bool) -> None:
    """The count family's relay pass as Soft Core Mapping runs it (``_apply_depth_balancing_relays``: the dead
    in-segment reads turned off and the unread neurons marked, then the relays), for the LIF semantics given."""
    from mimarsinan.chip_simulation.activation_semantics import fold_spiking_axes
    from mimarsinan.config_schema.recipe_fold import fold_conversion_recipe
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import SoftCoreMappingStep

    cfg = {"spiking_family": "lif", "spiking_variant": "synchronized" if retimed else "streamed",
           "weight_bits": hg.WQ_BITS, "thresholding_mode": "<=", "firing_mode": "Default",
           "simulation_steps": hg.T, "spike_generation_mode": "Uniform"}
    if retimed:
        cfg["lif_per_hop_retiming"] = True
    cfg = declaring_fire_while_integrating(cfg)  # [composed head, rule 12] U-LIF2 W16: the form fires in its window
    fold_spiking_axes(cfg)
    fold_conversion_recipe(cfg, resolve_spiking_point(cfg), explicit_keys=set(cfg))
    step = SoftCoreMappingStep.__new__(SoftCoreMappingStep)
    step.pipeline = SimpleNamespace(config=cfg)
    plan = DeploymentPlan.resolve(cfg)

    def relays(ir, *, thresholding_mode, q_max):
        enabled, record = step._apply_depth_balancing_relays(ir, plan)
        step._require_cycle_windows(ir, plan, enabled)
        return int(record["relay_cores_inserted"])

    monkeypatch.setattr(hg, "insert_depth_balancing_relays", relays)


def _marked_before_kept(ir) -> bool:
    """Whether a node holding a register per neuron has a marked neuron before a kept one."""
    for node in ir.get_neural_cores():
        mask = [bool(m) for m in (getattr(node, "pruned_col_mask", None) or [])]
        if np.ndim(node.threshold) == 1 and True in mask and False in mask and \
                mask.index(True) < len(mask) - 1 - mask[::-1].index(False):
            return True
    return False


@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
@pytest.mark.parametrize("gauges", ("one", "drawn"))
def test_the_count_family_reads_equal_behind_a_marked_neuron_under_per_neuron_thresholds(
        semantics, gauges, monkeypatch, per_neuron_grids):
    """The fold check's generated tree 481 at seed 1 (the check's per-neuron lane read 2 to 4 of 32 apart here): the
    walk equals the deployed read element by element, and the harness's per-core window counts (the kept neurons on
    the NF side) are bit-exact."""
    _soft_core_mapping_order(monkeypatch, hg.SEMANTICS[semantics][0])
    program = pf.count_program("tree481", 1, semantics, gauges, wq=True)
    assert _marked_before_kept(program.ir), "the form holds a marked neuron under a per-neuron threshold"
    pf.assert_equal_reads(pf.count_read(program, semantics, 1))
    if gauges == "one":  # the harness's NF capture reads one gauge per perceptron
        mismatches, compared, out_diff = hg.compare(program, semantics, n=16, seed=1)
        assert compared > 0 and all(v == 0 for v in mismatches.values()), mismatches
        assert out_diff <= 1e-5  # the count grid's float noise (count_read's rule)


@pytest.mark.parametrize("seed", (5,))
def test_the_timing_family_reads_equal_behind_a_marked_neuron_under_per_neuron_thresholds(seed, per_neuron_grids):
    """The generated tree 481 at seed 5 with Weight Quantization (the merge cut reads 0 elements apart, the composed
    head 6): the walk equals the deployed read element by element."""
    read = pf.timing_read("tree481", seed, "one", wq=True)
    assert _marked_before_kept(read["ir"]), "the form holds a marked neuron under a per-neuron threshold"
    pf.assert_equal_reads(read)


@pytest.mark.parametrize("family", ("streamed_lif", "windowed_lif", "timing"))
def test_ir_pruning_refuses_per_neuron_thresholds_by_name_before_any_compaction(family, per_neuron_grids):
    """IR pruning reads one threshold per core (its folds and its liveness), so a program holding per-neuron
    registers is refused by name before any column is compacted, in each family's program (its compaction carries
    the registers all the same: the test of ``_compact_node`` above)."""
    from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
    from mimarsinan.models.nn.lif_kernels import PerNeuronThresholdRefusal

    if family == "timing":
        ir = pf.timing_read("tree481", 1, "one", wq=True)["ir"]
    else:
        ir = pf.count_program("tree481", 1, family, "one", wq=True).ir
    assert any(np.ndim(n.threshold) == 1 for n in ir.get_neural_cores())
    with pytest.raises(PerNeuronThresholdRefusal, match="IR pruning"):
        prune_ir_graph(ir, simulation_steps=hg.T, spiking_mode="lif")


def test_a_bank_instance_takes_its_window_of_every_value_the_bank_holds_per_neuron():
    """A shared bank's instances (a convolution's positions, a layer's tiles): each instance's neurons are its window
    of the bank's columns, so it holds its window of the bank's bias, the window its matrix resolves through."""
    from mimarsinan.mapping.ir_mapping_class import IRMapping

    ir = IRMapping(hardware_bias=True)
    w = np.arange(24.0).reshape(6, 4)
    bank_id = ir.register_weight_bank(weights=w, biases=np.append(BIAS[:5], 6.0))
    sources = np.array([IRSource(-2, i) for i in range(4)], dtype=object)
    for window in ((0, 2), (2, 6)):
        ir.add_shared_neural_core(input_sources=sources, weight_bank_id=bank_id, has_bias=True, weight_row_slice=window)
        node = ir.nodes[-1]
        assert np.asarray(node.hardware_bias).tolist() == (BIAS[:5].tolist() + [6.0])[window[0]:window[1]]


def test_a_packed_hard_core_s_register_column_is_read_for_its_used_neurons(monkeypatch):
    """A packed program (hard cores of 7 neurons, soft cores split across them): each hard core's register column
    covers its 7 neurons, and SANA-FE's synthesis gives each used neuron its own register of that column (reading the
    column as the used neurons' alone is refused by name, never taken at its first entries)."""
    import mimarsinan.chip_simulation.sanafe.net_synth as net_synth
    from mimarsinan.chip_simulation.sanafe.net_synth.build import build_network_for_segment
    from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
    from unit.chip_simulation.test_per_neuron_threshold_backends import _FakeNetwork
    from unit.mapping.test_per_neuron_weight_scales import GEOMETRIES, _program

    monkeypatch.setattr(net_synth, "_sanafe", lambda: SimpleNamespace(Network=_FakeNetwork))
    program = _program("pool_downsample", 0, register=True, geometry=GEOMETRIES[1])
    packed = build_hybrid_hard_core_mapping(
        ir_graph=program.ir, cores_config=[{"max_axons": 256, "max_neurons": 7, "count": 400, "has_bias": True}],
        strategy=MappingStrategy.from_permissions(allow_neuron_splitting=True, allow_coalescing=False),
        retimed_level_stages=True)
    partial = 0
    for stage in packed.stages:
        if stage.kind != "neural":
            continue
        mapping = stage.hard_core_mapping
        arch = SimpleNamespace(tiles=[SimpleNamespace(cores=[object() for _ in range(len(mapping.cores))])])
        _net, core_to_group, _, _ = build_network_for_segment(arch, mapping, tile_offset=0, core_offset=0,
                                                              cores_per_tile=len(mapping.cores), simulation_length=4,
            point=point_of_mode("lif", "cascaded"))
        for core_idx, group in core_to_group.items():
            core = mapping.cores[core_idx]
            if np.ndim(core.threshold) != 1:
                continue
            assert np.size(core.threshold) == int(core.neurons_per_core)
            partial += len(group) < int(core.neurons_per_core)
            got = [n.attributes["model_attributes"]["threshold"] for n in group]
            assert got == np.asarray(core.threshold, dtype=np.float64)[:len(group)].tolist()
    assert partial, "a hard core with unused neurons holds a register column"


def test_a_lane_reduction_reads_its_producers_one_register_or_refuses_by_name():
    """A pool lane's reduction sums producer columns into a new core of one register (the producers' theta times the
    window): a producer holding a register per neuron is refused by name (``core_threshold``), never an unnamed error
    (the lanes fuse at mapping, before Weight Quantization writes any register per neuron)."""
    from mimarsinan.mapping.lanes.forms import PoolForm
    from mimarsinan.mapping.lanes.ir.reduce_cores import reduced_lane_cores
    from mimarsinan.models.nn.lif_kernels import PerNeuronThresholdRefusal

    op = SimpleNamespace(input_shape=(1, 2, 2), output_shape=(1, 1, 1), name="pool")
    form = PoolForm(kernel=(2, 2), stride=(2, 2), padding=(0, 0))
    sources = np.array([IRSource(0, i) for i in range(4)], dtype=object)
    producer = NeuralCore(id=0, name="p", input_sources=np.array([IRSource(-2, 0)], dtype=object),
                          core_matrix=np.ones((1, 4)), threshold=2.0)
    built, _outputs = reduced_lane_cores(op, sources, {0: producer}, None, form=form, first_id=10)
    assert [float(c.threshold) for c in built] == [8.0]
    producer.threshold = np.array([2.0, 3.0, 2.0, 3.0])
    with pytest.raises(PerNeuronThresholdRefusal, match="reduction 'pool'"):
        reduced_lane_cores(op, sources, {0: producer}, None, form=form, first_id=10)
