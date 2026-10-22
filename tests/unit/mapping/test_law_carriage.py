"""E282.S carriage: a law rides per column through conversion, splitting, fusion, packing and compaction."""

from __future__ import annotations

import copy
import pickle
import pickletools

import numpy as np
import pytest
import torch.nn as nn

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.law_stamp import core_channels
from mimarsinan.mapping.ir import NeuralCore, neural_core_to_soft_core
from mimarsinan.mapping.ir.core_law import CoreLaw, ReadoutLawMixedCoreError
from mimarsinan.mapping.packing.softcore import HardCore, SoftCore
from mimarsinan.models.nn.activations.readout_quantizer import LAW_FIELDS
from unit.chip_simulation.value_run.integer_programs import (
    build_integer_program,
    hcm_law_digests,
    lenet5,
    mlp,
)


def _keep_columns(core: NeuralCore) -> list:
    mask = getattr(core, "pruned_col_mask", None) if core.weight_bank_id is not None else None
    count = int(core.get_output_count())
    return [c for c in range(count) if not (mask is not None and mask[c])]


def _assert_hcm_matches_ir(program, hybrid) -> None:
    """Every packed column's law equals its IR core's law over the surviving columns, in IR column order."""
    packed = hcm_law_digests(hybrid)
    checked = 0
    for core in program.ir.nodes:
        if not isinstance(core, NeuralCore):
            continue
        keep = _keep_columns(core)
        want = core.readout_law.take_columns(keep).column_digests()
        got = packed[int(core.id)]
        assert [got[i] for i in range(len(keep))] == want, core.name
        checked += len(keep)
    assert checked > 0


@pytest.fixture(scope="module")
def mlp_program():
    return build_integer_program(mlp(), (8,), num_classes=4)


@pytest.fixture(scope="module")
def pruned_lenet():
    return build_integer_program(lenet5(), (1, 28, 28), num_classes=10, prune={0: [1, 3], 2: [5, 6, 7], 3: [2, 9]})


class TestConversion:
    def test_soft_core_shares_the_neural_core_law(self, mlp_program):
        core = next(n for n in mlp_program.ir.nodes if isinstance(n, NeuralCore))
        assert neural_core_to_soft_core(core, mlp_program.ir).readout_law is core.readout_law

    def test_a_core_without_a_law_converts_without_the_attribute(self, mlp_program):
        core = copy.copy(next(n for n in mlp_program.ir.nodes if isinstance(n, NeuralCore)))
        del core.readout_law
        assert "readout_law" not in vars(neural_core_to_soft_core(core, mlp_program.ir))


class TestPacking:
    def test_identity_packing_shares_one_law_per_core(self, mlp_program):
        _assert_hcm_matches_ir(mlp_program, mlp_program.packed_mapping(axons=256, neurons=64, split=False))

    def test_neuron_splitting_slices_and_reassembles_the_columns(self, mlp_program):
        hybrid = mlp_program.packed_mapping(axons=256, neurons=5, split=True)
        _assert_hcm_matches_ir(mlp_program, hybrid)
        splits = [p for stage in hybrid.stages if stage.kind == "neural"
                  for per_core in stage.hard_core_mapping.soft_core_placements_per_hard_core
                  for p in per_core if "split_group_id" in p]
        assert splits, "5-neuron cores must split the 16- and 12-wide layers"

    def test_multi_softcore_packing_splices_each_neuron_range(self, pruned_lenet):
        hybrid = pruned_lenet.packed_mapping(axons=1024, neurons=512, split=False)
        _assert_hcm_matches_ir(pruned_lenet, hybrid)
        shared = [hc for stage in hybrid.stages if stage.kind == "neural"
                  for hc in stage.hard_core_mapping.cores if len(hc.matrix_placements) > 1]
        assert shared, "the pool must pack several soft cores into one hard core"

    def test_axon_fusion_keeps_the_law(self):
        wide = nn.Sequential(nn.Linear(300, 12), nn.ReLU(), nn.Linear(12, 4)).eval()
        program = build_integer_program(wide, (300,), num_classes=4, max_axons=128, allow_coalescing=True)
        hybrid = program.packed_mapping(axons=128, neurons=64, split=False, coalescing=True)
        fused = [hc for stage in hybrid.stages if stage.kind == "neural"
                 for hc in stage.hard_core_mapping.cores if hc.fused_component_axons]
        assert fused, "the 300-axon layer must ride fused hard cores"
        _assert_hcm_matches_ir(program, hybrid)

    def test_a_permutation_of_columns_permutes_the_laws(self, mlp_program):
        law = next(n for n in mlp_program.ir.nodes if isinstance(n, NeuralCore)).readout_law
        order = np.random.default_rng(0).permutation(law.columns)
        assert law.take_columns(order).column_digests() == [law.column_digests()[i] for i in order]


class TestCompaction:
    def test_bank_slice_positions_reference_one_law_object(self, pruned_lenet):
        by_slice = {}
        for core in pruned_lenet.ir.nodes:
            if isinstance(core, NeuralCore) and core.weight_bank_id is not None:
                by_slice.setdefault((core.weight_bank_id, core.weight_row_slice), set()).add(id(core.readout_law))
        assert by_slice and all(len(ids) == 1 for ids in by_slice.values())

    def test_compacted_bank_cores_keep_the_digest_over_surviving_columns(self, pruned_lenet):
        hybrid = pruned_lenet.packed_mapping(axons=1024, neurons=512)
        _assert_hcm_matches_ir(pruned_lenet, hybrid)
        identity = pruned_lenet.identity().hybrid_mapping
        _assert_hcm_matches_ir(pruned_lenet, identity)
        pruned_banks = [c for c in pruned_lenet.ir.nodes if isinstance(c, NeuralCore)
                        and c.weight_bank_id is not None and len(_keep_columns(c)) < c.get_output_count()]
        assert pruned_banks, "the fixture must prune bank-backed columns"
        for stage in identity.stages:
            for hc in (stage.hard_core_mapping.cores if stage.kind == "neural" else ()):
                assert hc.readout_law.columns == hc.neurons_per_core
                if hc.hardware_bias is not None:
                    assert hc.readout_law.columns == len(hc.hardware_bias)

    def test_take_columns_returns_one_object_per_keep_content(self, pruned_lenet):
        law = next(c for c in pruned_lenet.ir.nodes if isinstance(c, NeuralCore)).readout_law
        keep = list(range(0, law.columns, 2))
        assert law.take_columns(keep) is law.take_columns(np.array(keep))
        assert law.take_columns(keep) is not law.take_columns(keep[:-1])

    def test_identity_packed_positions_share_compacted_laws(self, pruned_lenet):
        identity = pruned_lenet.identity().hybrid_mapping
        laws = [hc.readout_law for stage in identity.stages if stage.kind == "neural"
                for hc in stage.hard_core_mapping.cores]
        assert len({id(law) for law in laws}) < len(laws) / 10

    def test_pickle_stores_each_distinct_law_once(self, pruned_lenet):
        with_laws = pruned_lenet.identity().hybrid_mapping

        def laws_of(hybrid):
            return [hc.readout_law for stage in hybrid.stages if stage.kind == "neural"
                    for hc in stage.hard_core_mapping.cores]

        distinct = {id(law): law for law in laws_of(with_laws)}
        restored = pickle.loads(pickle.dumps(with_laws))
        assert len({id(law) for law in laws_of(restored)}) == len(distinct) < len(laws_of(restored))
        bare = copy.deepcopy(with_laws)
        for stage in bare.stages:
            for hc in (stage.hard_core_mapping.cores if stage.kind == "neural" else ()):
                del hc.readout_law
        def replaced(make):
            """``with_laws`` with every reference to one distinct law replaced by one stand-in object."""
            copied, stand_ins = copy.deepcopy(bare), {}
            for hc, original in zip(laws_of_cores(copied), laws_of(with_laws)):
                hc.readout_law = stand_ins.setdefault(id(original), make(len(stand_ins)))
            return copied

        def laws_of_cores(hybrid):
            return [hc for stage in hybrid.stages if stage.kind == "neural" for hc in stage.hard_core_mapping.cores]

        shared = ("readout_law",)
        sizes = {name: len(pickle.dumps(m)) for name, m in (
            ("bare", bare), ("one_sentinel", replaced(lambda k: shared)),
            ("one_column_laws", replaced(lambda k: CoreLaw(**dict(TestCoreLawPickle.ONE_COLUMN, bias=[k])))),
            ("laws", with_laws))}
        laws_bytes = sum(len(pickle.dumps(law)) for law in distinct.values())
        print(f"pickle sizes {sizes} B; {len(distinct)} distinct laws ({laws_bytes} B) over "
              f"{len(laws_of(with_laws))} references")
        # Over the same references to as many distinct one-column laws (the same memo structure), the laws add
        # at most their own bytes: each is stored once, as one buffer (CoreLaw.__reduce__), never one numpy
        # array per field. (A single shared sentinel can sit below memo index 256, where a reference costs a
        # 2-byte BINGET instead of a 5-byte LONG_BINGET: a memo-numbering effect, printed, not law content.)
        assert sizes["bare"] < sizes["one_sentinel"] < sizes["laws"]
        assert sizes["laws"] - sizes["one_column_laws"] <= laws_bytes

    def test_nf_and_ir_laws_agree_over_the_surviving_columns(self, pruned_lenet):
        perceptrons = pruned_lenet.model.get_perceptrons()
        compared = 0
        for core in pruned_lenet.ir.nodes:
            if not isinstance(core, NeuralCore):
                continue
            stamp = perceptrons[core.perceptron_index].scaler.stamped
            channels = core_channels(core)
            nf = CoreLaw(**{name: stamp.law[name][channels] for name in LAW_FIELDS})
            assert nf.column_digests() == core.readout_law.column_digests(), core.name
            compared += len(channels)
        assert compared > 0

    def test_pruned_non_bank_cores_are_solved_on_their_surviving_columns(self, pruned_lenet):
        owned = [c for c in pruned_lenet.ir.nodes if isinstance(c, NeuralCore) and c.weight_bank_id is None]
        assert owned
        for core in owned:
            assert core.readout_law.columns == core.get_output_count()
        assert any(core.get_output_count() < 120 for core in owned if core.perceptron_index == 2)


class TestCoreLawPickle:
    ONE_COLUMN = dict(input_step=[0.0], m0=[0], shift=[0], bias=[0], lo=[0], hi=[0], output_step=[0.0], adc_shift=[0],
                      adc_levels=[0])
    LAW = dict(input_step=[0.5, 0.25], m0=[3, 4], shift=[1, 2], bias=[-5, 6], lo=[-127, -127], hi=[127, 127],
               output_step=[0.1, 0.2], adc_shift=[0, 1], adc_levels=[7, 7])

    def test_a_law_round_trips_with_its_dtypes_and_a_fresh_memo(self):
        law = CoreLaw(**self.LAW)
        restored = pickle.loads(pickle.dumps(law))
        assert restored.digest() == law.digest() and restored.columns == 2
        assert restored.input_step.dtype == np.float64 and restored.m0.dtype == np.int64
        assert not restored.m0.flags.writeable
        assert restored.take_columns([1]) is restored.take_columns([1])

    def test_each_further_law_adds_a_few_memo_entries_not_one_per_field_array(self):
        laws = [CoreLaw(**dict(self.LAW, bias=[i, -i])) for i in range(10)]

        def memo_entries(objects):
            return sum(1 for op, _arg, _pos in pickletools.genops(pickle.dumps(objects)) if op.name == "MEMOIZE")

        assert memo_entries(laws) - memo_entries(laws[:1]) <= 3 * 9


def _soft(id_, *, law: bool) -> SoftCore:
    core = SoftCore(core_matrix=np.ones((2, 2)), axon_sources=[SpikeSource(-2, i, True, False) for i in range(2)],
                    id=id_)
    if law:
        core.readout_law = CoreLaw.blank(2)
    return core


class TestMixedHardCoreRefusal:
    @pytest.mark.parametrize("first_law", [True, False])
    def test_a_hard_core_mixing_law_and_no_law_soft_cores_refuses(self, first_law):
        hard = HardCore(8, 8)
        hard.add_softcore(_soft(0, law=first_law))
        with pytest.raises(ReadoutLawMixedCoreError, match="readout law"):
            hard.add_softcore(_soft(1, law=not first_law))

    def test_soft_cores_without_laws_leave_no_attribute(self):
        hard = HardCore(8, 8)
        hard.add_softcore(_soft(0, law=False))
        hard.add_softcore(_soft(1, law=False))
        assert "readout_law" not in vars(hard)
