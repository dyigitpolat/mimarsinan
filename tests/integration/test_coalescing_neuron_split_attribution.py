"""GAP-1: per-neuron attribution reassembly under coalescing + output-tiling.

When a wide perceptron is BOTH coalesced (fan-in fused onto one wider crossbar)
AND output-tiled (its neurons split across hard cores), the packer scatters one
perceptron's neurons across several hard cores, each carrying a
``perceptron_output_slice`` (its neuron range in the original layer). The
attribution reassembler (``integration._split_reassembly._reassemble`` and the
production ``nf_scm_parity._group_record_by_perceptron``) must stitch those
fragments back in *original-layer neuron order*.

The cracked keying concatenated a perceptron's cores in ``sorted(ir_node_id)``
order, which equals the slice order ONLY when the IR-id assignment happens to be
monotone in the output slice. At VGG scale, IR-graph pruning/compaction reorders
ids, so ``sorted(ir_id) != slice order`` and ~2% of neurons are mis-attributed —
while the deployed value (wired positionally through ``output_sources``) and the
aggregate spike counts stay bit-exact.

This file locks both:

1. ``test_coalescing_output_tile_end_to_end_reassembly_is_exact`` — a genuine
   real-model LIF run through the coalescing+output-tile packing: the reassembled
   per-neuron counts equal the torch ground truth. (Value-domain regression
   guard: this holds before AND after the fix, since the converter assigns ids in
   slice order.)
2. ``test_scrambled_id_order_reassembly_is_exact`` — the genuine collision: real
   ``SoftCore`` tiles placed through the real ``HardCoreMapping.merge_softcore_into``
   with ir-ids OUT of slice order (the compaction-reordered state), driven by a
   genuine ``RunRecord``. With the old ``sorted(ir_id)`` keying this scrambles the
   per-neuron attribution; the joint ``(ir_core_id, neuron_range)`` keying makes
   it bit-exact.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.recording.records import (
    CoreSpikeCounts,
    RunRecord,
    SegmentSpikeRecord,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.softcore.hard_core import HardCore
from mimarsinan.mapping.packing.softcore.hard_core_mapping import HardCoreMapping
from mimarsinan.mapping.packing.softcore.soft_core import SoftCore
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

from integration._split_reassembly import (
    hcm_per_perceptron_counts,
    torch_lif_node_counts,
)
from integration._torch_sim_fidelity import (
    MappingConfig,
    assert_config_triggered,
    build_torch_and_hcm,
    mapping_structure,
)

T = 8
INPUT_SHAPE = (16,)
NUM_CLASSES = 10

# ir_max_axons=16 < every on-chip fan-in (>=21) -> coalescing.
# ir_max_neurons=8 < every on-chip width (>=20) -> output tiling.
# core_max_axons=16 < fan-in -> the coalesced tiles FUSE (N 16-axon cores -> one
# wider crossbar). core_max_neurons=64 wide -> a fused tile fits without an extra
# HCM neuron-split (coalesced cores are not HCM-splittable by design).
_COALESCE_TILE_CFG = MappingConfig(
    "coalesce_outtile",
    ir_max_axons=16,
    ir_max_neurons=8,
    core_max_axons=16,
    core_max_neurons=64,
    allow_neuron_splitting=True,
    allow_coalescing=True,
)


class _WideOnChip(nn.Module):
    """input -> fc1 (encoding host) -> on-chip wide layers that fuse AND tile."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 24)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(24, 20)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 18)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(18, 10)
        self.act4 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return self.act4(self.fc4(x))


def _samples(n, seed):
    torch.manual_seed(seed)
    return torch.rand(n, 16)


def test_coalescing_output_tile_genuinely_fuses_and_tiles():
    """Pin that the trigger config really fuses AND output-tiles the same
    perceptron (so the reassembly test is not vacuous)."""
    torch.manual_seed(0)
    flow, hcm, hybrid, nodes = build_torch_and_hcm(
        _WideOnChip(), INPUT_SHAPE, NUM_CLASSES,
        spiking_mode="lif", config=_COALESCE_TILE_CFG, T=T,
    )
    s = mapping_structure(hybrid)
    assert s["fused_cores"] > 0, ("config did not fuse — coalescing not triggered", s)
    assert s["psum_partials"] == 0, ("firing partials must never appear", s)

    # A perceptron must span >1 fused hard core (genuine output tiling under fuse).
    perceptron_core_count: dict[int, int] = {}
    perceptron_fused = set()
    for stage in hybrid.stages:
        if stage.hard_core_mapping is None:
            continue
        placements = stage.hard_core_mapping.soft_core_placements_per_hard_core
        for ci, plist in enumerate(placements):
            fused = bool(
                getattr(stage.hard_core_mapping.cores[ci], "fused_component_axons", None)
            )
            for pl in plist:
                pi = pl.get("perceptron_index")
                if pi is None or pi < 0:
                    continue
                perceptron_core_count[pi] = perceptron_core_count.get(pi, 0) + 1
                if fused:
                    perceptron_fused.add(pi)
    tiled = {pi for pi, c in perceptron_core_count.items() if c > 1}
    assert tiled, ("no perceptron was output-tiled across cores", perceptron_core_count)
    assert tiled & perceptron_fused, (
        "the output-tiled perceptron(s) are not also fused — coalescing+tiling did "
        "not co-occur on the same perceptron",
        (tiled, perceptron_fused),
    )


def test_coalescing_output_tile_end_to_end_reassembly_is_exact():
    """Genuine real-model LIF run: reassembled per-neuron counts == torch.

    The deployed value path wires positionally through ``output_sources``, so this
    holds before and after the attribution fix — it is the value-domain-unchanged
    regression guard for the converter-order (slice-monotone id) case.
    """
    torch.manual_seed(0)
    flow, hcm, hybrid, nodes = build_torch_and_hcm(
        _WideOnChip(), INPUT_SHAPE, NUM_CLASSES,
        spiking_mode="lif", config=_COALESCE_TILE_CFG, T=T,
    )
    assert_config_triggered(hybrid, "axon_fuse")

    sample = _samples(1, seed=13).float()
    torch_counts = torch_lif_node_counts(
        lambda: chip_aligned_segment_forward(flow, sample, T), nodes, sample, T,
    )
    with torch.no_grad():
        _, record = hcm.forward_with_recording(sample, sample_index=0)
    hcm_counts = hcm_per_perceptron_counts(record, hybrid)

    compared = [pi for pi in torch_counts if pi in hcm_counts]
    assert compared, "no on-chip perceptron compared"
    for pi in compared:
        t, h = torch_counts[pi], hcm_counts[pi]
        assert t.shape == h.shape, (pi, t.shape, h.shape)
        assert np.array_equal(t, h), (
            f"perceptron {pi}: {int((t != h).sum())}/{t.size} neurons mis-attributed "
            f"(aggregate torch={int(t.sum())} hcm={int(h.sum())})"
        )


# --- the genuine collision: ir-id order decoupled from slice order ---


def _place_tile(hcm_map, core_idx, *, ir_id, pi, slice_, coal_gid, counts):
    """Place one genuine output-tile SoftCore through the real placement code.

    The tile carries genuine coalescing-master + ``perceptron_output_slice``
    provenance. ``counts`` are this tile's per-neuron output spike counts (the
    ground-truth values for its slice of the original layer).
    """
    width = slice_[1] - slice_[0]
    soft = SoftCore(
        core_matrix=np.zeros((1, width), dtype=np.float64),
        axon_sources=[SpikeSource(-1, 0, is_input=True, is_off=False)],
        id=ir_id,
    )
    soft.coalescing_group_id = coal_gid
    soft.coalescing_role = "master"
    soft.perceptron_index = pi
    soft.perceptron_output_slice = (int(slice_[0]), int(slice_[1]))
    soft.latency = 0
    hardcore = HardCore(axons_per_core=1, neurons_per_core=width, has_bias_capability=False)
    hcm_map.cores.append(hardcore)
    hcm_map.merge_softcore_into(core_idx, hardcore, soft)
    return CoreSpikeCounts(
        core_index=core_idx,
        n_in_used=1,
        n_out_used=width,
        core_latency=0,
        has_hardware_bias=False,
        n_always_on_axons=0,
        input_spike_count=np.zeros(1, dtype=np.int64),
        output_spike_count=np.asarray(counts, dtype=np.int64),
    )


def test_scrambled_id_order_reassembly_is_exact():
    """The compaction-reordered state: a coalesced+output-tiled perceptron whose
    tiles' ir-ids are NOT monotone in the output slice.

    Old keying (``sorted(ir_id)``) concatenates the tiles in id order, scrambling
    the per-neuron attribution; the joint ``(ir_core_id, neuron_range)`` keying
    orders by the original-layer neuron position and is bit-exact.
    """
    # One perceptron, width 12, output-tiled into three slices [0,4) [4,8) [8,12).
    # Ground-truth per-neuron values, distinct per neuron so any mis-order shows.
    ground_truth = np.arange(1, 13, dtype=np.int64)  # 1..12
    tiles = [
        # (ir_id, slice, counts) — ir-ids DELIBERATELY out of slice order.
        (30, (0, 4), ground_truth[0:4]),
        (10, (4, 8), ground_truth[4:8]),
        (20, (8, 12), ground_truth[8:12]),
    ]
    # sorted(ir_id) = [10, 20, 30] -> slices [4,8),[8,12),[0,4): a real scramble.

    hcm_map = HardCoreMapping(chip_cores=[])
    core_records = []
    for ci, (ir_id, slc, counts) in enumerate(tiles):
        core_records.append(
            _place_tile(
                hcm_map, ci, ir_id=ir_id, pi=1, slice_=slc, coal_gid=0, counts=counts,
            )
        )

    # Genuine single-stage hybrid mapping + record (the real reassembler reads these).
    class _Stage:
        hard_core_mapping = hcm_map

    class _Hybrid:
        stages = [_Stage()]

    seg = SegmentSpikeRecord(
        stage_index=0,
        stage_name="seg",
        schedule_segment_index=None,
        schedule_pass_index=None,
        seg_input_rates=np.zeros((1, 1), dtype=np.float32),
        seg_input_spike_count=np.zeros(1, dtype=np.int64),
        seg_output_spike_count=np.zeros(12, dtype=np.int64),
        cores=core_records,
    )
    record = RunRecord(sample_index=0, T=T, segments={0: seg})

    reassembled = hcm_per_perceptron_counts(record, _Hybrid())
    assert set(reassembled) == {1}, reassembled
    got = reassembled[1]
    assert np.array_equal(got, ground_truth), (
        f"per-neuron attribution scrambled under coalescing+output-tiling with "
        f"out-of-slice ir-id order:\n got={got.tolist()}\n exp={ground_truth.tolist()}\n"
        f"mis-attributed {int((got != ground_truth).sum())}/{ground_truth.size} neurons"
    )
