"""[W04] The core-state index answers a bank-affine softcore too: the cores that host its bank, then one core per state.

A bank-affine softcore's placement cost is (0 on a core that hosts its bank, else its area; the remaining capacity).
The best core is therefore the tightest feasible core among those hosting the bank, first in candidate order on a tie,
when one exists, and otherwise the bankless best fit, which the state index already answers exactly. W04 gave layout
specs their bank under the name the engine reads (``weight_bank_id``), so every bank-backed spec of the shape-only
plane scanned every used core: the same unscheduled pack of EfficientNet-B0 on 256 x 256 (14,112 specs) took 1.26 s
before W04 and 11.99 s after, 5.2 million predicate calls (``RA/11_wave2/pack/parity/probe_speed``). The index must
take exactly the decisions of the full scan (the reference engine) and read only the cores the cost can prefer.
"""

from __future__ import annotations

import random

import pytest

import mimarsinan.mapping.layout.layout_packer as lp
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.canonical import canonical_is_mapping_possible
from mimarsinan.mapping.packing.core_packing import greedy_pack_softcores
from mimarsinan.mapping.packing.placement_cost import note_resident_bank

from unit.mapping.packing import reference_engine as ref
from unit.mapping.packing.test_engine_equivalence import Hard, Soft, _fuse, _outcome, _place, _split


def _bank_heavy(seed):
    """Many instances of few banks, mixed classes and latencies, a few bankless and empty softcores, and used cores
    that already host banks when the pack starts."""
    rng = random.Random(seed)
    banks = list(range(rng.randint(1, 5)))
    palette = [(8, 8), (16, 4), (4, 16), (12, 12), (24, 6), (40, 8), (0, 3), (5, 0)]
    softs = []
    for i in range(rng.randint(1, 120)):
        a, b = rng.choice(palette) if rng.random() < 0.8 else (rng.randint(0, 70), rng.randint(0, 70))
        softs.append(Soft(a, b, f"s{i}", tg=rng.choice([None, 0, 1]) if rng.random() < 0.85 else None,
                          latency=rng.choice([None, 0, 1]),
                          bank=rng.choice(banks) if rng.random() < 0.9 else None))
    used = []
    for j in range(rng.randint(0, 6)):
        hard = Hard(rng.choice([32, 48, 64]), rng.choice([32, 48, 64]), f"u{j}", tg=rng.choice([None, 0, 1]))
        hard.available_axons -= rng.randint(0, 20)
        hard.available_neurons -= rng.randint(0, 20)
        for bank in rng.sample(banks, rng.randint(0, len(banks))):
            note_resident_bank(hard, Soft(1, 1, "seed", bank=bank))
        used.append(hard)
    types = rng.sample([(32, 32), (64, 64), (64, 16), (16, 64), (48, 48)], rng.randint(1, 3))
    unused = [Hard(a, b, f"h{t}.{k}") for t, (a, b) in enumerate(types) for k in range(rng.randint(0, 40))]
    rng.shuffle(unused)
    return softs, used, unused, rng


def _run(engine, seed, *, split, fuse, indexed):
    softs, used, unused, rng = _bank_heavy(seed)
    trace = []
    kwargs = dict(softcores=softs, used_hardcores=used, unused_hardcores=unused,
                  is_mapping_possible=canonical_is_mapping_possible, place=_place(trace),
                  fuse_hardcores=_fuse if fuse else None, split_softcore=_split if split else None,
                  split_threshold=rng.choice([0.0, 0.2, 0.5]))
    if indexed:
        kwargs["canonical_feasibility"] = True
    try:
        engine(**kwargs)
        outcome = None
    except RuntimeError as exc:
        outcome = str(exc)
    return {"trace": trace, "outcome": outcome,
            "used": [(h.label, h.available_axons, h.available_neurons) for h in used],
            "unused": [h.label for h in unused], "left": sorted(s.label for s in softs)}


MODES = [dict(split=s, fuse=f) for s in (False, True) for f in (False, True)]


@pytest.mark.parametrize("mode", MODES, ids=lambda m: f"split{int(m['split'])}fuse{int(m['fuse'])}")
def test_the_index_takes_the_full_scans_decisions_for_bank_affine_softcores(mode):
    for seed in range(200):
        want = _run(ref.reference_greedy_pack_softcores, seed, indexed=False, **mode)
        assert _run(greedy_pack_softcores, seed, indexed=True, **mode) == want, seed
        assert _run(greedy_pack_softcores, seed, indexed=False, **mode) == want, seed


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("coalesce", [False, True])
def test_pack_layout_of_bank_backed_specs_equals_the_reference(split, coalesce):
    for seed in range(120):
        rng = random.Random(1000 + seed)
        banks = list(range(rng.randint(1, 4)))
        specs = [LayoutSoftCoreSpec(rng.choice([16, 48, 96, 144, 300]), rng.choice([8, 16, 40, 300]),
                                    residency_class_id=rng.choice([0, 1, 2]), latency_tag=rng.choice([0, 1]),
                                    bank_id=rng.choice(banks) if rng.random() < 0.85 else None)
                 for _ in range(rng.randint(0, 90))]
        types = [LayoutHardCoreType(a, b, rng.choice([1, 3, 12, 40, 200]))
                 for a, b in rng.sample([(256, 256), (128, 128), (256, 64), (64, 256)], rng.randint(1, 2))]
        kw = dict(softcores=specs, core_types=types, allow_neuron_splitting=split, allow_coalescing=coalesce,
                  collect_placements=True)
        assert _outcome(lp.pack_layout, **kw) == _outcome(ref.reference_pack_layout, **kw), seed


def test_a_bank_affine_best_fit_reads_the_cores_of_its_bank_and_one_core_per_state(monkeypatch):
    """The bank-affine twin of ``test_best_fit_reads_one_core_per_state``: 6,000 conv-position specs of 60 banks in
    3 classes read a bounded number of cores each, not every used core of their class (about 63 each before)."""
    calls = {"n": 0}

    def counting(soft, hard):
        calls["n"] += 1
        return canonical_is_mapping_possible(soft, hard)

    monkeypatch.setattr(lp, "canonical_is_mapping_possible", counting)
    n = 6000
    specs = [LayoutSoftCoreSpec(16, 16, residency_class_id=i % 3, latency_tag=0, bank_id=i // 100) for i in range(n)]
    result = lp.pack_layout(softcores=specs, core_types=[LayoutHardCoreType(256, 256, n)])
    assert result.feasible
    assert calls["n"] <= 12 * n
