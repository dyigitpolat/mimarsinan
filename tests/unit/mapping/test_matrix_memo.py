"""Content-keyed materialization memo: share resolved grids, never recompute them.

Measured defect (real 4,925-core ViT, 2026-08-07): every ``get_core_matrix()``
returned a FRESH array, so one snapshot walk materialized 92.9 GB in 49.6 s and
every downstream identity-keyed cache missed — while those 4,925 grids are only
**27 distinct payloads**. The memo keys on ``core_matrix_key()`` (content
identity, with payload re-check) so equal content resolves once.

Contract: a resolved grid MAY be shared, so callers must not mutate it — the
exact-fit plain path already returned the shared bank payload before this memo.
"""

import numpy as np
import pytest

from mimarsinan.mapping.packing.softcore.hard_core import HardCore
from mimarsinan.mapping.packing.softcore.matrix_memo import (
    matrix_memo_stats,
    reset_matrix_memo,
)
from mimarsinan.mapping.packing.softcore.matrix_placement import (
    MatrixPlacement,
    resolve_core_matrix,
)


@pytest.fixture(autouse=True)
def _clean_memo():
    reset_matrix_memo()
    yield
    reset_matrix_memo()


def _bank(rows=6, cols=4, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, cols)).astype(np.float64)


def _core_on(bank, *, keep_rows=None, keep_cols=None, axons=None, neurons=None):
    """A hard core holding one compacted placement over ``bank``."""
    kr = None if keep_rows is None else np.asarray(keep_rows, dtype=np.int64)
    kc = None if keep_cols is None else np.asarray(keep_cols, dtype=np.int64)
    n_ax = axons if axons is not None else (len(kr) if kr is not None else bank.shape[0])
    n_ne = neurons if neurons is not None else (len(kc) if kc is not None else bank.shape[1])
    core = HardCore(n_ax, n_ne)
    core.matrix_placements = [MatrixPlacement(
        source=bank, keep_rows=kr, keep_cols=kc,
        axon_offset=0, neuron_offset=0, axons=n_ax, neurons=n_ne,
    )]
    core.available_axons = 0
    core.available_neurons = 0
    return core


class TestTheMemoShares:
    def test_repeat_calls_return_the_same_object(self):
        bank = _bank()
        core = _core_on(bank, keep_rows=[0, 2, 4], keep_cols=[1, 3])
        a, b = core.get_core_matrix(), core.get_core_matrix()
        assert a is b, "a second resolve must hit the memo, not re-materialize"

    def test_equal_content_across_cores_shares_one_array(self):
        """The real-vehicle case: 197 instances of one bank slice = 1 payload."""
        bank = _bank()
        cores = [_core_on(bank, keep_rows=[0, 2, 4], keep_cols=[1, 3])
                 for _ in range(8)]
        grids = [c.get_core_matrix() for c in cores]
        assert all(g is grids[0] for g in grids)
        assert matrix_memo_stats()["entries"] == 1

    def test_distinct_content_is_not_conflated(self):
        bank = _bank()
        a = _core_on(bank, keep_rows=[0, 2, 4], keep_cols=[1, 3])
        b = _core_on(bank, keep_rows=[1, 3, 5], keep_cols=[1, 3])
        ga, gb = a.get_core_matrix(), b.get_core_matrix()
        assert ga is not gb
        assert not np.array_equal(ga, gb)
        assert matrix_memo_stats()["entries"] == 2


class TestTheMemoIsBitExact:
    @pytest.mark.parametrize("keep", [
        ([0, 2, 4], [1, 3]), ([1, 3, 5], [0, 2, 3]), (None, None),
    ])
    def test_memoized_equals_uncached_materialization(self, keep):
        bank = _bank()
        kr, kc = keep
        core = _core_on(bank, keep_rows=kr, keep_cols=kc)
        got = core.get_core_matrix()
        want = resolve_core_matrix(
            None, core.matrix_placements,
            core.axons_per_core, core.neurons_per_core, "oracle",
        )
        assert got.dtype == want.dtype and got.shape == want.shape
        assert [float(v).hex() for v in got.flatten()] == \
               [float(v).hex() for v in want.flatten()]

    def test_composite_multi_placement_grid_matches(self):
        bank = _bank(rows=4, cols=3)
        core = HardCore(8, 6)
        core.matrix_placements = [
            MatrixPlacement(source=bank, keep_rows=None, keep_cols=None,
                            axon_offset=0, neuron_offset=0, axons=4, neurons=3),
            MatrixPlacement(source=bank, keep_rows=None, keep_cols=None,
                            axon_offset=4, neuron_offset=3, axons=4, neurons=3),
        ]
        core.available_axons = 0
        core.available_neurons = 0
        got = core.get_core_matrix()
        want = resolve_core_matrix(None, core.matrix_placements, 8, 6, "oracle")
        assert np.array_equal(got, want)
        assert got[0, 3] == 0.0 and got[4, 0] == 0.0, "padding stays zero"


class TestTheMemoIsBounded:
    def test_eviction_keeps_results_correct(self, capsys):
        """A tiny budget must evict, never corrupt — and must SAY it evicted."""
        reset_matrix_memo(budget_bytes=1)          # forces eviction every insert
        bank = _bank()
        first = _core_on(bank, keep_rows=[0, 2, 4], keep_cols=[1, 3])
        second = _core_on(bank, keep_rows=[1, 3, 5], keep_cols=[0, 2])
        g1 = first.get_core_matrix()
        g2 = second.get_core_matrix()
        again = first.get_core_matrix()
        want = resolve_core_matrix(
            None, first.matrix_placements, first.axons_per_core,
            first.neurons_per_core, "oracle",
        )
        assert np.array_equal(again, want), "an evicted entry must re-materialize correctly"
        assert not np.array_equal(g1, g2)
        assert matrix_memo_stats()["evictions"] >= 1
        assert "MatrixMemo" in capsys.readouterr().out, "eviction must be logged, never silent"


class TestPayloadIdentityIsRechecked:
    def test_a_stale_key_does_not_serve_a_wrong_grid(self):
        """Keys embed payload ``id``s; a re-used id must not alias."""
        bank = _bank()
        core = _core_on(bank, keep_rows=[0, 2, 4], keep_cols=[1, 3])
        core.get_core_matrix()
        stats_before = matrix_memo_stats()["entries"]
        # Same key shape, DIFFERENT payload object with different values.
        other = _bank(seed=99)
        core.matrix_placements[0].source = other
        got = core.get_core_matrix()
        want = resolve_core_matrix(
            None, core.matrix_placements, core.axons_per_core,
            core.neurons_per_core, "oracle",
        )
        assert np.array_equal(got, want), "payload swap must invalidate the entry"
        assert matrix_memo_stats()["entries"] >= stats_before


class TestLegacyPickleStatesLoad:
    """[verifier major] pre-descriptor pickles carry no ``matrix_placements``."""

    def test_owned_dense_legacy_state_resolves(self):
        core = HardCore(3, 2)
        legacy = {
            "axons_per_core": 3, "neurons_per_core": 2,
            "has_bias_capability": True,
            "core_matrix": np.arange(6, dtype=np.float64).reshape(3, 2),
            "axon_sources": [], "available_axons": 0, "available_neurons": 0,
            "hardware_bias": None, "threshold": 1.0, "latency": 0,
            "unusable_space": 0,
        }
        core.__setstate__(legacy)
        assert core.matrix_placements == []
        assert np.array_equal(core.get_core_matrix(), legacy["core_matrix"])
        assert core.core_matrix_key() is not None
        assert core.core_matrix_dtype() == np.dtype(np.float64)


class TestOutputSourcesPickleCompressed:
    """[verifier major] Unit 4 compressed HardCore.axon_sources but not
    HardCoreMapping.output_sources, which still pickled as object soup."""

    @staticmethod
    def _mapping_with_outputs(n):
        from mimarsinan.mapping.packing.softcore.hard_core_mapping import (
            HardCoreMapping,
        )
        from mimarsinan.code_generation.cpp_chip_model_types import SpikeSource

        hcm = HardCoreMapping([])
        hcm.output_sources = np.array(
            [SpikeSource(0, i % 7, is_input=False, is_off=False) for i in range(n)],
            dtype=object,
        )
        return hcm

    def test_round_trip_preserves_every_field(self):
        import pickle

        hcm = self._mapping_with_outputs(64)
        got = pickle.loads(pickle.dumps(hcm))
        want = list(np.asarray(hcm.output_sources).flatten())
        have = list(np.asarray(got.output_sources).flatten())
        assert len(have) == len(want)
        for a, b in zip(have, want):
            assert (a.core_, a.neuron_, a.is_input_, a.is_off_) == \
                   (b.core_, b.neuron_, b.is_input_, b.is_off_)

    def test_pickle_grows_sublinearly_in_source_count(self):
        import pickle

        small = len(pickle.dumps(self._mapping_with_outputs(200)))
        large = len(pickle.dumps(self._mapping_with_outputs(4000)))
        per_source = (large - small) / 3800
        assert per_source < 8, (
            f"{per_source:.1f} B/source — output_sources are still pickling "
            f"as objects (~27.7 B/source)"
        )
