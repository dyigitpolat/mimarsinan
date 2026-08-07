"""[§17 pricing] the vectorized ChipLatency walk is bit-equal to the recursive
reference on randomized DAG mappings (the recursive walk stays the oracle)."""

from __future__ import annotations

from fake_cores import FakeCore
from types import SimpleNamespace

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.latency.chip import ChipLatency


def _make_core(axon_sources, weights):
    return FakeCore(
        axon_sources=list(axon_sources),
        core_matrix=np.asarray(weights, dtype=np.float32),
        latency=None,
    )


def _random_dag_mapping(rng, n_cores=6, max_axons=5, max_neurons=4):
    cores = []
    for ci in range(n_cores):
        n_ax = int(rng.integers(1, max_axons + 1))
        n_out = int(rng.integers(1, max_neurons + 1))
        sources = []
        for _ in range(n_ax):
            kind = rng.integers(0, 4)
            if kind == 0 or ci == 0:
                sources.append(SpikeSource(-2, int(rng.integers(0, 4)), False, False))
            elif kind == 1:
                sources.append(SpikeSource(-1, 0, True, False))
            elif kind == 2:
                sources.append(SpikeSource(-1, 0, False, True))
            else:
                src_core = int(rng.integers(0, ci))
                src_neuron = int(
                    rng.integers(0, cores[src_core].core_matrix.shape[1]))
                sources.append(SpikeSource(src_core, src_neuron, False, False))
        weights = rng.integers(-2, 3, size=(n_ax, n_out)).astype(np.float32)
        cores.append(_make_core(sources, weights))
    outputs = []
    for _ in range(4):
        ci = int(rng.integers(0, n_cores))
        outputs.append(SpikeSource(
            ci, int(rng.integers(0, cores[ci].core_matrix.shape[1])),
            False, False))
    return SimpleNamespace(cores=cores, output_sources=outputs)


def test_vectorized_delays_match_the_recursive_oracle_on_random_dags():
    rng = np.random.default_rng(0)
    for _trial in range(20):
        mapping = _random_dag_mapping(rng)

        vec = ChipLatency(mapping)
        vec.memo = {}
        vec._compute_all_delays()

        ref = ChipLatency(mapping)
        ref.memo = {}
        for ci, core in enumerate(mapping.cores):
            for j in range(core.core_matrix.shape[1]):
                expected = ref.get_delay_for(SpikeSource(ci, j, False, False))
                assert vec.memo[(ci, j)] == expected, (
                    f"core {ci} neuron {j}: vectorized {vec.memo[(ci, j)]} "
                    f"!= recursive {expected}"
                )


def test_calculate_end_state_unchanged_on_a_random_dag():
    rng = np.random.default_rng(7)
    mapping_a = _random_dag_mapping(rng)
    rng = np.random.default_rng(7)
    mapping_b = _random_dag_mapping(rng)

    result = ChipLatency(mapping_a).calculate()

    ref = ChipLatency(mapping_b)
    ref.memo = {}
    per_core = {}
    for src in mapping_b.output_sources:
        ref.get_delay_for(src)
    for ci, core in enumerate(mapping_b.cores):
        for j in range(core.core_matrix.shape[1]):
            d = ref.get_delay_for(SpikeSource(ci, j, False, False))
            per_core[ci] = max(per_core.get(ci, 0), d - 1)
    assert result >= 0
    for ci, core in enumerate(mapping_a.cores):
        if core.latency is not None and ci in per_core:
            assert core.latency >= per_core[ci]
