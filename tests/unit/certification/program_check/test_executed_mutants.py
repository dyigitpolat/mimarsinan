"""[W15] Wrong programs the check must name: each mutant changes one program element an executor reads (a delivery
delay on a host-encoded slice, the level flag, a weight bank, a memo, a span cache), and, where the vehicle
discriminates, changes the HCM's output against the identity program's; the check fails it by name.

The first check established every one of them (``RA/11_wave2/exact/review_soundness.md``, B1 to B3 and M1)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from mimarsinan.mapping.packing.pass_timing import carried_nodes
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

from unit.certification.program_check import programs as P
from unit.certification.program_check.test_mutants import _failed
from unit.mapping.residency.executed import runs as R
from unit.mapping.residency.executed import vehicles as V

RUNS = {"streamed_lif": R.STREAMED_LIF, "cascaded_ttfs": R.CASCADED_TTFS, "windowed_lif": R.WINDOWED_LIF}
CHAIN_CORES = ({"max_axons": 8, "max_neurons": 8, "count": 2},)


def differing_entries(a, b, semantics, *, width: int = V.IN, batches: int = 4, **extra) -> int:
    """Output entries of the HCM run of ``b`` that differ from ``a``'s on ``batches`` seeded batches of 6."""
    n = 0
    for batch in range(batches):
        x = torch.rand(6, width, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            ya = SpikingHybridCoreFlow((width,), a, simulation_length=V.T, thresholding_mode="<=", **semantics,
                                       **extra)(x)
            yb = SpikingHybridCoreFlow((width,), b, simulation_length=V.T, thresholding_mode="<=", **semantics,
                                       **extra)(x)
        n += int((ya != yb).sum())
    return n


def live_slices(stage) -> set:
    """The input slices of ``stage`` some core reads with a nonzero weight."""
    of = {}
    for k, s in enumerate(stage.input_map):
        for i in range(int(s.offset), int(s.offset) + int(s.size)):
            of[i] = k
    out = set()
    for core in stage.hard_core_mapping.cores:
        grid = np.asarray(core.get_core_matrix())
        for a, src in enumerate(core.axon_sources):
            if src.is_input_ and not src.is_off_ and not src.is_always_on_ and a < grid.shape[0] and grid[a].any():
                out.add(of.get(int(src.neuron_)))
    return out - {None}


def delay_an_encoded_slice(program):
    """``replay_delays[k] = -1`` on the first host-encoded (not carried) slice a live row reads."""
    for stage in P.neural(program):
        carried = carried_nodes(program.stages, stage)
        delays = dict(getattr(stage, "replay_delays", None) or {})
        for k in sorted(live_slices(stage)):
            if int(stage.input_map[k].node_id) not in carried and k not in delays:
                delays[k] = -1
                stage.replay_delays = delays
                return stage, k
    raise AssertionError("no host-encoded slice is read")


@pytest.mark.parametrize("semantics", ["streamed_lif", "cascaded_ttfs"])
def test_a_delivery_delay_on_a_host_encoded_slice(semantics):
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    delay_an_encoded_slice(bad)
    assert differing_entries(P.identity_of(ir), bad, RUNS[semantics]) > 0
    _failed(P.run(bad, ir, P.decl(semantics)), "R2", "delay_on_encoded_slice")


def test_the_level_flag_on_a_streamed_pass_with_delayed_slices():
    """``is_retimed_level`` set on a streamed pass: the executor skips its raster replay and its delays."""
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    target = next(s for s in P.neural(bad) if getattr(s, "replay_delays", None))
    target.is_retimed_level = True
    assert differing_entries(P.identity_of(ir), bad, R.STREAMED_LIF) > 0
    report = P.run(bad, ir, P.decl("streamed_lif"))
    _failed(report, "H0", "level_flag")
    _failed(report, "H5-collapse", "wire_delivery")


def test_the_level_flag_on_a_pass_reading_a_host_train():
    """The flag on the first pass: it reads the encoder's cached train as a count re-encode, which the identity's
    stage reads verbatim (the per-slice kind, R4)."""
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    P.neural(bad)[0].is_retimed_level = True
    report = P.run(bad, ir, P.decl("streamed_lif"))
    _failed(report, "R4", "slice_kind")
    _failed(report, "H0", "level_flag")


def test_the_level_flag_cleared_on_the_level_units_of_a_windowed_program():
    """A level unit without its flag would consume cached trains, carried rasters and delays; named by its flag."""
    program, ir = P.windowed_multipass()
    bad = copy.deepcopy(program)
    flipped = 0
    for stage in P.neural(bad):
        for level in getattr(stage, "retimed_level_stages", None) or ():
            level.is_retimed_level, flipped = False, flipped + 1
    assert flipped
    report = P.run(bad, ir, P.decl("windowed_lif"), identity=P.identity_of(ir, retimed=True))
    _failed(report, "R5", "level_flag")


def _banks(program):
    for stage in P.neural(program):
        yield stage.hard_core_mapping.weight_banks


def test_bank_content_that_is_not_the_grid():
    """Every bank raised by 7 on its data rows, the matrix placements untouched: the HCM uploads the bank."""
    program, ir = P.resident_chain()
    bad = copy.deepcopy(program)
    for banks in _banks(bad):
        for bid in list(banks):
            fresh = np.array(banks[bid], dtype=np.float64, copy=True)
            fresh[:4, :] += 7.0
            banks[bid] = fresh
    assert differing_entries(P.identity_of(ir), bad, R.STREAMED_LIF, width=24) > 0
    report = P.run(bad, ir, P.decl("streamed_lif", core_types=CHAIN_CORES))
    _failed(report, "tiles", "bank_content")


def _widened(program):
    """Every bank widened by 4 random columns after its own, the records' ranges kept: the same program."""
    out = copy.deepcopy(program)
    rng = np.random.default_rng(3)
    for banks in _banks(out):
        for bid in list(banks):
            bank = np.asarray(banks[bid], dtype=np.float64)
            banks[bid] = np.concatenate([bank, np.round(rng.normal(size=bank.shape) * 9)], axis=1)
    return out


def test_a_bank_range_shifted_at_equal_extent():
    program, ir = P.resident_chain()
    wide = _widened(program)
    decl = P.decl("streamed_lif", core_types=CHAIN_CORES)
    assert P.run(wide, ir, decl).established("theorem1")
    bad = copy.deepcopy(wide)
    stage = P.neural(bad)[0]
    record = stage.hard_core_mapping.soft_core_placements_per_hard_core[0][0]
    n0, n1 = record["bank_neuron_range"]
    record["bank_neuron_range"] = (n0 + 2, n1 + 2)
    _failed(P.run(bad, ir, decl), "tiles", "bank_content")


def test_a_bank_range_one_axon_short_of_the_used_extent_fails_by_name():
    """[exact V4] A bank-path core whose record keeps ``axons`` at the used extent but names a bank axon range one
    axon short: the block the record names is not the core's shape, which the check names (``bank_content``) instead
    of stopping on the sweep's broadcast error."""
    from mimarsinan.mapping.packing.executed_reads import bank_read

    program, ir = P.resident_chain()
    decl = P.decl("streamed_lif", core_types=CHAIN_CORES)
    bad = copy.deepcopy(program)
    hit = None
    for stage in P.neural(bad):
        hcm = stage.hard_core_mapping
        for h, records in enumerate(hcm.soft_core_placements_per_hard_core):
            read = bank_read(hcm, h)
            if read is not None and read.axons[1] - read.axons[0] > 1:
                records[0]["bank_axon_range"] = (read.axons[0], read.axons[1] - 1)
                hit = (stage.name, h)
                break
        if hit:
            break
    assert hit is not None
    report = P.run(bad, ir, decl)
    _failed(report, "tiles", "bank_content")
    assert any("has shape" in v.text() for v in report.results["tiles"].violations)


def _encoded_host_node(program) -> int:
    """A host op whose output a pass encodes (the split-level vehicle's host_1, read by Y's pass)."""
    hosts = {int(s.compute_op.id) for s in program.stages if s.kind == "compute" and s.compute_op is not None}
    return next(int(sl.node_id) for s in P.neural(program) for sl in s.input_map if int(sl.node_id) in hosts)


def test_a_channel_table_keyed_off_the_identity_under_a_dithered_encode():
    """[E1] The program's channel table keys a host-encoded node's outputs on other channels than the identity
    program's (as a builder keying them on stage-input positions would): the dithered run departs, R4 names it."""
    ir = P.integer_ir(V.split_level_repr())
    program = V.build(ir, 1, None)
    bad = copy.deepcopy(program)
    bad.node_phase_channels = {**bad.node_phase_channels, _encoded_host_node(bad): 3}
    assert differing_entries(P.identity_of(ir), bad, R.STREAMED_LIF, phase_dither=True) > 0
    _failed(P.run(bad, ir, P.decl("streamed_lif", dithered_encode=True)), "R4", "phase_channel")


def test_a_stale_carried_output_memo():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    bad._carried_output_cache = {}
    assert differing_entries(P.identity_of(ir), bad, R.STREAMED_LIF) > 0
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "R3", "stale_memo")


def _live_rows(core):
    grid = np.asarray(core.get_core_matrix())
    used = int(core.axons_per_core) - int(core.available_axons)
    return [a for a in range(used) if np.any(grid[a] != 0) and not core.axon_sources[a].is_off_], grid


def test_a_stale_axon_span_cache():
    """Two live, differently weighted rows swapped in a core's cached spans only (``get_axon_source_spans``)."""
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    for stage in P.neural(bad):
        for core in stage.hard_core_mapping.cores:
            live, grid = _live_rows(core)
            pair = next(((x, y) for x in live for y in live if x < y and not np.array_equal(grid[x], grid[y])
                         and core.axon_sources[x] != core.axon_sources[y]), None)
            if pair is not None:
                swapped = list(core.axon_sources)
                swapped[pair[0]], swapped[pair[1]] = swapped[pair[1]], swapped[pair[0]]
                core._axon_source_spans = compress_spike_sources(swapped)
                break
        else:
            continue
        break
    assert differing_entries(P.identity_of(ir), bad, R.STREAMED_LIF) > 0
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "tiles", "stale_memo")


def test_a_stale_output_span_cache():
    """Two output positions of the last pass swapped in its cached spans only (``get_output_source_spans``)."""
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    hcm = P.neural(bad)[-1].hard_core_mapping
    out = list(np.asarray(hcm.output_sources).flatten())
    i, j = next((i, j) for i in range(len(out)) for j in range(i + 1, len(out)) if out[i] != out[j])
    out[i], out[j] = out[j], out[i]
    hcm._output_source_spans = compress_spike_sources(out)
    assert differing_entries(P.identity_of(ir), bad, R.STREAMED_LIF) > 0
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "tiles", "stale_memo")


def test_a_stale_consumer_count_memo():
    """A refcount one short frees a state buffer before its last reader."""
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    from mimarsinan.mapping.packing.executed_reads import consumer_counts

    counts = dict(consumer_counts(bad))
    node = next(n for n, c in counts.items() if c > 1)
    counts[node] -= 1
    bad._consumer_counts_cache = counts
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H9", "stale_memo")
