"""[U-SFR] In real SANA-FE: the spike trace taken as text, one string per simulated timestep, reads as SANA-FE's own.

SANA-FE's memory trace holds every spike twice at the end of ``chip.sim`` (its C++ list of addresses and one Python
address object per spike, about 200 bytes per spike together, measured on S4's 1,024-core segments). Its trace writer
for a file-like object writes the same spikes in the same order, one ``<group>.<offset>,<timestep>`` line each and one
write per timestep. ``simulate_chip_bounded(spike_trace=SPIKE_TRACE_COMPACT)`` hands SANA-FE such an object and returns
a per-timestep sequence of the ``<group>.<offset>`` strings: ``str()`` of SANA-FE's own address objects, which every
reader of the trace parses (``_spike_event_group_and_index``, ``str(event)``).
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest


def _have_sanafe() -> bool:
    try:
        import sanafe  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _have_sanafe(), reason="SANA-FE not installed (scripts/bootstrap_sanafe.sh)"),
    pytest.mark.slow,
    pytest.mark.integration,
]


def _both_traces(monkeypatch, units=2, saturating=False):
    """Each neural stage of the chain simulated twice on fresh chips of the same network: the memory trace and the
    compact one; returns [(memory results, compact results, compact traced bytes, memory traced bytes)] per stage."""
    import integration.test_sanafe_flat_segments as F
    import mimarsinan.chip_simulation.sanafe.runner.neural_stage as ns
    from mimarsinan.chip_simulation.sanafe.runner.bounded_sim import SPIKE_TRACE_COMPACT, simulate_chip_bounded

    seen = []

    def both(sanafe_module, arch, net, timesteps, *, spike_trace, potential_trace, message_trace, timeout_s):
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        chip_m, mem = simulate_chip_bounded(sanafe_module, arch, net, timesteps, spike_trace=True,
                                            potential_trace=potential_trace, message_trace=False, timeout_s=timeout_s)
        mem_bytes = tracemalloc.get_traced_memory()[0] - before
        del chip_m
        before = tracemalloc.get_traced_memory()[0]
        chip_c, compact = simulate_chip_bounded(sanafe_module, arch, net, timesteps, spike_trace=SPIKE_TRACE_COMPACT,
                                                potential_trace=potential_trace, message_trace=False,
                                                timeout_s=timeout_s)
        compact_bytes = tracemalloc.get_traced_memory()[0] - before
        tracemalloc.stop()
        seen.append((mem, compact, compact_bytes, mem_bytes))
        return chip_c, simulate_chip_bounded(sanafe_module, arch, net, timesteps, spike_trace=spike_trace,
                                             potential_trace=potential_trace, message_trace=message_trace,
                                             timeout_s=timeout_s)[1]

    monkeypatch.setattr(ns, "simulate_chip_bounded", both)
    mapping = F._chain(units, saturating=saturating)
    pipeline = F._pipeline(__import__("pathlib").Path(__import__("tempfile").mkdtemp()), mapping)
    F._default_record(pipeline, mapping, F._sample())
    monkeypatch.undo()
    return seen


def test_the_spike_trace_taken_as_text_reads_as_sanafes_own(monkeypatch):
    from mimarsinan.chip_simulation.sanafe.analysis import _spike_event_group_and_index

    seen = _both_traces(monkeypatch)
    assert len(seen) == 2
    for mem, compact, _cb, _mb in seen:
        m, c = mem["spike_trace"], compact["spike_trace"]
        assert len(c) == len(m) > 0 and bool(c)
        assert sum(len(ev) for ev in m) > 1000
        for t in range(len(m)):
            assert list(c[t]) == [str(e) for e in m[t]], t
            assert [_spike_event_group_and_index(e) for e in c[t]] == [_spike_event_group_and_index(e) for e in m[t]]
        assert [list(evs) for evs in c] == [[str(e) for e in evs] for evs in m]
        assert [list(evs) for evs in c[2:5]] == [[str(e) for e in evs] for evs in m[2:5]]
        assert len(c[:3]) == 3 and len(c[1:]) == len(m) - 1
        for key in ("spikes", "energy", "sim_time", "packets_sent", "neurons_fired", "neurons_updated"):
            assert compact[key] == mem[key], key


def test_the_compact_trace_holds_a_spike_in_a_few_bytes(monkeypatch):
    """The memory trace retains over 40 bytes per spike in traced Python objects alone (its C++ copies beside them are
    not traced); the compact one at most 24 in all."""
    seen = _both_traces(monkeypatch, units=2, saturating=True)
    for mem, compact, compact_bytes, mem_bytes in seen:
        spikes = sum(len(ev) for ev in mem["spike_trace"])
        assert spikes > 10_000
        assert mem_bytes / spikes > 40, mem_bytes / spikes
        assert compact_bytes / spikes <= 24, compact_bytes / spikes


def test_an_empty_timestep_reads_as_an_empty_list():
    from mimarsinan.chip_simulation.sanafe.runner.segment_release import CompactSpikeTrace

    trace = CompactSpikeTrace()
    for text in ("core0.1,1\ncore3_in.12,1\n", "", "core1.0,3\n"):
        trace.write(text)
    view = trace.trace()
    assert len(view) == 3 and [list(v) for v in view] == [["core0.1", "core3_in.12"], [], ["core1.0"]]
    assert not CompactSpikeTrace().trace() and len(CompactSpikeTrace().trace()) == 0
    with pytest.raises(ValueError, match="timestep"):
        CompactSpikeTrace().write("core0.1,1\ncore0.2,2\n")
