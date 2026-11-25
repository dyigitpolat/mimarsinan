"""[U-DG] The process-lifetime CUDA peak: a per-phase reset of torch's peak statistics no longer hides the session's
peak. ``diagnostics.reset_cuda_peak_stats`` folds the running peaks into the lifetime maxima before it resets, every
reset site in src resets through it, and ``diagnostics.lifetime_cuda_peaks`` returns the maxima with the running peak.
Per-phase readings stay as they were; without CUDA nothing changes and the lifetime reading is a named empty result."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

import mimarsinan.common.diagnostics as diagnostics

SRC = Path(diagnostics.__file__).resolve().parents[1]
MIB = 1 << 20


class FakeCuda:
    """torch.cuda's peak statistics for two devices: current and peak bytes, allocated and reserved."""

    def __init__(self, current_device: int = 0):
        self.device = current_device
        self.cur = {d: {"allocated": 0, "reserved": 0} for d in (0, 1)}
        self.peak = {d: {"allocated": 0, "reserved": 0} for d in (0, 1)}
        self.resets: list[int] = []

    def _index(self, device) -> int:
        if device is None:
            return self.device
        if isinstance(device, int):
            return device
        d = torch.device(device)
        return self.device if d.index is None else int(d.index)

    def alloc(self, n_bytes: int, device: int | None = None, reserve_extra: int = 0) -> None:
        d = self._index(device)
        self.cur[d]["allocated"] += n_bytes
        self.cur[d]["reserved"] = max(self.cur[d]["reserved"], self.cur[d]["allocated"] + reserve_extra)
        for k in ("allocated", "reserved"):
            self.peak[d][k] = max(self.peak[d][k], self.cur[d][k])

    def free(self, n_bytes: int, device: int | None = None) -> None:
        d = self._index(device)
        self.cur[d]["allocated"] -= n_bytes

    def empty_cache(self) -> None:
        for d in self.cur:
            self.cur[d]["reserved"] = self.cur[d]["allocated"]

    def install(self, monkeypatch) -> "FakeCuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: self.device)
        monkeypatch.setattr(torch.cuda, "empty_cache", self.empty_cache)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated",
                            lambda device=None: self.peak[self._index(device)]["allocated"])
        monkeypatch.setattr(torch.cuda, "max_memory_reserved",
                            lambda device=None: self.peak[self._index(device)]["reserved"])

        def reset(device=None):
            d = self._index(device)
            self.resets.append(d)
            self.peak[d] = dict(self.cur[d])

        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
        return self


@pytest.fixture
def fresh_lifetime(monkeypatch):
    """Each test starts from an empty process-lifetime record (absent before U-DG: the test then fails at its call)."""
    monkeypatch.setattr(diagnostics, "_LIFETIME_PEAKS", {}, raising=False)


def _two_phases(fake: FakeCuda, lines: list) -> None:
    """Phase 1 peaks at 900 MiB allocated (1,000 MiB reserved), phase 2 at 100 MiB, as the pipeline runs them."""
    with diagnostics.phase_profiler("tag", "first", sink=lines.append):
        fake.alloc(900 * MIB, reserve_extra=100 * MIB)
        fake.free(900 * MIB)
    fake.empty_cache()
    with diagnostics.phase_profiler("tag", "second", sink=lines.append):
        fake.alloc(100 * MIB)
        fake.free(100 * MIB)


class TestLifetimePeakOverPhases:
    def test_the_lifetime_peak_is_the_first_phases_peak(self, monkeypatch, fresh_lifetime):
        fake = FakeCuda().install(monkeypatch)
        lines: list = []
        _two_phases(fake, lines)
        peaks = diagnostics.lifetime_cuda_peaks(0)
        assert peaks.absent is None
        assert peaks.device == 0
        assert peaks.max_memory_allocated_bytes == 900 * MIB
        assert peaks.max_memory_reserved_bytes == 1000 * MIB
        # torch's own reading after the session is the last phase's only: what the lifetime reading corrects
        assert torch.cuda.max_memory_allocated(0) == 100 * MIB

    def test_per_phase_readings_stay_as_they_were(self, monkeypatch, fresh_lifetime):
        fake = FakeCuda().install(monkeypatch)
        lines: list = []
        _two_phases(fake, lines)
        # the line the profiler printed before U-DG: its own phase's peak, in the same format
        assert len(lines) == 2
        assert re.fullmatch(r"\[tag\] phase=first t=[0-9.]+s rss_delta=[+-][0-9.]+ MiB cuda_peak=900\.0 MiB", lines[0])
        assert re.fullmatch(r"\[tag\] phase=second t=[0-9.]+s rss_delta=[+-][0-9.]+ MiB cuda_peak=100\.0 MiB", lines[1])
        assert fake.resets == [0, 0]

    def test_the_running_peak_counts_without_a_later_reset(self, monkeypatch, fresh_lifetime):
        fake = FakeCuda().install(monkeypatch)
        _two_phases(fake, [])
        fake.alloc(2000 * MIB)  # after the last phase, before any reset
        assert diagnostics.lifetime_cuda_peaks(0).max_memory_allocated_bytes == 2000 * MIB

    def test_each_device_keeps_its_own_lifetime_peak(self, monkeypatch, fresh_lifetime):
        fake = FakeCuda(current_device=1).install(monkeypatch)
        _two_phases(fake, [])  # the profiler resets the current device, 1
        assert diagnostics.lifetime_cuda_peaks(1).max_memory_allocated_bytes == 900 * MIB
        assert diagnostics.lifetime_cuda_peaks(0).max_memory_allocated_bytes == 0
        assert diagnostics.lifetime_cuda_peaks(torch.device("cuda", 1)).max_memory_allocated_bytes == 900 * MIB
        assert diagnostics.lifetime_cuda_peaks(None).device == 1

    def test_the_helper_resets_the_device_it_names(self, monkeypatch, fresh_lifetime):
        fake = FakeCuda().install(monkeypatch)
        fake.alloc(300 * MIB, device=1)
        fake.free(300 * MIB, device=1)
        diagnostics.reset_cuda_peak_stats(1)
        assert fake.resets == [1]
        assert torch.cuda.max_memory_allocated(1) == 0
        assert diagnostics.lifetime_cuda_peaks(1).max_memory_allocated_bytes == 300 * MIB

    def test_the_record_is_a_plain_mapping_for_json(self, monkeypatch, fresh_lifetime):
        fake = FakeCuda().install(monkeypatch)
        _two_phases(fake, [])
        assert diagnostics.lifetime_cuda_peaks(0).as_dict() == {
            "scope": "process lifetime", "device": 0, "max_memory_allocated_bytes": 900 * MIB,
            "max_memory_reserved_bytes": 1000 * MIB, "absent": None}


class TestEveryResetSiteUsesTheHelper:
    def test_phase_profiler_resets_through_the_helper(self, monkeypatch, fresh_lifetime):
        FakeCuda().install(monkeypatch)
        calls = []
        real = diagnostics.reset_cuda_peak_stats
        monkeypatch.setattr(diagnostics, "reset_cuda_peak_stats", lambda device=None: (calls.append(device),
                                                                                         real(device)))
        with diagnostics.phase_profiler("tag", "p", sink=lambda _line: None):
            pass
        assert calls == [None]

    def test_torch_mapping_step_resets_through_the_helper(self, monkeypatch, fresh_lifetime):
        from mimarsinan.pipelining.pipeline_steps.config import torch_mapping_step as tms

        fake = FakeCuda().install(monkeypatch)
        fake.alloc(700 * MIB)  # an earlier step's peak, still the running peak when the step starts
        fake.free(700 * MIB)

        class Stop(Exception):
            pass

        step = object.__new__(tms.TorchMappingStep)

        def first_entry(_key):
            raise Stop

        monkeypatch.setattr(step, "get_entry", first_entry, raising=False)
        with pytest.raises(Stop):
            step.process()
        assert fake.resets == [0]  # the step still resets the peak (its own reading stays per step)
        assert diagnostics.lifetime_cuda_peaks(0).max_memory_allocated_bytes == 700 * MIB

    def test_no_site_in_src_resets_the_peak_statistics_directly(self):
        direct = re.compile(r"torch\.cuda\.(reset_peak_memory_stats|reset_max_memory_allocated|"
                            r"reset_max_memory_cached)\s*\(")
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if direct.search(line):
                    offenders.append(f"{path.relative_to(SRC)}:{n}")
        assert offenders == _helper_reset_sites(), offenders


def _helper_reset_sites() -> list:
    """The one direct reset allowed: inside reset_cuda_peak_stats itself (none where the helper is absent)."""
    lines = Path(diagnostics.__file__).read_text(encoding="utf-8").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("def reset_cuda_peak_stats")]
    if not starts:
        return []
    n = next(i for i in range(starts[0], len(lines)) if "torch.cuda.reset_peak_memory_stats(" in lines[i])
    return [f"common/diagnostics.py:{n + 1}"]


class TestWithoutCuda:
    def test_no_cuda_gives_a_named_empty_result(self, monkeypatch, fresh_lifetime):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        peaks = diagnostics.lifetime_cuda_peaks(0)
        assert peaks.absent == diagnostics.CUDA_UNAVAILABLE == "CUDA_UNAVAILABLE"
        assert (peaks.device, peaks.max_memory_allocated_bytes, peaks.max_memory_reserved_bytes) == (None, None, None)
        assert peaks.as_dict()["scope"] == "none (CUDA_UNAVAILABLE)"

    def test_the_helper_touches_no_cuda_function_without_cuda(self, monkeypatch, fresh_lifetime):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        def forbidden(*_a, **_k):
            raise AssertionError("a CUDA statistic was read or reset without CUDA")

        for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved", "synchronize",
                     "current_device"):
            monkeypatch.setattr(torch.cuda, name, forbidden)
        diagnostics.reset_cuda_peak_stats()
        lines: list = []
        with diagnostics.phase_profiler("tag", "cpu", sink=lines.append):
            pass
        assert len(lines) == 1 and "cuda_peak" not in lines[0]
        assert diagnostics._LIFETIME_PEAKS == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
class TestRealCuda:
    """On a real device (run once on a GPU of our server under the GPU rule): two profiled phases, the first larger."""

    def test_two_phases_on_the_device(self, fresh_lifetime):
        dev = torch.cuda.current_device()
        torch.cuda.synchronize()
        diagnostics.reset_cuda_peak_stats()
        base = torch.cuda.memory_allocated(dev)
        with diagnostics.phase_profiler("real", "first", sink=lambda _l: None):
            big = torch.empty(64 * MIB, dtype=torch.uint8, device=f"cuda:{dev}")
            del big
        torch.cuda.empty_cache()
        with diagnostics.phase_profiler("real", "second", sink=lambda _l: None):
            small = torch.empty(8 * MIB, dtype=torch.uint8, device=f"cuda:{dev}")
            del small
        torch.cuda.synchronize()
        last_phase = torch.cuda.max_memory_allocated(dev)
        peaks = diagnostics.lifetime_cuda_peaks(dev)
        assert last_phase < base + 64 * MIB
        assert peaks.max_memory_allocated_bytes >= base + 64 * MIB
        assert peaks.max_memory_reserved_bytes >= peaks.max_memory_allocated_bytes
        assert peaks.max_memory_allocated_bytes >= last_phase
