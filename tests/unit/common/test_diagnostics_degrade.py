"""diagnostics telemetry degrades through best_effort, not silent broad catches."""

import logging

import mimarsinan.common.diagnostics as diagnostics


class TestRssMib:
    def test_returns_nonnegative_float(self):
        assert diagnostics._rss_mib() >= 0.0

    def test_unreadable_proc_degrades_to_zero_and_logs(self, monkeypatch, caplog):
        def exploding_open(*args, **kwargs):
            raise RuntimeError("no /proc here")

        monkeypatch.setattr("builtins.open", exploding_open)
        with caplog.at_level(logging.DEBUG, logger="mimarsinan.best_effort"):
            assert diagnostics._rss_mib() == 0.0
        assert any("statm" in record.getMessage() for record in caplog.records)


class TestPhaseProfiler:
    def test_reports_single_line_to_sink(self):
        lines = []
        with diagnostics.phase_profiler("tag", "my-phase", sink=lines.append):
            pass
        assert len(lines) == 1
        assert "phase=my-phase" in lines[0]

    def test_exception_in_block_propagates_and_still_reports(self):
        lines = []
        try:
            with diagnostics.phase_profiler("tag", "boom-phase", sink=lines.append):
                raise ValueError("boom")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError to propagate")
        assert len(lines) == 1


class TestCudaGuard:
    def test_cpu_passthrough(self):
        with diagnostics.cuda_guard("noop"):
            pass

    def test_exception_propagates_when_disabled(self):
        try:
            with diagnostics.cuda_guard("noop", enabled=False):
                raise ValueError("boom")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError to propagate")
