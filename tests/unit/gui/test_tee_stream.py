"""``TeeStream`` is installed as ``sys.stdout``/``sys.stderr`` for the whole
pipeline run; it must never raise from ``write()``/``flush()`` regardless of
failures in the wrapped original stream or the console callback — a raise
here would crash the pipeline over a GUI side-effect.
"""

from __future__ import annotations

from mimarsinan.gui.tee_stream import TeeStream


class _BoomStream:
    def write(self, s):
        raise RuntimeError("original stream write boom")

    def flush(self):
        raise RuntimeError("original stream flush boom")


class _RecordingStream:
    def __init__(self) -> None:
        self.written: list = []

    def write(self, s):
        self.written.append(s)

    def flush(self) -> None:
        pass


class TestTeeStreamWriteResilience:
    def test_write_does_not_raise_when_original_stream_write_fails(self) -> None:
        lines: list[str] = []
        tee = TeeStream(_BoomStream(), lines.append)
        tee.write("hello\n")  # must not raise
        assert lines == ["hello"]

    def test_write_does_not_raise_when_original_stream_flush_fails(self) -> None:
        class _FlushBoom:
            def write(self, s):
                pass

            def flush(self):
                raise RuntimeError("flush boom")

        lines: list[str] = []
        tee = TeeStream(_FlushBoom(), lines.append)
        tee.write("hello\n")  # must not raise
        assert lines == ["hello"]

    def test_write_does_not_raise_when_callback_fails(self) -> None:
        def boom_callback(line: str) -> None:
            raise RuntimeError("callback boom")

        original = _RecordingStream()
        tee = TeeStream(original, boom_callback)
        tee.write("hello\n")  # must not raise
        assert original.written == ["hello\n"]

    def test_write_processes_later_lines_after_a_failing_callback(self) -> None:
        """A single failing line callback must not prevent later lines in
        the same write() call from being delivered."""
        seen: list[str] = []

        def flaky_callback(line: str) -> None:
            if line == "bad":
                raise RuntimeError("boom")
            seen.append(line)

        original = _RecordingStream()
        tee = TeeStream(original, flaky_callback)
        tee.write("bad\ngood\n")
        assert seen == ["good"]

    def test_flush_does_not_raise_when_original_flush_fails(self) -> None:
        tee = TeeStream(_BoomStream(), lambda line: None)
        tee.flush()  # must not raise

    def test_flush_remaining_does_not_raise_when_callback_fails(self) -> None:
        def boom_callback(line: str) -> None:
            raise RuntimeError("boom")

        tee = TeeStream(_RecordingStream(), boom_callback)
        tee.write("partial line without newline")
        tee.flush_remaining()  # must not raise
