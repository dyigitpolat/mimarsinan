"""[ODIN P7a] the device seam: one payload, two transports, no silent silicon.

The board is a TRANSPORT SWAP, and the claim is testable here: the RTL
cosimulation transport and the XRT transport are handed byte-identical program
payloads for the same export. The XRT half runs against the SHIPPED fake
``pyxrt`` injected at the import seam — the same file the HACC package carries,
so there is exactly ONE fake in this repository and
``test_odin_pyxrt_surface.py`` proves it is a subset of the real binding. What
these tests prove is the CALL CONTRACT (order, buffer sizes, bytes) — structure,
not silicon. Silicon is P7b.

NO REGISTER READ APPEARS BELOW, and that is the point. pyxrt binds none, so the
capacities are DECLARED and the kernel's verdict is the capture header it DMAs
back: a run that refused comes home with the host's no-verdict sentinel still in
place.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from integration.odin_fpga_harness import fixture_export

from mimarsinan.chip_simulation.odin_fpga.cosim_transport import RtlCosimTransport
from mimarsinan.chip_simulation.odin_fpga.factory import TRANSPORTS, build_transport
from mimarsinan.chip_simulation.odin_fpga.payload import (
    op_codes,
    payload_bytes,
    program_payload,
    program_plan,
    run_plan,
    split_payloads,
    stimulus_ops,
)
from mimarsinan.chip_simulation.odin_fpga.transport import (
    DeviceSession,
    DeviceTransport,
    DeviceTransportError,
)
from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (
    ARG_CAPTURE,
    ARG_PROGRAM,
    ARG_STIMULUS,
    CAPTURE_HEADER_WORDS,
    CAPTURE_NO_VERDICT,
    CAPTURE_RECORD_WORDS,
    KERNEL_ARGS,
    KERNEL_NAME,
    SHIPPED_CAPTURE_EVENTS,
    SHIPPED_CAPTURE_WORDS,
    SHIPPED_KERNEL_CORES,
    KernelCapacity,
    OdinFpgaCaptureTruncated,
    OdinFpgaKernelError,
    WORD_BYTES,
    capture_words,
    decode_capture,
    no_verdict_header,
    require_kernel_verdict,
    stimulus_base_word,
)
from mimarsinan.chip_simulation.odin_fpga.kernel_sim import SHIPPED_FIFO_WORDS
from mimarsinan.chip_simulation.odin_fpga.xrt_transport import (
    OdinFpgaDependencyError,
    XrtTransport,
    load_pyxrt,
)
from mimarsinan.chip_simulation.odin_rtl.limits.configurations import (
    wrapper_shipped_depths,
)
from mimarsinan.chip_simulation.odin_rtl.stimulus import OP_END, OP_SPI_W

ENGINE = "iverilog"

REPO = Path(__file__).resolve().parents[3]
FAKE_PATH = (
    REPO / "scripts" / "hacc" / "package" / "host"
    / "fake_pyxrt_for_selftest.py")

#: The fixture export programs two ODIN cores, so a bitstream that could run it
#: declares NC=2. Nothing can read that back off a card — it is declared here.
TWO_CORES = 2


@pytest.fixture(scope="module")
def export():
    return fixture_export()


def _load_fake():
    """A FRESH instance of the SHIPPED fake: its arming state is module-level."""
    spec = importlib.util.spec_from_file_location("unit_fake_pyxrt", FAKE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake(monkeypatch):
    module = _load_fake()
    monkeypatch.setitem(sys.modules, "pyxrt", module)
    return module


def _board(**overrides):
    fields = {
        "xclbin_path": "/tmp/odin.xclbin",
        "capacity": KernelCapacity(
            cores=TWO_CORES,
            provenance="unit test: a declared NC=2 geometry"),
    }
    fields.update(overrides)
    return XrtTransport(**fields)


def _inputs():
    return [[{0: (0,) * 5, 1: (0,) * 4}]]


class TestThePayloadHasExactlyOneEncoder:
    def test_the_programming_payload_is_a_prefix_of_the_whole_run(self, export):
        program = program_plan(export)
        full = run_plan(export, _inputs(), latencies=(0, 1))
        assert full.ops[:len(program.ops)] == program.ops

    def test_the_split_refuses_a_run_built_from_another_export(self, export):
        other = program_plan(export)
        full = run_plan(export, [], latencies=(0, 0))
        mismatched = SimpleNamespace(ops=full.ops[1:])
        with pytest.raises(ValueError, match="does not start with"):
            stimulus_ops(other, mismatched)

    def test_the_payload_is_little_endian_device_words_ending_in_END(self, export):
        payload = program_payload(export)
        words = op_codes(payload)
        assert words[-1] == OP_END
        assert words[0] == OP_SPI_W
        assert len(payload) == WORD_BYTES * len(words)

    def test_op_codes_refuses_a_ragged_payload(self):
        with pytest.raises(ValueError, match="whole number"):
            op_codes(b"\x01\x02\x03")


class TestTheTransportsReceiveTheSameProgramBytes:
    """The transport-swap claim, made testable: same export, same bytes."""

    def test_cosim_and_xrt_are_handed_byte_identical_program_payloads(
        self, export, fake,
    ):
        cosim = RtlCosimTransport(engine=ENGINE)
        cosim.open()
        cosim_receipt = cosim.program(export)
        cosim.close()

        board = _board()
        board.open()
        board_receipt = board.program(export)
        board.close()

        assert cosim_receipt.payload == board_receipt.payload
        assert cosim_receipt.ops == board_receipt.ops
        assert cosim_receipt.cores == board_receipt.cores
        # ... and the payload is the real thing, not two empty buffers.
        assert len(cosim_receipt.payload) > WORD_BYTES * 1000

    def test_the_walls_are_measured_and_their_basis_is_named_per_transport(
        self, export, fake,
    ):
        cosim = RtlCosimTransport(engine=ENGINE)
        cosim.open()
        receipt = cosim.program(export)
        cosim.close()
        assert receipt.wall_s >= 0.0
        assert "rtl_cosim" in receipt.basis and "simulated" in receipt.basis

        board = _board()
        board.open()
        board_receipt = board.program(export)
        board.close()
        assert "xrt" in board_receipt.basis and "SPI" in board_receipt.basis


class TestTheXrtSessionMakesTheCallsTheBoardNeeds:
    def test_the_session_opens_by_introspecting_before_it_loads(
        self, export, fake,
    ):
        board = _board(device_index=1)
        board.open()
        board.program(export)
        log = fake.call_log()
        assert log[0] == ("xclbin", "/tmp/odin.xclbin")
        assert log[1] == ("get_kernels",)
        assert log[2] == ("get_mems",)
        assert log[3] == ("device", 1)
        assert log[4] == ("load_xclbin", "/tmp/odin.xclbin")
        assert log[5] == ("kernel", KERNEL_NAME, "exclusive")
        writes = [entry for entry in log if entry[0] == "write"]
        assert writes[0][1] == ARG_PROGRAM
        assert ("sync", ARG_PROGRAM, "XCL_BO_SYNC_BO_TO_DEVICE",
                writes[0][2], 0) in log

    def test_no_call_in_the_whole_session_is_a_register_access(
        self, export, fake,
    ):
        board = _board(capture_events=8)
        board.open()
        board.program(export)
        fake.arm(capture_words=(0, 4242))
        board.run_samples(_inputs(), latencies=(0, 1))
        kinds = {entry[0] for entry in fake.call_log()}
        assert not any("register" in kind for kind in kinds)

    def test_a_run_dmas_the_stimulus_starts_the_kernel_and_syncs_the_capture_back(
        self, export, fake,
    ):
        board = _board(capture_events=8)
        board.open()
        receipt = board.program(export)
        fake.arm(capture_words=(0, 4242))
        run = board.run_samples(_inputs(), latencies=(0, 1))

        _program_bytes, stimulus, _plan = split_payloads(
            export, _inputs(), latencies=(0, 1))
        log = fake.call_log()
        kinds = [entry[0] for entry in log]
        assert kinds.index("start") > kinds.index("write")
        start = log[kinds.index("start")]
        assert start[1][:3] == (ARG_PROGRAM, ARG_STIMULUS, ARG_CAPTURE)
        assert start[1][3] == len(receipt.payload) // WORD_BYTES
        assert start[1][4] == len(stimulus) // WORD_BYTES
        assert start[1][5] == 8
        assert any(entry[0] == "wait" for entry in log)
        assert any(e[0] == "sync" and e[1] == ARG_CAPTURE
                   and e[2] == "XCL_BO_SYNC_BO_FROM_DEVICE" for e in log)
        assert run.device_cycles == 4242
        assert run.counts == {}

    def test_the_capture_header_carries_the_sentinel_before_the_kernel_runs(
        self, export, fake,
    ):
        board = _board(capture_events=8)
        board.open()
        board.program(export)
        fake.arm(capture_words=(0, 0))
        board.run_samples(_inputs(), latencies=(0, 1))
        log = fake.call_log()
        started_at = [entry[0] for entry in log].index("start")
        poison = [index for index, entry in enumerate(log)
                  if entry[0] == "write" and entry[1] == ARG_CAPTURE
                  and entry[2] == len(no_verdict_header())]
        assert poison and max(poison) < started_at

    def test_the_buffers_are_sized_to_the_payloads_they_carry(self, export, fake):
        board = _board(capture_events=8)
        board.open()
        receipt = board.program(export)
        fake.arm(capture_words=(0, 0))
        board.run_samples(_inputs(), latencies=(0, 1))
        log = fake.call_log()
        writes = {entry[1]: entry[2] for entry in log if entry[0] == "write"}
        assert writes[ARG_PROGRAM] == len(receipt.payload)
        reads = [entry for entry in log if entry[0] == "read"]
        assert reads[0][2] == WORD_BYTES * (
            CAPTURE_HEADER_WORDS + CAPTURE_RECORD_WORDS * 8)

    def test_captured_events_become_per_cycle_counts(self, export, fake):
        # header (1 event, 7 device cycles) + one record: tag 1 (= sample 0,
        # cycle 0), device cycle 3, core 1, neuron 2.
        board = _board(capture_events=4)
        board.open()
        board.program(export)
        fake.arm(capture_words=(1, 7, 1, 3, 1, 2))
        run = board.run_samples(_inputs(), latencies=(0, 1))
        assert run.counts == {(0, 0, 1, 2): 1}
        assert run.device_cycles == 7


class TestTheVerdictArrivesInMemoryOrNotAtAll:
    """A1/A4: with no register to read, the capture header IS the status word."""

    def test_a_header_still_carrying_the_sentinel_is_a_kernel_error(self):
        words = capture_words(no_verdict_header())
        with pytest.raises(OdinFpgaKernelError, match="NO-VERDICT sentinel"):
            require_kernel_verdict(words, transport="xrt")

    def test_the_refusal_names_why_no_host_can_read_the_err_bit(self):
        with pytest.raises(OdinFpgaKernelError) as exc:
            require_kernel_verdict(
                (CAPTURE_NO_VERDICT, CAPTURE_NO_VERDICT), transport="xrt")
        message = str(exc.value)
        assert "0x4C" in message and "read_register" in message

    def test_one_sentinel_word_alone_is_not_a_refusal(self):
        events, cycles = decode_capture((0, CAPTURE_NO_VERDICT), capacity=4)
        assert events == () and cycles == CAPTURE_NO_VERDICT

    def test_a_kernel_that_refused_the_run_refuses_the_transport(
        self, export, fake,
    ):
        board = _board()
        board.open()
        board.program(export)
        fake.arm(fault="kernel_err")
        with pytest.raises(OdinFpgaKernelError, match="NO-VERDICT sentinel"):
            board.run_samples(_inputs(), latencies=(0, 1))

    def test_a_run_that_never_completed_refuses_before_the_capture_is_read(
        self, export, fake,
    ):
        board = _board()
        board.open()
        board.program(export)
        fake.arm(fault="not_completed", capture_words=(0, 0))
        with pytest.raises(DeviceTransportError, match="never saw ap_done"):
            board.run_samples(_inputs(), latencies=(0, 1))
        log = fake.call_log()
        assert not any(entry[0] == "read" for entry in log)

    def test_a_truncated_capture_refuses_instead_of_reporting_silent_neurons(self):
        with pytest.raises(OdinFpgaCaptureTruncated, match="silent neurons"):
            decode_capture((9, 100), capacity=4)

    def test_a_capture_exactly_at_capacity_refuses_too(self):
        # At capacity the fabric's own count is indistinguishable from one that
        # dropped the next event, so the honest answer is a refusal.
        with pytest.raises(OdinFpgaCaptureTruncated, match="capacity of 4"):
            decode_capture((4, 100), capacity=4)

    def test_a_capture_shorter_than_its_header_refuses(self):
        with pytest.raises(DeviceTransportError, match="two-word header"):
            decode_capture((0,), capacity=4)

    def test_an_overflowing_capture_refuses_with_the_capacity_in_force(
        self, export, fake,
    ):
        board = _board()
        board.open()
        board.program(export)
        fake.arm(capture_words=(SHIPPED_CAPTURE_EVENTS + 1, 9))
        with pytest.raises(
            OdinFpgaCaptureTruncated, match=str(SHIPPED_CAPTURE_EVENTS),
        ):
            board.run_samples(_inputs(), latencies=(0, 1))


class TestTheCapacitiesAreDeclaredAndSayWhereTheyCameFrom:
    """No pyxrt call can ask a card what it was compiled with."""

    def test_the_default_declaration_is_the_rtl_shipped_geometry(self):
        capacity = KernelCapacity()
        assert capacity.capture_events == SHIPPED_CAPTURE_EVENTS
        assert capacity.cores == SHIPPED_KERNEL_CORES
        assert "NOT read back from the card" in capacity.provenance
        assert "PROG_WORDS" not in capacity.provenance, (
            "the fabric stores no program, so nothing may declare a capacity "
            "for one")

    def test_open_takes_the_min_of_the_host_ceiling_and_the_declaration(
        self, fake,
    ):
        board = _board(capture_events=1 << 20)
        board.open()
        assert board.capture_capacity == SHIPPED_CAPTURE_EVENTS
        assert board.capacity.cores == TWO_CORES

    def test_a_smaller_host_declaration_wins(self, fake):
        board = _board(capture_events=16)
        board.open()
        assert board.capture_capacity == 16

    def test_a_package_declaring_no_storage_refuses_at_open(self, fake):
        with pytest.raises(DeviceTransportError, match="silent empty one"):
            _board(capacity=KernelCapacity(
                cores=0, capture_events=0,
                provenance="unit test: nothing at all")).open()

    def test_the_capture_buffer_is_sized_from_the_declared_capacity(
        self, export, fake,
    ):
        board = _board(capture_events=1 << 20)
        board.open()
        board.program(export)
        fake.arm(capture_words=(0, 0))
        run = board.run_samples(_inputs(), latencies=(0, 1))
        log = fake.call_log()
        reads = [entry for entry in log if entry[0] == "read"]
        assert reads[0][2] == WORD_BYTES * (
            CAPTURE_HEADER_WORDS + CAPTURE_RECORD_WORDS * SHIPPED_CAPTURE_EVENTS)
        start = next(entry for entry in log if entry[0] == "start")
        assert start[1][5] == SHIPPED_CAPTURE_EVENTS
        assert run.detail["capture_capacity"] == SHIPPED_CAPTURE_EVENTS
        assert run.detail["capacity"]["provenance"] == (
            "unit test: a declared NC=2 geometry")

    def test_no_program_length_can_be_refused_for_not_fitting(
        self, export, fake,
    ):
        """The fabric stores no program, so there is nothing for one to outgrow.

        A run this long used to be refused before it started against a declared
        program RAM; the stream is now bounded only by the word-count arguments,
        and the ONLY thing left that can refuse a run for size is the capture.
        """
        board = _board(capture_events=1 << 20)
        board.open()
        board.program(export)
        fake.arm(capture_words=(0, 0))
        board.run_samples(_inputs(), latencies=(0, 1))
        assert any(entry[0] == "start" for entry in fake.call_log())

    def test_the_stimulus_lands_on_the_programming_payloads_terminator(self):
        assert stimulus_base_word(4) == 3
        assert stimulus_base_word(1) == 0
        assert stimulus_base_word(0) == 0


class TestTheXclbinIsIntrospectedBeforeItIsTrusted:
    def test_the_kernels_and_banks_the_bitstream_declares_are_recorded(
        self, fake,
    ):
        board = _board()
        board.open()
        assert board.xclbin_kernels == {KERNEL_NAME: KERNEL_ARGS}
        assert board.memory_banks[0]["tag"] == "DDR[0]"
        assert board.memory_banks[0]["used"] is True

    def test_an_xclbin_without_our_kernel_refuses_before_the_card_sees_it(
        self, fake,
    ):
        fake.arm(fault="wrong_kernel")
        with pytest.raises(DeviceTransportError, match="none of them"):
            _board().open()
        assert not any(
            entry[0] == "load_xclbin" for entry in fake.call_log())


class TestEveryEntryRefusesLoudWithoutXrt:
    def test_the_module_never_imports_pyxrt_at_import_time(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyxrt", None)
        sys.modules.pop("mimarsinan.chip_simulation.odin_fpga.xrt_transport", None)
        import importlib

        module = importlib.import_module(
            "mimarsinan.chip_simulation.odin_fpga.xrt_transport")
        assert module.TRANSPORT_NAME == "xrt"

    def test_load_pyxrt_names_the_dependency_the_runtime_and_the_alternative(
        self, monkeypatch,
    ):
        monkeypatch.setitem(sys.modules, "pyxrt", None)
        with pytest.raises(OdinFpgaDependencyError) as exc:
            load_pyxrt()
        message = str(exc.value)
        assert "pyxrt" in message
        assert "Xilinx Runtime" in message and "RUNBOOK" in message
        assert "cosimulation" in message

    def test_open_refuses_when_the_runtime_is_absent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyxrt", None)
        with pytest.raises(OdinFpgaDependencyError, match="pyxrt"):
            _board().open()

    def test_program_before_open_refuses_by_name(self, export, fake):
        with pytest.raises(OdinFpgaDependencyError, match="before a live session"):
            _board().program(export)

    def test_a_missing_xclbin_declaration_refuses_at_construction(self):
        with pytest.raises(OdinFpgaDependencyError, match="odin_fpga_xclbin_path"):
            XrtTransport(xclbin_path="")


class TestTheCosimTransportGuardsItsSession:
    def test_program_before_open_refuses(self, export):
        with pytest.raises(DeviceTransportError, match="before open"):
            RtlCosimTransport(engine=ENGINE).program(export)

    def test_run_before_program_refuses_by_naming_the_empty_cores(self):
        transport = RtlCosimTransport(engine=ENGINE)
        transport.open()
        with pytest.raises(DeviceTransportError, match="no weights"):
            transport.run_samples([], latencies=())

    def test_an_absent_simulator_refuses_with_the_toolchain_diagnostic(
        self, monkeypatch,
    ):
        import mimarsinan.chip_simulation.odin_fpga.cosim_transport as module
        from mimarsinan.chip_simulation.odin_rtl.toolchain import SimulatorUnavailable

        def _absent():
            raise SimulatorUnavailable("no simulator in <dir>")

        monkeypatch.setattr(module, "available_engine", _absent)
        with pytest.raises(DeviceTransportError, match="no simulator in <dir>"):
            RtlCosimTransport().open()

    def test_the_session_closes_even_when_the_body_raises(self, export):
        transport = RtlCosimTransport(engine=ENGINE)
        with pytest.raises(RuntimeError):
            with DeviceSession(transport):
                raise RuntimeError("boom")
        with pytest.raises(DeviceTransportError, match="before open"):
            transport.program(export)


class TestTheFactoryResolvesTheDeclaredDevice:
    def test_the_default_is_the_cosimulation(self):
        assert isinstance(build_transport({}), RtlCosimTransport)
        assert TRANSPORTS[0] == "rtl_cosim"

    def test_the_board_is_built_from_the_declared_xclbin(self):
        transport = build_transport({
            "odin_fpga_transport": "xrt",
            "odin_fpga_xclbin_path": "/data/u/odin.xclbin",
            "odin_fpga_device_index": 2,
        })
        assert isinstance(transport, XrtTransport)
        assert transport.xclbin_path == "/data/u/odin.xclbin"
        assert transport.device_index == 2

    def test_an_unknown_transport_refuses_by_key(self):
        with pytest.raises(DeviceTransportError, match="odin_fpga_transport"):
            build_transport({"odin_fpga_transport": "jtag"})

    def test_both_transports_satisfy_the_declared_protocol(self, fake):
        assert isinstance(RtlCosimTransport(), DeviceTransport)
        assert isinstance(XrtTransport(xclbin_path="/x.xclbin"), DeviceTransport)


def test_the_payload_of_an_empty_op_list_is_just_the_terminator():
    assert payload_bytes(()) == struct.pack("<I", OP_END)


class TestTheShippedDepthsAreOneNumber:
    """CROSS-LANGUAGE CONTRACT: the host's copies of the RTL's two depths.

    No host may read these back — pyxrt binds no register access — so a host
    constant that drifted from the fabric would size buffers and print advice
    for a kernel nobody built, and the ONE thing that would notice is the
    capture header coming home still carrying the no-verdict sentinel.
    """

    def test_the_capture_constant_is_the_rtl_default(self):
        assert wrapper_shipped_depths()["CAP_WORDS"] == SHIPPED_CAPTURE_WORDS

    def test_the_stream_fifo_constant_is_the_rtl_default(self):
        assert wrapper_shipped_depths()["FIFO_WORDS"] == SHIPPED_FIFO_WORDS

    def test_the_event_capacity_is_the_wrapper_s_own_arithmetic(self):
        assert SHIPPED_CAPTURE_EVENTS == (
            SHIPPED_CAPTURE_WORDS - CAPTURE_HEADER_WORDS) // CAPTURE_RECORD_WORDS

    def test_the_depth_is_a_whole_number_of_block_ram_tiles(self):
        """A tile is 1,024 words of 36 bits: the shipped depth is chosen so the
        capture buffer costs whole tiles and not a partial one plus fabric."""
        assert SHIPPED_CAPTURE_WORDS % 1024 == 0
