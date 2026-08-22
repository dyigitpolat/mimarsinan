"""[ODIN P7a] the device seam: one payload, two transports, no silent silicon.

The board is a TRANSPORT SWAP, and the claim is testable here: the RTL
cosimulation transport and the XRT transport are handed byte-identical program
payloads for the same export. The XRT half runs against a fake ``pyxrt``
injected at the import seam, so what these tests prove is the CALL CONTRACT
(order, buffer sizes, bytes) — structure, not silicon. Silicon is P7b.
"""

from __future__ import annotations

import struct
import sys
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
    ADDR_CAPTURE_CAPACITY,
    ADDR_PROGRAM_CAPACITY,
    ADDR_STATUS,
    ARG_CAPTURE,
    ARG_PROGRAM,
    ARG_STIMULUS,
    CAPTURE_HEADER_WORDS,
    CAPTURE_RECORD_WORDS,
    CTRL_OFFSET,
    STATUS_ERR_BIT,
    OdinFpgaCaptureTruncated,
    OdinFpgaKernelError,
    OdinFpgaProgramTooLarge,
    WORD_BYTES,
    decode_capture,
    decode_status,
    stimulus_base_word,
)
from mimarsinan.chip_simulation.odin_fpga.xrt_transport import (
    OdinFpgaDependencyError,
    XrtTransport,
    load_pyxrt,
)
from mimarsinan.chip_simulation.odin_rtl.stimulus import OP_END, OP_SPI_W

ENGINE = "iverilog"


@pytest.fixture(scope="module")
def export():
    return fixture_export()


# ---------------------------------------------------------------------------
# A fake pyxrt: every object records the calls the board would receive.
# ---------------------------------------------------------------------------


class _FakeBo:
    def __init__(self, log, size, group):
        self.log = log
        self.size = int(size)
        self.group = int(group)
        self.data = bytearray(int(size))

    def write(self, payload, offset):
        self.log.append(("write", self.group, len(payload), int(offset)))
        self.data[offset:offset + len(payload)] = payload

    def sync(self, direction, size, offset):
        self.log.append(("sync", self.group, direction, int(size), int(offset)))

    def read(self, size, offset):
        self.log.append(("read", self.group, int(size), int(offset)))
        return bytes(self.data[offset:offset + size])


class _FakeRun:
    def __init__(self, log):
        self.log = log

    def wait(self):
        self.log.append(("wait",))


#: The register images of the committed kernel geometry at the fixture's two
#: cores: a 65536-word capture RAM holds (65536 - 2) // 4 = 16383 records, and
#: PROG_WORDS is NC * 262144 (hw/fpga/kernel/odin_fpga_kernel_top.v).
FABRIC_CAPTURE_EVENTS = 16383
FABRIC_PROGRAM_WORDS = 2 * 262144


class _FakeKernel:
    exclusive = "exclusive"

    def __init__(self, log, capture_words, registers):
        self.log = log
        self._capture_words = capture_words
        self._registers = dict(registers)

    def group_id(self, arg):
        return int(arg)

    def read_register(self, offset):
        self.log.append(("read_register", int(offset)))
        return int(self._registers.get(int(offset), 0))

    def __call__(self, *args):
        self.log.append(("start", tuple(
            a.group if isinstance(a, _FakeBo) else a for a in args)))
        capture = args[ARG_CAPTURE]
        words = list(self._capture_words)
        packed = struct.pack(f"<{len(words)}I", *words)
        capture.data[0:len(packed)] = packed
        return _FakeRun(self.log)


class _FakeDevice:
    def __init__(self, log, index):
        self.log = log
        self.index = index

    def load_xclbin(self, xclbin):
        self.log.append(("load_xclbin", xclbin.path))
        return "uuid"


def fake_pyxrt(
    log,
    capture_words=(0, 0),
    *,
    capture_capacity=FABRIC_CAPTURE_EVENTS,
    program_capacity=FABRIC_PROGRAM_WORDS,
    status=0,
):
    """A module object shaped exactly like the pyxrt surface the transport uses."""
    registers = {
        CTRL_OFFSET: 0b0110,
        ADDR_STATUS: status,
        ADDR_CAPTURE_CAPACITY: capture_capacity,
        ADDR_PROGRAM_CAPACITY: program_capacity,
    }
    module = SimpleNamespace()
    module.device = lambda index: (
        log.append(("device", int(index))) or _FakeDevice(log, index))
    module.xclbin = lambda path: SimpleNamespace(path=path)
    module.kernel = lambda device, uuid, name, mode: (
        log.append(("kernel", name, mode))
        or _FakeKernel(log, capture_words, registers))
    module.kernel.exclusive = _FakeKernel.exclusive
    module.bo = lambda device, size, kind, group: _FakeBo(log, size, group)
    module.bo.normal = "normal"
    module.xclBOSyncDirection = SimpleNamespace(
        XCL_BO_SYNC_BO_TO_DEVICE="to", XCL_BO_SYNC_BO_FROM_DEVICE="from")
    return module


@pytest.fixture
def injected_pyxrt(monkeypatch):
    log: list = []
    monkeypatch.setitem(sys.modules, "pyxrt", fake_pyxrt(log))
    return log


class TestThePayloadHasExactlyOneEncoder:
    def test_the_programming_payload_is_a_prefix_of_the_whole_run(self, export):
        program = program_plan(export)
        full = run_plan(export, [[{0: (0,) * 5, 1: (0,) * 4}]], latencies=(0, 1))
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
        self, export, injected_pyxrt,
    ):
        cosim = RtlCosimTransport(engine=ENGINE)
        cosim.open()
        cosim_receipt = cosim.program(export)
        cosim.close()

        board = XrtTransport(xclbin_path="/tmp/odin.xclbin")
        board.open()
        board_receipt = board.program(export)
        board.close()

        assert cosim_receipt.payload == board_receipt.payload
        assert cosim_receipt.ops == board_receipt.ops
        assert cosim_receipt.cores == board_receipt.cores
        # ... and the payload is the real thing, not two empty buffers.
        assert len(cosim_receipt.payload) > WORD_BYTES * 1000

    def test_the_walls_are_measured_and_their_basis_is_named_per_transport(
        self, export, injected_pyxrt,
    ):
        cosim = RtlCosimTransport(engine=ENGINE)
        cosim.open()
        receipt = cosim.program(export)
        cosim.close()
        assert receipt.wall_s >= 0.0
        assert "rtl_cosim" in receipt.basis and "simulated" in receipt.basis

        board = XrtTransport(xclbin_path="/tmp/odin.xclbin")
        board.open()
        board_receipt = board.program(export)
        board.close()
        assert "xrt" in board_receipt.basis and "SPI" in board_receipt.basis


class TestTheXrtSessionMakesTheCallsTheBoardNeeds:
    def test_the_call_order_is_open_load_kernel_allocate_write_sync(
        self, export, injected_pyxrt,
    ):
        board = XrtTransport(xclbin_path="/tmp/odin.xclbin", device_index=1)
        board.open()
        board.program(export)
        assert injected_pyxrt[0] == ("device", 1)
        assert injected_pyxrt[1] == ("load_xclbin", "/tmp/odin.xclbin")
        assert injected_pyxrt[2] == ("kernel", "odin_fpga_kernel_top", "exclusive")
        assert injected_pyxrt[3] == ("read_register", ADDR_CAPTURE_CAPACITY)
        assert injected_pyxrt[4] == ("read_register", ADDR_PROGRAM_CAPACITY)
        assert injected_pyxrt[5][0] == "write"
        assert injected_pyxrt[6] == (
            "sync", ARG_PROGRAM, "to", injected_pyxrt[5][2], 0)

    def test_a_run_dmas_the_stimulus_starts_the_kernel_and_syncs_the_capture_back(
        self, export, monkeypatch,
    ):
        log: list = []
        monkeypatch.setitem(sys.modules, "pyxrt", fake_pyxrt(log, (0, 4242)))
        board = XrtTransport(xclbin_path="/tmp/odin.xclbin", capture_events=8)
        board.open()
        receipt = board.program(export)
        run = board.run_samples([[{0: (0,) * 5, 1: (0,) * 4}]], latencies=(0, 1))

        _program_bytes, stimulus, _plan = split_payloads(
            export, [[{0: (0,) * 5, 1: (0,) * 4}]], latencies=(0, 1))
        kinds = [entry[0] for entry in log]
        assert kinds.index("start") > kinds.index("write")
        start = log[kinds.index("start")]
        assert start[1][:3] == (ARG_PROGRAM, ARG_STIMULUS, ARG_CAPTURE)
        assert start[1][3] == len(receipt.payload) // WORD_BYTES
        assert start[1][4] == len(stimulus) // WORD_BYTES
        assert ("wait",) in log
        assert any(e[0] == "sync" and e[1] == ARG_CAPTURE and e[2] == "from"
                   for e in log)
        assert run.device_cycles == 4242
        assert run.counts == {}

    def test_the_buffers_are_sized_to_the_payloads_they_carry(
        self, export, monkeypatch,
    ):
        log: list = []
        monkeypatch.setitem(sys.modules, "pyxrt", fake_pyxrt(log, (0, 0)))
        board = XrtTransport(xclbin_path="/tmp/odin.xclbin", capture_events=8)
        board.open()
        receipt = board.program(export)
        board.run_samples([[{0: (0,) * 5, 1: (0,) * 4}]], latencies=(0, 1))
        writes = {entry[1]: entry[2] for entry in log if entry[0] == "write"}
        assert writes[ARG_PROGRAM] == len(receipt.payload)
        reads = [entry for entry in log if entry[0] == "read"]
        assert reads[0][2] == WORD_BYTES * (
            CAPTURE_HEADER_WORDS + CAPTURE_RECORD_WORDS * 8)

    def test_captured_events_become_per_cycle_counts(self, export, monkeypatch):
        # header (1 event, 7 device cycles) + one record: tag 1 (= sample 0,
        # cycle 0), device cycle 3, core 1, neuron 2.
        words = (1, 7, 1, 3, 1, 2)
        log: list = []
        monkeypatch.setitem(sys.modules, "pyxrt", fake_pyxrt(log, words))
        board = XrtTransport(xclbin_path="/tmp/odin.xclbin", capture_events=4)
        board.open()
        board.program(export)
        run = board.run_samples([[{0: (0,) * 5, 1: (0,) * 4}]], latencies=(0, 1))
        assert run.counts == {(0, 0, 1, 2): 1}
        assert run.device_cycles == 7

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


class TestTheCapacitiesComeFromTheBitstreamNotFromAnAssumption:
    """A1/A4: the kernel is asked what it can hold and what went wrong."""

    def test_open_reads_both_capacity_registers_and_takes_the_minimum(
        self, monkeypatch,
    ):
        log: list = []
        monkeypatch.setitem(sys.modules, "pyxrt", fake_pyxrt(log))
        board = XrtTransport(xclbin_path="/x.xclbin", capture_events=1 << 20)
        board.open()
        assert board.capture_capacity == FABRIC_CAPTURE_EVENTS
        assert board.program_capacity == FABRIC_PROGRAM_WORDS

    def test_a_smaller_host_declaration_wins_over_the_fabric(self, monkeypatch):
        log: list = []
        monkeypatch.setitem(sys.modules, "pyxrt", fake_pyxrt(log))
        board = XrtTransport(xclbin_path="/x.xclbin", capture_events=16)
        board.open()
        assert board.capture_capacity == 16

    def test_a_bitstream_that_reports_no_storage_refuses_at_open(self, monkeypatch):
        log: list = []
        monkeypatch.setitem(
            sys.modules, "pyxrt", fake_pyxrt(log, capture_capacity=0))
        with pytest.raises(DeviceTransportError, match="silent empty one"):
            XrtTransport(xclbin_path="/x.xclbin").open()

    def test_the_capture_buffer_is_sized_from_the_capacity_the_kernel_reports(
        self, export, monkeypatch,
    ):
        log: list = []
        monkeypatch.setitem(sys.modules, "pyxrt", fake_pyxrt(log, (0, 0)))
        board = XrtTransport(xclbin_path="/x.xclbin", capture_events=1 << 20)
        board.open()
        board.program(export)
        run = board.run_samples([[{0: (0,) * 5, 1: (0,) * 4}]], latencies=(0, 1))
        reads = [entry for entry in log if entry[0] == "read"]
        assert reads[0][2] == WORD_BYTES * (
            CAPTURE_HEADER_WORDS
            + CAPTURE_RECORD_WORDS * FABRIC_CAPTURE_EVENTS)
        start = next(entry for entry in log if entry[0] == "start")
        assert start[1][5] == FABRIC_CAPTURE_EVENTS
        assert run.detail["capture_capacity"] == FABRIC_CAPTURE_EVENTS

    def test_an_overflowing_capture_refuses_with_the_reported_capacity(
        self, export, monkeypatch,
    ):
        # One event past the fabric's own record capacity.
        log: list = []
        monkeypatch.setitem(
            sys.modules, "pyxrt",
            fake_pyxrt(log, (FABRIC_CAPTURE_EVENTS + 1, 9)))
        board = XrtTransport(xclbin_path="/x.xclbin")
        board.open()
        board.program(export)
        with pytest.raises(
            OdinFpgaCaptureTruncated, match=str(FABRIC_CAPTURE_EVENTS),
        ):
            board.run_samples([[{0: (0,) * 5, 1: (0,) * 4}]], latencies=(0, 1))

    def test_the_status_register_is_read_after_the_run_and_err_refuses(
        self, export, monkeypatch,
    ):
        log: list = []
        monkeypatch.setitem(
            sys.modules, "pyxrt",
            fake_pyxrt(log, (0, 0), status=STATUS_ERR_BIT | 17))
        board = XrtTransport(xclbin_path="/x.xclbin")
        board.open()
        board.program(export)
        with pytest.raises(OdinFpgaKernelError, match="17 event"):
            board.run_samples([[{0: (0,) * 5, 1: (0,) * 4}]], latencies=(0, 1))
        kinds = [entry for entry in log if entry[0] == "read_register"]
        assert (("read_register", ADDR_STATUS)) in kinds
        # ... and it is read BEFORE the capture is synced back, so a refused
        # run never decodes a buffer the kernel did not fill.
        order = [entry[0] for entry in log]
        status_at = max(
            index for index, entry in enumerate(log)
            if entry[0] == "read_register" and entry[1] == ADDR_STATUS)
        assert "read" not in order[status_at:]

    def test_a_run_that_outgrows_the_program_ram_refuses_before_starting(
        self, export, monkeypatch,
    ):
        log: list = []
        monkeypatch.setitem(
            sys.modules, "pyxrt", fake_pyxrt(log, (0, 0), program_capacity=64))
        board = XrtTransport(xclbin_path="/x.xclbin")
        board.open()
        board.program(export)
        with pytest.raises(OdinFpgaProgramTooLarge, match="program RAM holds 64"):
            board.run_samples([[{0: (0,) * 5, 1: (0,) * 4}]], latencies=(0, 1))
        assert not any(entry[0] == "start" for entry in log)

    def test_the_stimulus_lands_on_the_programming_payloads_terminator(self):
        assert stimulus_base_word(4) == 3
        assert stimulus_base_word(1) == 0
        assert stimulus_base_word(0) == 0

    def test_the_status_word_splits_into_err_and_the_events_the_fabric_saw(self):
        assert decode_status(0) == (False, 0)
        assert decode_status(STATUS_ERR_BIT | 5) == (True, 5)
        assert decode_status(5) == (False, 5)


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
            XrtTransport(xclbin_path="/tmp/odin.xclbin").open()

    def test_program_before_open_refuses_by_name(self, export, injected_pyxrt):
        with pytest.raises(OdinFpgaDependencyError, match="before a live session"):
            XrtTransport(xclbin_path="/tmp/odin.xclbin").program(export)

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

    def test_both_transports_satisfy_the_declared_protocol(self, injected_pyxrt):
        assert isinstance(RtlCosimTransport(), DeviceTransport)
        assert isinstance(XrtTransport(xclbin_path="/x.xclbin"), DeviceTransport)


def test_the_payload_of_an_empty_op_list_is_just_the_terminator():
    assert payload_bytes(()) == struct.pack("<I", OP_END)
