"""[ODIN P7b] the SHIPPED board driver, gated here so the cluster never gates it.

``scripts/hacc/package/host/odin_board_driver.py`` is the only host code the
HACC package carries, and it is deliberately standalone: standard library plus
``pyxrt``, no mimarsinan import, no repository on the board node. That
independence is exactly why it needs a gate HERE — nothing else in the suite
would notice it rotting.

What these tests drive is the file that gets zipped, loaded by path, against the
fake ``pyxrt`` that gets zipped next to it: the seal, the refusals a corrupted
upload must earn, the window rule, and the certificate the owner will read off
a board log. Silicon is still the board's job; the CALL CONTRACT is ours.

THE CONTRACT CHANGED ON 2026-08-25. pyxrt binds no register access at all
(``test_odin_pyxrt_surface.py`` holds the evidence), so the call order asserted
below has no CSR read in it: capacities are DECLARED, and the kernel's verdict
arrives as the capture header it DMAs back — or, when it refuses, as the
host's no-verdict sentinel coming home untouched.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "scripts" / "hacc" / "package"
DRIVER_PATH = PACKAGE / "host" / "odin_board_driver.py"
FAKE_PATH = PACKAGE / "host" / "fake_pyxrt_for_selftest.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load(DRIVER_PATH, "packaged_odin_board_driver")


@pytest.fixture
def fake(driver):
    """A FRESH fake per test: the driver caches one module object per path."""
    driver._FAKE_MODULES.clear()
    module = driver.load_pyxrt(str(FAKE_PATH))
    yield module
    driver._FAKE_MODULES.clear()


def _fixture(driver, *, cores=1, samples=2):
    """A minimal fixture in the shipped schema, sealed by the driver's own seal.

    Two cycles per sample, one core of two neurons at latency 0, and a count of
    2 on one neuron so the fold is visibly a SUM over cycles rather than a copy.
    """
    program = bytes(range(64)) * 4
    stimulus = bytes(range(32)) * 2
    per_cycle = []
    for sample in range(samples):
        per_cycle.append([sample, 0, 0, 0, 1])
        per_cycle.append([sample, 1, 0, 0, 2])
        per_cycle.append([sample, 1, 0, 1, 1])
    window = [[[3, 1]] for _ in range(samples)]
    document = {
        "schema": driver.SCHEMA,
        "name": "unit_fixture",
        "title": "unit",
        "description": "a synthetic fixture in the shipped schema",
        "source": "tests/unit/chip_simulation/test_odin_hacc_package_driver.py",
        "provenance": {"generating_commit": "0" * 40, "worktree_dirty": True,
                       "rtl_sha256": "0" * 64, "cosim_engine": "none",
                       "generator": "unit test"},
        "kernel": dict(driver.KERNEL_TABLE),
        "program": driver.encode_payload(program),
        "stimulus": driver.encode_payload(stimulus),
        "run": {
            "cores": cores, "samples": samples, "cycles_per_sample": 2,
            "first_tag": 1, "barrier_cycles": 8, "latencies": [0],
            "neurons": [2], "simulation_length": 2,
            "capture_events_needed": 4 * samples,
            "program_words_needed": len(program) // 4 + len(stimulus) // 4 - 1,
        },
        "expected": {"per_cycle": per_cycle, "window": window,
                     "device_cycles_cosim": 4242},
        "witnesses": {"synthetic": True},
    }
    return driver.seal(document)


def _capacity(driver, *, cores=1, **overrides):
    """A DECLARED geometry: no host can read one back off a card."""
    fields = {
        "cores": cores,
        "provenance": f"unit test: a declared NC={cores} geometry",
    }
    fields.update(overrides)
    return driver.KernelCapacity(**fields)


def _session(driver, *, cores=1, capacity=None, **overrides):
    session = driver.BoardSession(
        xclbin_path="/unit/none.xclbin", fake_pyxrt=str(FAKE_PATH),
        capacity=capacity if capacity is not None else _capacity(
            driver, cores=cores),
        **overrides)
    session.open()
    return session


class TestTheSealIsOneImplementation:
    def test_a_sealed_fixture_loads(self, driver, tmp_path):
        path = tmp_path / "unit_fixture.json"
        path.write_text(json.dumps(_fixture(driver)), encoding="utf-8")
        assert driver.load_fixture(str(path))["name"] == "unit_fixture"

    def test_one_flipped_count_makes_the_self_hash_refuse(self, driver, tmp_path):
        document = _fixture(driver)
        document["expected"]["window"][0][0][0] += 1
        path = tmp_path / "unit_fixture.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(driver.OdinFixtureCorrupt, match="self-hash"):
            driver.load_fixture(str(path))

    def test_a_damaged_payload_refuses_after_the_self_hash_passes(
        self, driver, tmp_path,
    ):
        document = _fixture(driver)
        document["program"]["sha256"] = "0" * 64
        path = tmp_path / "unit_fixture.json"
        path.write_text(json.dumps(driver.seal(document)), encoding="utf-8")
        loaded = driver.load_fixture(str(path))
        with pytest.raises(driver.OdinFixtureCorrupt, match="damaged"):
            driver.decode_payload(loaded["program"], what="program")

    def test_a_drifted_protocol_table_refuses_by_name(self, driver, tmp_path):
        document = _fixture(driver)
        document["kernel"]["capture_header_words"] = 3
        path = tmp_path / "unit_fixture.json"
        path.write_text(json.dumps(driver.seal(document)), encoding="utf-8")
        with pytest.raises(driver.OdinFixtureCorrupt, match="device-protocol"):
            driver.load_fixture(str(path))

    def test_a_fixture_from_the_register_era_refuses_on_its_schema(
        self, driver, tmp_path,
    ):
        document = _fixture(driver)
        document["schema"] = "odin_hacc_fixture/1"
        path = tmp_path / "unit_fixture.json"
        path.write_text(json.dumps(driver.seal(document)), encoding="utf-8")
        with pytest.raises(driver.OdinFixtureCorrupt, match="schema"):
            driver.load_fixture(str(path))


class TestTheCertifyPathAgainstTheFake:
    def test_a_clean_run_certifies_at_zero_delta_in_the_house_format(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault=None)
        session = _session(driver)
        result = session.run_fixture(fixture)
        assert result["passed"]
        assert result["window_counts"] == fixture["expected"]["window"]
        assert result["device_cycles"] == 4242
        line = result["certificate_line"]
        assert line.startswith("[SpikeCountCertificate] spike-count certificate "
                               "[odin_fpga/exact]: PASS ")
        assert "exact=1.000000" in line and "max|dcount|=0 " in line
        assert "over 4 neuron-windows, 2 sample(s)" in line

    def test_the_call_order_is_arguments_and_memory_and_no_register(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault=None)
        session = _session(driver)
        session.run_fixture(fixture)
        log = fake.call_log()
        # The xclbin's own metadata is read BEFORE anything is pushed to a card.
        assert log[0] == ("xclbin", "/unit/none.xclbin")
        assert log[1] == ("get_kernels",)
        assert log[2] == ("get_mems",)
        assert log[3] == ("device", 0)
        assert log[4] == ("load_xclbin", "/unit/none.xclbin")
        assert log[5] == ("kernel", driver.KERNEL_NAME, "exclusive")
        kinds = [entry[0] for entry in log]
        assert kinds.index("write") < kinds.index("start") < kinds.index("wait")
        # No register is touched anywhere: the fake could not answer one.
        assert not any(kind.endswith("_register") for kind in kinds)

    def test_the_capture_header_is_poisoned_before_the_kernel_starts(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault=None)
        session = _session(driver)
        session.run_fixture(fixture)
        log = fake.call_log()
        started_at = [entry[0] for entry in log].index("start")
        sentinel_bytes = len(driver.no_verdict_header())
        poison = [
            index for index, entry in enumerate(log)
            if entry[0] == "write" and entry[1] == driver.ARG_CAPTURE
            and entry[2] == sentinel_bytes
        ]
        assert poison and max(poison) < started_at, (
            "the capture header must carry the no-verdict sentinel before the "
            "kernel runs, or a refusing kernel would be indistinguishable from "
            "a silent one")

    def test_the_capture_is_synced_back_only_after_the_wait(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault=None)
        session = _session(driver)
        session.run_fixture(fixture)
        log = fake.call_log()
        waited_at = [entry[0] for entry in log].index("wait")
        back = next(
            index for index, entry in enumerate(log)
            if entry[0] == "sync" and entry[1] == driver.ARG_CAPTURE
            and entry[2] == "XCL_BO_SYNC_BO_FROM_DEVICE")
        read_at = next(index for index, entry in enumerate(log)
                       if entry[0] == "read")
        assert waited_at < back < read_at

    def test_the_six_arguments_go_in_the_frozen_order(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault=None)
        session = _session(driver)
        result = session.run_fixture(fixture)
        start = next(entry for entry in fake.call_log() if entry[0] == "start")
        assert start[1][:3] == (
            driver.ARG_PROGRAM, driver.ARG_STIMULUS, driver.ARG_CAPTURE)
        assert start[1][3] == result["program_words"]
        assert start[1][4] == result["stimulus_words"]
        assert start[1][5] == session.capture_capacity

    def test_one_extra_spike_from_the_device_fails_the_certificate(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault="extra_spike")
        session = _session(driver)
        result = session.run_fixture(fixture)
        assert not result["passed"]
        assert "FAIL" in result["certificate_line"]
        assert result["certificate"]["max_abs_delta"] == 1.0
        # (sample, core, neuron, expected, got) for the one window that moved.
        assert result["certificate"]["divergent"] == [(0, 0, 0, 3, 4)]


class TestEveryTypedRefusalHasADeviceWayToHappen:
    """Each refusal below names the ONE thing a card can do to earn it."""

    def test_a_package_that_declares_no_storage_refuses_at_open(
        self, driver, fake, tmp_path,
    ):
        capacity = _capacity(
            driver, cores=0, capture_events=0,
            provenance="unit test: a package declaring no storage")
        with pytest.raises(driver.OdinDriverError, match="silent empty one"):
            _session(driver, capacity=capacity)

    def test_the_refusal_says_where_the_capacity_number_came_from(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver, cores=2)
        fake.arm(fixture, fault=None)
        session = _session(driver, capacity=_capacity(
            driver, cores=1,
            provenance="unit test: an NC=1 geometry, declared by hand"))
        with pytest.raises(driver.OdinFixtureNeedsMoreCores) as exc:
            session.run_fixture(fixture)
        assert "declared by hand" in str(exc.value)

    def test_no_program_length_can_be_refused_for_not_fitting(
        self, driver, fake, tmp_path,
    ):
        """The fabric stores no program, so no run is too long to deliver."""
        assert not hasattr(driver, "OdinFpgaProgramTooLarge")
        assert not hasattr(driver, "require_program_fits")
        fixture = _fixture(driver)
        fake.arm(fixture, fault=None)
        session = _session(driver)
        session.run_fixture(fixture)
        assert any(entry[0] == "start" for entry in fake.call_log())

    def test_a_kernel_that_wrote_no_verdict_refuses_as_a_kernel_error(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault="kernel_err")
        session = _session(driver)
        with pytest.raises(
            driver.OdinFpgaKernelError, match="NO-VERDICT sentinel",
        ):
            session.run_fixture(fixture)

    def test_a_run_that_never_reached_ap_done_refuses_by_its_ert_state(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault="not_completed")
        session = _session(driver)
        with pytest.raises(driver.OdinFpgaKernelError, match="never saw ap_done"):
            session.run_fixture(fixture)

    def test_a_truncated_capture_refuses_instead_of_reporting_silent_neurons(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault="truncated_capture")
        session = _session(driver)
        with pytest.raises(
            driver.OdinFpgaCaptureTruncated, match="silent neurons",
        ):
            session.run_fixture(fixture)

    def test_an_xclbin_without_our_kernel_refuses_before_the_card_sees_it(
        self, driver, fake, tmp_path,
    ):
        fake.arm(fault="wrong_kernel")
        with pytest.raises(driver.OdinDriverError, match="none of them"):
            _session(driver)
        assert not any(
            entry[0] == "load_xclbin" for entry in fake.call_log())

    def test_a_bitstream_with_fewer_cores_skips_instead_of_running(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver, cores=2)
        fake.arm(fixture, fault=None)
        session = _session(driver, cores=1)
        with pytest.raises(driver.OdinFixtureNeedsMoreCores, match="NC 1"):
            session.run_fixture(fixture)


class TestTheHeaderLayoutIsTheRtlsOwn:
    def test_the_header_is_two_words_and_the_record_is_four(self, driver):
        assert driver.CAPTURE_HEADER_WORDS == 2
        assert driver.CAPTURE_RECORD_WORDS == 4
        assert (driver.HEADER_EVENTS_SEEN, driver.HEADER_DEVICE_CYCLES) == (0, 1)
        assert (driver.RECORD_TAG, driver.RECORD_CYCLE,
                driver.RECORD_CORE, driver.RECORD_NEURON) == (0, 1, 2, 3)

    def test_the_sentinel_fills_exactly_the_header(self, driver):
        assert len(driver.no_verdict_header()) == (
            driver.CAPTURE_HEADER_WORDS * driver.WORD_BYTES)
        assert driver.capture_words(driver.no_verdict_header()) == [
            driver.CAPTURE_NO_VERDICT] * driver.CAPTURE_HEADER_WORDS

    def test_a_capture_shorter_than_its_header_refuses(self, driver):
        with pytest.raises(driver.OdinDriverError, match="two-word header"):
            driver.decode_capture((0,), 4)

    def test_a_capture_exactly_at_capacity_refuses_too(self, driver):
        with pytest.raises(driver.OdinFpgaCaptureTruncated, match="capacity of 4"):
            driver.decode_capture((4, 100), 4)

    def test_a_half_written_header_is_not_read_as_a_refusal(self, driver):
        """Only BOTH words being the sentinel means no verdict; one does not."""
        events, cycles = driver.decode_capture(
            (0, driver.CAPTURE_NO_VERDICT), 4)
        assert events == [] and cycles == driver.CAPTURE_NO_VERDICT


class TestTheDeclaredCapacityNamesItsSource:
    def test_the_shipped_declaration_is_the_rtl_default_at_nc1(self, driver):
        capacity = driver.KernelCapacity()
        assert capacity.capture_events == (
            driver.SHIPPED_CAPTURE_WORDS - 2) // 4
        assert capacity.cores == driver.SHIPPED_KERNEL_CORES

    def test_the_provenance_says_it_was_not_read_off_the_card(self, driver):
        provenance = driver.KernelCapacity().provenance
        assert "odin_fpga_kernel_top.v" in provenance
        assert "NOT read back from the card" in provenance

    def test_a_command_line_override_is_recorded_in_the_provenance(self, driver):
        options = driver.build_parser().parse_args(
            ["--xclbin", "/x", "--declare-cores", "4"])
        capacity = driver.capacity_from_options(options)
        assert capacity.cores == 4
        assert "OVERRIDDEN on the command line" in capacity.provenance

    def test_the_host_ceiling_can_only_lower_the_declared_one(self, driver):
        capacity = driver.KernelCapacity()
        assert capacity.ceiling(16) == 16
        assert capacity.ceiling(1 << 30) == capacity.capture_events


class TestB0IsARoundTripNotACsrRead:
    def test_the_probe_runs_a_null_program_and_reads_its_header_back(
        self, driver, fake, tmp_path,
    ):
        code = driver.main([
            "--probe", "--xclbin", "/unit/none.xclbin",
            "--fake-pyxrt", str(FAKE_PATH), "--results", str(tmp_path)])
        assert code == 0
        report = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))
        assert report["null_run"]["events_seen"] == 0
        assert report["kernel"] == driver.KERNEL_NAME
        assert report["cu_access_mode"] == "exclusive"
        assert report["xclbin_kernels"][driver.KERNEL_NAME] == driver.KERNEL_ARGS
        assert report["memory_banks"][0]["tag"] == "DDR[0]"
        assert any("EXCLUSIVE" in claim for claim in report["proves"])
        assert "any spike count" in report["does_not_prove"]
        assert "0x4C stays unreadable" in report["does_not_prove"]

    def test_the_probe_refuses_an_xclbin_that_is_not_ours(
        self, driver, fake, tmp_path,
    ):
        fake.arm(fault="wrong_kernel")
        code = driver.main([
            "--probe", "--xclbin", "/unit/none.xclbin",
            "--fake-pyxrt", str(FAKE_PATH), "--results", str(tmp_path)])
        assert code == 2
        assert not (tmp_path / "probe.json").exists()

    def test_a_fabric_that_spikes_with_no_network_loaded_refuses(
        self, driver, fake, tmp_path,
    ):
        fake.arm(capture_words=(3, 99))
        session = _session(driver)
        with pytest.raises(driver.OdinDriverError, match="no network"):
            session.null_run()


class TestTheWindowRuleIsNevresimsAndNotAnAverage:
    def test_counts_sum_over_each_cores_own_window(self, driver):
        run = {"samples": 1, "cycles_per_sample": 4, "latencies": [1],
               "neurons": [2], "simulation_length": 2, "first_tag": 1}
        counts = {(0, 0, 0, 0): 5, (0, 1, 0, 0): 1, (0, 2, 0, 1): 3,
                  (0, 3, 0, 0): 7}
        # Cycle 0 is before the core's latency and cycle 3 is past its window:
        # both are drain, not deployment, and neither is counted.
        assert driver.window_counts(counts, run) == [[[1, 3]]]

    def test_tags_decode_to_the_sample_and_cycle_the_sequencer_wrote(self, driver):
        run = {"samples": 2, "cycles_per_sample": 3, "first_tag": 1}
        events = [(1, 0, 0, 0), (4, 0, 0, 1), (6, 0, 0, 1)]
        assert driver.fold_events(events, run) == {
            (0, 0, 0, 0): 1, (1, 0, 0, 1): 1, (1, 2, 0, 1): 1}

    def test_a_vacuous_certificate_refuses(self, driver):
        with pytest.raises(driver.OdinDriverError, match="ZERO neuron-windows"):
            driver.Certificate([], [], samples=0)


class TestTheDriverStaysUploadable:
    def test_it_imports_nothing_outside_the_standard_library(self):
        source = DRIVER_PATH.read_text(encoding="utf-8")
        banned = ("import numpy", "import torch", "from mimarsinan",
                  "import mimarsinan", "import pytest")
        for needle in banned:
            assert needle not in source, needle

    def test_pyxrt_is_imported_inside_the_seam_and_refuses_by_name(self, driver):
        source = DRIVER_PATH.read_text(encoding="utf-8")
        assert "    import pyxrt" in source, "pyxrt must not be a module import"
        with pytest.raises(driver.OdinFpgaDependencyError, match="Xilinx Runtime"):
            driver.load_pyxrt(None)

    def test_it_reads_a_numpy_style_buffer_the_way_pyxrt_answers(self, driver):
        """``pyxrt.bo.read`` returns a numpy array, not bytes."""
        pytest.importorskip("numpy")
        import numpy as np

        raw = np.frombuffer(driver.no_verdict_header(), dtype=np.int8)
        assert driver.capture_words(raw) == [
            driver.CAPTURE_NO_VERDICT] * driver.CAPTURE_HEADER_WORDS
