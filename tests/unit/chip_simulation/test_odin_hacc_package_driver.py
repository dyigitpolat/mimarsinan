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
        "kernel": {
            "name": driver.KERNEL_NAME,
            "arg_program": driver.ARG_PROGRAM,
            "arg_stimulus": driver.ARG_STIMULUS,
            "arg_capture": driver.ARG_CAPTURE,
            "ctrl_offset": driver.CTRL_OFFSET,
            "addr_status": driver.ADDR_STATUS,
            "addr_capture_capacity": driver.ADDR_CAPTURE_CAPACITY,
            "addr_program_capacity": driver.ADDR_PROGRAM_CAPACITY,
            "status_err_bit": driver.STATUS_ERR_BIT,
            "capture_header_words": driver.CAPTURE_HEADER_WORDS,
            "capture_record_words": driver.CAPTURE_RECORD_WORDS,
            "word_bytes": driver.WORD_BYTES,
        },
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


def _session(driver, fixture, tmp_path, **overrides):
    session = driver.BoardSession(
        xclbin_path="/unit/none.xclbin", fake_pyxrt=str(FAKE_PATH),
        **overrides)
    del tmp_path
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

    def test_a_drifted_register_table_refuses_by_name(self, driver, tmp_path):
        document = _fixture(driver)
        document["kernel"]["addr_status"] = 0x40
        path = tmp_path / "unit_fixture.json"
        path.write_text(json.dumps(driver.seal(document)), encoding="utf-8")
        with pytest.raises(driver.OdinFixtureCorrupt, match="register table"):
            driver.load_fixture(str(path))


class TestTheCertifyPathAgainstTheFake:
    def test_a_clean_run_certifies_at_zero_delta_in_the_house_format(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault=None, cores=1)
        session = _session(driver, fixture, tmp_path)
        result = session.run_fixture(fixture)
        assert result["passed"]
        assert result["window_counts"] == fixture["expected"]["window"]
        assert result["device_cycles"] == 4242
        line = result["certificate_line"]
        assert line.startswith("[SpikeCountCertificate] spike-count certificate "
                               "[odin_fpga/exact]: PASS ")
        assert "exact=1.000000" in line and "max|dcount|=0 " in line
        assert "over 4 neuron-windows, 2 sample(s)" in line

    def test_the_call_order_is_the_audited_one(self, driver, fake, tmp_path):
        fixture = _fixture(driver)
        fake.arm(fixture, fault=None, cores=1)
        session = _session(driver, fixture, tmp_path)
        session.run_fixture(fixture)
        log = fake.call_log()
        assert log[0] == ("device", 0)
        assert log[1] == ("load_xclbin", "/unit/none.xclbin")
        assert log[2] == ("kernel", driver.KERNEL_NAME, "exclusive")
        assert log[3] == ("read_register", driver.ADDR_CAPTURE_CAPACITY)
        assert log[4] == ("read_register", driver.ADDR_PROGRAM_CAPACITY)
        kinds = [entry[0] for entry in log]
        assert kinds.index("write") < kinds.index("start") < kinds.index("wait")
        # The status register is read BEFORE the capture is synced back, so a
        # refused run never decodes a buffer the kernel did not fill.
        started_at = kinds.index("start")
        status_at = next(
            index for index, entry in enumerate(log)
            if index > started_at and entry[0] == "read_register"
            and entry[1] == driver.ADDR_STATUS)
        read_at = next(index for index, entry in enumerate(log)
                       if index > started_at and entry[0] == "read")
        assert status_at < read_at

    def test_one_extra_spike_from_the_device_fails_the_certificate(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault="extra_spike", cores=1)
        session = _session(driver, fixture, tmp_path)
        result = session.run_fixture(fixture)
        assert not result["passed"]
        assert "FAIL" in result["certificate_line"]
        assert result["certificate"]["max_abs_delta"] == 1.0
        # (sample, core, neuron, expected, got) for the one window that moved.
        assert result["certificate"]["divergent"] == [(0, 0, 0, 3, 4)]

    @pytest.mark.parametrize("fault,error,needle", [
        ("no_storage", "OdinDriverError", "silent empty one"),
        ("kernel_err", "OdinFpgaKernelError", "raised err"),
        ("truncated_capture", "OdinFpgaCaptureTruncated", "silent neurons"),
        ("tiny_program_ram", "OdinFpgaProgramTooLarge", "program RAM holds 64"),
    ])
    def test_every_typed_refusal_fires_before_a_count_is_reported(
        self, driver, fake, tmp_path, fault, error, needle,
    ):
        fixture = _fixture(driver)
        fake.arm(fixture, fault=fault, cores=1)
        with pytest.raises(getattr(driver, error), match=needle):
            session = _session(driver, fixture, tmp_path)
            session.run_fixture(fixture)

    def test_a_bitstream_with_fewer_cores_skips_instead_of_running(
        self, driver, fake, tmp_path,
    ):
        fixture = _fixture(driver, cores=2)
        fake.arm(fixture, fault=None, cores=1)
        session = _session(driver, fixture, tmp_path)
        with pytest.raises(driver.OdinFixtureNeedsMoreCores, match="NC 1"):
            session.run_fixture(fixture)


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
