"""[ODIN multi-core, Stage 1] NC cores on the card: placement, declaration, per-core decode.

A bitstream built at NC > 1 holds NC cores behind ONE sequencer; nothing routes
between them in fabric, so spikes between cores cross at segment boundaries
through the host. What the host side must then do, gated here with the fake
pyxrt (structure, not silicon):

  * CORE SELECTION is data. A program addresses LOGICAL cores 0..k-1 (a fixture's
    whole mapping, or core 0 of a pass's one-core mapping); a PLACEMENT maps each
    to a FABRIC core, and the token stream is retargeted by rewriting the core
    argument of every core-addressed op (SPI_W, SPI_R, AER, SHADOW, PROG). The
    repository's encoder (``odin_rtl/stimulus.retarget_ops``) and the shipped
    driver's byte-level copy (``retarget_payload``) are the SAME rule, and the
    identity placement leaves every byte where it was;
  * the DECLARED NC is the build's own (``ODIN_KERNEL_CORES``, the variable the
    bitstream was built with), named in the capacity's provenance;
  * the capture is DECODED PER CORE: each record's fabric core is mapped back to
    its logical core, a record from a core the placement did not program is
    refused, and every run reports a certificate and a readout per core;
  * the board driver can run every fixture at every rotation of the fabric
    (``--placements rotations``), so each fabric core certifies each fixture;
  * the deployment executor places pass k on fabric core (position of k in the
    pass order) mod NC: at NC >= passes every core of the network is resident on
    its own fabric core, and at NC = 1 the archived single-core run is unchanged
    byte for byte.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from integration.odin_hacc_harness import stage_package_host

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle
from mimarsinan.chip_simulation.odin_rtl import stimulus as stim
from mimarsinan.chip_simulation.odin_rtl.stimulus import Op

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "scripts" / "hacc" / "package"
DRIVER_PATH = PACKAGE / "host" / "odin_board_driver.py"
FAKE_PATH = PACKAGE / "host" / "fake_pyxrt_for_selftest.py"
BUNDLE_NAME = "nc1_two_core_passes.json"
REPLAY_NAME = "nc1_two_core_passes_capture.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load(DRIVER_PATH, "multicore_odin_board_driver")


@pytest.fixture
def fake(driver):
    driver._FAKE_MODULES.clear()
    module = driver.load_pyxrt(str(FAKE_PATH))
    module.arm(fault=None, capture_words=None, deployment=None)
    yield module
    module.arm(fault=None, capture_words=None, deployment=None)
    driver._FAKE_MODULES.clear()


def _every_op():
    """One of every opcode the encoder knows, on cores 0 and 1."""
    return [
        Op(stim.OP_SPI_W, (0, 0x12345, 0x00ABC)),
        Op(stim.OP_SPI_W, (1, 0x00001, 0x00002)),
        Op(stim.OP_SPI_R, (1, 0x80000, 0)),
        Op(stim.OP_SHADOW, (0, 3, 1)),
        Op(stim.OP_PROG, (1, 2, 300, 7)),
        Op(stim.OP_TAG, (9,)),
        Op(stim.OP_AER, (0, 0x507)),
        Op(stim.OP_AER, (1, 0x7F)),
        Op(stim.OP_WAIT, (64,)),
    ]


def _bytes(ops) -> bytes:
    return bundle.tokens_to_bytes(stim.encode_ops(ops))


# ---------------------------------------------------------------------------
# One retargeting rule, two implementations that must agree
# ---------------------------------------------------------------------------


class TestCoreSelectionIsOneRule:

    def test_the_encoder_names_which_ops_address_a_core(self):
        assert stim.CORE_ADDRESSED_OPS == frozenset(
            {stim.OP_SPI_W, stim.OP_SPI_R, stim.OP_AER, stim.OP_SHADOW, stim.OP_PROG})

    def test_retargeting_rewrites_the_core_argument_and_nothing_else(self):
        moved = stim.retarget_ops(_every_op(), {0: 3, 1: 2})
        for before, after in zip(_every_op(), moved):
            assert after.code == before.code
            if before.code in stim.CORE_ADDRESSED_OPS:
                assert after.args[0] == {0: 3, 1: 2}[before.args[0]]
                assert after.args[1:] == before.args[1:]
            else:
                assert after.args == before.args

    def test_the_identity_placement_is_the_identity(self):
        assert stim.retarget_ops(_every_op(), {0: 0, 1: 1}) == tuple(_every_op())

    def test_a_core_the_placement_does_not_name_is_refused(self):
        with pytest.raises(stim.StimulusError, match="core 1"):
            stim.retarget_ops(_every_op(), {0: 0})

    def test_two_logical_cores_on_one_fabric_core_is_refused(self):
        with pytest.raises(stim.StimulusError, match="same fabric core"):
            stim.retarget_ops(_every_op(), {0: 1, 1: 1})

    def test_the_driver_carries_the_encoders_opcode_table(self, driver):
        assert driver.OP_ARITY == stim.OP_ARITY
        assert driver.CORE_ADDRESSED_OPS == stim.CORE_ADDRESSED_OPS

    @pytest.mark.parametrize("placement", [{0: 0, 1: 1}, {0: 1, 1: 0}, {0: 3, 1: 2}])
    def test_the_driver_retargets_bytes_exactly_as_the_encoder_retargets_ops(
            self, driver, placement):
        expected = _bytes(stim.retarget_ops(_every_op(), placement))
        assert driver.retarget_payload(_bytes(_every_op()), placement) == expected

    def test_the_identity_placement_returns_the_very_bytes(self, driver):
        payload = _bytes(_every_op())
        assert driver.retarget_payload(payload, {0: 0, 1: 1}) == payload

    @pytest.mark.parametrize("tokens, needle", [
        ([8, 0], "unknown opcode 8"),
        ([stim.OP_SPI_W, 0, 1], "ends inside"),
        ([stim.OP_TAG, 1], "not END-terminated"),
        ([stim.OP_END, stim.OP_TAG], "after its END"),
    ])
    def test_a_stream_the_encoder_would_never_write_is_refused(self, driver, tokens, needle):
        with pytest.raises(driver.OdinDriverError, match=needle):
            driver.retarget_payload(bundle.tokens_to_bytes(tokens), {0: 1})

    def test_the_fake_walks_the_same_grammar(self, driver, fake):
        assert fake.OP_ARITY == stim.OP_ARITY
        assert fake.CORE_ADDRESSED_OPS == stim.CORE_ADDRESSED_OPS


class TestPlacementsAreRotationsOfTheFabric:

    def test_one_core_on_one_core_has_one_placement(self, driver):
        assert driver.rotation_placements(1, 1) == [{0: 0}]

    def test_each_rotation_puts_every_logical_core_on_a_distinct_fabric_core(self, driver):
        placements = driver.rotation_placements(2, 4)
        assert placements == [{0: 0, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 3}, {0: 3, 1: 0}]
        hosts = [sorted(p.values()) for p in placements]
        assert all(len(set(h)) == 2 for h in hosts)
        assert {core for p in placements for core in p.values()} == {0, 1, 2, 3}

    def test_more_logical_cores_than_the_fabric_holds_is_refused(self, driver):
        with pytest.raises(driver.OdinFixtureNeedsMoreCores, match="NC 2"):
            driver.rotation_placements(3, 2)


# ---------------------------------------------------------------------------
# The declared NC
# ---------------------------------------------------------------------------


class TestTheDeclaredCoresAreTheBuildsOwn:

    def test_the_build_variable_declares_the_cores(self, driver, monkeypatch):
        monkeypatch.setenv("ODIN_KERNEL_CORES", "4")
        options = driver.build_parser().parse_args(["--xclbin", "/x"])
        capacity = driver.capacity_from_options(options)
        assert capacity.cores == 4
        assert "ODIN_KERNEL_CORES=4" in capacity.provenance

    def test_the_command_line_still_wins_and_says_so(self, driver, monkeypatch):
        monkeypatch.setenv("ODIN_KERNEL_CORES", "4")
        options = driver.build_parser().parse_args(["--xclbin", "/x", "--declare-cores", "2"])
        capacity = driver.capacity_from_options(options)
        assert capacity.cores == 2
        assert "OVERRIDDEN on the command line" in capacity.provenance

    def test_without_either_the_package_declares_one_core(self, driver, monkeypatch):
        monkeypatch.delenv("ODIN_KERNEL_CORES", raising=False)
        options = driver.build_parser().parse_args(["--xclbin", "/x"])
        assert driver.capacity_from_options(options).cores == 1

    def test_a_build_variable_that_is_not_a_core_count_is_refused(self, driver, monkeypatch):
        monkeypatch.setenv("ODIN_KERNEL_CORES", "four")
        options = driver.build_parser().parse_args(["--xclbin", "/x"])
        with pytest.raises(driver.OdinDriverError, match="ODIN_KERNEL_CORES"):
            driver.capacity_from_options(options)

    def test_no_text_claims_the_build_makes_one_core_only(self, driver):
        source = DRIVER_PATH.read_text()
        assert "the only geometry" not in source
        assert "builds NC=1 only" not in source


# ---------------------------------------------------------------------------
# A two-core fixture with a real token stream, run at several placements
# ---------------------------------------------------------------------------


def _two_core_fixture(driver, *, samples=2):
    """Two cores, two neurons each, one AER event per core per cycle."""
    program = _bytes([
        Op(stim.OP_SPI_W, (0, 0x00010, 0x00001)),
        Op(stim.OP_SPI_W, (1, 0x00010, 0x00001)),
    ])
    stimulus = _bytes([
        Op(stim.OP_TAG, (1,)), Op(stim.OP_AER, (0, 0x007)), Op(stim.OP_AER, (1, 0x107)),
        Op(stim.OP_WAIT, (16,)), Op(stim.OP_TAG, (2,)), Op(stim.OP_WAIT, (16,)),
    ])
    per_cycle = []
    for sample in range(samples):
        per_cycle.append([sample, 0, 0, 0, 1])
        per_cycle.append([sample, 1, 0, 1, 2])
        per_cycle.append([sample, 1, 1, 0, 3])
    window = [[[1, 2], [3, 0]] for _ in range(samples)]
    document = {
        "schema": driver.SCHEMA, "name": "two_core_unit", "title": "unit",
        "description": "a synthetic two-core fixture with a real token stream",
        "source": "tests/unit/chip_simulation/test_odin_multicore_host.py",
        "provenance": {"generating_commit": "0" * 40, "worktree_dirty": True,
                       "rtl_sha256": "0" * 64, "cosim_engine": "none",
                       "generator": "unit test"},
        "kernel": dict(driver.KERNEL_TABLE),
        "program": driver.encode_payload(program),
        "stimulus": driver.encode_payload(stimulus),
        "run": {
            "cores": 2, "samples": samples, "cycles_per_sample": 2,
            "first_tag": 1, "barrier_cycles": 16, "latencies": [0, 0],
            "neurons": [2, 2], "simulation_length": 2,
            "capture_events_needed": 6 * samples,
            "program_words_needed": len(program) // 4 + len(stimulus) // 4 - 1,
        },
        "expected": {"per_cycle": per_cycle, "window": window,
                     "device_cycles_cosim": 777},
        "witnesses": {"synthetic": True},
    }
    return driver.seal(document)


def _session(driver, cores):
    session = driver.BoardSession(
        xclbin_path="/unit/none.xclbin", fake_pyxrt=str(FAKE_PATH),
        capacity=driver.KernelCapacity(
            cores=cores, provenance=f"unit test: a declared NC={cores} geometry"))
    session.open()
    return session


class TestTheCaptureIsDecodedPerCore:

    @pytest.mark.parametrize("placement", [{0: 0, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 0}])
    def test_a_placed_fixture_certifies_and_says_where_it_ran(
            self, driver, fake, placement):
        fixture = _two_core_fixture(driver)
        session = _session(driver, 3)
        result = session.run_fixture(fixture, placement=placement)
        assert result["passed"], result["certificate_line"]
        assert result["placement"] == {str(k): v for k, v in placement.items()}
        assert [row["logical_core"] for row in result["per_core"]] == [0, 1]
        assert [row["fabric_core"] for row in result["per_core"]] == [
            placement[0], placement[1]]
        assert all(row["certificate"]["passed"] for row in result["per_core"])
        assert [row["events"] for row in result["per_core"]] == [6, 6]
        assert result["per_core"][0]["window_counts"] == [[1, 2], [1, 2]]

    def test_the_fabric_was_sent_the_retargeted_stream(self, driver, fake):
        """The fake answers only if the program it receives IS the placed one."""
        fixture = _two_core_fixture(driver)
        session = _session(driver, 3)
        result = session.run_fixture(fixture, placement={0: 2, 1: 1})
        assert result["passed"]
        start = next(entry for entry in fake.call_log() if entry[0] == "start")
        # The word counts are the frozen payload's: retargeting moves no word.
        assert start[1][3] == len(driver.decode_payload(
            fixture["program"], what="p")) // 4
        assert start[1][4] == len(driver.decode_payload(
            fixture["stimulus"], what="s")) // 4

    def test_the_fake_answers_only_the_stream_of_its_placement(self, driver, fake):
        """A fabric expecting the UNPLACED streams is sent the placed ones: no verdict."""
        fixture = _two_core_fixture(driver)
        fake.arm(fixture, fault="ignore_placement")
        session = _session(driver, 3)
        with pytest.raises(driver.OdinFpgaKernelError, match="NO-VERDICT"):
            session.run_fixture(fixture, placement={0: 2, 1: 1})

    def test_a_record_from_a_core_the_placement_never_programmed_is_refused(
            self, driver, fake):
        fixture = _two_core_fixture(driver)
        fake.arm(fixture, fault="stray_core")
        session = _session(driver, 3)
        with pytest.raises(driver.OdinCaptureFromUnplacedCore, match="fabric core"):
            session.run_fixture(fixture, placement={0: 0, 1: 1})

    def test_one_extra_spike_fails_exactly_the_core_it_came_from(self, driver, fake):
        fixture = _two_core_fixture(driver)
        fake.arm(fixture, fault="extra_spike")
        session = _session(driver, 2)
        result = session.run_fixture(fixture, placement={0: 1, 1: 0})
        assert not result["passed"]
        verdicts = {row["logical_core"]: row["certificate"]["passed"]
                    for row in result["per_core"]}
        # The fake duplicates the first frozen record: sample 0, cycle 0, core 0.
        assert verdicts == {0: False, 1: True}

    def test_a_placement_onto_a_core_past_the_declared_nc_is_refused(self, driver, fake):
        fixture = _two_core_fixture(driver)
        session = _session(driver, 2)
        with pytest.raises(driver.OdinFixtureNeedsMoreCores, match="fabric core 2"):
            session.run_fixture(fixture, placement={0: 0, 1: 2})

    def test_the_identity_run_is_the_archived_one(self, driver, fake):
        fixture = _two_core_fixture(driver)
        session = _session(driver, 2)
        result = session.run_fixture(fixture)
        assert result["placement"] == {"0": 0, "1": 1}
        assert result["passed"]


class TestTheBoardRunCoversEveryFabricCore:

    def _write(self, driver, tmp_path, fixture):
        directory = tmp_path / "fixtures"
        directory.mkdir(exist_ok=True)
        (directory / f"{fixture['name']}.json").write_text(json.dumps(fixture))
        return directory

    def test_rotations_run_each_fixture_on_every_fabric_core(self, driver, fake, tmp_path):
        fixtures = self._write(driver, tmp_path, _two_core_fixture(driver))
        code = driver.main([
            "--xclbin", "/unit/none.xclbin", "--fake-pyxrt", str(FAKE_PATH),
            "--fixtures", str(fixtures), "--results", str(tmp_path / "out"),
            "--declare-cores", "3", "--placements", "rotations"])
        assert code == 0
        summary = json.loads((tmp_path / "out" / "summary_board.json").read_text())
        rows = summary["rows"]
        assert [row["placement"] for row in rows] == [
            {"0": 0, "1": 1}, {"0": 1, "1": 2}, {"0": 2, "1": 0}]
        assert all(row["passed"] for row in rows)
        assert summary["fabric_cores_certified"] == [0, 1, 2]
        assert (tmp_path / "out" / "fixture_two_core_unit@r1.json").is_file()

    def test_the_default_runs_each_fixture_once_where_it_always_ran(
            self, driver, fake, tmp_path):
        fixtures = self._write(driver, tmp_path, _two_core_fixture(driver))
        code = driver.main([
            "--xclbin", "/unit/none.xclbin", "--fake-pyxrt", str(FAKE_PATH),
            "--fixtures", str(fixtures), "--results", str(tmp_path / "out"),
            "--declare-cores", "3"])
        assert code == 0
        summary = json.loads((tmp_path / "out" / "summary_board.json").read_text())
        assert [row["placement"] for row in summary["rows"]] == [{"0": 0, "1": 1}]
        assert (tmp_path / "out" / "fixture_two_core_unit.json").is_file()

    def test_a_fixture_wider_than_the_fabric_is_skipped_by_name(self, driver, fake, tmp_path):
        fixtures = self._write(driver, tmp_path, _two_core_fixture(driver))
        single = _two_core_fixture(driver)
        code = driver.main([
            "--xclbin", "/unit/none.xclbin", "--fake-pyxrt", str(FAKE_PATH),
            "--fixtures", str(fixtures), "--results", str(tmp_path / "out"),
            "--declare-cores", "1", "--placements", "rotations"])
        summary = json.loads((tmp_path / "out" / "summary_board.json").read_text())
        assert summary["rows"][0]["skipped"]
        assert "ODIN_KERNEL_CORES" in summary["rows"][0]["reason"]
        assert code == 1 and single["run"]["cores"] == 2


# ---------------------------------------------------------------------------
# The deployment executor: one pass per core, each on its own fabric core
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def staged(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("odin_multicore_pkg")
    stage_package_host(root)
    (root / "deployment").mkdir()
    for name in (BUNDLE_NAME, REPLAY_NAME):
        shutil.copyfile(PACKAGE / "deployment" / name, root / "deployment" / name)
    return root


def _deploy(staged: Path, results: str, *args: str):
    completed = subprocess.run(
        [sys.executable, "host/odin_deployment_executor.py",
         "--xclbin", "/selftest/no-such.xclbin",
         "--fake-pyxrt", "host/fake_pyxrt_for_selftest.py",
         "--bundle", f"deployment/{BUNDLE_NAME}",
         "--replay", f"deployment/{REPLAY_NAME}",
         "--results", results, *args],
        cwd=str(staged), capture_output=True, text=True)
    report_path = staged / results / "deployment_report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    return completed, report


class TestPassesArePlacedOnFabricCores:

    def test_the_placement_rule_is_round_robin_over_the_pass_order(self, staged):
        executor = _load(staged / "host" / "odin_deployment_executor.py",
                         "multicore_executor_rule")
        document = bundle.load_bundle(str(staged / "deployment" / BUNDLE_NAME))
        order = list(document["pass_order"])
        assert executor.pass_placement(document, 1) == {core: 0 for core in order}
        assert executor.pass_placement(document, 2) == {
            core: position for position, core in enumerate(order)}
        assert executor.pass_placement(document, 4) == {
            core: position for position, core in enumerate(order)}
        with pytest.raises(executor.OdinDeploymentRefusal, match="no fabric core"):
            executor.pass_placement(document, 0)

    @pytest.mark.parametrize("arm", ["as_run", "resident"])
    def test_at_two_cores_every_pass_runs_resident_on_its_own_core(self, staged, arm):
        completed, report = _deploy(staged, f"results/nc2_{arm}",
                                    "--declare-cores", "2", "--arm", arm)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert report["passed"]
        assert report["placement"] == {"fabric_cores": 2, "pass_to_fabric": {"0": 0, "1": 1}}
        assert [row["fabric_core"] for row in report["per_core"]] == [0, 1]
        assert all(row["passed"] for row in report["certificates"])
        assert {row["fabric_core"] for row in report["certificates"]} == {0, 1}
        assert all(row["matches_frozen"] for row in report["readout"])

    def test_at_one_core_the_archived_run_is_unchanged(self, staged):
        completed, report = _deploy(staged, "results/nc1", "--declare-cores", "1")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert report["placement"] == {"fabric_cores": 1, "pass_to_fabric": {"0": 0, "1": 0}}
        assert [row["fabric_core"] for row in report["per_core"]] == [0, 0]
        assert report["passed"]

    def test_the_readout_is_reported_per_core(self, staged):
        _completed, report = _deploy(staged, "results/nc2_per_core", "--declare-cores", "2")
        rows = {row["core"]: row for row in report["per_core"]}
        for core, row in rows.items():
            assert row["certificates"] == sum(
                1 for cert in report["certificates"] if cert["core"] == core)
            assert row["certificates_passed"] == row["certificates"]
            assert row["passes"] == len(report["campaign"]["samples_run"])

    def test_a_record_from_another_fabric_core_fails_the_pass(self, staged, tmp_path):
        executor = _load(staged / "host" / "odin_deployment_executor.py",
                         "multicore_executor_decode")
        driver_module = executor.driver
        words = [1, 10, 1, 0, 1, 0]  # one record: tag 1, cycle 0, fabric core 1, neuron 0
        with pytest.raises(driver_module.OdinCaptureFromUnplacedCore, match="fabric core 1"):
            executor.decode_pass_capture(words, capture_events=16, fabric_core=0,
                                         transport="unit")
        events, cycles = executor.decode_pass_capture(
            words, capture_events=16, fabric_core=1, transport="unit")
        assert events == [(1, 0, bundle.PASS_CORE, 0)] and cycles == 10


class TestTheBundleNamesItsPassCore:

    def test_every_pass_addresses_the_bundles_pass_core(self):
        document = bundle.load_bundle(str(PACKAGE / "deployment" / BUNDLE_NAME))
        for plan in document["cores"]:
            program = bundle.decode_payload(plan["program"], what="program")
            words = [int.from_bytes(program[i:i + 4], "little")
                     for i in range(0, len(program), 4)]
            cores = {op.args[0] for op in stim.decode_ops(words)
                     if op.code in stim.CORE_ADDRESSED_OPS}
            assert cores == {bundle.PASS_CORE}


# ---------------------------------------------------------------------------
# Packaging refuses a driver whose grammar drifted from the encoder's
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def packager():
    path = REPO / "scripts" / "hacc" / "make_package.py"
    spec = importlib.util.spec_from_file_location("multicore_make_package", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestPackagingGatesTheRetargetRule:

    def test_the_shipped_driver_agrees_with_the_encoder(self, packager):
        packager.require_retarget_agrees()

    def test_a_drifted_opcode_table_refuses_the_package(self, packager, monkeypatch):
        drifted = dict(packager.driver.OP_ARITY)
        drifted[stim.OP_PROG] = 3
        monkeypatch.setattr(packager.driver, "OP_ARITY", drifted)
        with pytest.raises(packager.PackagingRefusal, match="opcode"):
            packager.require_retarget_agrees()

    def test_a_drifted_retarget_rule_refuses_the_package(self, packager, monkeypatch):
        monkeypatch.setattr(packager.driver, "retarget_payload",
                            lambda payload, placement: payload)
        with pytest.raises(packager.PackagingRefusal, match="retarget"):
            packager.require_retarget_agrees()


class TestTheProvenanceProbeStaysOutOfSubmodules:
    """[targeted pass, review m3] the packager's git probe never walks into a submodule."""

    def test_the_status_probe_ignores_submodules(self, packager, monkeypatch):
        calls = []

        class _Done:
            stdout = ""
            returncode = 0

        def record(command, **_kwargs):
            calls.append(list(command))
            return _Done()

        monkeypatch.setattr(packager.subprocess, "run", record)
        packager.git_head()
        status = [command for command in calls if "status" in command]
        assert status, calls
        assert all("--ignore-submodules=all" in command for command in status), status


# ---------------------------------------------------------------------------
# [targeted pass, review m4] the declared NC is tied to the bitstream's record
# ---------------------------------------------------------------------------


def _xclbin_with_record(tmp_path: Path, **fields):
    """A stand-in xclbin, and (unless ``fields`` is None) the identity record beside it."""
    import hashlib

    path = tmp_path / "odin_fpga_hw.xclbin"
    path.write_bytes(b"stand-in xclbin\n")
    record = {"xclbin_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "xclbin_uuid": f"uuid-of-{path}", "kernel_cores": "4",
              "compute_units": "1", "chip_config": "odin_stock_256x256",
              "target": "hw"}
    record.update(fields)
    (tmp_path / "odin_fpga_hw.xclbin.identity").write_text(
        "".join(f"{key}={value}\n" for key, value in record.items()))
    return path


def _open_declaring(driver, path: Path, cores: int):
    session = driver.BoardSession(
        xclbin_path=str(path), fake_pyxrt=str(FAKE_PATH),
        capacity=driver.KernelCapacity(cores=cores, provenance=f"unit test: NC={cores}"))
    session.open()
    return session


class TestTheDeclarationIsCheckedAgainstTheBuildRecord:

    def test_a_declaration_that_disagrees_with_the_build_is_refused(self, driver, fake, tmp_path):
        path = _xclbin_with_record(tmp_path, kernel_cores="4")
        with pytest.raises(driver.OdinBuildRecordDisagrees, match=r"4.*2|2.*4"):
            _open_declaring(driver, path, cores=2)

    def test_a_declaration_that_agrees_opens_and_names_the_record(self, driver, fake, tmp_path):
        path = _xclbin_with_record(tmp_path, kernel_cores="4")
        session = _open_declaring(driver, path, cores=4)
        try:
            record = session.device_stamp()["build_record"]
            assert record["kernel_cores"] == 4
            assert record["verified"] is True
            assert record["path"].endswith(".identity")
        finally:
            session.close()

    def test_a_record_of_another_bitstream_is_refused(self, driver, fake, tmp_path):
        path = _xclbin_with_record(tmp_path, xclbin_sha256="0" * 64)
        with pytest.raises(driver.OdinBuildRecordDisagrees, match="sha256"):
            _open_declaring(driver, path, cores=4)

    def test_a_record_whose_uuid_is_not_the_loaded_one_is_refused(self, driver, fake, tmp_path):
        path = _xclbin_with_record(tmp_path, xclbin_uuid="84843941-cc20-521a-9211-6107cb163395")
        with pytest.raises(driver.OdinBuildRecordDisagrees, match="UUID"):
            _open_declaring(driver, path, cores=4)

    def test_without_a_record_the_declaration_stands_and_says_so(self, driver, fake, tmp_path):
        path = tmp_path / "odin_fpga_hw.xclbin"
        path.write_bytes(b"stand-in xclbin\n")
        session = _open_declaring(driver, path, cores=2)
        try:
            record = session.device_stamp()["build_record"]
            assert record["verified"] is False
            assert "no build record" in record["note"]
        finally:
            session.close()
