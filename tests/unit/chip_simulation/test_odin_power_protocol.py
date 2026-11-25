"""[ODIN H0] the card power protocol: resident arm, plan, gate, estimate, session.

STRUCTURE, NOT SILICON. Everything here drives the SHIPPED host files from a
staged package directory against the fake pyxrt: the resident arm's call
contract (one programming run, then ``program_words = 1``), the transfer
control (the same buffer writes and syncs, never ``ap_start``), the plan's
validation, xclbin identity pins and card-second budget, the pilot gate's
three verdicts with the warm-up taken off the counts, the estimator's three
outcomes on synthetic records with a known differential under the labels the
card boundary allows, the session's budget-capped child timeouts, and whole
fake sessions through ``odin_board.sbatch`` with ``ODIN_MODE=power``.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from integration.odin_hacc_harness import stage_package_host

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "scripts" / "hacc" / "package"
BUNDLE_NAME = "nc1_two_core_passes.json"
REPLAY_NAME = "nc1_two_core_passes_capture.json"
SHIPPED_PLAN = PACKAGE / "power" / "plan_hacc_u55c_tcad.json"
HOST_MODULES = ("odin_board_driver", "odin_deployment_bundle", "odin_deployment_encoding",
                "odin_deployment_executor", "odin_power_sampler", "odin_power_estimate",
                "odin_power_protocol")


def _stage(root: Path) -> Path:
    stage_package_host(root)
    (root / "deployment").mkdir(exist_ok=True)
    for name in (BUNDLE_NAME, REPLAY_NAME):
        shutil.copyfile(PACKAGE / "deployment" / name, root / "deployment" / name)
    (root / "power").mkdir(exist_ok=True)
    shutil.copyfile(SHIPPED_PLAN, root / "power" / SHIPPED_PLAN.name)
    (root / "scripts" / "hacc").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PACKAGE / "sbatch" / "odin_board.sbatch",
                    root / "scripts" / "hacc" / "odin_board.sbatch")
    return root


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    """The staged host/ modules, imported by their shipped names, then forgotten."""
    root = _stage(tmp_path_factory.mktemp("odin_power_pkg"))
    saved = {name: sys.modules.pop(name) for name in HOST_MODULES if name in sys.modules}
    sys.path.insert(0, str(root / "host"))
    try:
        modules = {name: importlib.import_module(name) for name in HOST_MODULES}
        yield SimpleNamespace(root=root, fake=str(root / "host" / "fake_pyxrt_for_selftest.py"),
                              driver=modules["odin_board_driver"],
                              executor=modules["odin_deployment_executor"],
                              estimator=modules["odin_power_estimate"],
                              protocol=modules["odin_power_protocol"])
    finally:
        sys.path.remove(str(root / "host"))
        for name in HOST_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def _session(host, document):
    driver = host.driver
    driver._FAKE_MODULES.clear()
    options = SimpleNamespace(
        xclbin="/unit/none.xclbin", device_index=0,
        capture_events=driver.DEFAULT_CAPTURE_EVENTS, declare_cores=0,
        capture_ram_events=0, chip="odin_stock_256x256", run_timeout_ms=0,
        fake_pyxrt=host.fake)
    session = driver.open_session(options)
    replay = json.loads((host.root / "deployment" / REPLAY_NAME).read_text())
    session.arm_fake(deployment=replay)
    fake = driver.load_pyxrt(host.fake)
    return session, fake


def _starts(fake):
    return [entry[1] for entry in fake.call_log() if entry[0] == "start"]


def _keys(node):
    """Every dictionary key anywhere in a decoded JSON tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


class TestTheResidentArm:
    @pytest.fixture
    def document(self, host):
        return host.executor.load_bundle(str(host.root / "deployment" / BUNDLE_NAME))

    def test_as_run_still_declares_the_whole_program_every_pass(self, host, document):
        session, fake = _session(host, document)
        campaign = host.executor.Campaign(document, session, samples=[0, 1, 2, 3])
        campaign.run()
        words = {int(plan["core"]): len(host.executor.decode(plan["program"], what="p")) // 4
                 for plan in host.executor.bundle.pass_plans(document)}
        starts = _starts(fake)
        assert len(starts) == len(campaign.pass_rows) == 8
        assert sorted({start[3] for start in starts}) == sorted(set(words.values()))
        assert all("programming_run" not in row for row in campaign.core_rows)

    def test_resident_programs_once_then_declares_one_word(self, host, document):
        session, fake = _session(host, document)
        campaign = host.executor.Campaign(document, session, samples=[0, 1, 2, 3],
                                          arm=host.executor.ARM_RESIDENT)
        campaign.run()
        starts = _starts(fake)
        # Per core: one programming run (whole program, one-word END stimulus),
        # then one run per sample declaring a single program word.
        assert len(starts) == 2 + 8
        programming = [start for start in starts if start[4] == 1]
        assert len(programming) == 2 and all(start[3] > 1 for start in programming)
        passes = [start for start in starts if start[4] != 1]
        assert len(passes) == 8 and {start[3] for start in passes} == {1}
        assert all(row["passed"] for row in campaign.certificates)
        rows, _accuracy = campaign.readout()
        assert all(row["matches_frozen"] for row in rows)
        for row in campaign.core_rows:
            run = row["programming_run"]
            assert run["capture_events"] == 0 and run["stimulus_words"] == 1

    def test_an_unknown_arm_refuses(self, host, document):
        session, _fake = _session(host, document)
        with pytest.raises(host.executor.OdinDeploymentRefusal, match="arm"):
            host.executor.Campaign(document, session, samples=[0], arm="hot")

    def test_the_timed_loop_runs_whole_certified_campaigns(self, host, document):
        session, _fake = _session(host, document)
        campaign = host.executor.Campaign(document, session, samples=[0, 1, 2, 3],
                                          arm=host.executor.ARM_RESIDENT)
        campaign.quiet = True
        loop = host.executor.timed_loop(campaign, 0.05)
        assert loop["campaigns"] >= 1 and loop["inferences"] == 4 * loop["campaigns"]
        assert loop["passes"] == 8 * loop["campaigns"]
        assert loop["programming_runs"] == 2 * loop["campaigns"]
        assert loop["passed"] and loop["certificates_failed"] == 0
        assert 0.0 < loop["kernel_duty"] <= 1.0 and loop["wall_s"] >= 0.05


def _transfer_entries(log):
    """Every buffer write and sync of a call log, in order (what crosses PCIe)."""
    return [entry for entry in log if entry[0] in ("write", "sync")]


class TestTheTransferControl:
    """The control condition: an arm's host-card transfers, never ``ap_start``."""

    @pytest.fixture
    def document(self, host):
        return host.executor.load_bundle(str(host.root / "deployment" / BUNDLE_NAME))

    @pytest.mark.parametrize("arm", ["as_run", "resident"])
    def test_it_replays_the_certified_transfers_without_starting_the_kernel(
            self, host, document, arm):
        session, fake = _session(host, document)
        campaign = host.executor.Campaign(document, session, samples=[0, 1, 2, 3], arm=arm)
        campaign.quiet = True
        campaign.record_transfers = True
        campaign.run()
        certified = list(fake.call_log())
        control = host.executor.TransferControl(campaign)
        mark = len(fake.call_log())
        stats = host.executor.transfer_loop(control, 0.0)
        replay = fake.call_log()[mark:]
        # Never ap_start: the kernel is not called once during the control.
        assert not [entry for entry in replay if entry[0] in ("start", "wait")]
        assert stats["ap_starts"] == 0 and stats["kernel_duty"] == 0.0 and stats["run_s"] == 0.0
        # The same buffer writes and syncs, in sizes and order, as the certified campaign.
        assert _transfer_entries(replay) == _transfer_entries(certified)
        assert stats["campaigns"] == 1 and stats["passes"] == 8
        assert stats["inferences"] == 4.0 and stats["passed"] is True
        assert stats["programming_replays"] == (2 if arm == "resident" else 0)
        assert stats["bytes_to_device"] > 0 and stats["bytes_from_device"] > 0
        assert stats["arm"] == arm and stats["source_certified"] is True

    def test_each_pass_is_paced_by_the_wall_the_certified_pass_took(self, host, document):
        session, _fake = _session(host, document)
        campaign = host.executor.Campaign(document, session, samples=[0, 1, 2, 3])
        campaign.quiet = True
        campaign.record_transfers = True
        campaign.run()
        control = host.executor.TransferControl(campaign)
        recorded = sum(row["pause_s"] for row in control.transfers.values())
        stats = host.executor.transfer_loop(control, 0.0)
        assert stats["paced_s"] == pytest.approx(recorded)
        for row in campaign.pass_rows:
            pause = control.transfers[(row["core"], row["sample"])]["pause_s"]
            transfers = row["bo_write_s"] + row["sync_s"] + row["readback_s"]
            assert pause == pytest.approx(max(0.0, row["pass_total_s"] - transfers))

    def test_a_control_window_ends_at_a_pass_boundary(self, host, document):
        session, _fake = _session(host, document)
        campaign = host.executor.Campaign(document, session, samples=[0, 1, 2, 3])
        campaign.quiet = True
        campaign.record_transfers = True
        campaign.run()
        stats = host.executor.transfer_loop(host.executor.TransferControl(campaign), 1e-9)
        assert stats["passes"] == 1 and stats["campaigns"] == 0
        assert stats["inferences"] == pytest.approx(0.5)  # one pass of a two-core sample

    def test_a_campaign_that_recorded_nothing_refuses(self, host, document):
        session, _fake = _session(host, document)
        campaign = host.executor.Campaign(document, session, samples=[0, 1])
        campaign.quiet = True
        campaign.run()
        with pytest.raises(host.executor.OdinDeploymentRefusal, match="record_transfers"):
            host.executor.TransferControl(campaign)

    def test_recording_leaves_the_certified_campaign_unchanged(self, host, document):
        rows = {}
        for record in (False, True):
            session, _fake = _session(host, document)
            campaign = host.executor.Campaign(document, session, samples=[0, 1, 2, 3])
            campaign.quiet = True
            campaign.record_transfers = record
            campaign.run()
            readout, _accuracy = campaign.readout()
            rows[record] = ([c["passed"] for c in campaign.certificates], readout)
        assert rows[False] == rows[True]


def _run_executor(root: Path, *args: str, env=None):
    return subprocess.run(
        [sys.executable, "host/odin_deployment_executor.py",
         "--xclbin", "/selftest/no-such.xclbin",
         "--fake-pyxrt", "host/fake_pyxrt_for_selftest.py",
         "--bundle", f"deployment/{BUNDLE_NAME}", "--replay", f"deployment/{REPLAY_NAME}",
         *args], cwd=str(root), capture_output=True, text=True,
        env=dict(os.environ, **(env or {})))


class TestTheExecutorCommandLine:
    def test_resident_from_the_environment_is_certified_and_recorded(self, host):
        completed = _run_executor(host.root, "--results", "results/resident",
                                  env={"ODIN_ARM": "resident"})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        report = json.loads((host.root / "results" / "resident" /
                             "deployment_report.json").read_text())
        assert report["campaign"]["arm"] == "resident"
        assert report["campaign"]["arm_source"] == "ODIN_ARM=resident"
        assert all(row["programming_run"] for row in report["per_core"])
        identity = report["device"]["identity"]
        assert identity["bdf"] == "0000:81:00.1" and identity["xclbin_uuid"]

    def test_the_default_arm_is_as_run_and_says_so(self, host):
        completed = _run_executor(host.root, "--results", "results/default")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        report = json.loads((host.root / "results" / "default" /
                             "deployment_report.json").read_text())
        assert report["campaign"]["arm"] == "as_run"
        assert "loop" not in report

    def test_a_loop_is_reported_apart_from_the_campaign(self, host):
        completed = _run_executor(host.root, "--results", "results/loop", "--loop-s", "0.2")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        report = json.loads((host.root / "results" / "loop" /
                             "deployment_report.json").read_text())
        assert report["loop"]["passed"] and report["loop"]["campaigns"] >= 1
        assert report["walls"]["passes"] == 8

    def test_a_bad_arm_in_the_environment_refuses(self, host):
        completed = _run_executor(host.root, "--results", "results/bad",
                                  env={"ODIN_ARM": "cold"})
        assert completed.returncode == 2 and "ODIN_ARM" in completed.stderr


class TestThePlan:
    def test_a_plan_path_expands_the_environment(self, host, tmp_path, monkeypatch):
        """A plan names its files by ${USER} rather than by one user's directory."""
        monkeypatch.setenv("USER", "someone")
        protocol = host.protocol
        assert protocol.resolve(str(tmp_path), "/data/${USER}/odin/x.xclbin") == \
            "/data/someone/odin/x.xclbin"
        assert protocol.resolve(str(tmp_path), "staged/${USER}.json") == \
            os.path.normpath(os.path.join(str(tmp_path), "staged", "someone.json"))
        plan = protocol.load_plan(str(SHIPPED_PLAN))
        assert plan["designs"]["stock"]["xclbin"] == "/data/someone/odin_stock/odin_fpga_hw.xclbin"

    def test_the_shipped_plan_fits_its_budget(self, host):
        plan = host.protocol.load_plan(str(SHIPPED_PLAN))
        budget = host.protocol.budget(plan)
        assert budget["fits"] and budget["budget_s"] == 3300
        assert budget["worst_path_s"] == max(budget["go_path_s"], budget["reduced_path_s"])
        assert budget["worst_path_s"] < 3600  # the U55C board partition's one-hour cap

    def test_the_transfer_control_costs_its_windows_and_one_sample_per_round(self, host):
        plan = host.protocol.load_plan(str(SHIPPED_PLAN))
        budget = host.protocol.budget(plan)
        # Before the control: GO 2,843.3 s and REDUCED 2,606.3 s (h0.patch). The
        # control adds, per stock round, one block and at most one sample's wall.
        per_sample = 13.3 / 64
        assert budget["go_path_s"] == pytest.approx(2843.3 + 5 * (60 + per_sample), abs=1e-6)
        assert budget["reduced_path_s"] == pytest.approx(2606.3 + 5 * (110 + per_sample),
                                                         abs=1e-6)
        assert budget["worst_path_s"] <= 3300 - 100  # headroom left for slow loads

    def test_the_shipped_plan_pins_the_archived_bundles(self, host):
        plan = json.loads(SHIPPED_PLAN.read_text())
        assert plan["designs"]["stock"]["bundle_self_hash"].startswith("sha256:2c5de8eb")
        assert plan["designs"]["wide"]["bundle_self_hash"].startswith("sha256:e1b44e06")
        kinds = [step["kind"] for step in plan["steps"]]
        assert kinds[0] == "shell_only" and kinds[1] == "pilot"

    def test_the_shipped_plan_runs_the_transfer_control_on_stock_only(self, host):
        plan = json.loads(SHIPPED_PLAN.read_text())
        protocols = [step for step in plan["steps"] if step["kind"] == "protocol"]
        assert protocols
        for step in protocols:
            assert ("transfer_as_run" in step["conditions"]) == (step["design"] == "stock")

    def test_the_shipped_plan_pins_the_archived_xclbin_builds(self, host):
        plan = json.loads(SHIPPED_PLAN.read_text())
        # MEASURED: "Successfully wrote (N bytes)" in each link's v++.log (misc/ tarballs).
        sizes = {name: design["xclbin_identity"]["archived"]["size_bytes"]
                 for name, design in plan["designs"].items()}
        assert sizes == {"stock": 48395756, "wide": 47559659}
        for design in plan["designs"].values():
            identity = design["xclbin_identity"]
            assert identity["rebuilt"] is False
            assert "2026-08-31" in identity["archived"]["written"]

    def test_the_shipped_plan_carries_odin_0_and_hmss_0_side_by_side(self, host):
        plan = json.loads(SHIPPED_PLAN.read_text())
        assert {name: (design["vivado"]["odin_0_dynamic_w"], design["vivado"]["hmss_0_dynamic_w"])
                for name, design in plan["designs"].items()} == {
            "stock": (0.068, 2.223), "wide": (0.145, 2.199)}
        assert not [key for key in _keys(plan) if "ratio" in key.lower()]

    @pytest.mark.parametrize("mutate, needle", [
        (lambda p: p.update(schema="odin_power_plan/0"), "schema"),
        (lambda p: p["steps"].insert(2, {"kind": "shell_only", "seconds": 5}), "after an xclbin"),
        (lambda p: p["designs"]["stock"]["campaign_s"].pop("resident"), "campaign_s"),
        (lambda p: p["steps"][4].update(conditions=["loop_as_run"]), "kernel_idle"),
        (lambda p: p["steps"][4].update(when="sometimes"), "when"),
        (lambda p: p.update(budget_s=0), "budget_s"),
        (lambda p: p["steps"][4].update(conditions=["kernel_idle", "loop_resident",
                                                    "transfer_as_run"]), "transfer_as_run"),
        (lambda p: p["designs"]["stock"].pop("xclbin_identity"), "xclbin_identity"),
        (lambda p: p["designs"]["stock"]["xclbin_identity"].update(rebuilt=True),
         "rebuilt"),
    ], ids=["schema", "late-shell", "no-campaign-wall", "no-idle", "bad-when", "no-budget",
            "control-without-its-arm", "no-xclbin-identity", "rebuilt-without-uuid"])
    def test_a_malformed_plan_refuses(self, host, tmp_path, mutate, needle):
        plan = json.loads(SHIPPED_PLAN.read_text())
        mutate(plan)
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan))
        with pytest.raises(host.protocol.OdinPowerRefusal, match=needle):
            host.protocol.load_plan(str(path))

    def test_an_over_budget_plan_refuses_before_the_card(self, host, tmp_path):
        plan = json.loads(SHIPPED_PLAN.read_text())
        plan["budget_s"] = 1000
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan))
        with pytest.raises(host.protocol.OdinPowerRefusal, match="worst path"):
            host.protocol.Session(str(path), str(tmp_path / "results")).run()
        assert not (tmp_path / "results" / "markers.jsonl").exists()

    def test_rounds_rotate_the_condition_order(self, host):
        conditions = ["kernel_idle", "loop_as_run", "loop_resident"]
        orders = [tuple(host.protocol.rotate(conditions, r)) for r in range(3)]
        assert len(set(orders)) == 3
        assert all(sorted(order) == sorted(conditions) for order in orders)


def _xclbin_design(tmp_path, *, size=64, archived=64, uuid=None, rebuilt=False):
    path = tmp_path / "odin_fpga_hw.xclbin"
    path.write_bytes(b"\0" * size)
    identity = {"archived": {"size_bytes": archived, "written": "2026-08-31 test"},
                "uuid": uuid, "rebuilt": rebuilt}
    return {"xclbin": str(path), "xclbin_identity": identity}


class TestTheXclbinIdentity:
    def test_the_archived_size_and_uuid_pass(self, host, tmp_path):
        design = _xclbin_design(tmp_path, uuid="6A1B-77")
        result = host.protocol.require_xclbin_identity(design, metadata_uuid="6a1b77",
                                                       name="stock")
        assert result["status"] == "archived" and result["uuid_checked"] is True

    def test_before_the_load_only_the_file_is_checked(self, host, tmp_path):
        design = _xclbin_design(tmp_path, uuid="6a1b77")
        result = host.protocol.require_xclbin_identity(design, name="stock")
        assert result["status"] == "archived" and result["uuid_checked"] is False

    def test_an_undeclared_uuid_is_recorded_as_size_only(self, host, tmp_path):
        design = _xclbin_design(tmp_path)
        result = host.protocol.require_xclbin_identity(design, metadata_uuid="6a1b77",
                                                       name="stock")
        assert result["status"] == "archived" and result["uuid_checked"] is False
        assert "size only" in result["note"]

    def test_another_size_refuses_and_names_the_way_out(self, host, tmp_path):
        design = _xclbin_design(tmp_path, size=65)
        with pytest.raises(host.protocol.OdinPowerRefusal, match="rebuilt") as caught:
            host.protocol.require_xclbin_identity(design, name="stock")
        assert "65 bytes" in str(caught.value) and "64 bytes" in str(caught.value)

    def test_another_uuid_refuses(self, host, tmp_path):
        design = _xclbin_design(tmp_path, uuid="6a1b77")
        with pytest.raises(host.protocol.OdinPowerRefusal, match="UUID"):
            host.protocol.require_xclbin_identity(design, metadata_uuid="ffff00", name="stock")

    def test_a_missing_xclbin_refuses_by_name(self, host, tmp_path):
        design = _xclbin_design(tmp_path)
        os.remove(design["xclbin"])
        with pytest.raises(host.protocol.OdinPowerRefusal, match="odin_fpga_hw.xclbin"):
            host.protocol.require_xclbin_identity(design, name="stock")

    def test_a_declared_rebuild_is_labelled_rebuilt_not_refused(self, host, tmp_path):
        design = _xclbin_design(tmp_path, size=65, uuid="0ddba11", rebuilt=True)
        result = host.protocol.require_xclbin_identity(design, metadata_uuid="0ddba11",
                                                       name="stock")
        assert result["status"] == "rebuilt" and result["uuid_checked"] is True
        assert "report_power" in result["note"]
        # A rebuild is still pinned: another file under the rebuilt label refuses.
        with pytest.raises(host.protocol.OdinPowerRefusal, match="UUID"):
            host.protocol.require_xclbin_identity(design, metadata_uuid="feed", name="stock")

    def test_preflight_checks_files_and_pins_without_a_card(self, host, tmp_path):
        plan = json.loads(SHIPPED_PLAN.read_text())
        for name in ("stock", "wide"):
            design = plan["designs"][name]
            xclbin = tmp_path / f"{name}.xclbin"
            xclbin.write_bytes(b"\0" * 16)
            design["xclbin"] = str(xclbin)
            design["xclbin_identity"]["archived"]["size_bytes"] = 16
            design["bundle"] = str(host.root / "deployment" / BUNDLE_NAME)
            design.pop("bundle_self_hash")
        plan["designs"]["wide"]["xclbin_identity"]["archived"]["size_bytes"] = 17
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan))
        report = host.protocol.preflight(host.protocol.load_plan(str(path)))
        assert report["designs"]["stock"]["ok"] is True
        assert report["designs"]["wide"]["ok"] is False
        assert "17 bytes" in " ".join(report["designs"]["wide"]["problems"])
        assert report["ok"] is False
        assert "xclbinutil --info --input" in report["designs"]["stock"]["uuid_command"]


class TestTheSessionClosesInsideItsBudget:
    def test_a_child_timeout_is_capped_at_the_budget_left_minus_teardown(
            self, host, tmp_path, monkeypatch):
        session = host.protocol.Session(str(SHIPPED_PLAN), str(tmp_path / "results"))
        monkeypatch.setattr(session, "elapsed", lambda: 3000.0)
        # 2 x 1,346.8 + 300 s is far beyond 3,300 - 3,000 - 60 = 240 s.
        assert session.child_timeout(2 * 1346.8 + 300) == pytest.approx(240.0)
        monkeypatch.setattr(session, "elapsed", lambda: 100.0)
        assert session.child_timeout(500.0) == pytest.approx(500.0)
        monkeypatch.setattr(session, "elapsed", lambda: 3290.0)
        assert session.child_timeout(500.0) == pytest.approx(1.0)  # never zero or negative

    def test_a_design_step_gets_the_capped_timeout(self, host, tmp_path, monkeypatch):
        session = host.protocol.Session(str(SHIPPED_PLAN), str(tmp_path / "results"))
        seen = []

        def child(argv, *, log, timeout):
            seen.append(timeout)
            return -9, f"timed out after {timeout:.0f} s and was killed"
        monkeypatch.setattr(session, "child", child)
        monkeypatch.setattr(session, "elapsed", lambda: 1500.0)
        row = session.run_step(4, session.plan["steps"][4])
        assert seen == [pytest.approx(3300 - 1500 - 60)]
        assert row["status"] == "failed" and row["timeout_s"] == pytest.approx(1740.0)

    def test_the_sampler_stop_fits_inside_the_teardown(self, host, tmp_path):
        session = host.protocol.Session(str(SHIPPED_PLAN), str(tmp_path / "results"))
        graceful, terminate = session.stop_waits()
        assert graceful + terminate <= 0.75 * 60  # the plan books 60 s of teardown


def _synthetic(host, tmp_path, name, **kwargs):
    samples, markers = tmp_path / f"{name}_s.jsonl", tmp_path / f"{name}_m.jsonl"
    host.protocol._synthetic(str(samples), str(markers), **kwargs)
    return str(samples), str(markers)


PLAN = {"steps": [{"kind": "protocol", "design": "stock", "rounds": 5, "block_s": 50,
                   "when": "go"}],
        "thresholds": {"pilot": {"expected_differential_w": {"stock": 0.3},
                                 "samples_per_condition_min": 100}}}


#: The declared quantities, word for word (run_plan.md, Idle-Subtracted Protocol).
WORKLOAD = ("workload increment at the card boundary (kernel plus its HBM, interconnect "
            "and PCIe activity), an upper bound on the kernel's dynamic power, at kernel "
            "duty D")
USER_REGION = ("user-region increment over the shell (xclbin loaded and idle, minus shell "
               "after reset, valid only when the reset read back a zero UUID)")
VIVADO = {"odin_0_dynamic_w": 0.068, "hmss_0_dynamic_w": 2.223,
          "class": "ESTIMATE: Vivado vectorless, confidence Low",
          "describes": "the archived 2026-08-31 build"}


class TestTheEstimator:
    @pytest.mark.parametrize("dynamic, sigma, seed, expected", [
        (0.30, 0.02, 11, "NUMBER"), (0.012, 0.05, 12, "BOUND"),
        (0.0, 0.05, 13, "NOT_SEPARATED")])
    def test_a_known_differential_earns_its_declared_outcome(
            self, host, tmp_path, dynamic, sigma, seed, expected):
        samples, markers = _synthetic(host, tmp_path, expected, dynamic=dynamic,
                                      static=0.4, sigma=sigma, seed=seed)
        result = host.estimator.estimate(samples, markers, PLAN)
        row = result["designs"]["stock"]["arms"]["as_run"]["workload_increment_w"]
        assert row["outcome"] == expected
        assert row["lo"] <= dynamic <= row["hi"]
        assert row["outcome"] in host.estimator.OUTCOMES
        # The declared interval is never narrower than either of its two parts.
        assert row["lo"] <= min(row["bootstrap"]["lo"], row["t_interval"]["lo"]) + 1e-12
        assert row["hi"] >= max(row["bootstrap"]["hi"], row["t_interval"]["hi"]) - 1e-12

    def test_energy_per_inference_is_power_times_wall_over_inferences(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "energy", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=21)
        result = host.estimator.estimate(samples, markers, PLAN)
        arm = result["designs"]["stock"]["arms"]["as_run"]
        power, energy = arm["workload_increment_w"], arm["workload_energy_per_inference_mJ"]
        # Synthetic windows are 201 readings x 0.25 s long and hold 64 inferences.
        assert energy["point"] == pytest.approx(power["point"] * 1e3 * 50.25 / 64, rel=1e-6)
        assert arm["card_energy_per_inference_mJ"] > energy["point"]

    def test_every_quantity_says_what_the_card_boundary_holds(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "labels", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=26)
        result = host.estimator.estimate(samples, markers, PLAN)
        design = result["designs"]["stock"]
        power = design["arms"]["as_run"]["workload_increment_w"]
        assert power["quantity"] == WORKLOAD.replace("duty D", "duty D = 0.900")
        assert power["statement"].startswith(power["quantity"] + ": ")
        assert power["class"] == "MEASURED"
        energy = design["arms"]["as_run"]["workload_energy_per_inference_mJ"]
        assert "upper bound on the kernel's dynamic energy per inference" in energy["quantity"]
        region = design["user_region_increment_w"]
        assert region["quantity"] == USER_REGION and region["interleaved"] is False
        assert host.estimator.QUANTITIES["workload_increment_w"] == WORKLOAD
        assert host.estimator.QUANTITIES["user_region_increment_w"] == USER_REGION
        # The retired names appear nowhere: neither as keys nor in the text.
        keys = set(_keys(result))
        assert not keys & {"dynamic_w", "static_w", "energy_per_inference_mJ"}
        text = json.dumps(result) + host.estimator.format_estimate(result)
        for retired in ("kernel static", "kernel dynamic", " dynamic_w", " static_w"):
            assert retired not in text, retired

    @pytest.mark.parametrize("verified, needle", [
        (False, "left an xclbin loaded"), (None, "zero UUID")], ids=["non-zero", "unread"])
    def test_the_user_region_increment_needs_a_zero_uuid_reset(
            self, host, tmp_path, verified, needle):
        samples, markers = _synthetic(host, tmp_path, "unverified", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=27, shell_verified=verified)
        result = host.estimator.estimate(samples, markers, PLAN)
        assert needle in result["designs"]["stock"]["user_region_increment_w"]["refused"]
        assert "stock/user_region_increment_w" not in result["family"]

    def test_the_transfer_control_splits_the_workload_increment(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "control", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=25, transfer=0.05)
        result = host.estimator.estimate(samples, markers, PLAN)
        design = result["designs"]["stock"]
        control = design["controls"]["as_run"]["transfer_increment_w"]
        net = design["arms"]["as_run"]["net_of_transfers_w"]
        assert control["lo"] <= 0.05 <= control["hi"] and net["lo"] <= 0.25 <= net["hi"]
        assert "without ap_start" in control["quantity"]
        assert net["quantity"].startswith("workload increment net of the transfer control")
        assert "an upper bound on the kernel's dynamic power" in net["quantity"]
        assert design["arms"]["as_run"]["workload_increment_w"]["lo"] <= 0.30
        net_energy = design["arms"]["as_run"]["net_of_transfers_energy_per_inference_mJ"]
        assert net_energy["point"] == pytest.approx(net["point"] * 1e3 * 50.25 / 64, rel=1e-6)
        assert {"stock/as_run/workload_increment_w", "stock/as_run/net_of_transfers_w",
                "stock/as_run/transfer_increment_w",
                "stock/user_region_increment_w"} <= set(result["family"])

    def test_the_vivado_comparison_is_odin_0_against_an_upper_bound_never_a_ratio(
            self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "vivado", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=28, transfer=0.05)
        plan = dict(PLAN, designs={"stock": {"vivado": VIVADO}})
        result = host.estimator.estimate(samples, markers, plan)
        comparison = result["designs"]["stock"]["vivado_comparison"]
        row = comparison["arms"]["as_run"]
        arm = result["designs"]["stock"]["arms"]["as_run"]
        assert row["odin_0_at_duty_w"] == pytest.approx(0.068 * 0.9)
        assert row["odin_0_dynamic_w"] == 0.068 and row["hmss_0_dynamic_w"] == 2.223
        # The tightest MEASURED upper bound the session holds: net of the transfers.
        assert row["upper_bound_w"] == arm["net_of_transfers_w"]["hi"]
        assert row["upper_bound_quantity"] == "net_of_transfers_w"
        assert row["odin_0_within_bound"] is True
        assert "odin_0" in row["reading"] and "hmss_0" in row["reading"]
        assert not [key for key in _keys(result) if "ratio" in key.lower()]

    def test_a_rebuilt_xclbin_refuses_the_archived_vivado_comparison(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "rebuilt", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=29)
        with open(markers, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "xclbin_loaded", "design": "stock",
                                     "xclbin_status": "rebuilt", "t_mono": 1.0}) + "\n")
        plan = dict(PLAN, designs={"stock": {"vivado": VIVADO}})
        result = host.estimator.estimate(samples, markers, plan)
        design = result["designs"]["stock"]
        assert design["xclbin_status"] == "rebuilt"
        assert "re-extract report_power" in design["vivado_comparison"]["refused"]
        assert "rebuilt" in design["arms"]["as_run"]["workload_increment_w"]["statement"]

    def test_a_window_with_a_red_certificate_is_not_used(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "red", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=22)
        lines = [json.loads(line) for line in Path(markers).read_text().splitlines()]
        for record in lines:
            if record.get("event") == "window_end" and "inferences" in record:
                record["passed"] = False
        Path(markers).write_text("".join(json.dumps(r) + "\n" for r in lines))
        result = host.estimator.estimate(samples, markers, PLAN)
        assert "refused" in result["designs"]["stock"]["arms"]["as_run"]
        assert len(result["excluded_windows"]) == 5

    def test_a_drift_reference_outside_its_band_downgrades_a_number(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "drift", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=23)
        rows = [json.loads(line) for line in Path(samples).read_text().splitlines()]
        windows = host.estimator.windows_of(host.estimator.load_jsonl(markers))
        drift = next(w for w in windows if w["kind"] == "drift")
        for row in rows:
            if not row.get("header") and drift["t_start"] <= row["t_mono"] <= drift["t_end"]:
                row["power_w"] += 0.5
        Path(samples).write_text("".join(json.dumps(r) + "\n" for r in rows))
        result = host.estimator.estimate(samples, markers, PLAN)
        assert result["drift"]["inside_band"] is False
        row = result["designs"]["stock"]["arms"]["as_run"]["workload_increment_w"]
        assert row["outcome"] == "BOUND" and "drift" in row["statement"]

    def test_the_session_can_overrule_the_reset_marker(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "shell", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=24)
        result = host.estimator.estimate(samples, markers, PLAN, shell_verified=False)
        assert "refused" in result["designs"]["stock"]["user_region_increment_w"]

    def test_holm_steps_down_and_stops(self, host):
        # 0.001 <= 0.05/3 rejects; 0.03 > 0.05/2 stops, so 0.04 is kept too.
        rejected = host.estimator.holm({"a": 0.001, "b": 0.03, "c": 0.04}, 0.05)
        assert rejected == {"a": True, "b": False, "c": False}
        assert host.estimator.holm({"a": 0.001, "b": 0.02, "c": 0.04}, 0.05) == {
            "a": True, "b": True, "c": True}

    @pytest.mark.parametrize("df, critical", [(1, 12.706), (2, 4.303), (4, 2.776),
                                              (10, 2.228), (1000, 1.962)])
    def test_student_t_critical_values(self, host, df, critical):
        assert host.estimator.t_critical(df) == pytest.approx(critical, abs=2e-3)


class TestThePilotGate:
    def test_a_quiet_sensor_goes(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "quiet", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=31)
        verdict = host.estimator.pilot(samples, markers, PLAN)
        assert verdict["verdict"] == "GO" and verdict["predictions"][0]["ok"]

    def test_a_noisy_sensor_is_reduced_with_the_numbers_that_decided(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "noisy", dynamic=0.3, static=0.4,
                                      sigma=1.5, seed=32)
        verdict = host.estimator.pilot(samples, markers, PLAN)
        assert verdict["verdict"] == "REDUCED"
        assert verdict["predictions"][0]["predicted_half_width_w"] > 0.03

    def test_a_frozen_reading_is_no_go(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "frozen", dynamic=0.3, static=0.4,
                                      sigma=0.0, seed=33)
        assert host.estimator.pilot(samples, markers, PLAN)["verdict"] == "NO_GO"

    def test_missing_readings_are_no_go(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "gaps", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=34)
        rows = [json.loads(line) for line in Path(samples).read_text().splitlines()]
        for index, row in enumerate(rows):
            if not row.get("header") and index % 10 == 0:
                row.update(ok=False, refusal="OdinTelemetryMissing")
                row.pop("power_w")
        Path(samples).write_text("".join(json.dumps(r) + "\n" for r in rows))
        verdict = host.estimator.pilot(samples, markers, PLAN)
        assert verdict["verdict"] == "NO_GO"
        assert any("availability" in reason for reason in verdict["reasons"])

    def test_the_shipped_thresholds_are_the_declared_ones(self, host):
        plan = json.loads(SHIPPED_PLAN.read_text())
        gate = host.estimator.thresholds_of(plan)
        assert gate["pilot"]["relative_half_width_max"] == 0.10
        assert gate["pilot"]["samples_per_condition_min"] == 300
        assert gate["pilot"]["availability_min"] == 0.95
        assert gate["pilot"]["expected_differential_w"] == {"stock": 0.0626, "wide": 0.0652}
        assert gate["estimate"]["alpha"] == 0.05
        assert gate["estimate"]["warmup_samples"] == 5
        assert gate["estimate"]["warmup_refresh_multiple"] == 2.0

    def test_the_planned_counts_drop_the_warm_up_of_every_window(self, host):
        step = {"rounds": 5, "block_s": 60}
        # At a 1 s read interval the old count was exactly 300 against 300.
        assert host.estimator.planned_counts(step, 1.0, 1.0, warmup=5) == (275.0, 275.0)
        polled, independent = host.estimator.planned_counts(step, 0.25, 1.0, warmup=8)
        assert polled == pytest.approx(5 * (240 - 8))
        assert independent == pytest.approx(5 * (60 - 8 * 0.25) / 1.0)

    @pytest.mark.parametrize("interval, refresh, readings", [
        (0.25, 1.0, 8), (0.25, 0.5, 5), (1.0, 1.0, 5), (0.25, None, 5), (0.25, 2.0, 16),
        (0.5, 1.3, 6)])
    def test_the_warm_up_is_the_larger_of_five_readings_and_two_refreshes(
            self, host, interval, refresh, readings):
        gate = host.estimator.thresholds_of({})["estimate"]
        assert host.estimator.warmup_readings(gate, interval_s=interval,
                                              refresh_s=refresh) == readings

    def test_one_second_polling_misses_the_sample_floor(self, host):
        summary = {"availability": 1.0, "changes": 200, "refresh_s": 1.0,
                   "sigma_w": 0.001, "quantum_w": 0.001}
        plan = {"steps": [{"kind": "protocol", "design": "stock", "rounds": 5, "block_s": 60,
                           "when": "go"}],
                "thresholds": {"pilot": {"expected_differential_w": {"stock": 0.0626}}}}
        verdict = host.estimator.pilot_verdict(summary, plan, interval_s=1.0)
        row = verdict["predictions"][0]
        assert row["polled_per_condition"] == pytest.approx(275.0)
        assert row["samples_ok"] is False and verdict["verdict"] == "REDUCED"
        assert verdict["warmup"]["readings"] == 5

    def test_the_pilot_and_the_estimate_record_the_warm_up_they_used(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "warm", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=35)
        pilot = host.estimator.pilot(samples, markers, PLAN)
        warmup = pilot["warmup"]
        assert warmup["readings"] == max(5, math.ceil(2 * warmup["refresh_s"] / 0.25 - 1e-9))
        assert warmup["seconds"] == pytest.approx(warmup["readings"] * warmup["interval_s"])
        assert "larger of 5 readings and 2 x the measured refresh" in warmup["rule"]
        result = host.estimator.estimate(samples, markers, PLAN, pilot_result=pilot)
        assert result["warmup"]["readings"] == warmup["readings"]

    def test_a_slow_refresh_lengthens_the_warm_up_the_estimate_drops(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "slow", dynamic=0.3, static=0.4,
                                      sigma=0.02, seed=36)
        pilot = {"summary": {"refresh_s": 2.0}}
        result = host.estimator.estimate(samples, markers, PLAN, pilot_result=pilot)
        assert result["warmup"]["readings"] == 16
        counts = result["designs"]["stock"]["arms"]["as_run"]["readings"]
        assert counts["loop"] == 5 * (200 - 16)  # 200 readings per synthetic window


@pytest.fixture(scope="module")
def overlay(tmp_path_factory):
    out = tmp_path_factory.mktemp("overlay")
    completed = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "hacc" / "make_power_overlay.py"),
         "--stage-only", "--out", str(out)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return out / "odin_power_overlay"


class TestTheOverlay:
    def test_it_carries_every_file_by_hash(self, overlay):
        manifest = json.loads((overlay / "MANIFEST.json").read_text())
        assert "host/odin_power_protocol.py" in manifest["files"]
        assert "host/odin_deployment_bundle.py" in manifest["files"]
        assert "scripts/hacc/odin_board.sbatch" in manifest["files"]
        assert "power/plan_hacc_u55c_tcad.json" in manifest["files"]
        import hashlib
        for relative, digest in manifest["files"].items():
            assert hashlib.sha256((overlay / relative).read_bytes()).hexdigest() == digest

    @pytest.mark.timeout(240)
    def test_the_selftest_is_green_from_a_fresh_overlay(self, overlay, tmp_path):
        completed = subprocess.run(
            [sys.executable, "-I", str(overlay / "host" / "odin_power_protocol.py"),
             "selftest", "--results", str(tmp_path / "selftest")],
            capture_output=True, text=True, cwd=str(tmp_path))
        assert completed.returncode == 0, completed.stdout[-6000:] + completed.stderr[-2000:]
        result = json.loads((tmp_path / "selftest" / "selftest_power.json").read_text())
        assert result["passed"] and len(result["rows"]) >= 48
        names = {row["check"] for row in result["rows"]}
        for needed in (
                "reset: the fake card reads back a zero UUID (shell verified)",
                "identity: the declared xclbin UUID matched the file (archived)",
                "control: the transfer windows never started the kernel",
                "estimate: the transfer control recovers the plant's 0.05 W",
                "estimate: net of the transfers recovers the plant's 0.25 W",
                "estimate: the user-region increment recovers the plant's 0.40 W",
                "estimate: odin_0 and hmss_0 beside the upper bound, no ratio",
                "pilot: the warm-up is recorded with its rule"):
            assert needed in names, needed


class TestTheBoardJobInPowerMode:
    @pytest.mark.timeout(240)
    def test_odin_mode_power_runs_a_whole_fake_session(self, overlay, tmp_path):
        stubs = tmp_path / "xrt_bin"
        stubs.mkdir()
        (stubs / "xbutil").write_text("#!/bin/sh\necho 'stub xbutil examine (no card)'\n")
        (stubs / "xbutil").chmod(0o755)
        setup = tmp_path / "setup.sh"
        setup.write_text("")
        sys.path.insert(0, str(overlay / "host"))
        saved = {name: sys.modules.pop(name) for name in HOST_MODULES if name in sys.modules}
        try:
            protocol = importlib.import_module("odin_power_protocol")
            plan = protocol.selftest_plan(str(tmp_path), absent=False)
        finally:
            sys.path.remove(str(overlay / "host"))
            for name in HOST_MODULES:
                sys.modules.pop(name, None)
            sys.modules.update(saved)
        results = tmp_path / "results" / "board_power"
        env = dict(os.environ, ODIN_PKG=str(overlay), ODIN_MODE="power",
                   ODIN_POWER_PLAN=plan, ODIN_RESULTS=str(results),
                   ODIN_LOG=str(tmp_path / "results" / "phase_power.log"),
                   XRT_SETUP=str(setup), XRT_BIN=str(stubs))
        env.pop("ODIN_XCLBIN", None)
        completed = subprocess.run(
            ["bash", str(overlay / "scripts" / "hacc" / "odin_board.sbatch")],
            capture_output=True, text=True, env=env, cwd=str(tmp_path))
        assert completed.returncode == 0, completed.stdout[-6000:] + completed.stderr[-2000:]
        record = json.loads((results / "session.json").read_text())
        assert record["pilot"]["verdict"] == "GO" and record["failures"] == 0
        assert record["shell_verified"] is True
        estimate = json.loads((results / "estimate.json").read_text())
        dyn = estimate["designs"]["stock"]["arms"]["resident"]["workload_increment_w"]
        assert dyn["lo"] <= 0.30 <= dyn["hi"]
        control = estimate["designs"]["stock"]["controls"]["as_run"]["transfer_increment_w"]
        assert control["lo"] <= 0.05 <= control["hi"]
        assert estimate["designs"]["stock"]["xclbin_status"] == "archived"
        assert (results / "xbutil_examine.log").read_text().startswith("stub xbutil")
        log = (tmp_path / "results" / "phase_power.log").read_text()
        assert "PILOT VERDICT GO" in log and "mode    power" in log
