"""[ODIN route D] the host drives N compute units at once and keeps them apart.

STRUCTURE, NOT SILICON: the shipped host files, staged as a package's host/,
against the fake pyxrt. The claims:

  * the driver names unit k exactly as the link names it (``odin_<k>``,
    ``scripts/hacc/gen_link_cfg.py``), opens it by its CU-qualified kernel name,
    stamps which unit a session holds, and can load an xclbin without taking any
    unit (so N units can then be opened by N processes);
  * the deployment executor with ``--compute-units N`` runs the same certified
    campaign on every unit at once, one process per unit, and keeps one
    certificate set and one readout set per unit, each stamped with its unit;
    N = 1 is the archived behaviour (no per-unit keys);
  * the power protocol's plan declares N per design as data; a protocol step on
    an N-unit design drives all N units in every loop window and replays all N
    units' buffer traffic in every transfer window without starting a kernel;
    the window records carry the per-unit rows and their sums;
  * the estimator names N in every quantity of an N-unit design, adds the
    per-unit figure (the increment divided by N, named so), and, when the plan
    declares the sensor's step, refuses to offer an interval end below one step
    as a bound and to call a point under the declared number of steps a NUMBER.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
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
    return root


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    root = _stage(tmp_path_factory.mktemp("odin_route_d_pkg"))
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


def _options(host, **overrides):
    values = dict(xclbin="/unit/none.xclbin", device_index=0,
                  capture_events=host.driver.DEFAULT_CAPTURE_EVENTS, declare_cores=0,
                  capture_ram_events=0, chip="odin_stock_256x256", run_timeout_ms=0,
                  fake_pyxrt=host.fake)
    values.update(overrides)
    return SimpleNamespace(**values)


def _kernel_calls(fake):
    return [entry for entry in fake.call_log() if entry[0] == "kernel"]


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


class TestTheDriverHoldsOneNamedUnit:

    def test_unit_names_are_the_links_own(self, host):
        spec = importlib.util.spec_from_file_location(
            "gen_link_cfg_for_names", REPO / "scripts" / "hacc" / "gen_link_cfg.py")
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        base = (REPO / "scripts" / "hacc" / "odin_u55c.cfg").read_text()
        assert host.driver.compute_unit_names(10) == gen.compute_unit_names(base, 10)
        assert host.driver.compute_unit_name(0) == "odin_0"

    def test_a_named_unit_is_opened_by_its_cu_qualified_kernel_name(self, host):
        host.driver._FAKE_MODULES.clear()
        session = host.driver.open_session(_options(host, cu="odin_3"))
        fake = host.driver.load_pyxrt(host.fake)
        try:
            calls = _kernel_calls(fake)
            assert calls == [("kernel", "odin_fpga_kernel_top:{odin_3}", "exclusive")]
            assert session.identity["compute_unit"] == "odin_3"
            stamp = session.device_stamp()
            assert stamp["compute_unit"]["name"] == "odin_3"
            assert set(stamp["compute_unit"]["group_ids"]) == {"0", "1", "2"}
        finally:
            session.close()

    def test_the_default_session_is_the_archived_one(self, host):
        host.driver._FAKE_MODULES.clear()
        session = host.driver.open_session(_options(host))
        fake = host.driver.load_pyxrt(host.fake)
        try:
            assert _kernel_calls(fake) == [("kernel", "odin_fpga_kernel_top", "exclusive")]
            assert "compute_unit" not in session.identity
            assert "compute_unit" not in session.device_stamp()
        finally:
            session.close()

    def test_a_load_only_session_takes_no_unit_and_refuses_buffers(self, host):
        host.driver._FAKE_MODULES.clear()
        session = host.driver.open_session(_options(host), open_kernel=False)
        fake = host.driver.load_pyxrt(host.fake)
        try:
            assert ("load_xclbin", "/unit/none.xclbin") in fake.call_log()
            assert _kernel_calls(fake) == []
            assert session.identity.get("xclbin_uuid")
            with pytest.raises(host.driver.OdinDriverError, match="no compute unit"):
                session.allocate(64, host.driver.ARG_PROGRAM)
        finally:
            session.close()

    def test_a_unit_the_xclbin_does_not_hold_refuses_by_name(self, host):
        host.driver._FAKE_MODULES.clear()
        fake = host.driver.load_pyxrt(host.fake)
        fake.arm(compute_units=2)
        try:
            with pytest.raises(host.driver.OdinDriverError, match="odin_2"):
                host.driver.open_session(_options(host, cu="odin_2"))
        finally:
            fake.arm(compute_units=None)


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------


def _run_executor(root: Path, *args: str, env=None):
    return subprocess.run(
        [sys.executable, "host/odin_deployment_executor.py",
         "--xclbin", "/selftest/no-such.xclbin",
         "--fake-pyxrt", "host/fake_pyxrt_for_selftest.py",
         "--bundle", f"deployment/{BUNDLE_NAME}", "--replay", f"deployment/{REPLAY_NAME}",
         *args], cwd=str(root), capture_output=True, text=True,
        env=dict(os.environ, **(env or {})))


class TestTheExecutorRunsEveryUnitAtOnce:

    @pytest.mark.timeout(180)
    def test_three_units_each_certify_their_own_campaign(self, host):
        completed = _run_executor(host.root, "--results", "results/cu3",
                                  "--compute-units", "3", "--arm", "resident")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        results = host.root / "results" / "cu3"
        report = json.loads((results / "deployment_report.json").read_text())
        assert report["compute_units"] == 3 and report["passed"] is True
        names = [row["compute_unit"] for row in report["per_cu"]]
        assert names == ["odin_0", "odin_1", "odin_2"]
        single = json.loads((results / "cu0" / "deployment_report.json").read_text())
        per_unit_certificates = len(single["certificates"])
        assert per_unit_certificates > 0
        assert len(report["certificates"]) == 3 * per_unit_certificates
        assert len(report["readout"]) == 3 * len(single["readout"])
        for name in names:
            mine = [row for row in report["certificates"] if row["compute_unit"] == name]
            assert len(mine) == per_unit_certificates and all(row["passed"] for row in mine)
            readouts = [row for row in report["readout"] if row["compute_unit"] == name]
            assert readouts and all(row["matches_frozen"] for row in readouts)
        for index, row in enumerate(report["per_cu"]):
            unit = json.loads((results / f"cu{index}" / "deployment_report.json").read_text())
            assert unit["compute_unit"] == {"name": f"odin_{index}", "index": index, "of": 3}
            assert unit["device"]["identity"]["compute_unit"] == f"odin_{index}"
            assert row["passed"] is True and row["identity"]["compute_unit"] == f"odin_{index}"
        # All at once: every unit's campaign was running while every other one was.
        windows = [row["campaign_window"] for row in report["per_cu"]]
        assert max(w["t_start"] for w in windows) < min(w["t_end"] for w in windows)
        # The xclbin was loaded once, by the parent, before any unit opened.
        assert report["device"]["identity"]["xclbin_uuid"]

    def test_one_unit_is_the_archived_report(self, host):
        completed = _run_executor(host.root, "--results", "results/cu1")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        report = json.loads((host.root / "results" / "cu1" / "deployment_report.json").read_text())
        for key in ("compute_units", "per_cu", "compute_unit", "campaign_window"):
            assert key not in report

    def test_a_count_below_one_refuses(self, host):
        completed = _run_executor(host.root, "--results", "results/cu0",
                                  "--compute-units", "0")
        assert completed.returncode == 2 and "compute units" in completed.stderr


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


class TestThePlanDeclaresUnitsAsData:

    @pytest.mark.parametrize("value", [0, -1, 1.5, "ten"])
    def test_a_count_that_is_not_a_positive_integer_refuses(self, host, tmp_path, value):
        plan = json.loads(SHIPPED_PLAN.read_text())
        plan["designs"]["stock"]["compute_units"] = value
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan))
        with pytest.raises(host.protocol.OdinPowerRefusal, match="compute_units"):
            host.protocol.load_plan(str(path))

    def test_no_count_is_one_unit(self, host):
        plan = host.protocol.load_plan(str(SHIPPED_PLAN))
        assert host.protocol.compute_units_of(plan["designs"]["stock"]) == 1


class TestTheAggregateOfUnits:

    def test_loop_windows_sum_their_units_and_keep_each(self, host):
        rows = [{"campaigns": 2, "inferences": 128, "passes": 256, "run_s": 9.0,
                 "device_cycles": 10, "programming_runs": 0, "programming_run_s": 0.0,
                 "programming_device_cycles": 0, "certificates": 100,
                 "certificates_failed": 0, "readouts_drifted": 0, "arm": "as_run",
                 "t_start": 10.0 + k, "t_end": 20.0 + k, "wall_s": 10.0,
                 "kernel_duty": 0.9, "passed": True} for k in range(3)]
        total = host.protocol.aggregate_units(rows, ["odin_0", "odin_1", "odin_2"])
        assert total["compute_units"] == 3
        assert total["inferences"] == 384 and total["passes"] == 768
        assert total["kernel_duty"] == pytest.approx(0.9)
        assert total["kernel_duty_sum"] == pytest.approx(2.7)
        assert total["t_start"] == 10.0 and total["t_end"] == 22.0
        assert total["passed"] is True
        assert [row["compute_unit"] for row in total["per_cu"]] == ["odin_0", "odin_1", "odin_2"]
        rows[1]["passed"] = False
        assert host.protocol.aggregate_units(rows, ["a", "b", "c"])["passed"] is False


def _unit_plan(host, work: Path, units: int) -> str:
    path = host.protocol.selftest_plan(str(work), absent=False)
    plan = json.loads(Path(path).read_text())
    plan["name"] = f"selftest_cu{units}"
    plan["designs"]["stock"]["compute_units"] = units
    out = work / f"plan_cu{units}.json"
    out.write_text(json.dumps(plan, indent=2, sort_keys=True))
    return str(out)


@pytest.fixture(scope="module")
def session(host, tmp_path_factory):
    """One whole fake session on a three-unit design, shared by the class below."""
    work = tmp_path_factory.mktemp("cu3_session")
    plan = _unit_plan(host, work, 3)
    results = work / "results"
    code = host.protocol.Session(plan, str(results)).run()
    record = json.loads((results / "session.json").read_text())
    return SimpleNamespace(code=code, record=record, results=results)


@pytest.mark.timeout(240)
class TestAWholeFakeSessionOnThreeUnits:

    def test_it_completes_with_no_failed_step(self, session):
        assert session.code == 0, session.record["steps"]
        assert session.record["pilot"]["verdict"] == "GO"

    def test_every_loop_window_drove_all_three_units_at_once(self, session):
        proto = next(s for s in session.record["steps"]
                     if s["kind"] == "protocol" and s["status"] == "done")
        step = json.loads((Path(proto["dir"]) / "step.json").read_text())
        loops = [w for w in step["windows"] if w["condition"].startswith("loop_")]
        assert loops
        for window in loops:
            assert window["compute_units"] == 3 and window["passed"] is True
            units = window["per_cu"]
            assert [u["compute_unit"] for u in units] == ["odin_0", "odin_1", "odin_2"]
            assert window["inferences"] == sum(u["inferences"] for u in units)
            assert window["kernel_duty_sum"] == pytest.approx(
                sum(u["kernel_duty"] for u in units))
            assert max(u["loop_t_start"] for u in units) < min(u["loop_t_end"] for u in units)

    def test_every_transfer_window_moved_all_units_buffers_without_a_start(self, session):
        proto = next(s for s in session.record["steps"]
                     if s["kind"] == "protocol" and s["status"] == "done")
        step = json.loads((Path(proto["dir"]) / "step.json").read_text())
        controls = [w for w in step["windows"] if w["condition"].startswith("transfer_")]
        assert controls
        for window in controls:
            units = window["per_cu"]
            assert len(units) == 3 and window["ap_starts"] == 0
            # Counted by each unit's own driver session, not written as a constant.
            assert all(u["ap_starts"] == 0 for u in units)
            assert all(u["passes"] > 0 and u["bytes_to_device"] > 0 for u in units)
            assert window["bytes_to_device"] == sum(u["bytes_to_device"] for u in units)
        # The same counter counts every start of the loop windows: one per pass
        # plus one per programming run, per unit.
        loops = [w for w in step["windows"] if w["condition"].startswith("loop_")]
        assert loops
        for window in loops:
            for unit in window["per_cu"]:
                assert unit["ap_starts"] == unit["passes"] + unit["programming_runs"] > 0
            assert window["ap_starts"] == sum(u["ap_starts"] for u in window["per_cu"])

    def test_the_recertification_certified_every_unit(self, session):
        recert = next(s for s in session.record["steps"] if s["kind"] == "recertify")
        report = json.loads((Path(recert["dir"]) / "deployment_report.json").read_text())
        assert report["compute_units"] == 3 and report["passed"] is True
        assert all(row["passed"] for row in report["per_cu"])

    def test_the_estimate_names_the_units_and_the_per_unit_figure(self, session):
        result = json.loads((session.results / "estimate.json").read_text())
        stock = result["designs"]["stock"]
        assert stock["compute_units"] == 3
        row = stock["arms"]["as_run"]["workload_increment_w"]
        assert "3 compute units" in row["quantity"]
        assert row["lo"] <= 0.30 <= row["hi"]
        per_unit = row["per_compute_unit"]
        assert per_unit["point"] == pytest.approx(row["point"] / 3)
        assert per_unit["lo"] == pytest.approx(row["lo"] / 3)
        assert "per compute unit" in per_unit["quantity"]
        assert per_unit["outcome"] == row["outcome"]


# ---------------------------------------------------------------------------
# The estimator at the sensor's step
# ---------------------------------------------------------------------------


def _synthetic(host, tmp_path, name, **kwargs):
    samples, markers = tmp_path / f"{name}_s.jsonl", tmp_path / f"{name}_m.jsonl"
    host.protocol._synthetic(str(samples), str(markers), **kwargs)
    return str(samples), str(markers)


def _plan(step_w=None, min_steps=3.0, units=None):
    plan = {"steps": [{"kind": "protocol", "design": "stock", "rounds": 5, "block_s": 50,
                       "when": "go"}],
            "thresholds": {"pilot": {"expected_differential_w": {"stock": 0.3},
                                     "samples_per_condition_min": 100},
                           "estimate": {}}}
    if step_w is not None:
        plan["thresholds"]["estimate"]["resolution"] = {
            "step_w": step_w, "number_min_steps": min_steps,
            "source": "unit test: a declared sensor step"}
    if units is not None:
        plan["designs"] = {"stock": {"compute_units": units}}
    return plan


class TestTheEstimatorAtTheSensorsStep:

    def test_without_a_declared_step_nothing_changes(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "plain", dynamic=0.012, static=0.4,
                                      sigma=0.05, seed=12)
        result = host.estimator.estimate(samples, markers, _plan())
        row = result["designs"]["stock"]["arms"]["as_run"]["workload_increment_w"]
        assert row["outcome"] == "BOUND" and "sensor step" not in row["statement"]
        assert "per_compute_unit" not in row
        assert "compute_units" not in result["designs"]["stock"]

    def test_a_number_needs_the_declared_number_of_steps(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "number", dynamic=0.30, static=0.4,
                                      sigma=0.02, seed=11)
        resolved = host.estimator.estimate(samples, markers, _plan(step_w=0.0854))
        row = resolved["designs"]["stock"]["arms"]["as_run"]["workload_increment_w"]
        assert row["outcome"] == "NUMBER"
        coarse = host.estimator.estimate(samples, markers, _plan(step_w=0.1099, min_steps=5))
        row = coarse["designs"]["stock"]["arms"]["as_run"]["workload_increment_w"]
        assert row["outcome"] == "BOUND"
        assert "5 sensor steps" in row["statement"]

    def test_an_upper_end_below_one_step_is_never_offered_as_a_bound(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "below", dynamic=0.0, static=0.4,
                                      sigma=0.05, seed=13)
        result = host.estimator.estimate(samples, markers, _plan(step_w=0.0854))
        row = result["designs"]["stock"]["arms"]["as_run"]["workload_increment_w"]
        assert row["outcome"] == "NOT_SEPARATED"
        assert row["hi"] < 0.0854
        assert "at most" not in row["statement"]
        assert "below one sensor step" in row["statement"]
        energy = result["designs"]["stock"]["arms"]["as_run"]["workload_energy_per_inference_mJ"]
        assert "at most" not in energy["statement"]

    def test_the_pilot_gate_uses_the_declared_quantum_floor(self, host):
        summary = {"availability": 1.0, "changes": 200, "refresh_s": 1.0,
                   "sigma_w": 0.0619, "quantum_w": 0.003456}
        plan = {"steps": [{"kind": "protocol", "design": "stock", "rounds": 5,
                           "block_s": 110, "when": "go"}],
                "thresholds": {"pilot": {"expected_differential_w": {"stock": 0.0626},
                                         "quantum_floor_w": 0.1099}}}
        verdict = host.estimator.pilot_verdict(summary, plan, interval_s=0.25)
        row = verdict["predictions"][0]
        assert row["quantum_used_w"] == pytest.approx(0.1099)
        assert row["quantum_ok"] is False
        plan["thresholds"]["pilot"]["expected_differential_w"]["stock"] = 0.626
        row = host.estimator.pilot_verdict(summary, plan, interval_s=0.25)["predictions"][0]
        assert row["quantum_ok"] is True and row["half_width_ok"] is True


class TestARebuiltXclbinIsPinnedByItsOwnHash:
    """A rebuilt design is held to its own UUID and, when declared, its own sha256."""

    def test_a_rebuilt_designs_declared_sha256_is_checked(self, host, tmp_path):
        import hashlib
        path = tmp_path / "rebuilt.xclbin"
        path.write_bytes(b"x" * 64)
        design = {"xclbin": str(path), "xclbin_identity": {
            "rebuilt": True, "uuid": "11111111-2222-3333-4444-555555555555",
            "sha256": "0" * 64}}
        result = host.protocol.xclbin_identity(design)
        assert result["status"] == "mismatch"
        assert any("hashes to" in problem for problem in result["problems"])
        design["xclbin_identity"]["sha256"] = hashlib.sha256(b"x" * 64).hexdigest()
        result = host.protocol.xclbin_identity(design)
        assert result["status"] == "rebuilt" and result["checks"]["sha256"]


class TestTheUnitClauseSaysWhatRanAtOnce:

    def test_a_loop_names_the_campaign_and_a_control_the_traffic(self, host):
        quantity = host.estimator.quantity
        loop = quantity("workload_increment_w", duty=0.9, units=10)
        assert loop.endswith("; 10 compute units running the same campaign at once, "
                             "D the mean kernel duty per unit (each unit's run walls "
                             "over its own loop wall)")
        control = quantity("transfer_increment_w", arm="as_run", units=10)
        assert control.endswith("; 10 compute units' buffer traffic replayed at once")
        energy = quantity("transfer_energy_per_inference_mJ", units=10)
        assert energy.endswith("replayed at once, inferences counted over all units")
        assert quantity("workload_increment_w", duty=0.9) == quantity(
            "workload_increment_w", duty=0.9, units=1)


# ---------------------------------------------------------------------------
# Fix round of the route D review (2026-10-27)
# ---------------------------------------------------------------------------


STEP_W = 0.10988


def _drift_plan(*, step_w=STEP_W, interleaved=True, rule="idle_trend"):
    plan = _plan(step_w=step_w)
    if interleaved:
        plan["thresholds"]["estimate"]["drift"] = {
            "interleaved": rule, "source": "unit test: the declared interleaved drift rule"}
    return plan


class TestTheDriftReferenceMatchesTheInterleavedContrasts:
    """M1: an interleaved contrast is judged by its own design's idle trend.

    The session-end idle against the pilot (the absolute gate) stays the drift
    reference of the one contrast that is not interleaved, the user-region
    increment; its quantum term is floored at the declared sensor step.
    """

    def test_without_the_declaration_the_absolute_gate_judges_every_contrast(
            self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "abs", dynamic=0.6, static=0.4,
                                      sigma=0.02, seed=21, trend_w_per_s=0.0003)
        result = host.estimator.estimate(samples, markers, _plan(step_w=STEP_W))
        assert result["drift"]["inside_band"] is False
        row = result["designs"]["stock"]["arms"]["as_run"]["workload_increment_w"]
        assert row["outcome"] == "BOUND"
        assert "drift reference outside its band" in row["statement"]
        assert "drift_trend" not in row and "idle_trend" not in result["designs"]["stock"]

    def test_a_slow_trend_leaves_an_interleaved_number_and_bounds_the_region(
            self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "slow", dynamic=0.6, static=0.4,
                                      sigma=0.02, seed=21, trend_w_per_s=0.0003)
        result = host.estimator.estimate(samples, markers, _drift_plan())
        assert result["drift"]["inside_band"] is False
        assert result["drift"]["applies_to"] == ["user_region_increment_w"]
        stock = result["designs"]["stock"]
        trend = stock["idle_trend"]
        assert trend["idle_windows"] == 5
        assert trend["slope_w_per_s"] == pytest.approx(0.0003, rel=0.25)
        row = stock["arms"]["as_run"]["workload_increment_w"]
        drift = row["drift_trend"]
        # Two conditions rotated over five rounds: the loop sits on average one
        # fifth of a window after its idle window.
        assert drift["mean_offset_s"] > 0
        assert drift["shift"] == pytest.approx(trend["slope_w_per_s"] * drift["mean_offset_s"])
        assert drift["band"] == pytest.approx(
            max(0.0, 0.10 * abs(row["point"]) - row["half_width"]))
        assert drift["inside_band"] is True
        assert row["outcome"] == "NUMBER", row["statement"]
        energy = stock["arms"]["as_run"]["workload_energy_per_inference_mJ"]
        assert energy["drift_trend"]["inside_band"] is True and energy["outcome"] == "NUMBER"
        region = stock["user_region_increment_w"]
        assert "drift_trend" not in region
        assert region["outcome"] == "BOUND"
        assert "drift reference outside its band" in region["statement"]

    def test_a_trend_that_moves_a_contrast_past_its_budget_makes_it_a_bound(
            self, host, tmp_path):
        # The same order in every round: the loop always follows its idle window,
        # so the trend does not cancel and shifts the point by slope x offset.
        samples, markers = _synthetic(host, tmp_path, "fixed", dynamic=0.6, static=0.4,
                                      sigma=0.02, seed=22, trend_w_per_s=0.0015,
                                      rotate_rounds=False)
        result = host.estimator.estimate(samples, markers, _drift_plan())
        row = result["designs"]["stock"]["arms"]["as_run"]["workload_increment_w"]
        drift = row["drift_trend"]
        assert row["half_width"] <= 0.10 * abs(row["point"])
        assert abs(drift["shift"]) > drift["band"] and drift["inside_band"] is False
        assert row["outcome"] == "BOUND"
        assert "idle trend" in row["statement"]
        assert "above 10% of the point" not in row["statement"]  # the half-width reason

    def test_both_reasons_are_named_when_both_rules_fail(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "both", dynamic=0.6, static=0.4,
                                      sigma=1.0, seed=23, trend_w_per_s=0.0015,
                                      rotate_rounds=False)
        result = host.estimator.estimate(samples, markers, _drift_plan())
        row = result["designs"]["stock"]["arms"]["as_run"]["workload_increment_w"]
        assert row["holm_separated"] is True and row["outcome"] == "BOUND"
        assert row["half_width"] > 0.10 * abs(row["point"])
        assert row["drift_trend"]["inside_band"] is False
        assert "above 10% of the point" in row["statement"]  # the half-width reason
        assert "idle trend" in row["statement"]

    def test_the_absolute_bands_quantum_is_floored_at_the_declared_step(
            self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "floor", dynamic=0.6, static=0.4,
                                      sigma=0.02, seed=24)
        floored = host.estimator.estimate(samples, markers, _drift_plan())["drift"]
        assert floored["quantum_w"] == pytest.approx(STEP_W)
        assert floored["quantum_measured_w"] < STEP_W
        assert floored["band_w"] >= STEP_W and floored["inside_band"] is True
        plain = host.estimator.estimate(samples, markers, _plan())["drift"]
        assert "quantum_w" not in plain and "applies_to" not in plain

    def test_an_undeclared_rule_name_refuses(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "rule", dynamic=0.6, static=0.4,
                                      sigma=0.02, seed=25)
        with pytest.raises(ValueError, match="idle_trend"):
            host.estimator.estimate(samples, markers, _drift_plan(rule="bracket"))

    def test_the_synthetic_records_are_unchanged_without_a_trend(self, host, tmp_path):
        first = _synthetic(host, tmp_path, "same_a", dynamic=0.3, static=0.4, sigma=0.02,
                           seed=11)
        second = _synthetic(host, tmp_path, "same_b", dynamic=0.3, static=0.4, sigma=0.02,
                            seed=11, trend_w_per_s=0.0, rotate_rounds=True)
        for a, b in zip(first, second):
            assert Path(a).read_bytes() == Path(b).read_bytes()


class TestTheWindowDutyOfNUnits:
    """m3: beside the mean per-unit duty D, the duty over the whole window."""

    def test_every_loop_window_records_its_busy_share_of_the_window(self, session):
        proto = next(s for s in session.record["steps"]
                     if s["kind"] == "protocol" and s["status"] == "done")
        step = json.loads((Path(proto["dir"]) / "step.json").read_text())
        loops = [w for w in step["windows"] if w["condition"].startswith("loop_")]
        assert loops
        for window in loops:
            busy = window["run_s"] + window["programming_run_s"]
            assert window["kernel_duty_window"] == pytest.approx(
                busy / (3 * window["seconds"]), rel=0.02)
            assert window["kernel_duty_window"] <= window["kernel_duty"] + 1e-9

    def test_the_estimate_carries_both_duties(self, session):
        result = json.loads((session.results / "estimate.json").read_text())
        row = result["designs"]["stock"]["arms"]["as_run"]
        assert 0.0 < row["kernel_duty_window"] <= row["kernel_duty"] + 1e-9


class TestKernelStartsAreCountedByTheDriver:
    """m7: a control's ap_start count is measured, so a control that started a
    kernel would say so."""

    def test_a_control_that_started_the_kernel_would_report_it(self, host, monkeypatch):
        host.driver._FAKE_MODULES.clear()
        session = host.driver.open_session(_options(host))
        replay = json.loads((host.root / "deployment" / REPLAY_NAME).read_text())
        session.arm_fake(deployment=replay)
        try:
            document = host.executor.load_bundle(str(host.root / "deployment" / BUNDLE_NAME))
            campaign = host.executor.Campaign(document, session, samples=[0, 1, 2, 3])
            campaign.quiet = True
            campaign.record_transfers = True
            campaign.run()
            clean = host.executor.transfer_loop(host.executor.TransferControl(campaign), 0.0)
            assert clean["ap_starts"] == 0 and clean["passes"] > 0
            original = host.executor.ResidentPass.transfers_only

            def leaky(self, stimulus, pause_s):
                moved = original(self, stimulus, pause_s)
                self.session.start_and_wait(
                    self._program_bo, self._stimulus_bo, self._capture_bo,
                    program_words=self.declared_program_words,
                    stimulus_words=len(stimulus) // 4, capture_events=self.capture_events)
                return moved

            monkeypatch.setattr(host.executor.ResidentPass, "transfers_only", leaky)
            leaked = host.executor.transfer_loop(host.executor.TransferControl(campaign), 0.0)
            assert leaked["ap_starts"] == leaked["passes"] > 0
        finally:
            session.close()


class TestTheUnitWorkersChannelAndLifecycle:
    """m2: nothing but replies reaches a worker's channel, a stray line never
    ends a step, and a step killed on its timeout takes its workers with it."""

    def test_stray_writes_to_fd_1_never_reach_the_channel(self, host):
        script = textwrap.dedent("""
            import json, os, sys
            sys.path.insert(0, sys.argv[1])
            import odin_power_protocol as protocol
            channel = protocol.open_unit_channel()
            os.write(1, b"stray C-level output\\n")
            print("a stray print")
            channel.write(json.dumps({"ready": True}) + "\\n")
            channel.flush()
        """)
        completed = subprocess.run([sys.executable, "-c", script, str(host.root / "host")],
                                   capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == '{"ready": true}\n'
        assert "stray C-level output" in completed.stderr
        assert "a stray print" in completed.stderr

    def test_a_line_that_is_not_json_is_skipped_and_recorded(self, host, tmp_path):
        reply = json.dumps({"ready": True, "compute_unit": "odin_0", "identity": {}})
        script = (f"import sys; print('library chatter'); print({reply!r}); "
                  f"sys.stdout.flush(); sys.stdin.read()")
        workers = host.protocol.UnitWorkers(plan_path="unused", index=0, units=1,
                                            results=str(tmp_path), fake_pyxrt=None)
        workers.argv_for = lambda cu_index: [sys.executable, "-c", script]
        try:
            ready = workers.start(timeout=30)
            assert ready[0]["ready"] is True and ready[0]["compute_unit"] == "odin_0"
            assert workers.stray == [{"compute_unit": "odin_0", "line": "library chatter"}]
        finally:
            workers.stop()

    def test_a_worker_that_ends_without_a_reply_refuses_by_name(self, host, tmp_path):
        workers = host.protocol.UnitWorkers(plan_path="unused", index=0, units=1,
                                            results=str(tmp_path), fake_pyxrt=None)
        workers.argv_for = lambda cu_index: [sys.executable, "-c", "print('not json')"]
        try:
            with pytest.raises(host.protocol.OdinPowerRefusal, match="odin_0"):
                workers.start(timeout=30)
        finally:
            workers.stop()

    @pytest.mark.timeout(60)
    def test_a_timed_out_child_takes_its_own_children_with_it(self, host, tmp_path):
        pidfile = tmp_path / "grandchild.pid"
        script = textwrap.dedent(f"""
            import subprocess, sys, time
            grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            with open({str(pidfile)!r}, "w") as handle:
                handle.write(str(grandchild.pid))
            time.sleep(120)
        """)
        code, note = host.protocol.run_child([sys.executable, "-c", script],
                                             log=str(tmp_path / "child.log"), timeout=4.0)
        assert code == -9 and "killed" in note
        pid = int(pidfile.read_text())

        def alive() -> bool:
            try:
                with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
                    state = handle.read().rsplit(")", 1)[1].split()[0]
            except OSError:
                return False
            return state != "Z"

        deadline = time.monotonic() + 15.0
        while alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not alive()


class TestTheVivadoAndPilotTextsOfARebuiltNUnitDesign:
    """m5: the ten-unit Vivado row names the sum, and the pilot names the
    report its expected differential came from."""

    def _record(self):
        return {"arms": {"as_run": {"kernel_duty": 0.9, "workload_increment_w": {
            "hi": 0.7, "outcome": "NUMBER"}}}}

    def test_the_vivado_reading_names_the_sum_of_the_units(self, host):
        declared = {"odin_0_dynamic_w": 0.68, "hmss_0_dynamic_w": 2.2, "re_extracted": True,
                    "class": "ESTIMATE: unit test"}
        out = host.estimator.vivado_comparison(self._record(), declared, status="rebuilt",
                                               units=10)
        reading = out["arms"]["as_run"]["reading"]
        assert reading.startswith("Vivado prices the 10 compute units (the sum of "
                                  "odin_0 to odin_9) at 680.0 mW dynamic")
        assert "mean kernel duty 0.900 per unit" in reading
        assert out["arms"]["as_run"]["compute_units"] == 10
        one = host.estimator.vivado_comparison(self._record(), declared, status="archived")
        assert one["arms"]["as_run"]["reading"].startswith(
            "Vivado prices odin_0 at 680.0 mW dynamic, 612.0 mW at kernel duty 0.900")

    def test_the_pilot_names_the_re_extracted_report_once_it_is(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "pilot_text", dynamic=0.6, static=0.4,
                                      sigma=0.02, seed=26)
        plan = _plan()
        plan["designs"] = {"stock": {"compute_units": 10,
                                     "xclbin_identity": {"rebuilt": True},
                                     "vivado": {"re_extracted": True}}}
        declared = "ESTIMATE: the rebuilt build's own ten odin_<k> rows x the sizing duty"
        plan["thresholds"]["pilot"]["expected_differential_class"] = declared
        result = host.estimator.pilot(samples, markers, plan)
        assert result["expected_differential_describes"] == declared
        assert any("re-extracted" in note for note in result["notes"])
        assert not any("comes from the archived report_power" in note
                       for note in result["notes"])
        plan["designs"]["stock"]["vivado"]["re_extracted"] = False
        result = host.estimator.pilot(samples, markers, plan)
        assert any("comes from the archived report_power" in note for note in result["notes"])

    def test_an_archived_design_keeps_the_archived_pilot_text(self, host, tmp_path):
        samples, markers = _synthetic(host, tmp_path, "pilot_old", dynamic=0.6, static=0.4,
                                      sigma=0.02, seed=27)
        plan = _plan()
        plan["thresholds"]["pilot"]["expected_differential_class"] = "anything"
        result = host.estimator.pilot(samples, markers, plan)
        assert result["expected_differential_describes"] == (
            "the archived builds' Vivado report_power (odin_0 x archived duty, ESTIMATE)")
        assert "notes" not in result


class TestTheUnitNamesComeFromTheLinksOwnOutput:
    """m8: the names the filled plan read from the xclbin (xclbinutil --info)
    must be the names the host opens, or the plan refuses."""

    def _plan_with(self, tmp_path, names, units=3):
        plan = json.loads(SHIPPED_PLAN.read_text())
        plan["designs"]["stock"]["compute_units"] = units
        plan["designs"]["stock"]["xclbin_identity"]["compute_unit_names"] = names
        path = tmp_path / "plan_names.json"
        path.write_text(json.dumps(plan))
        return str(path)

    def test_the_names_the_host_opens_are_accepted(self, host, tmp_path):
        plan = host.protocol.load_plan(self._plan_with(tmp_path, ["odin_0", "odin_1", "odin_2"]))
        assert plan["designs"]["stock"]["xclbin_identity"]["compute_unit_names"] == [
            "odin_0", "odin_1", "odin_2"]

    @pytest.mark.parametrize("names", [["odin_0", "odin_1", "cu_2"], ["odin_0", "odin_1"],
                                       ["odin_1", "odin_0", "odin_2"], "odin_0"])
    def test_other_names_refuse(self, host, tmp_path, names):
        with pytest.raises(host.protocol.OdinPowerRefusal, match="compute_unit_names"):
            host.protocol.load_plan(self._plan_with(tmp_path, names))
