"""[ODIN C3] a WIDE-fabric deployment, end to end, and the two gaps it closed.

The same tiny classifier the stock loop-closer freezes, declared against
`odin_wide_1024x256_mb16` instead: 1024 axon slots signed per SYNAPSE on a
16-bit membrane. Nothing about the network changes, which is the point — what
is under test is the FABRIC axis:

  * the pass is programmed through the generated core's configuration port
    (`OP_PROG`) rather than over SPI, and its axon events are worded as SLOT
    addresses rather than ``{physical row, 0x07}``;
  * the SHIPPED reader rebuilds that stimulus byte for byte from the bundle's
    own chip claims — which is the whole reason the wording is dispatched and
    not assumed;
  * a stock bundle and a wide one are refused on each other's bitstream.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from integration.odin_hacc_harness import (
    CLASSES,
    TIMESTEPS,
    WIDE_PLATFORM_RESOLVED,
    prepare_step,
    two_core_mapping,
    wide_config_overrides,
)

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle
from mimarsinan.chip_simulation import odin_deployment_encoding as aer
from mimarsinan.chip_simulation.odin_deployment_bundle import seal
from mimarsinan.chip_simulation.odin_hacc.artifact import render_bundle
from mimarsinan.chip_simulation.odin_fpga.chip_configs import (
    STOCK_CHIP,
    WIDE_CHIP,
    ChipConfigError,
    chip_config_named,
)
from mimarsinan.chip_simulation.odin_fpga.chip_selection import (
    chip_config_for,
    chip_config_of_bundle,
)
from mimarsinan.chip_simulation.odin_rtl.stimulus import (
    OP_AER,
    OP_PROG,
    decode_ops,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.export.odin.feasibility import (
    EMISSION_CEILING,
    OdinFeasibilityError,
)
from mimarsinan.mapping.export.odin_gen.feasibility import gate_variant_segment
from mimarsinan.pipelining.pipeline_steps.verification.odin_hacc_deployment_step import (
    OdinHaccDeploymentStep,
)

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "scripts" / "hacc" / "package"
CHIP_SIM = REPO / "src" / "mimarsinan" / "chip_simulation"
VERBATIM_MODULES = (
    CHIP_SIM / "odin_deployment_bundle.py",
    CHIP_SIM / "odin_deployment_encoding.py",
)
HOST_FILES = (
    "odin_board_driver.py", "fake_pyxrt_for_selftest.py",
    "odin_deployment_executor.py",
)


@pytest.fixture(scope="module")
def wide_export(tmp_path_factory):
    """One run of the REAL export step against the WIDE chip configuration."""
    monkeypatch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("odin_hacc_wide")
    try:
        pipeline, step = prepare_step(
            monkeypatch, OdinHaccDeploymentStep, working_directory=str(root),
            config_overrides=wide_config_overrides(),
            platform_resolved=WIDE_PLATFORM_RESOLVED)
        step.process()
    finally:
        monkeypatch.undo()
    stats = pipeline.cache["OdinHaccDeploymentStep.odin_hacc_deployment_bundle"]
    return step.document, stats


def _unpack(payload: bytes):
    """A device-endian token payload back into its 32-bit words."""
    return struct.unpack(f"<{len(payload) // 4}I", payload)


class TestTheBundleNamesTheWideFabricAndNothingElse:

    def test_its_claims_resolve_to_the_wide_chip_configuration(self, wide_export):
        document, _stats = wide_export
        assert chip_config_of_bundle(document).name == WIDE_CHIP

    def test_the_claims_are_not_the_stock_fabrics(self, wide_export):
        document, _stats = wide_export
        stock = chip_config_named(STOCK_CHIP).bundle_claims()
        assert chip_config_named(WIDE_CHIP).claims_of_bundle(
            document["chip_config"]) != stock

    def test_the_declared_envelope_is_the_wide_one(self, wide_export):
        document, _stats = wide_export
        chip = document["chip_config"]
        assert chip["weight_bits"] == 8
        assert chip["weight_sign_granularity"] == "per_synapse"
        assert chip["effective_max_axons"] == 1023
        assert chip["soma_law"]["membrane_bits"] == 16

    def test_the_stats_name_the_fabric_the_package_will_build(self, wide_export):
        _document, stats = wide_export
        assert stats["chip"] == WIDE_CHIP


class TestTheWideStimulusIsWordedForItsOwnFabric:

    def test_the_reader_dispatches_the_variant_wording(self, wide_export):
        document, _stats = wide_export
        encoding = aer.encoding_of_bundle(document)
        assert encoding.name == aer.AER_VARIANT
        assert encoding.address_bits == 10
        assert not encoding.emits_tref

    def test_no_frozen_stimulus_carries_the_stock_time_reference(
            self, wide_export):
        """0x7F on this fabric is a spike on slot 127, and it must not be there."""
        document, _stats = wide_export
        for entry in document["samples"]:
            payload = bundle.decode_payload(
                entry["stimulus"], what=f"sample {entry['index']}")
            words = [int(word) for word in
                     _unpack(payload)]
            ops = decode_ops(words)
            for op in ops:
                if op.code == OP_AER:
                    assert op.args[1] < (1 << 10), (
                        "an AER word outside the wide core's 10-bit axon "
                        "address space reached a frozen stimulus")

    def test_the_program_is_written_through_the_configuration_port(
            self, wide_export):
        document, _stats = wide_export
        for plan in document["cores"]:
            payload = bundle.decode_payload(
                plan["program"], what=f"core {plan['core']}/program")
            ops = decode_ops([int(word) for word in _unpack(payload)])
            codes = {op.code for op in ops}
            assert codes == {OP_PROG}, (
                "a generated core is programmed through its configuration "
                f"port alone; this program carries {sorted(codes)}")

    def test_every_pass_slot_maps_to_exactly_one_physical_row(self, wide_export):
        """A synapse cell that signs itself spends ONE row per logical slot."""
        document, _stats = wide_export
        for plan in document["cores"]:
            for slot, rows in plan["slot_rows"]:
                assert rows == [slot]


@pytest.fixture(scope="module")
def staged_wide(tmp_path_factory, wide_export):
    _document, stats = wide_export
    root = tmp_path_factory.mktemp("odin_hacc_wide_pkg")
    (root / "host").mkdir()
    (root / "deployment").mkdir()
    for name in HOST_FILES:
        shutil.copyfile(PACKAGE / "host" / name, root / "host" / name)
    for module in VERBATIM_MODULES:
        shutil.copyfile(module, root / "host" / module.name)
    shutil.copyfile(stats["paths"]["bundle"], root / "deployment" / "bundle.json")
    shutil.copyfile(stats["paths"]["capture"], root / "deployment" / "replay.json")
    return root


def run_executor(staged: Path, *args: str):
    return subprocess.run(
        [sys.executable, "host/odin_deployment_executor.py",
         "--xclbin", "/selftest/no-such.xclbin",
         "--fake-pyxrt", "host/fake_pyxrt_for_selftest.py",
         "--bundle", "deployment/bundle.json",
         "--replay", "deployment/replay.json",
         "--results", "results/deploy", *args],
        cwd=str(staged), capture_output=True, text=True)


class TestTheShippedExecutorRunsItOnTheRightFabricOnly:

    def test_it_runs_green_when_the_session_declares_the_wide_chip(
            self, staged_wide):
        proc = run_executor(staged_wide, "--chip", WIDE_CHIP)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        report = json.loads(
            (staged_wide / "results" / "deploy"
             / "deployment_report.json").read_text())
        assert report["accuracy"] == 1.0
        assert report["passed"] is True
        assert report["bundle"]["aer_encoding"]["encoding"] == aer.AER_VARIANT
        predicted = {int(row["predicted"]) for row in report["readout"]}
        assert len(predicted) == CLASSES, predicted

    def test_the_stock_fabric_refuses_a_wide_bundle_rather_than_answering(
            self, staged_wide):
        proc = run_executor(staged_wide, "--chip", STOCK_CHIP)
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert "word an axon event differently" in proc.stdout + proc.stderr

    def test_the_default_session_is_the_stock_fabric_and_refuses(
            self, staged_wide):
        proc = run_executor(staged_wide)
        assert proc.returncode == 2, proc.stdout + proc.stderr


class TestTheWiderCrossbarDoesNotLiftTheCountCurrency:
    """127 events per cycle is what a segment BOUNDARY carries, not a core."""

    def test_the_gate_reports_what_the_segment_can_emit(self):
        wide = chip_config_named(WIDE_CHIP)
        mapping = two_core_mapping()
        gate = gate_variant_segment(
            mapping, spec=wide.core_spec, membrane_init=0, cycles=TIMESTEPS)
        assert gate.peak_emission <= EMISSION_CEILING
        assert set(gate.thetas) == set(range(len(mapping.cores)))

    def test_a_neuron_over_the_ceiling_refuses_on_the_wide_fabric_too(self):
        wide = chip_config_named(WIDE_CHIP)
        mapping = two_core_mapping()
        # theta 1 with 200 unit inputs is 200 events in one cycle: inside the
        # 1023-slot crossbar and the [-128, 127] cell, outside the currency.
        core = mapping.cores[0]
        core.core_matrix = np.ones((200, core.neurons_per_core), dtype=np.float64)
        core.axon_sources = [
            SpikeSource(-2, index, is_input=True) for index in range(200)]
        core.axons_per_core = 200
        with pytest.raises(OdinFeasibilityError, match="count-currency ceiling"):
            gate_variant_segment(
                mapping, spec=wide.core_spec, membrane_init=0, cycles=TIMESTEPS)


@pytest.fixture(scope="module")
def auditor():
    """``scripts/hacc/selftest/deployment_envelope_audit.py`` as a module."""
    path = REPO / "scripts" / "hacc" / "selftest" / "deployment_envelope_audit.py"
    spec = importlib.util.spec_from_file_location("odin_envelope_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTheBundleSurvivesLosingItsProducer:
    """The audit checks a bundle against NOTHING but its own decoded bytes."""

    def test_a_clean_wide_bundle_audits_green(self, auditor, wide_export):
        _document, stats = wide_export
        report = auditor.audit(Path(stats["paths"]["bundle"]), samples=3)
        assert report["chip"] == WIDE_CHIP
        assert report["stimulus"]["wording"] == aer.AER_VARIANT
        assert report["frozen_truth"]["samples_re_derived"] == 3
        assert report["frozen_truth"]["peak_emission_bound"] <= EMISSION_CEILING
        low, high = report["envelope"]["weight_range"]
        assert report["envelope"]["peak_abs_weight"] <= max(abs(low), high)

    def test_one_frozen_count_too_many_is_a_finding(
            self, auditor, wide_export, tmp_path):
        _document, stats = wide_export
        document = json.loads(Path(stats["paths"]["bundle"]).read_text())
        sample = document["certification"]["samples"][0]
        core = sorted(document["certification"]["windows"][str(sample)])[0]
        document["certification"]["windows"][str(sample)][core][0][0] += 1
        mutated = tmp_path / "mutated.json"
        # RESEALED: the seal still verifies, so only the re-derivation can catch it.
        mutated.write_text(render_bundle(seal(document)), encoding="utf-8")
        with pytest.raises(auditor.AuditFinding, match="re-derived from the shipped"):
            auditor.audit(mutated, samples=3)

    def test_the_stock_fabric_is_declined_rather_than_half_audited(self, auditor):
        committed = (REPO / "scripts" / "hacc" / "package" / "deployment"
                     / "nc1_two_core_passes.json")
        with pytest.raises(auditor.AuditUndecodable, match="did not run"):
            auditor.audit(committed, samples=3)


class TestTheFabricIsSelectedByTheDeclaredEnvelope:

    def test_the_wide_envelope_selects_the_wide_chip(self):
        assert chip_config_for(
            weight_bits=8, weight_sign_granularity="per_synapse",
            effective_max_axons=1023, membrane_bits=16).name == WIDE_CHIP

    def test_the_stock_envelope_selects_the_stock_chip(self):
        assert chip_config_for(
            weight_bits=4, weight_sign_granularity="per_axon",
            effective_max_axons=127, membrane_bits=8).name == STOCK_CHIP

    def test_an_envelope_no_fabric_can_be_refuses_by_name(self):
        with pytest.raises(ChipConfigError, match="no chip configuration"):
            chip_config_for(
                weight_bits=8, weight_sign_granularity="per_synapse",
                effective_max_axons=511, membrane_bits=16)
