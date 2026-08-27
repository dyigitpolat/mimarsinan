"""[ODIN P7a/P8] the deployment bundle: one schema, one seal, one routing rule.

The bundle is the document a board node executes with no repository in reach, so
the module that READS it ships verbatim. Everything here defends that: it may
import nothing a board node lacks, the packager must ship it byte-identically,
its seal must be the fixtures' seal, and the transcode it defines must be the
very one the cycle-accurate twin executes — otherwise the host-mediated passes
and the evidence they are certified against would be two different networks.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle
from mimarsinan.chip_simulation.odin_fpga import kernel_registers
from mimarsinan.chip_simulation.odin_rtl import reference

REPO = Path(__file__).resolve().parents[3]
MODULE = REPO / "src" / "mimarsinan" / "chip_simulation" / "odin_deployment_bundle.py"
PACKAGE = REPO / "scripts" / "hacc" / "package"
BUNDLE_PATH = PACKAGE / "deployment" / "nc1_two_core_passes.json"
REPLAY_PATH = PACKAGE / "deployment" / "nc1_two_core_passes_capture.json"

#: What a board node is promised: python3 and XRT. Nothing else may be imported.
STDLIB_ONLY = {
    "__future__", "base64", "hashlib", "json", "struct", "typing", "zlib",
}


@pytest.fixture(scope="module")
def document():
    return bundle.load_bundle(str(BUNDLE_PATH))


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location(
        "bundle_gate_driver", PACKAGE / "host" / "odin_board_driver.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestItShipsWhereThereIsNothingToImport:

    def test_it_imports_nothing_outside_the_standard_library(self):
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= STDLIB_ONLY, (
            f"the shipped bundle module imports {sorted(imported - STDLIB_ONLY)}, "
            f"which a HACC board node is not promised")

    def test_the_packager_ships_it_verbatim(self):
        source = (REPO / "scripts" / "hacc" / "make_package.py").read_text()
        assert 'copy(BUNDLE_MODULE, "host/odin_deployment_bundle.py")' in source
        assert "BUNDLE_MODULE = (" in source
        assert "odin_deployment_bundle.py" in source

    def test_the_seal_is_the_fixtures_seal(self, driver):
        probe = {"schema": "probe", "a": [1, 2, {"b": "c"}], "d": True}
        assert bundle.self_hash(probe) == driver.self_hash(probe)
        payload = bytes(range(256)) * 3
        assert bundle.encode_payload(payload) == driver.encode_payload(payload)


class TestTheRoutingRuleHasOneHome:

    def test_the_twin_executes_the_shipped_gather(self):
        assert reference.gather_axon_counts.__module__.endswith("reference")
        assert reference.ReferenceTraceError is bundle.OdinRoutingRefusal

    def test_a_slot_reading_a_core_that_never_ran_refuses(self):
        with pytest.raises(bundle.OdinRoutingRefusal, match="reads core 3"):
            bundle.gather_axon_slots(
                [(3, 0)], [(), ()], [0], where="core 9")

    def test_a_slot_reading_a_neuron_that_does_not_exist_refuses(self):
        with pytest.raises(bundle.OdinRoutingRefusal, match="neuron 7"):
            bundle.gather_axon_slots(
                [(0, 7)], [(1, 1)], [0], where="core 9")

    def test_the_sentinels_deliver_zero_one_and_the_entry_raster(self):
        slots = bundle.gather_axon_slots(
            [(bundle.SOURCE_OFF, 0), (bundle.SOURCE_ALWAYS_ON, 0),
             (bundle.SOURCE_INPUT, 1), (0, 2)],
            [(5, 6, 7)], [9, 4], where="core 0")
        assert slots == (0, 1, 4, 7)


class TestTheSeal:

    def test_a_sealed_document_verifies(self):
        sealed = bundle.seal({"schema": bundle.SCHEMA, "x": 1})
        assert bundle.verify_seal(sealed, source="probe")["x"] == 1

    def test_one_flipped_field_refuses(self):
        sealed = bundle.seal({"schema": bundle.SCHEMA, "x": 1})
        sealed["x"] = 2
        with pytest.raises(bundle.OdinBundleCorrupt, match="self-hash"):
            bundle.verify_seal(sealed, source="probe")

    def test_a_foreign_schema_refuses_before_anything_is_read(self):
        sealed = bundle.seal({"schema": "odin_hacc_fixture/2", "x": 1})
        with pytest.raises(bundle.OdinBundleCorrupt, match="is not"):
            bundle.verify_seal(sealed, source="probe")

    def test_a_damaged_payload_refuses_after_the_self_hash_passes(self):
        entry = bundle.encode_payload(b"\x01\x02\x03\x04")
        entry["bytes"] = 8
        with pytest.raises(bundle.OdinBundleCorrupt, match="declares 8 bytes"):
            bundle.decode_payload(entry, what="probe")


class TestTheCommittedBundle:

    def test_it_seals_and_carries_this_kernels_protocol_table(self, document,
                                                              driver):
        assert document["schema"] == bundle.SCHEMA
        assert document["kernel"] == driver.KERNEL_TABLE
        assert document["kernel"]["name"] == kernel_registers.KERNEL_NAME

    def test_its_counts_were_measured_on_the_rtl(self, document):
        assert "run_cosim" in document["provenance"]["derivation"]
        assert document["provenance"]["cosim_engine"]
        assert document["provenance"]["generating_commit"]

    def test_the_pass_order_is_causal(self, document):
        seen = []
        for plan in bundle.pass_plans(document):
            producers = {kind for kind, _ in plan["routes"] if kind >= 0}
            assert producers <= set(seen), (
                f"core {plan['core']} reads {producers} before they have run")
            seen.append(plan["core"])

    def test_the_second_pass_is_fed_only_by_the_first(self, document):
        consumer = bundle.pass_plans(document)[1]
        kinds = {kind for kind, _ in consumer["routes"]}
        assert kinds == {0}, (
            "the consumer must be entirely host-delivered from core 0, or the "
            "fixture would not exercise the transcode at all")

    def test_the_readout_answers_more_than_one_class(self, document):
        predicted = {row["predicted"] for row in document["expected"]["final"]}
        assert len(predicted) >= 3, (
            f"the shipped bundle predicts {predicted}; a constant answer could "
            f"not tell a working transcode from a broken one")

    def test_the_frozen_labels_leave_the_accuracy_short_of_one(self, document):
        rows = document["expected"]["final"]
        correct = sum(1 for row in rows if row["predicted"] == row["label"])
        assert 0 < correct < len(rows), (
            "the synthetic labels must produce both a hit and a miss, so the "
            "accuracy accumulator is exercised rather than asserted")

    def test_every_certified_sample_carries_a_window_for_every_pass(self, document):
        for sample in bundle.certification_samples(document):
            windows = bundle.expected_pass_windows(document, sample)
            assert sorted(windows) == sorted(document["pass_order"])
            for core, rows in windows.items():
                assert len(rows) == 1 and rows[0], f"core {core} froze no counts"

    def test_the_certification_subset_is_a_strict_subset(self, document):
        certified = set(bundle.certification_samples(document))
        shipped = {int(entry["index"]) for entry in document["samples"]}
        assert certified and certified < shipped

    def test_the_replay_is_bound_to_these_bundle_bytes(self, document):
        replay = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
        assert replay["bundle_self_hash"] == document["self_hash"]
        stamped = {k: v for k, v in replay.items() if k != "self_hash"}
        assert bundle.self_hash(stamped) == replay["self_hash"]


@pytest.fixture(scope="module")
def rebuilt():
    """The committed network, re-exported here: the ENCODER half of the mirror."""
    sys.path.insert(0, str(REPO / "scripts" / "hacc"))
    try:
        import make_deployment_bundle as generator
        from deployment_bundle_builder import PassBuild
    finally:
        sys.path.pop(0)
    from mimarsinan.chip_simulation.odin_rtl.reference import simulate_cycles
    from mimarsinan.mapping.latency.chip import ChipLatency

    mapping = generator.two_core_network()
    latency = int(ChipLatency(mapping).calculate())
    traces = [
        simulate_cycles(
            mapping, soma_law=generator.ODIN_LAW,
            input_counts=[list(row) for row in raster],
            simulation_length=generator.TIMESTEPS, chip_latency=latency)
        for raster in generator.RASTERS
    ]
    builds = [
        PassBuild(mapping, index, traces[0],
                  weight_bits=generator.WEIGHT_BITS,
                  effective_max_axons=generator.EFFECTIVE_MAX_AXONS,
                  soma_law=generator.ODIN_LAW,
                  weight_sign_granularity="per_axon", membrane_init=0)
        for index in range(len(mapping.cores))
    ]
    return builds, traces


class TestTheStimulusMirrorStillMatchesTheEncoder:
    """The reader builds every consumer stimulus; the repo built the evidence."""

    def test_the_frozen_core0_streams_are_still_what_the_encoder_emits(
            self, document, rebuilt):
        builds, traces = rebuilt
        for entry, trace in zip(document["samples"], traces):
            frozen = bundle.decode_payload(
                entry["stimulus"], what=f"sample {entry['index']}")
            assert frozen == bundle.tokens_to_bytes(
                builds[0].reference_stimulus(trace))

    def test_the_reader_rebuilds_every_pass_byte_for_byte(self, document,
                                                          rebuilt):
        builds, traces = rebuilt
        cycles = int(document["cycles_per_sample"])
        for plan, build in zip(bundle.pass_plans(document), builds):
            for trace in traces:
                mirrored = bundle.pass_stimulus_tokens(
                    plan,
                    [trace.inputs[cycle][build.index] for cycle in range(cycles)],
                    sample=0, cycles_per_sample=cycles)
                assert mirrored == list(build.reference_stimulus(trace))

    def test_a_short_transcode_refuses_instead_of_running_a_truncated_pass(
            self, document):
        plan = bundle.pass_plans(document)[0]
        with pytest.raises(bundle.OdinBundleError, match="truncated network"):
            bundle.pass_stimulus_tokens(
                plan, [(0,) * len(plan["routes"])], sample=0,
                cycles_per_sample=int(document["cycles_per_sample"]))


class TestTheReadout:

    def test_the_declared_rule_is_the_only_one_executed(self, document):
        drifted = copy.deepcopy(document)
        drifted["readout"]["rule"] = "softmax"
        with pytest.raises(bundle.OdinBundleError, match="not one this reader"):
            bundle.predicted_label(drifted, [1, 2, 3])

    def test_the_first_maximum_wins_a_tie(self, document):
        assert bundle.predicted_label(document, [4, 4, 1]) == 0

    def test_scores_are_gathered_from_the_declared_core(self, document):
        core = int(document["readout"]["core"])
        neurons = len(document["readout"]["neurons"])
        scores = bundle.readout_scores(document, {core: list(range(neurons))})
        assert scores == list(range(neurons))
