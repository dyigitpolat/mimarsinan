"""WI-5e and W08 end to end: a lane chip's integer value program runs through PipelineSession to atol-0 value
certificates, its lanes on chip, and its deployment record carries the executed output stage; a value-domain
bundle runs its lanes by default (and none under its off ablation) and prints the output stage beside the priced
point."""

from __future__ import annotations

import copy
import glob
import json
import os
import re

import pytest

from conftest import MockDataProviderFactory

from mimarsinan.pipelining.session import PipelineSession

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "integer_value", "mvm_int_readout_lenet5.json")
SHAPE = (3, 16, 16)
#: name -> (model_type, model_config, host ops the lane subset leaves): a residual net (adds and the head pool
#: on chip) and a DenseNet-BC (each transition's pool reduced into its 1x1 conv; the pre-activation BatchNorm and
#: ReLU of every concatenation stay on the host, outside the built classes).
MEMBERS = {
    "resnet": ("cifar_resnet_family", {"blocks_per_stage": 1, "base_width": 4}, 0),
    "densenet": ("cifar_densenet_bc", {"depth": 10, "growth_rate": 4}, None),
}
#: Model Building's static on-chip majority estimate refuses this tiny DenseNet-BC before any lane step (16% of its
#: 2,584 parameters estimated on chip): a property of the estimate at this size, so the run declares it host-heavy.
EXTRA = {"densenet": {"onchip_majority_gate": False}, "resnet_scheduled": {"allow_scheduling": True}}
MEMBERS["resnet_scheduled"] = MEMBERS["resnet"]
#: The split arm (review M2, P2): base width 6 on cores of 16 columns, so the packer splits the 24-column fused-add
#: cores into a used core's spare columns; each fragment keeps its sibling's operand rows, off.
MEMBERS["resnet_split"] = ("cifar_resnet_family", {"blocks_per_stage": 1, "base_width": 6}, 0)
#: The scheduled arm: few declared cores, so the one lane segment runs in several passes and the C-edge compares
#: the values its passes carry.
SCHEDULED_CORES = [{"max_axons": 1024, "max_neurons": 256, "count": 6, "has_bias": True}]
SPLIT_CORES = [{"max_axons": 1024, "max_neurons": 16, "count": 40000, "has_bias": True}]


def _config(tmp_path, model_type, model_config, extra=None):
    config = copy.deepcopy(json.load(open(FIXTURE, encoding="utf-8")))
    config["generated_files_path"] = str(tmp_path)
    config["device"] = "cpu"
    config["data_provider_name"] = "CIFAR10_DataProvider"
    platform = config["platform_constraints"]
    platform["cores"] = [{"max_axons": 1024, "max_neurons": 256, "count": 4000, "has_bias": True}]
    platform.update(max_axons=1024, max_neurons=256, weight_bits=4, activation_bits=4)
    platform["onchip_arithmetic"] = {"classes": ["L", "E", "P"], "lanes_per_core": 4, "lane_block": "ng45_os_lane"}
    config["deployment_parameters"].update({
        "model_type": model_type, "model_config": dict(model_config), "pruning": False, "allow_scheduling": False,
        "num_workers": 0, "batch_size": 16, "training_epochs": 1, "tuning_budget_scale": 0.05,
        "pretrain_floor_chance_multiple": 0, "degradation_tolerance": 0.9, **(extra or {}),
    })
    return config


@pytest.mark.slow
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("member", sorted(MEMBERS))
def test_a_lane_program_certifies_at_atol_zero_and_records_its_output_stage(tmp_path, capsys, member):
    model_type, model_config, host_left = MEMBERS[member]
    config = _config(tmp_path, model_type, model_config, EXTRA.get(member))
    if member == "resnet_scheduled":
        config["platform_constraints"]["cores"] = SCHEDULED_CORES
    if member == "resnet_split":
        config["platform_constraints"].update(cores=SPLIT_CORES, max_neurons=16, allow_neuron_splitting=True)
    session = PipelineSession.from_config(
        config,
        data_provider_factory=MockDataProviderFactory(input_shape=SHAPE, num_classes=10, size=64),
    )
    session.run()
    out = capsys.readouterr().out
    assert "[ValueParityGate] PASS model≡identity (max|delta|=0.000e+00, atol=0.0e+00" in out
    certificate = next(line for line in out.splitlines() if line.startswith("[ValueTwinCertificate] PASS"))
    assert "max|delta|=0.000e+00 atol=0.0e+00 passed=True" in certificate and "windows=0 " not in certificate
    ir_line = next(line for line in out.splitlines() if line.startswith("[SoftCoreMappingStep] IR Graph:"))
    host_ops = int(re.search(r"(\d+) compute ops", ir_line).group(1))
    if host_left is not None:
        assert host_ops == host_left, ir_line
    paths = glob.glob(os.path.join(str(tmp_path), "**", "deployment_record.json"), recursive=True)
    assert paths, "the pipeline sealed no deployment record"
    stage = json.load(open(paths[0], encoding="utf-8"))["output_stage"]
    assert stage["label"] == "executed" and stage["lane_block"] == "ng45_os_lane"
    assert stage["deployed"]["host_ops"] == host_ops
    work = stage["deployed"]["lane_ops"]
    assert work["clamp"] > 0 and work["pool"] > 0
    if member.startswith("resnet"):
        assert work["add"] > 0 and stage["deployed"]["host_round_trips"] == 0
    if member == "resnet_scheduled":
        program = next(line for line in out.splitlines() if "Hybrid program (scheduled)" in line)
        passes = [int(count) for count in re.findall(r"seg \d+: (\d+) pass", program)]
        assert passes and max(passes) >= 2, program


BUNDLE = "generic_estimated_22nm"
OFF = {"onchip_arithmetic": {"value": {"classes": []}, "note": "the off ablation of the lane subset"}}


def _bundle_config(tmp_path, arm):
    config = _config(tmp_path, *MEMBERS["resnet"][:2])
    platform = {"target": BUNDLE, "activation_bits": 4,
                "readout_law": {"adc_bits": 32, "range": "calibrated", "mult_bits": 16}}
    if arm == "off":
        platform["target_overrides"] = OFF
    config["platform_constraints"] = platform
    return config


@pytest.mark.slow
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("arm", ["on", "off"])
def test_a_value_domain_bundle_runs_its_lanes_by_default_and_prices_the_output_stage(tmp_path, capsys, arm):
    session = PipelineSession.from_config(
        _bundle_config(tmp_path, arm),
        data_provider_factory=MockDataProviderFactory(input_shape=SHAPE, num_classes=10, size=64),
    )
    session.run()
    out = capsys.readouterr().out
    assert "[ValueParityGate] PASS model≡identity (max|delta|=0.000e+00, atol=0.0e+00" in out
    ir_line = next(line for line in out.splitlines() if line.startswith("[SoftCoreMappingStep] IR Graph:"))
    host_ops = int(re.search(r"(\d+) compute ops", ir_line).group(1))
    record = glob.glob(os.path.join(str(tmp_path), "**", "deployment_record.json"), recursive=True)
    report = glob.glob(os.path.join(str(tmp_path), "**", "deployment_cost_report.json"), recursive=True)
    assert record and report, "the pipeline sealed no record or no priced report"
    stage = json.load(open(record[0], encoding="utf-8"))["output_stage"]
    notes = [n for n in json.load(open(report[0], encoding="utf-8"))["notes"] if n.startswith("output stage")]
    assert len(notes) == 1 and "chip-only energy" in notes[0] and "host round trips" in notes[0], notes
    if arm == "on":
        assert stage["label"] == "executed" and stage["lane_block"] == "ng45_os_lane" and host_ops == 0
        assert notes[0].startswith("output stage (executed): lanes ELP via lane block ng45_os_lane")
        _assert_the_candidate_prices_the_recorded_lane_block(session, json.load(open(record[0], encoding="utf-8")),
                                                             json.load(open(report[0], encoding="utf-8")))
    else:
        assert stage["label"] == "no lanes" and host_ops > 0 and stage["deployed"]["host_ops"] == host_ops
        assert notes[0].startswith("output stage (no lanes): lanes none")


def _assert_the_candidate_prices_the_recorded_lane_block(session, record, report):
    """Review F1: the joint search's candidate of this run's own configuration (its introspection seam, as
    ``cost_identity_w1.py`` walks it) prices the lane block the record prices: equal chip area, equal lanes per
    tile and lane work, and the candidate's output stage beside its chip-only energy."""
    from types import SimpleNamespace

    from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
    from mimarsinan.pipelining.pipeline_steps.config.architecture_search_problem import build_search_problem

    config = dict(session.pipeline.config)
    problem, _ = build_search_problem(
        SimpleNamespace(config=config, data_provider_factory=None), search_mode="hardware",
        builder_cls=ModelRegistry.get_builder_cls(str(config["model_type"])), arch_options=[],
        model_config_assembler=lambda raw: dict(raw), seed=0)
    problem._requires_fragment = lambda fragment: True
    view = problem.candidate_layout({"model_config": {}, "platform_constraints": {}}).view
    quantities = view.quantities
    work = sum(int(n) for n in record["output_stage"]["deployed"]["lane_ops"].values())
    assert quantities.get("lane_ops").value == work and quantities.get("lane_ops").provenance == "modeled"
    lanes = int(record["output_stage"]["lanes_per_core"]) * int(
        record["identity"]["platform"]["cores_per_tile_resolved"])
    assert quantities.get("lanes_per_tile").value == lanes
    candidate = view.cost_report().to_dict()
    area = {t["name"]: t["value"] for t in candidate["area"]}["chip_area_mm2"]
    recorded = {t["name"]: t["value"] for t in report["area"]}["chip_area_mm2"]
    assert abs(area - recorded) <= 1e-9 * recorded, (area, recorded)
    stage = [n for n in candidate["notes"] if n.startswith("output stage (estimated)")]
    assert len(stage) == 1 and f"lane ops {work:,}" in stage[0] and "lane-free shape" in stage[0], stage
    assert any(t["name"] == "energy_chip_only_mj" for t in report["energy"]), "the record prices its chip-only energy"
    # The candidate prices no host wall, so its static power over e2e latency refuses; the chip-only term says so.
    assert any(t["name"] == "energy_chip_only_mj" for t in candidate["energy"]) or any(
        n.startswith("refused energy_chip_only_mj: ") for n in candidate["notes"]), candidate["notes"]
