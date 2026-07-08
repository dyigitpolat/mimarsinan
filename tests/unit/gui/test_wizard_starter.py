"""THE fresh-state contract: opening the wizard yields a complete, resolvable,
runnable baseline pipeline.

The starter draft is the packaged baseline document (data, not framework
code) + a fresh experiment name. Three contracts, forever:
(1) fresh draft -> resolve -> ZERO errors and a live pipeline preview;
(2) fresh draft -> emit -> the emitted config passes the same validation the
    representability test applies (schema-known, resolvable, widget-renderable);
(3) the template path (save -> load -> resolve) stays error-free for every
    tier config.
"""

import glob
import json
import os
from datetime import datetime

import pytest

from mimarsinan.config_schema.registry import (
    Category,
    FieldType,
    REGISTRY,
    parse_deployment_document,
)
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.gui.wizard.emit import emit_deployment_config
from mimarsinan.gui.wizard.starter import load_starter_baseline, starter_draft

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_TIER_CONFIG_PATHS = sorted(
    glob.glob(os.path.join(_REPO_ROOT, "test_configs", "tier*", "t*.json"))
)


def _assert_widget_renderable(document: dict) -> None:
    """Every declared value must render in the schema-driven widgets."""
    parsed = parse_deployment_document(document)
    assert parsed.unknown == []
    for key, value in parsed.known_flat_keys().items():
        entry = REGISTRY[key]
        assert entry.category is not Category.RUNTIME, key
        if entry.category is Category.DERIVED:
            assert entry.declarable, key
        if value is None:
            continue
        if entry.type is FieldType.ENUM:
            options = entry.resolved_options() or ()
            assert value in options, f"{key}={value!r} not in {options}"
        elif entry.type is FieldType.BOOL:
            assert isinstance(value, bool), key
        elif entry.type in (FieldType.INT, FieldType.FLOAT):
            assert isinstance(value, (int, float)) and not isinstance(value, bool), key
            if entry.bounds is not None:
                lo, hi = entry.bounds
                assert lo is None or value >= lo, f"{key}={value} < {lo}"
                assert hi is None or value <= hi, f"{key}={value} > {hi}"


class TestFreshDraftResolves:
    def test_starter_resolves_with_zero_errors(self):
        resolution = resolve_draft(starter_draft())
        assert resolution.errors == []
        assert resolution.unknown_keys == []

    def test_starter_yields_a_live_pipeline_preview(self):
        from mimarsinan.gui.wizard.schema_api import resolve_payload

        payload = resolve_payload(starter_draft())
        assert payload["ok"] is True
        steps = payload["pipeline"]["steps"]
        assert len(steps) >= 10
        assert "Pretraining" in steps
        assert len(payload["pipeline"]["semantic_groups"]) == len(steps)

    def test_starter_derives_the_baseline_deployment_family(self):
        resolution = resolve_draft(starter_draft())
        assert resolution.derived["weight_quantization"]["value"] is True
        assert resolution.derived["activation_quantization"]["value"] is True
        assert resolution.derived["pipeline_mode"]["value"] == "phased"


class TestFreshDraftEmits:
    def test_emitted_config_passes_shape_validation(self):
        emitted = emit_deployment_config(starter_draft())
        assert validate_deployment_config(emitted) == []

    def test_emitted_config_is_schema_pure_and_widget_renderable(self):
        _assert_widget_renderable(emit_deployment_config(starter_draft()))

    def test_emitted_config_resolves_with_zero_errors(self):
        resolution = resolve_draft(emit_deployment_config(starter_draft()))
        assert resolution.errors == []
        assert resolution.unknown_keys == []


class TestStarterIdentity:
    def test_experiment_name_is_fresh_per_draft(self):
        a = starter_draft(now=datetime(2026, 7, 8, 10, 0, 0))
        b = starter_draft(now=datetime(2026, 7, 8, 10, 0, 1))
        assert a["experiment_name"] != b["experiment_name"]

    def test_experiment_name_extends_the_baseline_name(self):
        base = load_starter_baseline()["experiment_name"]
        draft = starter_draft(now=datetime(2026, 7, 8, 10, 0, 0))
        assert draft["experiment_name"].startswith(base + "_")

    def test_run_control_scaffold(self):
        draft = starter_draft()
        assert draft["generated_files_path"]
        assert "start_step" in draft and draft["start_step"] is None

    def test_baseline_document_is_deterministic(self):
        assert load_starter_baseline() == load_starter_baseline()

    def test_drafts_are_independent_copies(self):
        a = starter_draft()
        a["deployment_parameters"]["model_config"]["fc_w_1"] = -1
        b = starter_draft()
        assert b["deployment_parameters"]["model_config"]["fc_w_1"] != -1


class TestStarterIsRunnable:
    def test_starter_model_maps_feasibly_onto_its_platform(self):
        """The baseline model must fit the baseline core grid (the CPU proxy
        for runnability the wizard's own verify endpoint applies)."""
        from mimarsinan.mapping.verification.wizard_layout_verify import (
            model_repr_from_wizard_body,
            verify_planned_mapping_performance,
        )

        draft = starter_draft()
        dp = draft["deployment_parameters"]
        pc = draft["platform_constraints"]
        # The baseline workload facts (MNIST): tests own workload literals.
        model_repr = model_repr_from_wizard_body({
            "model_type": dp["model_type"],
            "input_shape": [1, 28, 28],
            "num_classes": 10,
            "model_config": dp["model_config"],
            "target_tq": pc.get("target_tq", 32),
        })
        stats = verify_planned_mapping_performance(model_repr, pc)
        assert stats is not None and stats["feasible"] is True


@pytest.mark.parametrize(
    "path",
    _TIER_CONFIG_PATHS,
    ids=lambda p: os.path.relpath(p, os.path.join(_REPO_ROOT, "test_configs")),
)
def test_template_path_stays_error_free(path, tmp_path, monkeypatch):
    """save-as-template -> load-template -> resolve: zero errors for every
    tier config (the wizard template flow, exercised server-side)."""
    from mimarsinan.gui.templates import get_template, save_template

    monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    template_id = save_template(config["experiment_name"], config)
    loaded = get_template(template_id)
    assert loaded is not None
    resolution = resolve_draft(loaded)
    assert resolution.errors == []
    assert resolution.unknown_keys == []
