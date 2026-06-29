"""Template POST body parsing and minimal persistence."""

import json

from mimarsinan.config_schema.defaults import get_default_deployment_parameters
from mimarsinan.gui.templates import name_and_deployment_from_post_body, save_template


def test_name_and_deployment_from_post_body_wizard_shape() -> None:
    dep = {
        "experiment_name": "e",
        "pipeline_mode": "m",
        "deployment_parameters": {},
    }
    name, cfg = name_and_deployment_from_post_body({"name": "Display", "config": dep})
    assert name == "Display"
    assert cfg == dep


def test_name_and_deployment_from_post_body_uses_experiment_when_name_empty() -> None:
    dep = {"experiment_name": "from_exp", "pipeline_mode": "m", "deployment_parameters": {}}
    name, cfg = name_and_deployment_from_post_body({"name": "", "config": dep})
    assert name == "from_exp"
    assert cfg == dep


def test_name_and_deployment_from_post_body_flat() -> None:
    dep = {"experiment_name": "only", "pipeline_mode": "m", "deployment_parameters": {}}
    name, cfg = name_and_deployment_from_post_body(dep)
    assert name == "only"
    assert cfg == dep


def test_save_template_strips_derived_and_system_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MIMARSINAN_TEMPLATES_DIR", str(tmp_path))
    dp = get_default_deployment_parameters()
    dp.update({
        "model_config_mode": "user",
        "model_type": "mlp_mixer",
        "model_config": {},
        "spiking_mode": "ttfs_quantized",
        "activation_quantization": True,
        "pipeline_mode": "phased",
        "firing_mode": "TTFS",
        "spike_generation_mode": "TTFS",
        "thresholding_mode": "<=",
    })

    template_id = save_template(
        "Minimal Template",
        {
            "experiment_name": "old",
            "pipeline_mode": "phased",
            "deployment_parameters": dp,
            "platform_constraints": {},
        },
    )

    saved = json.loads((tmp_path / f"{template_id}.json").read_text(encoding="utf-8"))
    persisted = saved["deployment_parameters"]
    assert "pipeline_mode" not in saved
    assert "activation_quantization" not in persisted
    assert "firing_mode" not in persisted
    assert "kd_ce_alpha" not in persisted
    assert persisted["spiking_mode"] == "ttfs_quantized"
"""Template POST body parsing (wire format vs flat deployment)."""

from mimarsinan.gui.templates import name_and_deployment_from_post_body


def test_name_and_deployment_from_post_body_wizard_shape() -> None:
    dep = {
        "experiment_name": "e",
        "pipeline_mode": "m",
        "deployment_parameters": {},
    }
    name, cfg = name_and_deployment_from_post_body({"name": "Display", "config": dep})
    assert name == "Display"
    assert cfg == dep


def test_name_and_deployment_from_post_body_uses_experiment_when_name_empty() -> None:
    dep = {"experiment_name": "from_exp", "pipeline_mode": "m", "deployment_parameters": {}}
    name, cfg = name_and_deployment_from_post_body({"name": "", "config": dep})
    assert name == "from_exp"
    assert cfg == dep


def test_name_and_deployment_from_post_body_flat() -> None:
    dep = {"experiment_name": "only", "pipeline_mode": "m", "deployment_parameters": {}}
    name, cfg = name_and_deployment_from_post_body(dep)
    assert name == "only"
    assert cfg == dep
