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
