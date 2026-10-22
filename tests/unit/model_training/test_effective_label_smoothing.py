"""Every tracked config's trainers apply the historical label smoothing 0.1, whatever their recipes declare."""

# The recipe field was never read, so the 0 most tracked configs declare never took effect. Change
# EXPECTED_EFFECTIVE_LABEL_SMOOTHING only together with a deliberate migration of those configs.

import glob
import json
import os

import pytest
import torch.nn as nn

import mimarsinan.pipelining  # noqa: F401  the toolchain's import order (model_training first is a cycle)
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.data_handling.data_provider import ClassificationMode
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import build_recipe
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from conftest import MockDataProviderFactory

EXPECTED_EFFECTIVE_LABEL_SMOOTHING = 0.1
RECIPE_KEYS = ("training_recipe", "tuning_recipe")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PATTERNS = (
    "templates/**/*.json",
    "tests/fixtures/deployment_configs/*.json",
    "scripts/_probes/**/*.json",
)


def _config_paths():
    paths = set()
    for pattern in _PATTERNS:
        paths.update(glob.glob(os.path.join(_REPO_ROOT, pattern), recursive=True))
    out = []
    for path in sorted(paths):
        with open(path, encoding="utf-8") as f:
            document = json.load(f)
        if isinstance(document, dict) and "deployment_parameters" in document:
            out.append(path)
    return out


CONFIG_PATHS = _config_paths()


def _resolved_parameters(path: str) -> dict:
    """The pipeline's merged config; the raw declaration when the merge refuses the config."""
    with open(path, encoding="utf-8") as f:
        document = json.load(f)
    try:
        return merge_pipeline_config(
            dict(document["deployment_parameters"]), dict(document.get("platform_constraints", {})))
    except ValueError:  # a config the legality checks refuse never trains; its declaration is still pinned
        return dict(document["deployment_parameters"])


def _trainer(recipe):
    dlf = DataLoaderFactory(MockDataProviderFactory(input_shape=(1, 4, 4), num_classes=4), num_workers=0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 4))
    return BasicTrainer(model, "cpu", dlf, ClassificationMode(4).create_loss(), recipe=recipe)


def test_the_config_matrix_is_present():
    # 38 tier-0 templates, the other templates, 52 probes and 17 fixtures at the time of writing.
    assert len(CONFIG_PATHS) >= 120


def test_most_configs_declare_a_zero_that_never_took_effect():
    declared_zero = 0
    for path in CONFIG_PATHS:
        recipes = [build_recipe(_resolved_parameters(path), key=key) for key in RECIPE_KEYS]
        declared_zero += any(r is not None and r.label_smoothing == 0 for r in recipes)
    assert declared_zero >= 100


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: os.path.relpath(p, _REPO_ROOT))
def test_every_trainer_recipe_trains_at_the_effective_smoothing(path):
    params = _resolved_parameters(path)
    for key in RECIPE_KEYS:
        trainer = _trainer(build_recipe(params, key=key))
        try:
            assert trainer.loss_function.label_smoothing == EXPECTED_EFFECTIVE_LABEL_SMOOTHING, key
        finally:
            trainer.close()
