"""Golden resolution snapshot: rehoming constants must never change a resolved value.

The purity migration moves workload constants between homes (framework literals
-> registry profiles / config schema). This harness resolves every tier config
exactly the way ``DeploymentPipeline._initialize_config`` does (shared merge
functions, stubbed dataset I/O with the real provider classes) and pins the
full numeric surface — resolved config, DeploymentPlan, TuningBudget, recipe
and policy tables — against a checked-in snapshot. Regenerate deliberately with
``python scripts/regen_golden_resolution_snapshot.py``; the review diff of the
snapshot IS the bit-identity audit (pure additions = new keys; any changed line
= a changed resolved value).
"""

import dataclasses
import glob
import json
import os
from types import SimpleNamespace

import pytest

import mimarsinan.data_handling.data_providers  # noqa: F401 — registers providers
from mimarsinan.data_handling.data_provider import ClassificationMode, DataProvider
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import (
    apply_provider_facts,
    apply_workload_profiles,
    merge_pipeline_config,
)
from mimarsinan.tuning.orchestration.calibration_pipeline import encoder_scale_pin
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy
from mimarsinan.tuning.orchestration.tuning_budget import (
    min_step_for_smooth_adaptation,
    resolve_tuning_batch_size,
    tuning_budget_from_pipeline,
)
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG_PATHS = sorted(
    glob.glob(os.path.join(_REPO_ROOT, "test_configs", "tier*", "t*.json"))
)
SNAPSHOT_PATH = os.path.join(
    os.path.dirname(__file__), "golden_resolution_snapshot.json"
)
_REGEN_HINT = (
    "run `python scripts/regen_golden_resolution_snapshot.py` and audit the "
    "snapshot diff: additions are new keys; any CHANGED line is a changed "
    "resolved value and needs an explicit justification"
)


@dataclasses.dataclass(frozen=True)
class _ProviderFacts:
    """Static dataset facts (sizes as the real providers split them today) so
    resolution runs without dataset I/O; batch/budget math uses the REAL
    provider class methods on top of these."""

    base_input_shape: tuple
    num_classes: int
    train_size: int
    val_size: int
    test_size: int
    forced_preprocessing: dict | None = None


_IMAGENET_PREPROCESSING = {
    "normalize": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}

PROVIDER_FACTS = {
    "MNIST_DataProvider": _ProviderFacts((1, 28, 28), 10, 57000, 3000, 10000),
    "CIFAR10_DataProvider": _ProviderFacts((3, 32, 32), 10, 47500, 2500, 10000),
    "CIFAR100_DataProvider": _ProviderFacts((3, 32, 32), 100, 47500, 2500, 10000),
    "ImageNet_DataProvider": _ProviderFacts(
        (3, 224, 224), 1000, 1281167, 10000, 40000,
        forced_preprocessing=_IMAGENET_PREPROCESSING,
    ),
}


def stub_provider(provider_name: str, deployment_parameters: dict) -> DataProvider:
    """The registered provider class with dataset I/O replaced by static facts."""
    facts = PROVIDER_FACTS[provider_name]
    provider_cls = BasicDataProviderFactory._provider_registry[provider_name]
    preprocessing = (
        facts.forced_preprocessing
        if facts.forced_preprocessing is not None
        else deployment_parameters.get("preprocessing")
    )

    class _Stub(provider_cls):
        def __init__(self):
            DataProvider.__init__(
                self, "<golden-stub>", seed=0,
                preprocessing=preprocessing,
                batch_size=deployment_parameters.get("batch_size"),
            )

        def get_prediction_mode(self):
            return ClassificationMode(facts.num_classes)

        def get_training_set_size(self):
            return facts.train_size

        def get_validation_set_size(self):
            return facts.val_size

        def get_test_set_size(self):
            return facts.test_size

        def get_input_shape(self):
            spec = self._preprocessing_spec
            if spec is not None and spec.resize_to:
                return (facts.base_input_shape[0], spec.resize_to, spec.resize_to)
            return facts.base_input_shape

    return _Stub()


def _jsonify(value):
    return json.loads(json.dumps(value))


def build_config_entry(path: str) -> dict:
    """Resolve one tier config through the REAL pipeline resolution functions."""
    document = json.loads(open(path, encoding="utf-8").read())
    config = merge_pipeline_config(
        dict(document["deployment_parameters"]),
        dict(document["platform_constraints"]),
    )
    provider = stub_provider(
        document["data_provider_name"], document["deployment_parameters"]
    )
    apply_provider_facts(config, provider)
    apply_workload_profiles(config, provider)
    plan = DeploymentPlan.resolve(config)
    fake_pipeline = SimpleNamespace(
        config=config,
        data_provider_factory=SimpleNamespace(create=lambda: provider),
    )
    budget = tuning_budget_from_pipeline(fake_pipeline)
    return _jsonify({
        "resolved_config": config,
        "plan": {
            f.name: (
                dataclasses.asdict(value)
                if dataclasses.is_dataclass(value := getattr(plan, f.name))
                and not isinstance(value, type)
                else value
            )
            for f in dataclasses.fields(plan)
            if f.name != "config"
        },
        "tuning_budget": {
            **dataclasses.asdict(budget),
            "accuracy_se": budget.accuracy_se(),
        },
        "derived": {
            "training_batch_size": provider.get_training_batch_size(),
            "validation_batch_size": provider.get_validation_batch_size(),
            "test_batch_size": provider.get_test_batch_size(),
            "tuning_batch_size": resolve_tuning_batch_size(
                fake_pipeline, provider.get_training_batch_size()
            ),
            "min_step_for_smooth_adaptation": min_step_for_smooth_adaptation(
                fake_pipeline, budget
            ),
            "encoder_scale_pin": encoder_scale_pin(config),
            # The D1/D2 boundary-scale truth: hard-coded 1.0 at the two call
            # sites today; the input-scale fix rewires this to the resolved
            # workload value (MNIST cells must stay at 1.0).
            "boundary_input_data_scale": 1.0,
        },
    })


_RECIPE_MODES = (
    ("lif", None),
    ("ttfs", None),
    ("ttfs_quantized", None),
    ("ttfs_cycle_based", "cascaded"),
    ("ttfs_cycle_based", "synchronized"),
)


def build_global_surface() -> dict:
    """Mode-indexed recipe knobs + the frozen tuning policy (workload-silent SSOTs)."""
    recipes = {}
    for mode, schedule in _RECIPE_MODES:
        recipe = ConversionPolicy.derive(mode, schedule)
        recipes[f"{mode}|{schedule or 'none'}"] = {
            "driver": recipe.driver,
            "knobs": dict(recipe.knobs),
            "sim_enables": dict(recipe.sim_enables),
            "special_case": recipe.special_case,
        }
    return _jsonify({
        "conversion_recipes": recipes,
        "tuning_policy": dataclasses.asdict(TUNING_POLICY),
    })


def _config_id(path: str) -> str:
    return os.path.relpath(path, os.path.join(_REPO_ROOT, "test_configs"))


def build_snapshot() -> dict:
    return {
        "global": build_global_surface(),
        "configs": {_config_id(p): build_config_entry(p) for p in _CONFIG_PATHS},
    }


@pytest.fixture(scope="module")
def snapshot() -> dict:
    assert os.path.exists(SNAPSHOT_PATH), (
        f"golden snapshot missing at {SNAPSHOT_PATH}; {_REGEN_HINT}"
    )
    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        return json.load(f)


class TestGoldenResolutionSnapshot:
    def test_the_tier_matrix_is_present(self):
        assert len(_CONFIG_PATHS) == 61

    def test_snapshot_covers_exactly_the_tier_matrix(self, snapshot):
        assert set(snapshot["configs"]) == {_config_id(p) for p in _CONFIG_PATHS}, (
            _REGEN_HINT
        )

    def test_global_surface_is_bit_identical(self, snapshot):
        assert build_global_surface() == snapshot["global"], _REGEN_HINT

    @pytest.mark.parametrize("path", _CONFIG_PATHS, ids=_config_id)
    def test_resolution_is_bit_identical(self, snapshot, path):
        expected = snapshot["configs"][_config_id(path)]
        actual = build_config_entry(path)
        for section in expected:
            assert actual[section] == expected[section], (
                f"{section}: resolved values moved; {_REGEN_HINT}"
            )
        assert set(actual) == set(expected), _REGEN_HINT


class TestSharedResolutionSeam:
    """The harness resolves through the SAME functions the pipeline runs."""

    def test_apply_provider_facts_sets_exactly_the_provider_keys(self):
        provider = stub_provider("MNIST_DataProvider", {"batch_size": 128})
        config = {}
        apply_provider_facts(config, provider)
        assert config == {
            "input_shape": (1, 28, 28),
            "input_size": 784,
            "num_classes": 10,
        }

    def test_merge_pipeline_config_reproduces_the_documented_derivation(self):
        config = merge_pipeline_config(
            {"spiking_mode": "lif", "weight_quantization": True}, {}
        )
        assert config["activation_quantization"] is True
        assert config["pipeline_mode"] == "phased"
        assert config["target_tq"] == 32

    def test_stub_batch_policy_rides_the_real_provider_formulas(self):
        provider = stub_provider("MNIST_DataProvider", {})
        assert provider.get_training_batch_size() == 570
        assert provider.get_validation_batch_size() == 570
        provider_bs = stub_provider("MNIST_DataProvider", {"batch_size": 128})
        assert provider_bs.get_training_batch_size() == 128
