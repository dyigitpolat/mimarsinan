"""Characterization lock: the resolved step list of every fixture config.

Pins the mode->step-plan surface (applies_to filtering + backend tail) so the
core_semantics axis and later packaging waves cannot silently reshape any
existing deployment's pipeline.
"""

import glob
import json
import os

import pytest

from mimarsinan.pipelining.core.pipelines.deployment_pipeline import (
    merge_pipeline_config,
)
from mimarsinan.pipelining.core.pipelines.deployment_specs import (
    get_pipeline_step_specs,
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_FIXTURE_DIR = os.path.join(_REPO_ROOT, "tests", "fixtures", "deployment_configs")

_COMMON_LIF = [
    "Model Configuration", "Model Building", "Pretraining", "Torch Mapping",
    "Scale Migration", "Activation Analysis", "Activation Adaptation",
    "Clamp Adaptation", "Activation Shifting", "Activation Quantization",
    "LIF Adaptation", "Weight Quantization", "Quantization Verification",
    "Normalization Fusion", "Soft Core Mapping",
    "Core Quantization Verification", "Hard Core Mapping",
    "Simulation", "Loihi Simulation", "SANA-FE Simulation",
]

EXPECTED_STEP_LISTS = {
    "mvm_lenet5.json": [
        "Model Configuration", "Model Building", "Pretraining",
        "Torch Mapping", "Weight Quantization", "Quantization Verification",
        "Normalization Fusion", "Soft Core Mapping",
        "Core Quantization Verification", "Hard Core Mapping",
    ],
    "mvm_fp_mmixcore.json": [
        "Model Configuration", "Model Building", "Pretraining",
        "Torch Mapping", "Normalization Fusion", "Soft Core Mapping",
        "Hard Core Mapping",
    ],
    "lif_cifar_deepcnn.json": _COMMON_LIF,
    "lif_deepcnn.json": _COMMON_LIF,
    "lif_deepmlp.json": _COMMON_LIF,
    "lif_imagenet_resnet.json": [
        "Model Configuration", "Model Building", "Weight Preloading",
        "Torch Mapping", "Scale Migration", "Activation Analysis",
        "Activation Adaptation", "Clamp Adaptation", "Activation Shifting",
        "Activation Quantization", "LIF Adaptation", "Weight Quantization",
        "Quantization Verification", "Normalization Fusion",
        "Soft Core Mapping", "Core Quantization Verification",
        "Hard Core Mapping", "Simulation", "Loihi Simulation",
        "SANA-FE Simulation",
    ],
    "lif_lenet5_novena.json": [
        "Model Configuration", "Model Building", "Pretraining",
        "Torch Mapping", "Pruning Adaptation", "Activation Analysis",
        "Activation Adaptation", "Clamp Adaptation", "Activation Shifting",
        "Activation Quantization", "LIF Adaptation", "Normalization Fusion",
        "Soft Core Mapping", "Hard Core Mapping", "Simulation",
        "Loihi Simulation", "SANA-FE Simulation",
    ],
    "lif_mmixcore.json": _COMMON_LIF,
    "lif_simplemlp.json": [
        "Model Configuration", "Model Building", "Pretraining",
        "Scale Migration", "Activation Analysis", "Activation Adaptation",
        "Clamp Adaptation", "Activation Shifting", "Activation Quantization",
        "LIF Adaptation", "Weight Quantization", "Quantization Verification",
        "Normalization Fusion", "Soft Core Mapping",
        "Core Quantization Verification", "Hard Core Mapping", "Simulation",
        "Loihi Simulation", "SANA-FE Simulation",
    ],
    "sync_deepcnn.json": [
        "Model Configuration", "Model Building", "Pretraining",
        "Torch Mapping", "Scale Migration", "Activation Analysis",
        "Activation Adaptation", "Clamp Adaptation", "Activation Shifting",
        "Activation Quantization", "Weight Quantization",
        "Quantization Verification", "Normalization Fusion",
        "Soft Core Mapping", "Core Quantization Verification",
        "Hard Core Mapping", "SANA-FE Simulation",
    ],
    "sync_mmixcore_pruned.json": [
        "Model Configuration", "Model Building", "Pretraining",
        "Torch Mapping", "Pruning Adaptation", "Scale Migration",
        "Activation Analysis", "Activation Adaptation", "Clamp Adaptation",
        "Activation Shifting", "Activation Quantization",
        "Weight Quantization", "Quantization Verification",
        "Normalization Fusion", "Soft Core Mapping",
        "Core Quantization Verification", "Hard Core Mapping",
        "SANA-FE Simulation",
    ],
    "sync_simplemlp.json": [
        "Model Configuration", "Model Building", "Pretraining",
        "Scale Migration", "Activation Analysis", "Activation Adaptation",
        "Clamp Adaptation", "Activation Shifting", "Activation Quantization",
        "Weight Quantization", "Quantization Verification",
        "Normalization Fusion", "Soft Core Mapping",
        "Core Quantization Verification", "Hard Core Mapping",
        "SANA-FE Simulation",
    ],
    "ttfs_mmixcore.json": [
        "Model Configuration", "Model Building", "Pretraining",
        "Torch Mapping", "Scale Migration", "Activation Analysis",
        "Activation Adaptation", "Clamp Adaptation", "Weight Quantization",
        "Quantization Verification", "Normalization Fusion",
        "Soft Core Mapping", "Core Quantization Verification",
        "Hard Core Mapping", "Simulation", "SANA-FE Simulation",
    ],
    "ttfs_simplemlp_identity.json": [
        "Model Configuration", "Model Building", "Pretraining",
        "Activation Analysis", "Activation Adaptation", "Clamp Adaptation",
        "Normalization Fusion", "Soft Core Mapping", "Hard Core Mapping",
        "Simulation", "SANA-FE Simulation",
    ],
    "ttfsq_deepcnn.json": [
        "Model Configuration", "Model Building", "Pretraining",
        "Torch Mapping", "Scale Migration", "Activation Analysis",
        "Activation Adaptation", "Clamp Adaptation", "Activation Shifting",
        "Activation Quantization", "Weight Quantization",
        "Quantization Verification", "Normalization Fusion",
        "Soft Core Mapping", "Core Quantization Verification",
        "Hard Core Mapping", "Simulation", "SANA-FE Simulation",
    ],
    "ttfsq_mmixcore.json": [
        "Model Configuration", "Model Building", "Pretraining",
        "Torch Mapping", "Scale Migration", "Activation Analysis",
        "Activation Adaptation", "Clamp Adaptation", "Activation Shifting",
        "Activation Quantization", "Weight Quantization",
        "Quantization Verification", "Normalization Fusion",
        "Soft Core Mapping", "Core Quantization Verification",
        "Hard Core Mapping", "Simulation", "SANA-FE Simulation",
    ],
}


def _fixture_paths():
    return sorted(glob.glob(os.path.join(_FIXTURE_DIR, "*.json")))


def test_every_fixture_is_pinned():
    assert {os.path.basename(p) for p in _fixture_paths()} == set(
        EXPECTED_STEP_LISTS
    )


@pytest.mark.parametrize(
    "path", _fixture_paths(), ids=lambda p: os.path.basename(p)
)
def test_step_list_is_pinned(path):
    document = json.load(open(path, encoding="utf-8"))
    config = merge_pipeline_config(
        dict(document["deployment_parameters"]),
        dict(document["platform_constraints"]),
    )
    names = [name for name, _cls in get_pipeline_step_specs(config)]
    assert names == EXPECTED_STEP_LISTS[os.path.basename(path)]
