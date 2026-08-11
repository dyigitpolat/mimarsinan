"""PipelineSession owns the registry's ``PipelineSession/determinism`` contract."""

import hashlib
import random

import numpy as np
import pytest
import torch

from conftest import MockDataProviderFactory

from mimarsinan.pipelining.session import PipelineSession, apply_determinism


@pytest.fixture(autouse=True)
def _restore_process_globals():
    """Determinism is process-global by design; undo it so sibling tests keep
    their ambient RNG/backend state."""
    torch_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    precision = torch.get_float32_matmul_precision()
    yield
    torch.random.set_rng_state(torch_state)
    np.random.set_state(np_state)
    random.setstate(py_state)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    torch.backends.cudnn.benchmark = benchmark
    torch.set_float32_matmul_precision(precision)


HARDWARE_SEARCH = {
    "optimizer": "nsga2",
    "pop_size": 4,
    "generations": 2,
    "seed": 0,
    "num_core_types": 1,
    "core_axons_bounds": [64, 256],
    "core_neurons_bounds": [64, 256],
    "core_count_bounds": [8, 64],
}


def _config(tmp_path, *, seed, name, hardware_search=False):
    config = {
        "experiment_name": name,
        "data_provider_name": "MNIST_DataProvider",
        "generated_files_path": str(tmp_path / name),
        "seed": seed,
        "deployment_parameters": {
            "model_type": "simple_mlp",
            "spiking_mode": "lif",
            "model_config_mode": "user",
            "hw_config_mode": "fixed",
            "model_config": {
                "mlp_width_1": 16,
                "mlp_width_2": 16,
                "base_activation": "ReLU",
            },
        },
        "platform_constraints": {},
    }
    if hardware_search:
        config["deployment_parameters"]["hw_config_mode"] = "search"
        config["deployment_parameters"]["arch_search"] = dict(HARDWARE_SEARCH)
        config["platform_constraints"] = {
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 64}],
        }
    return config


def _session(tmp_path, *, seed, name, hardware_search=False):
    return PipelineSession.from_config(
        _config(tmp_path, seed=seed, name=name, hardware_search=hardware_search),
        data_provider_factory=MockDataProviderFactory(),
    )


def _state_dict_hash(model) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _model_building_state_hash(tmp_path, *, seed, name, hardware_search=False) -> str:
    session = _session(tmp_path, seed=seed, name=name, hardware_search=hardware_search)
    session.pipeline.run(stop_step="Model Building")
    model = session.pipeline.cache.get("Model Building.model")
    assert model is not None
    return _state_dict_hash(model)


class TestApplyDeterminism:
    def test_seeds_every_rng_family(self):
        apply_determinism(4242)
        torch_draw = torch.rand(3)
        np_draw = np.random.rand(3)
        py_draw = random.random()

        apply_determinism(4242)
        assert torch.equal(torch.rand(3), torch_draw)
        assert np.array_equal(np.random.rand(3), np_draw)
        assert random.random() == py_draw

    def test_pins_deterministic_and_fp32_backend_flags(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        apply_determinism(0)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        assert torch.get_float32_matmul_precision() == "highest"


class TestSessionDeterminismOwnership:
    def test_session_init_applies_determinism_before_any_step(self, tmp_path):
        """The block belongs to session init: after from_config (no step has
        run yet) the RNG is pinned to the config seed and fp32 flags are set."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        _session(tmp_path, seed=1234, name="det_order")

        assert torch.initial_seed() == 1234
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.are_deterministic_algorithms_enabled()

    def test_plan_seed_resolves_from_top_level_config_seed(self, tmp_path):
        session = _session(tmp_path, seed=7, name="det_plan")
        assert session.pipeline.plan.seed == 7

    def test_same_seed_reproduces_model_building_state_dict(self, tmp_path):
        first = _model_building_state_hash(tmp_path, seed=7, name="det_same_a")
        second = _model_building_state_hash(tmp_path, seed=7, name="det_same_b")
        assert first == second

    def test_different_seeds_diverge_model_building_state_dict(self, tmp_path):
        first = _model_building_state_hash(tmp_path, seed=7, name="det_diff_a")
        second = _model_building_state_hash(tmp_path, seed=8, name="det_diff_b")
        assert first != second

    def test_searching_the_hardware_deploys_the_weights_the_fixed_run_would(self, tmp_path):
        """Choosing the chip must not choose the weights.

        Candidate scoring seeds the world to its own scoring seed; before the
        search ran inside ``isolated_rng_stream`` that seed escaped, so turning
        hw_config_mode from ``fixed`` to ``search`` re-rolled every downstream
        draw — the searched arm trained a DIFFERENT model and then bet it
        against the same certificates. Measured end to end on t0_60 vs its
        t0_05 anchor at seed 1: identical Model Building and Pretraining state
        dicts, same 0.981 read, on two different chips.
        """
        fixed = _model_building_state_hash(tmp_path, seed=3, name="det_fixed_arm")
        searched = _model_building_state_hash(
            tmp_path, seed=3, name="det_searched_arm", hardware_search=True,
        )
        assert searched == fixed
