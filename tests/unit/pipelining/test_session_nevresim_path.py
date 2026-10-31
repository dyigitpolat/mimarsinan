"""[U-TT T-3] The library entry resolves the nevresim tree as the CLI entry does, through one helper.

The e2e driver builds its session through ``PipelineSession`` and never calls ``src/init.py``'s ``init()``, the only
place that set ``NevresimDriver.nevresim_path``; the Simulation step's ``prepare_all_segments``
(``chip_simulation/simulation_runner/emit.py``) then asserts on the unset path (the tuned C5 probe on RC4A, attempt 1,
defect D1). Both entries now resolve it with ``resolve_nevresim_path``: a path a caller set stays, an unset one takes
``nevresim_root()`` (``MIMARSINAN_NEVRESIM_ROOT``, else the tree beside the package).
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import MockDataProviderFactory

from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.common.env import nevresim_root
from mimarsinan.pipelining.session import PipelineSession

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _restore_process_globals():
    """A session applies determinism process-wide; put it back for the sibling tests."""
    torch_state, np_state, py_state = torch.random.get_rng_state(), np.random.get_state(), random.getstate()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    benchmark, precision = torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision()
    yield
    torch.random.set_rng_state(torch_state)
    np.random.set_state(np_state)
    random.setstate(py_state)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul_tf32, cudnn_tf32
    torch.backends.cudnn.benchmark = benchmark
    torch.set_float32_matmul_precision(precision)


def _session(tmp_path) -> PipelineSession:
    config = {
        "experiment_name": "nevresim_path",
        "data_provider_name": "MNIST_DataProvider",
        "generated_files_path": str(tmp_path / "generated"),
        "seed": 0,
        "deployment_parameters": {
            "model_type": "simple_mlp", "spiking_mode": "lif", "model_config_mode": "user", "hw_config_mode": "fixed",
            "model_config": {"mlp_width_1": 16, "mlp_width_2": 16, "base_activation": "ReLU"},
        },
        "platform_constraints": {},
    }
    return PipelineSession.from_config(config, data_provider_factory=MockDataProviderFactory())


def test_a_session_built_through_the_library_entry_resolves_the_nevresim_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(NevresimDriver, "nevresim_path", None)
    monkeypatch.delenv("MIMARSINAN_NEVRESIM_ROOT", raising=False)
    _session(tmp_path)
    assert NevresimDriver.nevresim_path == nevresim_root()
    assert Path(NevresimDriver.nevresim_path).is_dir()


def test_the_environment_override_reaches_the_library_entry(tmp_path, monkeypatch):
    tree = tmp_path / "another_tree"
    tree.mkdir()
    monkeypatch.setattr(NevresimDriver, "nevresim_path", None)
    monkeypatch.setenv("MIMARSINAN_NEVRESIM_ROOT", str(tree))
    _session(tmp_path)
    assert NevresimDriver.nevresim_path == str(tree.resolve())


def test_a_path_a_caller_set_stays(tmp_path, monkeypatch):
    monkeypatch.setattr(NevresimDriver, "nevresim_path", "/a/tree/the/caller/chose")
    _session(tmp_path)
    assert NevresimDriver.nevresim_path == "/a/tree/the/caller/chose"


def test_the_cli_entry_resolves_through_the_same_helper():
    init = next(n for n in ast.parse((REPO / "src" / "init.py").read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "init")
    called = {getattr(c.func, "id", getattr(c.func, "attr", None)) for c in ast.walk(init) if isinstance(c, ast.Call)}
    assert "resolve_nevresim_path" in called


def test_no_other_site_assigns_the_tree_itself():
    """One helper: no other module of the package, nor the CLI entry, assigns ``NevresimDriver.nevresim_path`` from
    ``nevresim_root()`` on its own (a runner handed an explicit tree still sets that tree)."""
    sites = []
    for path in [REPO / "src" / "init.py", *sorted((REPO / "src" / "mimarsinan").rglob("*.py"))]:
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Attribute) and t.attr == "nevresim_path" for t in node.targets)):
                continue
            if any(isinstance(c, ast.Call) and getattr(c.func, "id", getattr(c.func, "attr", None)) == "nevresim_root"
                   for c in ast.walk(node.value)):
                sites.append(str(path.relative_to(REPO)))
    assert sites == ["src/mimarsinan/chip_simulation/nevresim/nevresim_driver.py"], sites
