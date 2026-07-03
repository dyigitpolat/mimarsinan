"""DataLoaderFactory.for_pipeline is the single seam honoring config num_workers."""

import ast
from pathlib import Path

from conftest import MockPipeline

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"


class TestForPipelineSeam:
    def test_honors_config_num_workers_zero(self):
        pipeline = MockPipeline(config={"num_workers": 0})
        factory = DataLoaderFactory.for_pipeline(pipeline)
        assert factory._num_workers == 0
        assert factory._persistent_workers is False

    def test_defaults_to_four_workers_without_config_key(self):
        pipeline = MockPipeline(config={"device": "cpu"})
        factory = DataLoaderFactory.for_pipeline(pipeline)
        assert factory._num_workers == 4
        assert factory._persistent_workers is True

    def test_uses_pipeline_data_provider_factory(self):
        pipeline = MockPipeline(config={"num_workers": 0})
        factory = DataLoaderFactory.for_pipeline(pipeline)
        assert factory._data_provider_factory is pipeline.data_provider_factory


class TestAllPipelineSitesUseTheSeam:
    def test_no_direct_construction_from_a_pipeline(self):
        """Any DataLoaderFactory built from a pipeline must go through for_pipeline,
        else config num_workers is silently ignored (leaked persistent workers)."""
        offenders = []
        for path in SRC_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "DataLoaderFactory"):
                    continue
                args_src = ast.unparse(node)
                if "data_provider_factory" in args_src and "pipeline" in args_src:
                    offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno} {args_src}")
        assert offenders == [], (
            "DataLoaderFactory constructed directly from a pipeline "
            f"(use DataLoaderFactory.for_pipeline): {offenders}"
        )
