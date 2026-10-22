"""The hypothesis table's assertion sites: H9's stage-order refusal, and every base-tree anchor resolves."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch

import mimarsinan.certification.randomized.executors as ex
from mimarsinan.certification.randomized.arms import ARMS
from mimarsinan.certification.randomized.runner import deployment
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import StageOrderError, run_hybrid_stages
from mimarsinan.common.refusal import ToolchainRefusal

from .test_runner import _chain, _constraints

TABLE = Path(__file__).resolve().parents[3] / "fixtures" / "theorem_box" / "hypothesis_table.json"


def _two_stage_program(semantics):
    instance = _chain(semantics, host=True)
    _ir, mapping = deployment(instance, _constraints(), 0, ARMS["on"])
    return instance, mapping


class TestStageOrder:
    @pytest.mark.parametrize("semantics", ["streamed_lif", "value", "synchronized_ttfs"])
    def test_a_program_in_dependency_order_runs(self, semantics):
        instance, mapping = _two_stage_program(semantics)
        assert ex.run_nodes(mapping, instance)

    @pytest.mark.parametrize("semantics", ["streamed_lif", "value", "synchronized_ttfs"])
    def test_a_stage_before_its_producer_refuses_by_name(self, semantics):
        """Reversed stages: the chip stage reads the host encoder's output before it exists (or, in the
        value domain, the program's second pass reads its first). Every executor goes through the one loop."""
        instance, mapping = _two_stage_program(semantics)
        mapping.stages = list(reversed(mapping.stages))
        with pytest.raises(StageOrderError, match="no earlier stage produced") as caught:
            ex.run_nodes(mapping, instance)
        assert isinstance(caught.value, ToolchainRefusal) and isinstance(caught.value, AssertionError)

    def test_the_loop_checks_before_the_callback(self):
        _instance, mapping = _two_stage_program("value")
        seen = []
        mapping.stages = list(reversed(mapping.stages))
        with pytest.raises(StageOrderError):
            run_hybrid_stages(mapping, {-2: torch.zeros(1, 4)}, on_neural=lambda *a: seen.append(a),
                              on_compute=lambda *a: seen.append(a))
        assert seen == []


def _resolve(site: str):
    module_name, _, qualname = site.partition(":")
    target = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


class TestAnchors:
    def test_every_base_tree_site_resolves(self):
        table = json.loads(TABLE.read_text())
        sites = [s["site"] for h in table["hypotheses"] for s in h["sites"] if s["tree"] == "base"]
        assert len(sites) >= 8
        for site in sites:
            assert callable(_resolve(site)), site
