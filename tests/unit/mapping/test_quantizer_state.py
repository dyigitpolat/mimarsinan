"""WI-5a: the declared quantizer state, a sha-pinned map keyed by FX node name, read into the model's grids."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
import torch.fx
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.boundary_grids import install_boundary_grids
from mimarsinan.mapping.support.quantizer_state import (
    QUANTIZER_STATE_FORMAT,
    QuantizerStateError,
    declare_boundary_grids,
    declare_readout_ranges,
    load_quantizer_state,
    perceptron_fx_names,
    quantizer_state_sha256,
    verify_declared_state,
)
from mimarsinan.models.nn.activations.value_quantizer import BoundaryGrid, DeclaredBoundaryGrid
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
)
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import install_readout_quantizers
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from unit.chip_simulation.value_run.integer_programs import residual_block_net


def _flow(model=None, shape=(3, 16, 16)):
    model = model or residual_block_net()
    return convert_torch_model(model.eval(), shape, 10, device="cpu", packaging=MVM_PACKAGING).eval()


def write_state(tmp_path, perceptrons, name="state.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"format": QUANTIZER_STATE_FORMAT, "perceptrons": perceptrons}, sort_keys=True))
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class TestTheMap:
    def test_load_reads_ranges_and_entry_grids(self, tmp_path):
        declaration = write_state(tmp_path, {
            "stem": {"readout_range": 2.5, "entry": {"scale": 3.0, "offset": 1.5, "rounding": "floor",
                                                    "codes": [0, 15]}},
            "fc": {"readout_range": 4.0},
        })
        state = load_quantizer_state(declaration)
        assert state.sha256 == declaration["sha256"] == quantizer_state_sha256(declaration["file"])
        assert state.readout_ranges == {"stem": 2.5, "fc": 4.0}
        entry = state.entry_grids["stem"]
        assert (entry.scale, entry.offset, entry.rounding, entry.codes) == (3.0, 1.5, "floor", (0, 15))

    def test_absent_is_none(self):
        assert load_quantizer_state(None) is None

    def test_a_digest_other_than_the_file_refuses_by_name(self, tmp_path):
        declaration = write_state(tmp_path, {"fc": {"readout_range": 4.0}})
        declaration["sha256"] = "0" * 64
        with pytest.raises(QuantizerStateError, match="sha256") as info:
            load_quantizer_state(declaration)
        assert isinstance(info.value, ToolchainRefusal)

    @pytest.mark.parametrize("perceptrons,named", [
        ({"fc": {"readout_range": -1.0}}, "readout_range"),
        ({"fc": {"readout_range": float("inf")}}, "readout_range"),
        ({"fc": {"range": 1.0}}, "range"),
        ({"fc": {"entry": {"scale": 0.0}}}, "scale"),
        ({"fc": {"entry": {"scale": 1.0, "rounding": "up"}}}, "rounding"),
        ({"fc": {"entry": {"scale": 1.0, "codes": [3, 1]}}}, "codes"),
        ({"fc": {"entry": {"scale": 1.0, "codes": [0.5, 3]}}}, "codes"),
        ({"fc": {"entry": {"scale": 1.0, "step": 2}}}, "step"),
        ({"fc": {}}, "fc"),
    ])
    def test_a_malformed_entry_refuses_by_name(self, tmp_path, perceptrons, named):
        with pytest.raises(QuantizerStateError, match=named):
            load_quantizer_state(write_state(tmp_path, perceptrons))

    def test_a_document_of_another_format_refuses(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"format": "other", "perceptrons": {}}))
        with pytest.raises(QuantizerStateError, match="format"):
            load_quantizer_state({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def test_a_missing_file_refuses_by_name(self, tmp_path):
        with pytest.raises(QuantizerStateError, match="missing.json"):
            load_quantizer_state({"file": str(tmp_path / "missing.json"), "sha256": "0" * 64})


class TestFxNames:
    def test_every_perceptron_is_named_by_its_fx_node(self):
        model = residual_block_net()
        flow = _flow(copy.deepcopy(model))
        names = perceptron_fx_names(flow.get_mapper_repr())
        fx_nodes = {node.name for node in torch.fx.symbolic_trace(model).graph.nodes}
        assert set(names) <= fx_nodes
        assert len(names) == len(list(flow.get_perceptrons()))
        assert {"stem", "fc"} <= set(names)
        assert names["stem"] is next(iter(flow.get_perceptrons()))

    def test_a_linear_network_is_named_too(self):
        model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
        fx_nodes = {n.name for n in torch.fx.symbolic_trace(model).graph.nodes if n.op == "call_module"}
        flow = _flow(copy.deepcopy(model), shape=(8,))
        assert set(perceptron_fx_names(flow.get_mapper_repr())) == {"_0", "_2"} == fx_nodes - {"_1"}


def _quantized_flow(tmp_path, perceptrons, bits=5):
    flow = _flow()
    batches = [torch.randn(16, 3, 16, 16, generator=torch.Generator().manual_seed(0))]
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr()))
    grids = install_boundary_quantizers(entries, bits)
    calibrate_boundary_scales(flow, entries, grids, batches)
    all_perceptrons = list(flow.get_perceptrons())
    readouts = install_readout_quantizers(all_perceptrons, bits)
    for quantizer in readouts:
        quantizer.calibrate(1.0)
    declaration = write_state(tmp_path, perceptrons)
    state = load_quantizer_state(declaration)
    return flow, entries, grids, all_perceptrons, readouts, declaration, state


class TestDeclaringTheModel:
    def test_entry_grids_and_readout_ranges_are_installed(self, tmp_path):
        flow, entries, grids, perceptrons, readouts, declaration, state = _quantized_flow(tmp_path, {
            "stem": {"readout_range": 2.5, "entry": {"scale": 3.0, "offset": 1.5, "rounding": "floor",
                                                    "codes": [0, 15]}},
            "fc": {"readout_range": 4.0, "entry": {"scale": 6.0}},
        })
        names = perceptron_fx_names(flow.get_mapper_repr())
        assert declare_boundary_grids(flow, entries, grids, state) == 2
        stem_grid = grids[entries.index(names["stem"])].grid
        assert stem_grid == DeclaredBoundaryGrid(scale=3.0, bits=5, offset=1.5, rounding="floor", code_min=0,
                                                 code_max=15, state_sha256=state.sha256)
        assert grids[entries.index(names["fc"])].grid.entry_offset() == 0.0
        undeclared = [g for p, g in zip(entries, grids) if p not in (names["stem"], names["fc"])]
        assert all(type(g.grid) is BoundaryGrid for g in undeclared)
        assert declare_readout_ranges(flow, perceptrons, readouts, state) == 2
        assert float(readouts[perceptrons.index(names["stem"])].range) == 2.5
        assert float(readouts[perceptrons.index(names["fc"])].range) == 4.0
        assert {float(q.range) for p, q in zip(perceptrons, readouts) if p not in (names["stem"], names["fc"])} == {1.0}
        verify_declared_state(flow, declaration)

    def test_an_unknown_name_refuses_by_name(self, tmp_path):
        flow, entries, grids, *_rest, state = _quantized_flow(tmp_path, {"no_such_node": {"readout_range": 1.0}})
        with pytest.raises(QuantizerStateError, match="no_such_node"):
            declare_boundary_grids(flow, entries, grids, state)

    def test_an_entry_grid_on_a_core_fed_perceptron_refuses_by_name(self, tmp_path):
        class TwoHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b = nn.Linear(8, 6), nn.Linear(6, 3)

            def forward(self, x):
                y = self.a(x)
                return torch.cat([y, self.b(y)], dim=1)

        torch.manual_seed(0)
        flow = convert_torch_model(TwoHeads().eval(), (8,), 9, device="cpu", packaging=MVM_PACKAGING).eval()
        entries = list(segment_entry_perceptrons(flow.get_mapper_repr()))
        grids = install_boundary_quantizers(entries, 5)
        names = perceptron_fx_names(flow.get_mapper_repr())
        assert names["a"] in entries and names["b"] not in entries
        state = load_quantizer_state(write_state(tmp_path, {"b": {"entry": {"scale": 1.0}}}, "inner.json"))
        with pytest.raises(QuantizerStateError, match="'b'"):
            declare_boundary_grids(flow, entries, grids, state)

    def test_codes_beyond_the_dac_refuse_by_name(self, tmp_path):
        flow, entries, grids, *_rest, state = _quantized_flow(tmp_path, {
            "stem": {"entry": {"scale": 3.0, "codes": [0, 31]}}})
        with pytest.raises(ToolchainRefusal, match="code_max"):
            declare_boundary_grids(flow, entries, grids, state)


class TestStaleAndUnreadState:
    def _declared(self, tmp_path):
        flow, entries, grids, perceptrons, readouts, declaration, state = _quantized_flow(tmp_path, {
            "stem": {"readout_range": 2.5, "entry": {"scale": 3.0, "offset": 1.5}},
        })
        declare_boundary_grids(flow, entries, grids, state)
        declare_readout_ranges(flow, perceptrons, readouts, state)
        return flow, declaration

    def test_a_program_read_under_another_map_refuses(self, tmp_path):
        flow, _declaration = self._declared(tmp_path)
        other = write_state(tmp_path, {"stem": {"readout_range": 2.0, "entry": {"scale": 3.0}}}, "other.json")
        with pytest.raises(QuantizerStateError, match="stale"):
            verify_declared_state(flow, other)

    def test_a_program_carrying_state_with_no_declaration_refuses(self, tmp_path):
        flow, _declaration = self._declared(tmp_path)
        with pytest.raises(QuantizerStateError, match="no quantizer_state"):
            verify_declared_state(flow, None)

    def test_a_declaration_no_quantizer_read_refuses(self, tmp_path):
        flow, entries, grids, perceptrons, readouts, declaration, state = _quantized_flow(tmp_path, {
            "stem": {"readout_range": 2.5, "entry": {"scale": 3.0}}})
        declare_boundary_grids(flow, entries, grids, state)  # the readout range is never applied
        with pytest.raises(QuantizerStateError, match="readout_range"):
            verify_declared_state(flow, declaration)

    def test_soft_core_mapping_runs_the_check_and_stamps_declared_grids(self, tmp_path):
        flow, declaration = self._declared(tmp_path)
        repr_ = flow.get_mapper_repr()
        repr_.assign_perceptron_indices()
        ir = IRMapping(q_max=7, firing_mode="Default").map(repr_)
        assert install_boundary_grids(ir, flow, quantizer_state=declaration) > 0
        assert any(isinstance(getattr(n, "boundary_grid", None), DeclaredBoundaryGrid) for n in ir.nodes)
        with pytest.raises(QuantizerStateError):
            install_boundary_grids(ir, flow, quantizer_state=None)

    def test_an_undeclared_program_passes_with_no_declaration(self):
        flow = _flow()
        verify_declared_state(flow, None)
        assert np.isfinite(1.0)
