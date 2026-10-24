"""Both planes of one program at one declared point, as the plane-parity census asks them (W04).

The shape-only plane is the one the exploration and the ladder read: the layout walk's specs, the footprint
``packed_tiles``, ``compute_mapping_stats`` under the declared capabilities, then ``collect_noc_fragments`` (here on the
builder's segments, ``spec_runs``). The executed plane is the deployment's: ``IRMapping`` at the geometry, ``IRLatency`` (the soft-core mapping
step's), ``build_hybrid_hard_core_mapping`` at the budget. A refusal is the raised class's name on either plane; the shape-only
plane's ``compute_mapping_stats`` message states the same name (``refusal_name_of_message``).
"""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from mimarsinan.deployment_record.introspection.scaling_ladder import packed_tiles
from mimarsinan.deployment_record.introspection.scaling_ladder_types import ShapeCensus
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.residency_rules import spec_runs
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.torch_mapping.converter import trace_and_convert

try:
    from mimarsinan.mapping.packing.infeasibility_proofs import refusal_name_of_message
except ImportError:  # before W04 a shape-only message names no refusal: every name comparison fails, by design
    def refusal_name_of_message(message: str):
        return None

PASSES = 4096


class _Res(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.a, self.b, self.r = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1), nn.ReLU()

    def forward(self, x):
        return self.r(self.b(self.r(self.a(x))) + x)


def vehicle(name: str):
    """(model, input shape): small CNNs whose wide fan-ins, odd widths and banks exercise every packing decision."""
    torch.manual_seed(0)
    if name == "conv_odd":
        return nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 22, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(22, 22, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(22 * 16, 10)), (3, 4, 4)
    if name == "wide_fc":
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 80), nn.ReLU(), nn.Linear(80, 10)), (3, 8, 8)
    if name == "res":
        return nn.Sequential(nn.Conv2d(3, 10, 3, padding=1), nn.ReLU(), _Res(10), nn.Conv2d(10, 20, 3, stride=2, padding=1),
                             nn.ReLU(), _Res(20), nn.Flatten(), nn.Linear(20 * 4, 10)), (3, 4, 4)
    if name == "pw":
        return nn.Sequential(nn.Conv2d(3, 6, 1), nn.ReLU(), nn.Conv2d(6, 9, 1), nn.ReLU(), nn.Conv2d(9, 9, 3, padding=1),
                             nn.ReLU(), nn.Flatten(), nn.Linear(9 * 36, 10)), (3, 6, 6)
    raise KeyError(name)


def representation(name: str):
    model, shape = vehicle(name)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
        warnings.simplefilter("ignore")
        flow = trace_and_convert(model.eval(), shape, "cpu", encoding_layer_placement="subsume",
                                 packaging=MVM_PACKAGING).flow.eval()
    rep = flow.get_mapper_repr()
    rep.assign_perceptron_indices()
    return rep


@dataclass(frozen=True)
class Answer:
    ok: bool
    refusal: Optional[str] = None
    passes: Optional[int] = None
    cores: Optional[int] = None


def walk(rep, a: int, n: int):
    layout = LayoutIRMapping(max_axons=a, max_neurons=n, allow_coalescing=True, hardware_bias=True,
                             collect_wire_census=False)
    specs = list(layout.collect_layout_softcores(rep))
    rep.clear_ir_caches()
    return layout, specs


def footprint(specs, a: int, n: int) -> int:
    census = ShapeCensus(softcores=tuple(specs), wire_census=None, host_segments=0, segments=0, logical_cells=1,
                         as_mapped_cells=1, wall_s=0.0)
    return packed_tiles(census, {"cores": [{"max_axons": a, "max_neurons": n, "count": 1, "has_bias": True}],
                                 "allow_coalescing": True, "allow_neuron_splitting": True})


def shape_answer(layout, specs, a: int, n: int, count: int, *, scheduled: bool) -> Answer:
    caps = ChipCapabilities(allow_coalescing=True, allow_neuron_splitting=True, allow_scheduling=scheduled,
                            max_schedule_passes=PASSES)
    types = [LayoutHardCoreType(a, n, int(count))]
    stats, error = compute_mapping_stats(specs, types, **caps.layout_kwargs())
    try:
        frag = collect_noc_fragments(softcores=specs, core_types=types, census=None,
                                     runs=spec_runs(layout) if scheduled else None, **caps.layout_kwargs())
    except Exception as exc:  # noqa: BLE001 - the plane's refusal, by name
        assert not stats.feasible, f"compute_mapping_stats feasible where collect_noc_fragments refuses: {exc}"
        assert refusal_name_of_message(error or "") == type(exc).__name__, (error, type(exc).__name__)
        return Answer(False, type(exc).__name__)
    assert stats.feasible, f"collect_noc_fragments places a program compute_mapping_stats refuses: {error}"
    return Answer(True, passes=len(frag.pass_placements))


def physical(hybrid) -> list:
    return [sum(len(c.fused_component_axons or [0]) for c in s.hard_core_mapping.cores)
            for s in hybrid.stages if s.kind == "neural"]


def executed_answer(rep, a: int, n: int, count: int, *, scheduled: bool) -> Answer:
    ir = IRMapping(q_max=1.0, firing_mode="Default", max_axons=a, max_neurons=n, allow_coalescing=True,
                   hardware_bias=True).map(rep)
    rep.clear_ir_caches()
    IRLatency(ir).calculate()  # as the soft-core mapping step does before every build
    caps = ChipCapabilities(allow_coalescing=True, allow_neuron_splitting=True, allow_scheduling=scheduled,
                            max_schedule_passes=PASSES)
    try:
        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=[{"max_axons": a, "max_neurons": n, "count": int(count), "has_bias": True}],
            strategy=MappingStrategy.resolve(caps), max_schedule_passes=PASSES)
    except Exception as exc:  # noqa: BLE001 - the plane's refusal, by name
        return Answer(False, type(exc).__name__)
    per = physical(hybrid)
    return Answer(True, passes=len(per), cores=sum(per))
