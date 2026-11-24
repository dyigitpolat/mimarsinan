"""[U-E2 H-1] The lane plan on the shape-only layout walk, applied as the build applies it.

The builder fuses the plan into the IR at Soft Core Mapping (``ir/fusion.py``). The shape-only walk
(``LayoutIRMapping`` holding a ``LaneWalk``) applies the same plan while it walks, at the same call sites
(``ir/call_sites.py``: an op takes a planned site's role when it carries the site's payload under the site's name,
or its payload alone for an unnamed site), so the walk records the fused program's shapes and
cells: a clamp's readers read its producer's cells; a fused add's readers read the accumulator's cells, and each
accumulator core takes one operand row per column (every core of a bank the bank's whole block, ``operand_rows``),
placed as axons and counted as no cell (``operand_axons``); a pool lane's outputs are reduction cores over its
windows (``pool_cores``' windows and core groups); a reduction replaces its producer's cores by reduced cores
(``reduce_cores``' groups); a constant-zero pad's readers read its input's cells with an off source at each padded
element (``pad_wiring``). So the absorbed ops leave the host census and the segmentation, and the wires through
them are core-to-core. Weights, grids and laws stay the builder's. A plan the walk cannot apply is refused by the
fusion's own name (``LaneFusionError``, ``PoolLaneError``). [U-L2] A pool lane's banks: ``reduction_banks.WalkBanks``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.lanes.declaration import lanes_executed
from mimarsinan.mapping.lanes.forms import pool_form, zero_pad_form
from mimarsinan.mapping.lanes.install import lane_core_geometry
from mimarsinan.mapping.lanes.ir.call_sites import PAD, planned_call_sites
from mimarsinan.mapping.lanes.ir.pad_wiring import padded_sources
from mimarsinan.mapping.lanes.ir.pool_cores import PoolLaneError, pool_windows, window_rows
from mimarsinan.mapping.lanes.ir.reduction_banks import WalkBanks, declared_reduction_banks
from mimarsinan.mapping.lanes.ir.refusal import LaneFusionError
from mimarsinan.mapping.lanes.plan import LanePlan, plan_lanes
from mimarsinan.mapping.layout.layout_source_view_ops import node_ids_of, total_size
from mimarsinan.mapping.noc.wire_census import record_emission_census

_OFF = IRSource(node_id=-1, index=0)
_ON = IRSource(node_id=-3, index=0)


class LaneWalk:
    """One walk's lane fusion: the plan's call sites and their roles, and what applying them recorded (per
    accumulator core its operand per column, the reduction cores inserted, the producers a reduction removed, the
    lane work by kind)."""

    def __init__(self, plan: LanePlan, *, first_owner: int, max_axons: Optional[int],
                 max_neurons: Optional[int], reduction_banks: bool = False) -> None:
        self.plan, self.sites, self.banks = plan, planned_call_sites(plan, first_owner), WalkBanks(reduction_banks)
        self._payloads = {id(site.module) for site, _role in self.sites}
        self.max_axons, self.max_neurons = max_axons, max_neurons
        self.absorbed_units = len(plan.absorbed())
        self.lane_ops: Dict[str, int] = {"clamp": 0, "add": 0, "pool": 0}
        self.operands: Dict[int, Dict[int, IRSource]] = {}
        self.inserted: Set[int] = set()
        self.removed: Set[int] = set()
        #: Per kept core, its emission ordinal: its index in the lane-free walk (the builder's layout record).
        self.record_of: Dict[int, int] = {}
        self._holders = {id(lane.holder) for lane in plan.perceptron_lanes if lane.add or lane.reduce}
        self._keep: Optional[Set[int]] = None
        self._inputs: Dict[int, Any] = {}
        self._parts: Set[int] = set()
        self._matched: Set[int] = set()
        self._ordinal = 0
        self._inserting = False

    @classmethod
    def of(cls, model_repr: Any, platform: Mapping[str, Any]) -> Optional["LaneWalk"]:
        """The walk of the lanes ``platform`` runs on ``model_repr`` (``lanes_executed``), else None."""
        declaration = lanes_executed(platform)
        if declaration is None:
            return None
        max_axons, max_neurons = lane_core_geometry(platform)
        walk = cls(plan_lanes(model_repr, declaration), first_owner=len(list(model_repr.get_perceptrons())),
                   max_axons=max_axons, max_neurons=max_neurons, reduction_banks=declared_reduction_banks(platform))
        order, _, _ = model_repr._ensure_exec_graph()
        walk._keep = {int(n.perceptron_index) for n in order
                      if id(n) in walk._holders and getattr(n, "perceptron_index", None) is not None}
        return walk

    # ---------------------------------------------------------------- the walk's hooks
    def absorbs(self, params: Any, name: Optional[str] = None) -> bool:
        return isinstance(params, Mapping) and self._site(params.get("module"), name) is not None

    def _site(self, module: Any, name: Optional[str]) -> Optional[int]:
        """The planned call site an op of payload ``module`` named ``name`` is (``call_site_roles``' rule), else None;
        an op two sites claim is refused by name."""
        if id(module) not in self._payloads:
            return None
        claims = [k for k, (site, _role) in enumerate(self.sites) if site.module is module
                  and (not getattr(site, "name", None) or site.name == name)]
        if len(claims) > 1:
            raise LaneFusionError(f"lane {self.sites[claims[0]][1][0]} {name!r}: the op is already another planned "
                                  "call site's")
        return claims[0] if claims else None

    def on_emit(self, mapping: Any, node_id: int, input_sources: Any, perceptron_index: Optional[int],
                psum_group_id: Optional[int]) -> None:
        if psum_group_id is not None:
            self._parts.add(int(node_id))
        if self._inserting:
            self.inserted.add(int(node_id))
            return
        self.record_of[int(node_id)] = self._ordinal
        self._ordinal += 1
        if perceptron_index is not None and self._keep is not None and int(perceptron_index) in self._keep:
            self._inputs[int(node_id)] = input_sources

    def absorb(self, mapping: Any, input_sources: Any, params: Mapping[str, Any], input_shape: Any,
               output_shape: Any, name: Optional[str]) -> Any:
        """The absorbed op's outputs as the fused program holds them."""
        module = params["module"]
        site = self._site(module, name)
        if site is None:
            raise LaneFusionError(f"op {name!r} is no planned call site's")
        kind, argument = self.sites[site][1]
        if site in self._matched:
            raise LaneFusionError(f"lane {kind} {name!r}: its module runs at several ops of the walk and its call site "
                                  "has no name that tells them apart")
        self._matched.add(site)
        shape = (tuple(int(d) for d in output_shape) if output_shape is not None
                 else (total_size(input_sources),))
        if kind == "clamp":
            self._require_cores(mapping, input_sources, "clamp", name)
            self.lane_ops["clamp"] += total_size(input_sources)
            return input_sources.reshape(shape)
        if kind == "add":
            return self._add(mapping, input_sources, params, int(argument), name, shape)  # type: ignore[arg-type]
        if kind == PAD:
            return self._pad(mapping, input_sources, module, input_shape, name, shape)
        flat = np.asarray(input_sources, dtype=object).ravel()
        windows, out_shape = self._windows(module, input_shape, output_shape, name, kind)
        if kind == "pool":
            self._require_cores(mapping, flat, "pool", name)
            windows = [[k for k in window if not flat[k].is_off()] for window in windows]
            groups = window_rows(windows, self.max_axons, self.max_neurons)  # [U-L2] one rule with the fusion's
            outputs = self.banks.pool(mapping, groups, windows, flat, self._insert(mapping, name, "lane", groups, flat),
                                      int(argument))  # type: ignore[arg-type]
        else:
            outputs = self._reduce(mapping, flat, windows, name)
        self.lane_ops["pool"] += int(outputs.size)
        return outputs.reshape(out_shape)

    def finalize(self, mapping: Any) -> None:
        """After the walk: every planned op was met, operand rows enter their accumulators, reduced producers leave."""
        if len(self._matched) != len(self.sites):
            raise LaneFusionError(f"the lane plan absorbs {len(self.sites)} host ops but the walk carries "
                                  f"{len(self._matched)} of them: the IR was mapped from another model")
        self._refuse_split_banks(mapping)
        for node, columns in sorted(self.operands.items()):
            self._operand_rows(mapping, node, columns)
        self.banks.apply(mapping)  # [U-L2] the pool lanes' banks, before the reduced producers' indices move
        self._drop_reduced(mapping)

    # ---------------------------------------------------------------- fused ops
    def _require_cores(self, mapping: Any, sources: Any, role: str, name: Optional[str], *, off: bool = True) -> None:
        neural = mapping._node_is_neural
        ids = node_ids_of(sources)
        if ids and all(neural.get(i, False) for i in ids) and not isinstance(sources, np.ndarray):
            return
        flat = np.asarray(sources, dtype=object).ravel()
        bad = sorted({int(s.node_id) for s in flat if not neural.get(int(s.node_id), False)
                      and not (off and s.is_off())})
        if bad:
            raise LaneFusionError(f"lane {role} {name!r} reads nodes {bad[:4]} that are not cores (host or input "
                                  f"values): the planned lane has no on-chip producer")

    def _add(self, mapping: Any, sources: Any, params: Mapping[str, Any], operand_index: int,
             name: Optional[str], shape: Tuple[int, ...]) -> Any:
        shapes = [tuple(s) for s in params.get("input_shapes") or ()]
        if len(shapes) != 2 or shapes[0] != shapes[1]:
            raise LaneFusionError(f"lane add {name!r} has operand shapes {shapes}: a fused add takes two equal "
                                  "shapes")
        flat = np.asarray(sources, dtype=object).ravel()
        size = flat.size // 2
        halves = (slice(0, size), slice(size, 2 * size))
        acc, operand = flat[halves[1 - operand_index]], flat[halves[operand_index]]
        self._require_cores(mapping, acc, "add", name, off=False)
        self._require_cores(mapping, operand, "add operand", name)
        for a, b in zip(acc, operand):
            columns = self.operands.setdefault(int(a.node_id), {})
            if int(a.index) in columns:
                raise LaneFusionError(f"core {a.node_id} column {a.index} would take two operands (a lane fuses "
                                      "one add)")
            columns[int(a.index)] = b
        self.lane_ops["add"] += int(size)
        return sources.flatten()[halves[1 - operand_index]].reshape(shape)

    def _pad(self, mapping: Any, sources: Any, module: Any, input_shape: Any, name: Optional[str],
             shape: Tuple[int, ...]) -> Any:
        """A constant-zero pad as wiring (the fusion's ``pad``): its input's cells, an off source per padded
        element."""
        in_shape = tuple(int(d) for d in (input_shape or ()))
        form = zero_pad_form(module, in_shape or None)
        flat = np.asarray(sources, dtype=object).ravel()
        if form is None or int(np.prod(in_shape)) != flat.size:
            raise LaneFusionError(f"pad {name!r} over input {in_shape}: not a constant-zero pad of its input (the walk "
                                  "and the plan disagree)")
        self._require_cores(mapping, flat, "pad", name)
        return padded_sources(flat, in_shape, form).reshape(shape)

    def _windows(self, module: Any, input_shape: Any, output_shape: Any, name: Optional[str],
                 kind: str) -> Tuple[List[List[int]], Tuple[int, ...]]:
        shape = tuple(int(d) for d in (input_shape or ()))
        form = pool_form(module, shape) if len(shape) == 3 else None  # pyright: ignore[reportArgumentType]
        if form is None:
            raise PoolLaneError(f"{'pool lane' if kind == 'pool' else 'reduction'} {name!r} over input {shape}: no "
                                "fixed window with a constant divisor")
        out_shape, windows = pool_windows(form, shape)  # pyright: ignore[reportArgumentType] -- rank 3, checked
        if output_shape is not None and tuple(int(d) for d in output_shape) != out_shape:
            raise PoolLaneError(f"{name!r}: windows give {out_shape}, the op writes {tuple(output_shape)}")
        return windows, out_shape

    def _insert(self, mapping: Any, name: Optional[str], tag: str, groups: List[Tuple[List[int], List[int]]],
                flat: np.ndarray) -> np.ndarray:
        """One reduction core per ``(outputs, rows)`` group over ``flat``'s elements; the source of each output."""
        outputs = np.empty(sum(len(members) for members, _ in groups), dtype=object)
        for g, (members, rows) in enumerate(groups):
            node_id = mapping._alloc_node_id()
            self._inserting = True
            try:
                mapping._emit_softcore_record(node_id=node_id, input_count=len(rows), output_count=len(members),
                                              name=f"{name}_{tag}{g}", input_sources=flat[rows],
                                              perceptron_index=None)
            finally:
                self._inserting = False
            for column, member in enumerate(members):
                outputs[member] = IRSource(node_id=node_id, index=column)
        return outputs

    def _reduce(self, mapping: Any, flat: np.ndarray, windows: List[List[int]], name: Optional[str]) -> np.ndarray:
        self._require_cores(mapping, flat, "reduction", name, off=False)
        producers = sorted({int(s.node_id) for s in flat})
        specs, index = mapping.layout_softcores, mapping._node_id_to_softcore_idx
        for node in producers:
            spec = specs[index[node]]
            if node in self._parts or spec.coalescing_group_id is not None:
                raise LaneFusionError(f"core {spec.name!r} under reduction {name!r} is a partial-sum or coalescing "
                                      "part: a reduction sums whole accumulators")
            if node in self.operands or node in self.inserted:
                raise LaneFusionError(f"reduction {name!r} reads core {spec.name!r}, which another lane already "
                                      "rewrote")
            if node not in self._inputs:
                raise LaneFusionError(f"reduction {name!r} reads core {spec.name!r}, which is no planned lane's "
                                      "holder (its inputs were not kept)")
        perceptrons = {mapping._sc_idx_to_perceptron_index.get(index[n]) for n in producers}
        if len(perceptrons) != 1 or None in perceptrons:
            raise LaneFusionError(f"reduction {name!r} reads perceptrons {sorted(map(str, perceptrons))}: one "
                                  "producer")
        groups: Dict[Tuple[int, ...], List[int]] = {}
        for output, window in enumerate(windows):
            groups.setdefault(tuple(int(flat[k].node_id) for k in window), []).append(output)
        outputs = np.empty(len(windows), dtype=object)
        perceptron = perceptrons.pop()
        rows_of = {n: int(specs[index[n]].input_count) // int(mapping.rows_per_input) for n in producers}
        for g, (key, members) in enumerate(groups.items()):
            node_id = mapping._alloc_node_id()
            cells = [np.asarray(self._inputs[n], dtype=object).ravel() for n in key]
            biased = sum(rows_of[n] - c.size for n, c in zip(key, cells))
            rows = np.concatenate(cells + [np.array([_ON] * biased, dtype=object)]) if biased else np.concatenate(cells)
            self._inserting = True
            try:
                mapping._emit_softcore_record(node_id=node_id, input_count=int(rows.size), output_count=len(members),
                                              name=f"{name}_reduce{g}", input_sources=rows,
                                              perceptron_index=perceptron)
            finally:
                self._inserting = False
            for column, member in enumerate(members):
                outputs[member] = IRSource(node_id=node_id, index=column)
        self.removed.update(producers)
        return outputs

    # ---------------------------------------------------------------- after the walk
    def _operand_rows(self, mapping: Any, node: int, columns: Mapping[int, IRSource]) -> None:
        sc = mapping._node_id_to_softcore_idx[node]
        spec = mapping.layout_softcores[sc]
        if node in self._parts or spec.coalescing_group_id is not None:
            raise LaneFusionError(f"core {spec.name!r} is a partial-sum or coalescing part: a fused add needs the "
                                  "whole accumulator in one core")
        bank = mapping._sc_idx_to_bank_id.get(sc)
        rows = len(columns) if bank is None else int(mapping._layout_weight_banks[bank][1])
        lane = list(columns[j] for j in sorted(columns)) + [_OFF] * (rows - len(columns))
        mapping.layout_softcores[sc] = replace(spec, input_count=int(spec.input_count) + rows, operand_axons=rows)
        mapping._node_input_node_ids.setdefault(node, set()).update(
            int(s.node_id) for s in lane if int(s.node_id) >= 0)
        if mapping.collect_wire_census:
            own = np.asarray(self._inputs[node], dtype=object).ravel()
            record_emission_census(mapping._node_wire_census, node, np.concatenate([own, np.array(lane, dtype=object)]),
                                   int(spec.input_count) // int(mapping.rows_per_input) + rows)

    def _refuse_split_banks(self, mapping: Any) -> None:
        """A bank takes its operand rows once, for every core reading it: each such core is an accumulator of the
        lane with every column given an operand."""
        node_of = {sc: n for n, sc in mapping._node_id_to_softcore_idx.items()}
        banks = {mapping._sc_idx_to_bank_id[mapping._node_id_to_softcore_idx[n]] for n in self.operands
                 if mapping._node_id_to_softcore_idx[n] in mapping._sc_idx_to_bank_id}
        for sc, bank in sorted(mapping._sc_idx_to_bank_id.items()):
            spec = mapping.layout_softcores[sc]
            if bank in banks and len(self.operands.get(node_of[sc], {})) != int(spec.output_count):
                raise LaneFusionError(f"weight bank {bank} is shared by core {spec.name!r} outside its add lane: "
                                      "one operand row set per bank")

    def _drop_reduced(self, mapping: Any) -> None:
        if not self.removed:
            return
        node_of = {sc: n for n, sc in mapping._node_id_to_softcore_idx.items()}
        kept = [sc for sc in range(len(mapping.layout_softcores)) if node_of[sc] not in self.removed]
        moved = {sc: i for i, sc in enumerate(kept)}
        mapping.layout_softcores = [mapping.layout_softcores[sc] for sc in kept]
        mapping._node_id_to_softcore_idx = {n: moved[sc] for n, sc in mapping._node_id_to_softcore_idx.items()
                                            if sc in moved}
        for table in ("_sc_idx_to_perceptron_index", "_sc_idx_to_bank_id"):
            setattr(mapping, table, {moved[sc]: v for sc, v in getattr(mapping, table).items() if sc in moved})
        for node in self.removed:
            for table in ("_node_is_neural", "_node_input_node_ids", "_node_wire_census"):
                getattr(mapping, table).pop(node, None)
        for deps in mapping._node_input_node_ids.values():
            deps.difference_update(self.removed)
