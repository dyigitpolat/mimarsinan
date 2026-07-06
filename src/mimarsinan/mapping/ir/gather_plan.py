"""Precomputed gather plans for IR nodes: one indexed copy per source group.

The per-source Python walk in ``IRNode.gather_inputs`` cost one torch indexing
op per axon per forward (the W3b span-fill wall class). Sources are static per
node, so the walk collapses into index arrays built once and cached weakly per
node: one advanced-index copy for network inputs, one fill for always-on lanes,
and one copy per referenced producer buffer. Bit-equal to the reference walk by
construction (pure copies into the same default-dtype zeros tensor).
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

# Keyed by id(node): IR dataclasses are unhashable (eq=True); the finalizer
# evicts the entry during the node's dealloc, before its id can be reused.
_PLANS: Dict[int, "GatherPlan"] = {}


@dataclass
class GatherPlan:
    """Index arrays for one node's flattened ``input_sources``."""

    n_sources: int
    input_dst: Tuple[int, ...]
    input_src: Tuple[int, ...]
    on_dst: Tuple[int, ...]
    buffer_groups: Tuple[Tuple[int, Tuple[int, ...], Tuple[int, ...]], ...]
    _device_cache: Dict[torch.device, dict] = field(default_factory=dict)

    @property
    def referenced_node_ids(self) -> Tuple[int, ...]:
        return tuple(nid for nid, _, _ in self.buffer_groups)

    def _indices(self, device: torch.device) -> dict:
        cached = self._device_cache.get(device)
        if cached is None:
            cached = {
                "input_dst": torch.tensor(self.input_dst, dtype=torch.long, device=device),
                "input_src": torch.tensor(self.input_src, dtype=torch.long, device=device),
                "on_dst": torch.tensor(self.on_dst, dtype=torch.long, device=device),
                "buffers": [
                    (
                        nid,
                        torch.tensor(dst, dtype=torch.long, device=device),
                        torch.tensor(src, dtype=torch.long, device=device),
                    )
                    for nid, dst, src in self.buffer_groups
                ],
            }
            self._device_cache[device] = cached
        return cached

    def gather(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        batch_size = input_tensor.shape[0]
        device = input_tensor.device
        # Default dtype on purpose: the reference walk builds a default-dtype
        # zeros tensor, downcasting wider inputs on assignment.
        result = torch.zeros(batch_size, self.n_sources, device=device)
        idx = self._indices(device)
        if self.input_dst:
            result[:, idx["input_dst"]] = input_tensor[:, idx["input_src"]].to(result.dtype)
        if self.on_dst:
            result[:, idx["on_dst"]] = 1.0
        for nid, dst, src in idx["buffers"]:
            result[:, dst] = buffers[nid][:, src].to(result.dtype)
        return result

    def gather_referenced_buffers(self, buffers: Dict[int, "torch.Tensor"]) -> Dict[int, "torch.Tensor"]:
        """Subset of ``buffers`` this plan actually reads (for cheap dict conversions)."""
        return {nid: buffers[nid] for nid in self.referenced_node_ids}


def build_gather_plan(node) -> GatherPlan:
    sources = node.input_sources.flatten()
    input_dst: List[int] = []
    input_src: List[int] = []
    on_dst: List[int] = []
    by_node: Dict[int, Tuple[List[int], List[int]]] = {}
    for idx, src in enumerate(sources):
        if src.is_off():
            continue
        if src.is_input():
            input_dst.append(idx)
            input_src.append(int(src.index))
        elif src.is_always_on():
            on_dst.append(idx)
        else:
            dst_list, src_list = by_node.setdefault(int(src.node_id), ([], []))
            dst_list.append(idx)
            src_list.append(int(src.index))
    return GatherPlan(
        n_sources=len(sources),
        input_dst=tuple(input_dst),
        input_src=tuple(input_src),
        on_dst=tuple(on_dst),
        buffer_groups=tuple(
            (nid, tuple(dst), tuple(src)) for nid, (dst, src) in by_node.items()
        ),
    )


def gather_plan_for(node) -> GatherPlan:
    """Weakly-cached plan per node (sources are static after graph construction)."""
    key = id(node)
    plan = _PLANS.get(key)
    if plan is None:
        plan = build_gather_plan(node)
        _PLANS[key] = plan
        weakref.finalize(node, _PLANS.pop, key, None)
    return plan


def gather_inputs_reference(
    node,
    input_tensor: torch.Tensor,
    buffers: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """The original per-source walk, kept as the golden-test reference."""
    batch_size = input_tensor.shape[0]
    sources = node.input_sources.flatten()
    result = torch.zeros(batch_size, len(sources), device=input_tensor.device)

    for idx, src in enumerate(sources):
        if src.is_off():
            continue
        elif src.is_input():
            result[:, idx] = input_tensor[:, src.index]
        elif src.is_always_on():
            result[:, idx] = 1.0
        else:
            result[:, idx] = buffers[src.node_id][:, src.index]

    return result
