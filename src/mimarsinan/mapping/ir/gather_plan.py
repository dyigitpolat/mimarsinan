"""Precomputed, weakly-cached gather plans for IR nodes: bit-equal to the per-source ``IRNode.gather_inputs`` walk but built once as index arrays instead of one torch op per axon per forward.

The index arrays are columnar (coverage W9): int64 arrays built in one pass over the sources, which a CPU device's
index tensors share (``torch.as_tensor``), 16 B per gathered entry instead of two tuples of Python ints and their
tensor copies (about 64 B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

from mimarsinan.common.instance_memo import InstanceMemo

_PLANS: InstanceMemo["GatherPlan"] = InstanceMemo()


@dataclass
class GatherPlan:
    """Index arrays (int64) for one node's flattened ``input_sources``."""

    n_sources: int
    input_dst: np.ndarray
    input_src: np.ndarray
    on_dst: np.ndarray
    buffer_groups: Tuple[Tuple[int, np.ndarray, np.ndarray], ...]
    _device_cache: Dict[torch.device, dict] = field(default_factory=dict)

    @property
    def referenced_node_ids(self) -> Tuple[int, ...]:
        return tuple(nid for nid, _, _ in self.buffer_groups)

    def _indices(self, device: torch.device) -> dict:
        cached = self._device_cache.get(device)
        if cached is None:
            cached = {
                "input_dst": torch.as_tensor(self.input_dst, device=device),
                "input_src": torch.as_tensor(self.input_src, device=device),
                "on_dst": torch.as_tensor(self.on_dst, device=device),
                "buffers": [
                    (nid, torch.as_tensor(dst, device=device), torch.as_tensor(src, device=device))
                    for nid, dst, src in self.buffer_groups
                ],
            }
            self._device_cache[device] = cached
        return cached

    def gather(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
        dtype: "torch.dtype | None" = None,
    ) -> torch.Tensor:
        batch_size = input_tensor.shape[0]
        device = input_tensor.device
        # Default dtype on purpose: the reference walk builds a default-dtype
        # zeros tensor, downcasting wider inputs on assignment. An explicit
        # ``dtype`` (the value-domain fp64 path) widens the buffer instead.
        result = torch.zeros(batch_size, self.n_sources, device=device, dtype=dtype)
        idx = self._indices(device)
        if self.input_dst.size:
            result[:, idx["input_dst"]] = input_tensor[:, idx["input_src"]].to(result.dtype)
        if self.on_dst.size:
            result[:, idx["on_dst"]] = 1.0
        for nid, dst, src in idx["buffers"]:
            buf = buffers[nid]
            if buf.shape[0] != batch_size:
                # Fail loud: a silent broadcast here once masked a
                # single-sample producer output (the §16.12 attention
                # double-selection) smeared across the batch.
                raise ValueError(
                    f"gather: node {nid} buffer rows {buf.shape[0]} != "
                    f"batch {batch_size} (shape {tuple(buf.shape)})"
                )
            result[:, dst] = buf[:, src].to(result.dtype)
        return result

    def gather_referenced_buffers(self, buffers: Dict[int, "torch.Tensor"]) -> Dict[int, "torch.Tensor"]:
        """Subset of ``buffers`` this plan actually reads (for cheap dict conversions)."""
        return {nid: buffers[nid] for nid in self.referenced_node_ids}


def build_gather_plan(node) -> GatherPlan:
    """The plan of ``node.input_sources`` (flattened): off (-1) skipped, network input (-2) and always-on (-3) entries,
    and every other node id's entries as one group, groups in the order of their first entry, entries in order."""
    sources = node.input_sources.flatten()
    count = len(sources)
    ids = np.fromiter((src.node_id for src in sources), dtype=np.int64, count=count)
    indices = np.fromiter((src.index for src in sources), dtype=np.int64, count=count)
    is_input, is_on = ids == -2, ids == -3
    positions = np.flatnonzero((ids != -1) & ~is_input & ~is_on)
    by_id = np.argsort(ids[positions], kind="stable")
    grouped, ordered = positions[by_id], ids[positions][by_id]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]]) if ordered.size else np.zeros(0, np.int64)
    groups: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for start, end in zip(starts.tolist(), [*starts[1:].tolist(), int(ordered.size)]):
        dst = grouped[start:end]
        groups.append((int(ordered[start]), dst, indices[dst]))
    groups.sort(key=lambda group: int(group[1][0]))
    input_dst = np.flatnonzero(is_input)
    return GatherPlan(
        n_sources=count,
        input_dst=input_dst,
        input_src=indices[input_dst],
        on_dst=np.flatnonzero(is_on),
        buffer_groups=tuple(groups),
    )


def gather_plan_for(node) -> GatherPlan:
    """Weakly-cached plan per node (sources are static after graph construction)."""
    return _PLANS.get(node, build_gather_plan)


def invalidate_gather_plan(node) -> None:
    """Forget ``node``'s plan after a pass rewired its sources (lane fusion)."""
    _PLANS.evict(node)


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
