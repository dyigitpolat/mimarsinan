"""Wiring between a producer and a product operand, resolved to one flat index map by probing (N2).

A chain of stateless single-source mappers (reshape, permute, ``select``, ``getitem``, transpose) between a
producer and an operand only copies values; probed with two distinct-value inputs, each step proves itself a
gather of its source, and the chain composes to one map. The product then reads the producer directly.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

_PROBE_ERRORS = (RuntimeError, TypeError, ValueError, IndexError)


def mapper_output_shapes(model_repr, input_shape: Optional[Tuple[int, ...]] = None) -> Dict[int, Tuple[int, ...]]:
    """{id(mapper): its per-sample output shape} from one zero-input forward of batch 1 (tensors only); the input
    shape defaults to the graph's input mapper's."""
    order, deps, consumers = model_repr._ensure_exec_graph()
    if input_shape is None:
        root = next(node for node in order if not deps.get(node) and hasattr(node, "input_shape"))
        input_shape = tuple(int(d) for d in root.input_shape)
    first = next((p for node in order for p in node.parameters()), None)
    dtype = first.dtype if first is not None else torch.float32
    device = first.device if first is not None else None  # [U-RSHIFT] the zero sample on the model's device
    x = torch.zeros((1, *tuple(int(d) for d in input_shape)), dtype=dtype, device=device)
    values: dict = {}
    remaining = dict(consumers)
    shapes: Dict[int, Tuple[int, ...]] = {}
    with torch.no_grad():
        for node in order:
            sources = deps.get(node, [])
            arg = x if not sources else values[sources[0]] if len(sources) == 1 else tuple(values[s] for s in sources)
            values[node] = node.forward(arg)
            if isinstance(values[node], torch.Tensor):
                shapes[id(node)] = tuple(int(d) for d in values[node].shape[1:])
            for source in sources:
                remaining[source] -= 1
                if remaining[source] == 0:
                    values.pop(source, None)
    return shapes


def _stateless(mapper) -> bool:
    return next(mapper.parameters(), None) is None and next(mapper.buffers(), None) is None


def _probe(mapper, shape: Tuple[int, ...], values: np.ndarray) -> Optional[np.ndarray]:
    x = torch.as_tensor(values, dtype=torch.float64).reshape(1, *shape)
    try:
        with torch.no_grad():
            y = mapper.forward(x)
    except _PROBE_ERRORS:
        return None
    if not isinstance(y, torch.Tensor):
        return None
    return y.detach().reshape(-1).numpy()


def gather_step(mapper, source_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """The flat map ``out[i] = in[map[i]]`` of a stateless single-source mapper, or None when it is no gather."""
    size = int(math.prod(source_shape))
    if size == 0 or not _stateless(mapper):
        return None
    first = _probe(mapper, source_shape, np.arange(1, size + 1, dtype=np.float64))
    if first is None or not np.all(np.isin(first, np.arange(1, size + 1))):
        return None
    index = first.astype(np.int64) - 1
    permuted = np.random.default_rng(size).permutation(size).astype(np.float64) + 1.0
    second = _probe(mapper, source_shape, permuted)
    if second is None or second.shape != index.shape or not np.array_equal(second, permuted[index]):
        return None
    return index


def gather_chain(mapper, shapes: Dict[int, Tuple[int, ...]]) -> Tuple[object, np.ndarray]:
    """(the producer, the flat per-sample map): ``mapper``'s output is the producer's output gathered at the map."""
    node = mapper
    index = np.arange(int(math.prod(shapes[id(mapper)])), dtype=np.int64)
    while True:
        sources = node.get_source_mappers()
        if len(sources) != 1 or id(sources[0]) not in shapes:
            return node, index
        step = gather_step(node, shapes[id(sources[0])])
        if step is None:
            return node, index
        index = step[index]
        node = sources[0]
