"""Consumer groups (N2 design §2.3): a producer's channels partitioned by the readers that read them, the
coarsest partition in which every operand bank and every other consumer reads whole groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from mimarsinan.mapping.mappers.operand.operand_mapper import operand_mappers
from mimarsinan.mapping.support.operand_form import B_STATIONARY

_WHOLE = ("whole",)


@dataclass(frozen=True)
class ConsumerGroups:
    """Per output channel its group, and per group whether an operand bank writes it as its payload."""

    channels: np.ndarray
    payload: Tuple[bool, ...]

    @property
    def count(self) -> int:
        return len(self.payload)


def _channel_axis(perceptron, rank: int) -> int:
    axis = int(getattr(perceptron, "output_channel_axis", -1))
    return axis - 1 if axis > 0 else rank + axis


def channels_of(perceptron, shape: Tuple[int, ...], flat: np.ndarray) -> np.ndarray:
    """The output channel of each flat per-sample index of ``perceptron``'s output (of per-sample ``shape``)."""
    return np.unravel_index(np.asarray(flat, dtype=np.int64), shape)[_channel_axis(perceptron, len(shape))]


def bank_reads(mapper, role: int) -> np.ndarray:
    """(banks, elements) flat root indices each bank reads for operand ``role``."""
    form = mapper.lowering.form
    index = mapper.operand_map(role).reshape(mapper.operand_shapes[role])
    perm = form.a_perm if role == 0 else form.b_perm
    return index.transpose(perm).reshape(form.banks, -1)


def consumer_group_partition(model_repr) -> Dict[int, ConsumerGroups]:
    """{id(perceptron): its groups} for every static perceptron an operand product reads."""
    readers: Dict[int, Dict[int, set]] = {}
    owners: Dict[int, object] = {}
    sizes: Dict[int, int] = {}
    for mapper in operand_mappers(model_repr):
        for role in (0, 1):
            root = mapper.get_source_mappers()[mapper.operand_roots[role]]
            perceptron = getattr(root, "perceptron", None)
            if perceptron is None:
                continue
            shape = mapper.root_shapes[mapper.operand_roots[role]]
            owners[id(perceptron)], sizes[id(perceptron)] = perceptron, int(perceptron.output_channels)
            payload = (role == 1) == (mapper.lowering.stationary == B_STATIONARY)
            for bank, flat in enumerate(bank_reads(mapper, role)):
                key = (id(mapper), bank, "payload" if payload else "stream")
                for channel in np.unique(channels_of(perceptron, shape, flat)).tolist():
                    readers.setdefault(id(perceptron), {}).setdefault(int(channel), set()).add(key)
    consumers = model_repr.consumer_map()
    lowered = {id(m) for m in operand_mappers(model_repr)}
    for node in model_repr.execution_order():
        perceptron = getattr(node, "perceptron", None)
        if perceptron is None or id(perceptron) not in owners:
            continue
        others = [c for c in consumers.get(id(node), []) if id(c) not in lowered]
        if others or node is model_repr.output_layer_mapper:
            for channel in range(sizes[id(perceptron)]):
                readers[id(perceptron)].setdefault(channel, set()).add(_WHOLE)
    out: Dict[int, ConsumerGroups] = {}
    for key, by_channel in readers.items():
        signatures: List[frozenset] = []
        channels = np.zeros(sizes[key], dtype=np.int64)
        for channel in range(sizes[key]):
            signature = frozenset(by_channel.get(channel, set()))
            if signature not in signatures:
                signatures.append(signature)
            channels[channel] = signatures.index(signature)
        payload = tuple(any(isinstance(r, tuple) and r[-1] == "payload" for r in s) for s in signatures)
        out[key] = ConsumerGroups(channels=channels, payload=payload)
    return out
