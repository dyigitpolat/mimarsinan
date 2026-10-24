"""Canonical digests of value programs (IR, packed cores, outputs), for the Wave 2 grid declarations' identity test."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.ir import NeuralCore
from unit.chip_simulation.value_run.integer_programs import INSTANCES

IDENTITY_INSTANCES = ("mlp", "lenet5", "residual_block", "conv1d", "pruned_bank_conv")


def _feed(h, value) -> None:
    if value is None:
        h.update(b"none;")
    elif isinstance(value, np.ndarray):
        h.update(f"nd{value.dtype.str}{value.shape};".encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, torch.Tensor):
        _feed(h, value.detach().cpu().numpy())
    elif isinstance(value, float):
        h.update(f"f{value.hex()};".encode())
    else:
        h.update(f"{type(value).__name__}:{value!r};".encode())


def _sources(sources) -> list:
    return [(int(s.node_id), int(s.index)) for s in np.asarray(sources).flatten()]


def _ir_digest(ir) -> str:
    h = hashlib.sha256()
    for node in ir.nodes:
        _feed(h, f"{type(node).__name__}|{node.id}|{node.name}")
        _feed(h, repr(_sources(node.input_sources)))
        if isinstance(node, NeuralCore):
            _feed(h, np.asarray(node.get_core_matrix(ir)))
            _feed(h, float(node.threshold))
            _feed(h, node.hardware_bias)
            _feed(h, repr(node.boundary_grid))
            _feed(h, node.perceptron_index)
            law = getattr(node, "readout_law", None)
            _feed(h, None if law is None else law.digest())
    _feed(h, repr(_sources(ir.output_sources)))
    return h.hexdigest()[:16]


def _packed_digest(mapping) -> str:
    h = hashlib.sha256()
    for stage in mapping.stages:
        _feed(h, stage.kind)
        if stage.kind != "neural":
            continue
        for core in stage.hard_core_mapping.cores:
            _feed(h, np.asarray(core.get_core_matrix()))
            _feed(h, float(core.threshold))
            _feed(h, core.hardware_bias)
            _feed(h, repr(core.boundary_grid))
            law = getattr(core, "readout_law", None)
            _feed(h, None if law is None else law.digest())
            _feed(h, repr([(s.is_input_, s.is_off_, s.is_always_on_, int(s.core_), int(s.neuron_))
                           for s in core.axon_sources]))
    return h.hexdigest()[:16]


def _array_digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()[:16]


def program_digest(name: str) -> dict:
    build, geometry = INSTANCES[name]
    program = build()
    packed = program.packed_mapping(**geometry)
    x = torch.randint(-3, 4, (8, *program.input_shape), generator=torch.Generator().manual_seed(3)).double()
    with torch.no_grad():
        identity = program.identity()(x).numpy()
        packed_out = ValueHybridCoreFlow(packed, dtype=torch.float64)(x).numpy()
        nf = program.nf()(x).detach().numpy()
    codes, steps = run_golden(packed, x.numpy())
    return {"ir": _ir_digest(program.ir), "packed": _packed_digest(packed), "identity_out": _array_digest(identity),
            "packed_out": _array_digest(packed_out), "nf_out": _array_digest(nf),
            "golden": _array_digest(codes) + _array_digest(steps)}
