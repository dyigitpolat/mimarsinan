"""[U-LIF2 W9] Integer programs for the tests of integrate then emit's executors.

An X3a instance (integer weights, biases and thresholds, so every charge is exact) mapped as the toolchain maps it:
one fused execution stage per segment, or per-hop re-timed level stages, whole or cut into passes. Each neural execution
stage's input train is the one the flow's own encode hands it; the three executors of integrate then emit (the gated
packed executor, the gated reference loop and the closed form) are read per core and neuron on that train. Every flow
here states integrate then emit explicitly (its two-window flag), never through the declared set."""

from __future__ import annotations

import importlib
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from mimarsinan.certification.randomized import executors as ex
from mimarsinan.certification.randomized import program as prog
from mimarsinan.certification.randomized.instance import INPUT, NEURAL, SYNCHRONIZED_LIF, Instance, Layer, Network
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE
from mimarsinan.models.spiking.cycle_policy import cycle_neuron_policy
from mimarsinan.models.spiking.hybrid.executors.sync_counts import run_neural_segment_counts
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

GATED = "mimarsinan.models.spiking.hybrid.executors.gated_window"
GATED_REFERENCE = "mimarsinan.models.spiking.hybrid.executors.gated_reference"

#: Layer specs (name, sources, width): a chain of D levels, the skip shape (a -> b -> c, a -> c), and a stage whose
#: depth-2 core also reads the stage input (a, b(a), c(b, input), d(c, a)).
SKIP = (("a", (INPUT,), 6), ("b", ("a",), 6), ("c", ("a", "b"), 6))
INPUT_AT_DEPTH = (("a", (INPUT,), 5), ("b", ("a",), 5), ("c", ("b", INPUT), 5), ("d", ("c", "a"), 5))


def chain(depth: int, width: int = 6) -> Tuple[Tuple[str, Tuple[str, ...], int], ...]:
    names = [f"l{i}" for i in range(depth)]
    return tuple((n, ((INPUT,) if i == 0 else (names[i - 1],)), width) for i, n in enumerate(names))


def new(module: str, name: str) -> Any:
    """A name of the build, imported where a test reads it (absent before the build: the test is red on its own)."""
    return getattr(importlib.import_module(module), name)


def layered_instance(spec, *, timesteps: int, seed: int, input_width: int = 6, batch: int = 8,
                     hardware_bias: bool = True, weight=(-2, 3), bias=(-1, 2), theta=(2, 5)) -> Instance:
    """A seeded integer network of ``spec``'s layers (every layer an output, so every neuron is observed)."""
    rng = random.Random(seed)
    widths = {INPUT: input_width}
    layers = []
    for name, sources, width in spec:
        fan_in = sum(widths[s] for s in sources)
        layers.append(Layer(name=name, kind=NEURAL, sources=tuple(sources), width=width,
                            weights=tuple(tuple(rng.randint(*weight) for _ in range(fan_in)) for _ in range(width)),
                            bias=tuple(rng.randint(*bias) for _ in range(width)), theta=rng.randint(*theta)))
        widths[name] = width
    network = Network(input_width, tuple(layers), tuple(name for name, _sources, _width in spec))
    inputs = tuple(tuple(rng.randint(0, timesteps) for _ in range(input_width)) for _ in range(batch))
    return Instance(seed=seed, semantics=SYNCHRONIZED_LIF, timesteps=timesteps, hardware_bias=hardware_bias,
                    network=network, inputs=inputs)


def integer_ir(instance: Instance, *, max_axons: Optional[int] = None, max_neurons: Optional[int] = None):
    return prog.integer_ir(prog.mapper_graph(instance), max_axons=max_axons, max_neurons=max_neurons,
                           allow_coalescing=False, hardware_bias=instance.hardware_bias)


def identity_program(instance: Instance, *, retimed: bool):
    """The identity program: one fused execution stage per segment, or per-hop re-timed level stages."""
    return build_identity_hybrid_mapping(ir_graph=integer_ir(instance), retimed_level_stages=retimed)


def ite_flow(instance: Instance, mapping, *, lif_membrane_init: float = 0.0, phase_dither: bool = False,
             firing_mode: str = "Default", soma_law: SomaLaw = DEFAULT_SOMA_LAW) -> SpikingHybridCoreFlow:
    """A flow that integrates then emits (its two-window flag states the point explicitly)."""
    return SpikingHybridCoreFlow((instance.network.input_width,), mapping, simulation_length=instance.timesteps,
                                 firing_mode=firing_mode, thresholding_mode="<=", pass_transfer=COLLAPSE,
                                 phase_dither=phase_dither, spiking_mode="lif", cycle_accurate_lif_forward=True,
                                 lif_execution_synchronized=True, lif_membrane_init=lif_membrane_init,
                                 soma_law=soma_law)


def inputs(instance: Instance) -> torch.Tensor:
    return ex.input_tensor(instance)


def stage_runs(flow, x: torch.Tensor) -> List[Tuple[Any, torch.Tensor]]:
    """Run the flow once; every neural execution stage with the input train its own encode handed it."""
    seen: List[Tuple[Any, torch.Tensor]] = []
    original = flow._run_neural_segment_rate

    def spy(stage, *, input_spike_train, **kw):
        seen.append((stage, input_spike_train.detach().clone()))
        return original(stage, input_spike_train=input_spike_train, **kw)

    flow._run_neural_segment_rate = spy
    try:
        with torch.no_grad():
            flow(x)
    finally:
        del flow._run_neural_segment_rate
    return seen


def _policy(flow):
    return cycle_neuron_policy(flow.spiking_mode, flow.ttfs_cycle_schedule, flow.firing_mode,
                               integer_lattice=bool(getattr(flow, "membrane_integer_lattice", False)),
                               soma_law=flow.soma_law)


def window_counts(flow, stage, train: torch.Tensor, executor: str,
                  soma_law: Optional[SomaLaw] = None) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """``({core: (B, n) window counts}, segment output counts)`` of one execution stage under ``executor``:
    ``"packed"`` (the gated packed executor), ``"reference"`` (the gated reference loop) or ``"closed_form"``."""
    seg = flow._get_segment_tensors(stage, train.device)
    recorder: Dict[int, torch.Tensor] = {}
    common = dict(seg=seg, T=int(flow.simulation_length), batch_size=int(train.shape[1]), device=train.device)
    with torch.no_grad():
        if executor == "closed_form":
            out = run_neural_segment_counts(flow, train, soma_law=soma_law or flow.soma_law,
                                            neuron_count_recorder=recorder, **common)
        else:
            run = new(GATED, "run_neural_segment_gated") if executor == "packed" else new(
                GATED_REFERENCE, "run_neural_segment_gated_reference")
            out = run(flow, train, stage=stage, policy=_policy(flow), neuron_count_recorder=recorder, **common)
    return recorder, out


def differing_windows(got: Dict[int, torch.Tensor], want: Dict[int, torch.Tensor]) -> Tuple[int, int]:
    """``(neuron windows compared, windows that differ)`` over every core both record (atol 0)."""
    assert sorted(got) == sorted(want) and got, (sorted(got), sorted(want))
    windows = differ = 0
    for core in want:
        a, b = got[core], want[core]
        assert a.shape == b.shape, (core, a.shape, b.shape)
        windows += int(b.numel())
        differ += int((a != b).sum())
    return windows, differ


def neural_stages(mapping) -> list:
    return [s for s in mapping.stages if s.kind == "neural"]


def run_point(instance: Instance, mapping, *, executor_flow=None, **flow_kwargs) -> torch.Tensor:
    flow = executor_flow or ite_flow(instance, mapping, **flow_kwargs)
    with torch.no_grad():
        return flow(inputs(instance))


def soma(**keys: Any) -> SomaLaw:
    return SomaLaw.resolve(dict(keys))


def depth_of(seg) -> int:
    levels = [level for level in new(GATED, "stage_dependency_levels")(seg) if level is not None]
    return 1 + max(levels) if levels else 0


def chains(depths: Sequence[int]) -> List[Tuple[str, Any]]:
    return [(f"chain{d}", chain(d)) for d in depths]
