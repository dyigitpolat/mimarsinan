"""[U-TGAUGE4] The timing walk's cycle loop by the placement rule: every read of an element takes every quantity the
deployed program holds per producer from that producer, never from a layout or from the core that reads it.

[U-TGAUGE5, carried by U-PFOLD] The window, corrected (``mapping/latency/cycle_windows.py``): the deployed timing
programs of record run the cascaded schedule, a per-cycle cascade. Each core steps from its latency ``L``, which
``ChipLatency`` sets (the longest path over the stage's core reads, then the alignment that raises each core to its
deepest reader's latency minus one), and fills its axons at cycle ``L + k`` from each producer core's buffer of the
previous cycle: that producer's relative cycle ``k + g - 1`` at a latency gap ``g``. The mapping makes ``g = 1`` on
every live in-segment edge (the depth-balancing relays) or refuses the program by name, and the per-cycle executor
refuses a stage holding another gap.
The analytical, quantized and synchronized schedules run each core in dependency order over complete values
(``chip_simulation/ttfs/ttfs_segment.py``): no window. Per producer the deployed program holds the gauge its buffer
holds each element at (``output_decode._held_gauge``), its window (each core's ``latency``), and, at a reading core,
the fold that core's weights apply to each input (``per_input_scales``,
``PerceptronTransformer.get_effective_weight``). The walk reads each element so:

- a core at its relative cycle ``k`` reads each in-segment producer's spike of that producer's own relative cycle
  ``k``, through any layout between them, which is the deployed read at ``g = 1`` (a producer more than one hop
  shallower was read at the reader's previous cycle, late by the depth gap, its early spikes lost);
- a layout read outside the segment (an output, a host op's operand) is its element movement applied to its producers'
  decoded values, each over its own window at its own gauge (``region_read``; it was accumulated over the layout's
  own window, from its deepest producer's depth);
- a core folds each input at its fold over its own input scale (``per_input_scales / input_scale``), and an element
  whose producer lies outside the segment enters through a boundary encode at that producer's own gauge (every input
  was read at the core's one input scale, the mean of its sources). Where a core's fold is its one input scale for
  every input, the closed form stays (that scale, and the encode at it), which places every element alike.
"""

from __future__ import annotations

from typing import Any

import torch

from mimarsinan.mapping.map_model_to_ir import as_gauge_like, one_value, own_gauge_of
from mimarsinan.mapping.support.gauge_channels import same_gauge
from mimarsinan.mapping.mappers.source_gauges import read_source_gauges
from mimarsinan.mapping.support.gauge_placement import GaugePlacementError
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.spiking.region_read import region_producers, region_read
from mimarsinan.spiking.segment_partition import is_encoding_perceptron, perceptron_of
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.perceptron.position_fold import holds_positions, input_fold_view

__all__ = ["input_fold", "run_genuine_cycles"]


def input_fold(perceptron: Any) -> Any:
    """The deployed fold of ``perceptron``'s inputs over the one input scale its timing activation applies
    (``per_input_scales / input_scale``, viewed on the weight's input axis); None where the fold is that one scale for
    every input (``same_gauge``: one gauge read through two float32 tables), the closed form. The input scale is the
    boundary walk's, the mean of the reader's sources, so even a fold of one value can differ from it (a gather that
    holds one producer of a join)."""
    fold = getattr(perceptron, "per_input_scales", None)
    timing = [m for m in perceptron.modules() if isinstance(m, TTFSActivation)]
    if fold is None or not timing:  # no stamp, or no timing activation whose one input scale a fold corrects
        return None
    scale = torch.as_tensor(timing[0].input_scale).detach().reshape(-1)
    if not one_value(scale):
        raise GaugePlacementError(
            f"{getattr(perceptron, 'name', None)!r}: its timing activation holds {scale.numel()} input scales, so no "
            f"per-input fold can be read over one")
    flat = torch.as_tensor(fold).detach().reshape(-1)
    if one_value(flat) and same_gauge(float(flat[0]), float(scale[0])):
        return None
    if holds_positions(perceptron):  # [U-PFOLD] each element at its position's fold, read at the input's shape
        return lambda like, s=max(float(scale[0]), 1e-12): input_fold_view(perceptron, like) / s
    view = PerceptronTransformer().input_scale(perceptron)
    return torch.as_tensor(view).detach().to(torch.float64) / max(float(scale[0]), 1e-12)


def _lags(driver: Any, perceptrons: list, seg_set: set, depth: dict) -> dict:
    """Per in-segment producer, the most cycles a reading core of its segment trails it by (its spikes kept so long)."""
    lag = {p: 0 for p in perceptrons}
    for reader in perceptrons:
        stack = list(driver._deps.get(reader, []))
        seen: set = set()
        while stack:
            n = stack.pop()
            if n not in seg_set or id(n) in seen:
                continue
            seen.add(id(n))
            if perceptron_of(n) is not None:
                lag[n] = max(lag[n], depth[reader] - depth[n])
            else:
                stack.extend(driver._deps.get(n, []))
    return lag


def _forward(node: Any, deps: list, inputs: list, x: torch.Tensor) -> torch.Tensor:
    if not deps:
        return node.forward(x)
    return node.forward(inputs[0] if len(inputs) == 1 else tuple(inputs))


def run_genuine_cycles(policy: Any, driver: Any, seg_nodes: list, values: dict, x: torch.Tensor):
    """``TtfsSegmentPolicy``'s genuine segment run (the module docstring's reads)."""
    T = driver.T
    seg_set = set(seg_nodes)
    segment = driver._seg_of.get(seg_nodes[0])
    ext = driver.external_consumed(seg_nodes)
    depth = policy.segment_depths(driver, seg_nodes)
    zeros = policy._segment_output_zeros(driver, seg_nodes, values, x)
    perceptrons = [n for n in seg_nodes if perceptron_of(n) is not None]
    folds = {n: input_fold(perceptron_of(n)) for n in perceptrons if not is_encoding_perceptron(n)}
    lag = _lags(driver, perceptrons, seg_set, depth)
    spikes: dict = {n: {} for n in perceptrons}
    trains: dict = {}
    gauges: dict = {}

    def producer_gauge(src: Any) -> Any:
        """``src``'s own gauge per element of its value (the source walk's placed table), read once per run."""
        if "table" not in gauges:
            gauges["table"] = read_source_gauges(driver.repr)
        value = values[src]
        return as_gauge_like(own_gauge_of(gauges["table"].get(src), src, tuple(value.shape[1:])), value)

    def boundary_spikes(src: Any, k: int, reader: Any, by_producer: bool) -> torch.Tensor:
        if by_producer:  # the producer's own buffer: each element at its own gauge, whoever reads it
            key = (src, None)
            if key not in trains:
                rate = (values[src] / torch.as_tensor(producer_gauge(src), dtype=values[src].dtype,
                                                      device=values[src].device).clamp(min=1e-12)).clamp(0.0, 1.0)
                trains[key] = policy._single_spike_train(rate, T)
            return trains[key][k]
        scale = policy.consumer_wire_scale(driver, reader)
        key = (src, float(scale.float().mean()) if isinstance(scale, torch.Tensor) else float(scale))
        if key not in trains:
            trains[key] = policy._boundary_single_spike_train(values[src], scale, T)
        return trains[key][k]

    def read_at(src: Any, k: int, reader: Any, by_producer: bool, memo: dict) -> torch.Tensor:
        """``src``'s elements as a core at its relative cycle ``k`` reads them (``reader``: the node reading ``src``)."""
        if src not in seg_set:
            if is_encoding_perceptron(reader):
                return values[src]
            return boundary_spikes(src, k, reader, by_producer)
        if perceptron_of(src) is not None:
            return spikes[src].get(k, zeros[src])
        key = (src, k, by_producer)
        if key not in memo:
            deps = driver._deps.get(src, [])
            memo[key] = _forward(src, deps, [read_at(d, k, src, by_producer, memo) for d in deps], x)
        return memo[key]

    watch = set(n for n in ext if perceptron_of(n) is not None)
    for n in ext:
        if perceptron_of(n) is None:
            watch.update(region_producers(driver, n))
    if policy.node_value_recorder is not None:
        watch.update(perceptrons)
    latched: dict = {}
    accum: dict = {}
    for t in range(T + max(depth.values(), default=0)):
        memo: dict = {}
        for n in perceptrons:
            k = t - depth[n]
            if not 0 <= k < T:
                continue
            deps = driver._deps.get(n, [])
            fold = folds.get(n)
            inputs = [read_at(d, k, n, fold is not None, memo) for d in deps]
            if fold is not None:
                if len(inputs) != 1:
                    raise GaugePlacementError(
                        f"{type(n).__name__} reads {len(inputs)} inputs at a per-input fold: no one input axis holds it")
                view = torch.as_tensor(fold(inputs[0]) if callable(fold) else fold)
                inputs = [inputs[0] * view.to(device=inputs[0].device, dtype=inputs[0].dtype)]
            out = _forward(n, deps, inputs, x)
            spikes[n][k] = out
            if n in watch:
                latched[n] = out if n not in latched else torch.maximum(latched[n], out)
                accum[n] = latched[n] if n not in accum else accum[n] + latched[n]
        for p in perceptrons:  # a spike no reading core will read again is dropped
            spikes[p].pop(t - depth[p] - lag[p], None)

    decoded: dict = {}

    def producer_value(n: Any) -> torch.Tensor:
        """``n``'s values as the deployed program holds them: an outside node's own, a perceptron of the segment over
        its own window at its own gauge, a layout of the segment its producers' moved (``region_read``)."""
        if driver._seg_of.get(n) is not segment:
            return values[n]
        if n not in decoded:
            decoded[n] = (policy.decode(driver, n, accum[n] / float(T)) if perceptron_of(n) is not None
                          else region_read(driver, n, producer_value))
        return decoded[n]

    for n in ext:
        values[n] = producer_value(n)
    if policy.node_value_recorder is not None:
        for n in perceptrons:
            policy.node_value_recorder[n] = producer_value(n).detach()
    if driver._output in seg_set:
        return values[driver._output]
    return None
