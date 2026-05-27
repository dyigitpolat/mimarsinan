"""Codegen value types for nevresim chip models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


class SpikeSource:
    def __init__(self, core, neuron, is_input = False, is_off = False, is_always_on = False):
        self.is_input_: bool = is_input
        self.is_off_: bool = is_off
        self.is_always_on_: bool = is_always_on
        self.core_: int = core
        self.neuron_: int = neuron

    def get_string(self) -> str:
        if(self.is_off_):
            return "Src{off,0}"
        
        if(self.is_input_):
            return "Src{in," + str(self.neuron_) + "}"
        
        if(self.is_always_on_):
            return "Src{on,0}"
        
        return "Src{" + str(self.core_) + "," + str(self.neuron_) + "}"


@dataclass
class CodegenSpan:
    """A contiguous range of axon sources from a single origin."""
    core_str: str
    start: int
    count: int

    def get_string(self) -> str:
        return f"Span{{{self.core_str}, {self.start}, {self.count}}}"


def compress_sources_to_spans(sources: List[SpikeSource]) -> List[CodegenSpan]:
    """Run-length encode SpikeSource list into contiguous CodegenSpans."""
    if not sources:
        return []

    spans: List[CodegenSpan] = []
    i = 0
    n = len(sources)

    while i < n:
        src = sources[i]

        if src.is_off_:
            core_str, start = "off", 0
        elif src.is_input_:
            core_str, start = "in", int(src.neuron_)
        elif src.is_always_on_:
            core_str, start = "on", 0
        else:
            core_str, start = str(int(src.core_)), int(src.neuron_)

        count = 1
        prev_neuron = start
        j = i + 1
        while j < n:
            nxt = sources[j]
            if src.is_off_ and nxt.is_off_:
                pass
            elif src.is_always_on_ and nxt.is_always_on_:
                pass
            elif (src.is_input_ and nxt.is_input_
                  and int(nxt.neuron_) == prev_neuron + 1):
                prev_neuron = int(nxt.neuron_)
            elif (not src.is_off_ and not src.is_input_ and not src.is_always_on_
                  and not nxt.is_off_ and not nxt.is_input_ and not nxt.is_always_on_
                  and int(nxt.core_) == int(src.core_)
                  and int(nxt.neuron_) == prev_neuron + 1):
                prev_neuron = int(nxt.neuron_)
            else:
                break
            count += 1
            j += 1

        spans.append(CodegenSpan(core_str=core_str, start=start, count=count))
        i += count

    return spans


class Connection:
    def __init__(self, axon_sources_list):
        self.axon_sources: list[SpikeSource] = axon_sources_list
        self._spans: list[CodegenSpan] | None = None

    def get_spans(self) -> list[CodegenSpan]:
        if self._spans is None:
            self._spans = compress_sources_to_spans(self.axon_sources)
        return self._spans

    def get_string(self, idx) -> str:
        spans = self.get_spans()
        parts = [
            f"    cons[{idx}].spans_[{i}] = {span.get_string()};\n"
            for i, span in enumerate(spans)
        ]
        parts.append(f"    cons[{idx}].span_count_ = {len(spans)};\n")
        return ''.join(parts)


class Neuron:
    def __init__(self, weights_list, thresh = 1.0, bias = 0.0):
        self.weights: list[float] = weights_list
        self.thresh: float = thresh
        self.bias: float = bias

    def get_string(self) -> str:
        return ','.join(str(w) for w in self.weights)


class Core:
    def __init__(self, neurons_list, latency):
        self.latency: int = latency
        self.neurons: list[Neuron] = neurons_list

    def get_string(self, idx) -> str:
        parts = [
            f"    neurons[{i}] = Neu{{{{ {neuron.get_string()} }}}};\n"
            for i, neuron in enumerate(self.neurons)
        ]
        parts.append(f"    cores[{idx}] = Core{{neurons, {self.latency}}};\n")
        return ''.join(parts)
