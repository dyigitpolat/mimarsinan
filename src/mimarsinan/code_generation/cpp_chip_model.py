import json
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
    core_str: str   # C++ expression: "in", "off", "on", or a core id string
    start: int
    count: int

    def get_string(self) -> str:
        return f"Span{{{self.core_str}, {self.start}, {self.count}}}"


def _compress_sources_to_spans(sources: List[SpikeSource]) -> List[CodegenSpan]:
    """
    Run-length encode a list of SpikeSource into contiguous CodegenSpans.
    
    Consecutive sources from the same core with sequential neuron indices
    are merged.  For off/on sources, any consecutive run is merged regardless
    of neuron indices.
    """
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
            self._spans = _compress_sources_to_spans(self.axon_sources)
        return self._spans

    def get_string(self, idx) -> str:
        spans = self.get_spans()
        result = ""
        for i, span in enumerate(spans):
            result += "    cons[{}].spans_[{}] = {};\n".format(idx, i, span.get_string())
        result += "    cons[{}].span_count_ = {};\n".format(idx, len(spans))
        return result

class Neuron:
    def __init__(self, weights_list, thresh = 1.0, bias = 0.0):
        self.weights: list[float] = weights_list
        self.thresh: float = thresh
        self.bias: float = bias

    def get_string(self) -> str:
        result = ""
        for i, w in enumerate(self.weights):
            result += str(w)
            if(i < len(self.weights) - 1):
                result += ","
        
        return result

class Core:
    def __init__(self, neurons_list, latency):
        self.latency: int = latency
        self.neurons: list[Neuron] = neurons_list

    def get_string(self, idx) -> str:
        result = ""
        for i, neuron in enumerate(self.neurons):
            result += "    neurons[{}] = Neu{{{{ {} }}}};\n".format(i,neuron.get_string())
        
        result += "    cores[{}] = Core{{neurons, {}}};\n".format(idx, self.latency)
        return result

class ChipModel:
    def __init__(
        self, axons = 0, neurons = 0, cores = 0, inputs = 0, outputs = 0, leak = 0,
        connections_list = [], output_list = [], cores_list = [], weight_type = float):

        self.axon_count = axons
        self.neuron_count = neurons
        self.core_count = cores
        self.input_size = inputs
        self.output_size = outputs
        self.leak = leak
        self.weight_type = weight_type

        self.connections: list[Connection] = connections_list
        self.output_buffer: list[SpikeSource] = output_list
        self.cores: list[Core] = cores_list

    def _max_spans_per_core(self) -> int:
        if not self.connections:
            return 1
        return max(len(con.get_spans()) for con in self.connections)

    def get_string(self) -> str:
        max_spans = self._max_spans_per_core()

        result = "" 

        result += """
#pragma once

#include "common/constants.hpp"
#include "simulator/chip/neuron.hpp"
#include "simulator/chip/core.hpp"
#include "simulator/chip/chip.hpp"

namespace nevresim
{

template <typename Con, typename Span, size_t core_count, size_t in, size_t off, size_t on>
consteval auto generate_connections()
{
    std::array<Con, core_count> cons; 
        
"""

        for i, con in enumerate(self.connections):
            result += con.get_string(i)
        result += """
    return cons;
}

template <typename Src, size_t output_size>
consteval auto generate_outputs()
{
    std::array<Src, output_size> outs;
    
"""

        for i, out in enumerate(self.output_buffer):
            result += "    outs[{}] = {};\n".format(i, out.get_string())
        result += """
    return outs;
}}

template <typename ComputePolicy, typename WeightType>
consteval auto generate_chip()
{{
    using weight_t = WeightType;
    constexpr std::size_t axon_count{{{0}}};
    constexpr std::size_t neuron_count{{{1}}};
    constexpr std::size_t core_count{{{2}}};
    constexpr std::size_t input_size{{{3}}};
    constexpr std::size_t output_size{{{4}}};
    constexpr std::size_t max_spans_per_core{{{6}}};
    constexpr MembraneLeak<weight_t> leak{{{5}}};

    using Cfg = nevresim::ChipConfiguration<
        weight_t,
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        max_spans_per_core,
        leak
    >;

    using Map = nevresim::Mapping<Cfg>;
    using Src = nevresim::SpikeSource;
    using Span = nevresim::SourceSpan;
    using Con = nevresim::CoreSpanConnection<max_spans_per_core>;

    constexpr nevresim::core_id_t in = nevresim::k_input_buffer_id;
    constexpr nevresim::core_id_t off = nevresim::k_no_connection;
    constexpr nevresim::core_id_t on = nevresim::k_always_on;

    using Chip = nevresim::Chip<
        Cfg, 
        Map{{
            generate_connections<Con, Span, core_count, in, off, on>(),
            generate_outputs<Src, output_size>()}}, 
        ComputePolicy>;

    constexpr Chip chip{{}};
    return chip;
}}

}} // namespace nevresim """.format(
            self.axon_count,
            self.neuron_count,
            self.core_count,
            self.input_size,
            self.output_size,
            self.leak,
            max_spans,
        )
        return result

    def get_weights_string(self):
        result = "";
        for core in self.cores:
            result += str(core.latency) + ' '
            for neuron in core.neurons:
                result += str(self.weight_type(neuron.thresh)) + ' '
                result += str(self.weight_type(neuron.bias)) + ' '
                for w in neuron.weights:
                    result += str(self.weight_type(w)) + ' '
        return result

    def get_chip_json(self):
        result = { 
            "axon_count": int(self.axon_count), 
            "neuron_count": int(self.neuron_count), 
            "core_count": int(self.core_count), 
            "input_size": int(self.input_size), 
            "output_size": int(self.output_size), 
            "leak": self.weight_type(self.leak), 
            "core_parameters": [],
            "core_latencies": [],
            "core_connections": [],
            "output_buffer": []}

        for core in self.cores:
            core_params = []
            for neuron in core.neurons:
                core_params.append({
                    "threshold": self.weight_type(neuron.thresh),
                    "bias": self.weight_type(neuron.bias),
                    "weights": [self.weight_type(w) for w in neuron.weights]
                })
            result["core_parameters"].append(core_params)

        for core in self.cores:
            result["core_latencies"].append(core.latency)
        
        for con in self.connections:
            result["core_connections"].append([])
            cur = result["core_connections"][-1]
            for src in con.axon_sources:
                cur.append({
                    "is_input": bool(src.is_input_),
                    "is_off": bool(src.is_off_),
                    "is_on": bool(src.is_always_on_),
                    "source_core": int(src.core_),
                    "source_neuron": int(src.neuron_),
                })
        
        for out in self.output_buffer:
            result["output_buffer"].append({
                "is_input": bool(out.is_input_),
                "is_off": bool(out.is_off_),
                "is_on": bool(out.is_always_on_),
                "source_core": int(out.core_),
                "source_neuron": int(out.neuron_)
            })

        return json.dumps(result)

    def load_from_json(self, json_string):
        data = json.loads(json_string)
        self.axon_count = data["axon_count"]
        self.neuron_count = data["neuron_count"]
        self.core_count = data["core_count"]
        self.input_size = data["input_size"]
        self.output_size = data["output_size"]
        self.leak = data["leak"]

        self.connections = []
        for con in data["core_connections"]:
            src_list = []
            for src in con:
                src_list.append(
                    SpikeSource( 
                        src["source_core"], 
                        src["source_neuron"],
                        src["is_input"], 
                        src["is_off"],
                        src["is_on"]))

            self.connections.append(Connection(src_list))

        self.cores = []
        for core, latency in zip(data["core_parameters"], data["core_latencies"]):
            neurons = []
            for neuron in core:
                neurons.append(Neuron(neuron["weights"], neuron["threshold"], neuron["bias"]))
            self.cores.append(Core(neurons, latency))

        self.output_buffer = []
        for out in data["output_buffer"]:
            self.output_buffer.append(
                SpikeSource(
                    out["source_core"], 
                    out["source_neuron"],
                    out["is_input"], 
                    out["is_off"],
                    out["is_on"]))
