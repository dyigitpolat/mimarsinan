import json

from mimarsinan.code_generation.cpp_chip_model_types import (
    CodegenSpan,
    Connection,
    Core,
    Neuron,
    SpikeSource,
    compress_sources_to_spans,
)

__all__ = [
    "ChipModel",
    "CodegenSpan",
    "Connection",
    "Core",
    "Neuron",
    "SpikeSource",
    "compress_sources_to_spans",
]


class ChipModel:
    def __init__(
        self, axons = 0, neurons = 0, cores = 0, inputs = 0, outputs = 0, leak = 0,
        connections_list = [], output_list = [], cores_list = [],
        weight_type = float, threshold_type = None):

        self.axon_count = axons
        self.neuron_count = neurons
        self.core_count = cores
        self.input_size = inputs
        self.output_size = outputs
        self.leak = leak
        self.weight_type = weight_type
        self.threshold_type = threshold_type if threshold_type is not None else weight_type

        self.connections: list[Connection] = connections_list
        self.output_buffer: list[SpikeSource] = output_list
        self.cores: list[Core] = cores_list

    def _max_spans_per_core(self) -> int:
        if not self.connections:
            return 1
        return max(len(con.get_spans()) for con in self.connections)

    def get_string(self) -> str:
        max_spans = self._max_spans_per_core()

        parts: list[str] = []

        parts.append("""
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
        
""")

        for i, con in enumerate(self.connections):
            parts.append(con.get_string(i))
        parts.append("""
    return cons;
}

template <typename Src, size_t output_size>
consteval auto generate_outputs()
{
    std::array<Src, output_size> outs;
    
""")

        for i, out in enumerate(self.output_buffer):
            parts.append(f"    outs[{i}] = {out.get_string()};\n")
        parts.append("""
    return outs;
}}

template <typename ComputePolicy, typename WeightType, typename ThresholdType>
consteval auto generate_chip()
{{
    using weight_t = WeightType;
    using threshold_t = ThresholdType;
    constexpr std::size_t axon_count{{{0}}};
    constexpr std::size_t neuron_count{{{1}}};
    constexpr std::size_t core_count{{{2}}};
    constexpr std::size_t input_size{{{3}}};
    constexpr std::size_t output_size{{{4}}};
    constexpr std::size_t max_spans_per_core{{{6}}};
    constexpr MembraneLeak<weight_t> leak{{{5}}};

    using Cfg = nevresim::ChipConfiguration<
        weight_t,
        threshold_t,
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
        ))
        return ''.join(parts)

    def get_weights_string(self):
        wt = self.weight_type
        tt = self.threshold_type
        parts: list[str] = []
        for core in self.cores:
            parts.append(str(core.latency))
            for neuron in core.neurons:
                parts.append(str(tt(neuron.thresh)))
                parts.append(str(wt(neuron.bias)))
                parts.extend(str(wt(w)) for w in neuron.weights)
        return ' '.join(parts) + ' '

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
                    "threshold": self.threshold_type(neuron.thresh),
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
