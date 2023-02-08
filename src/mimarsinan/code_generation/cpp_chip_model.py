import json

class SpikeSource:
    def __init__(self, core, neuron, is_input = False, is_off = False):
        self.is_input_: bool = is_input
        self.is_off_: bool = is_off
        self.core_: int = core
        self.neuron_: int = neuron

    def get_string(self) -> str:
        if(self.is_input_):
            return "Src{in," + str(self.neuron_) + "}"
        
        if(self.is_off_):
            return "Src{off,0}"
        
        return "Src{" + str(self.core_) + "," + str(self.neuron_) + "}"

class Connection:
    def __init__(self, axon_sources_list):
        self.axon_sources: list[SpikeSource] = axon_sources_list

    def get_string(self, idx) -> str:
        result = ""
        for i, source in enumerate(self.axon_sources):
            result += "    cons[{}].sources_[{}] = {};\n" .format(idx, i, source.get_string())
        
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
    def __init__(self, neurons_list):
        self.neurons: list[Neuron] = neurons_list

    def get_string(self, idx) -> str:
        result = ""
        for i, neuron in enumerate(self.neurons):
            result += "    neurons[{}] = Neu{{{{ {} }}}};\n".format(i,neuron.get_string())
        
        result += "    cores[{}] = Core{{neurons}};\n".format(idx)
        return result

class ChipModel:
    def __init__(
        self, axons = 0, neurons = 0, cores = 0, inputs = 0, outputs = 0, leak = 0,
        connections_list = [], output_list = [], cores_list = []):

        self.axon_count = axons
        self.neuron_count = neurons
        self.core_count = cores
        self.input_size = inputs
        self.output_size = outputs
        self.leak = leak

        self.connections: list[Connection] = connections_list
        self.output_buffer: list[SpikeSource] = output_list
        self.cores: list[Core] = cores_list

    def get_string(self) -> str:
        result = "" 

        result += """
#pragma once

#include "common/constants.hpp"
#include "simulator/chip/neuron.hpp"
#include "simulator/chip/core.hpp"
#include "simulator/chip/chip.hpp"

namespace nevresim
{

template <typename Con, typename Src, size_t core_count, size_t in, size_t off>
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

template <typename ComputePolicy>
consteval auto generate_chip()
{{
    constexpr std::size_t axon_count{{{0}}};
    constexpr std::size_t neuron_count{{{1}}};
    constexpr std::size_t core_count{{{2}}};
    constexpr std::size_t input_size{{{3}}};
    constexpr std::size_t output_size{{{4}}};
    constexpr MembraneLeak<weight_t> leak{{{5}}};

    using Cfg = nevresim::ChipConfiguration<
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        leak
    >;

    using Map = nevresim::Mapping<Cfg>;
    using Src = nevresim::SpikeSource;
    using Con = nevresim::CoreConnection<axon_count>;

    constexpr nevresim::core_id_t in = nevresim::k_input_buffer_id;
    constexpr nevresim::core_id_t off = nevresim::k_no_connection;

    using Chip = nevresim::Chip<
        Cfg, 
        Map{{
            generate_connections<Con, Src, core_count, in, off>(),
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
            self.leak
        )
        return result

    def get_weights_string(self):
        result = "";
        for core in self.cores:
            for neuron in core.neurons:
                result += str(neuron.thresh) + ' '
                result += str(neuron.bias) + ' '
                for w in neuron.weights:
                    result += str(w) + ' '
        return result

    def get_chip_json(self):
        result = { 
            "axon_count": self.axon_count, 
            "neuron_count": self.neuron_count, 
            "core_count": self.core_count, 
            "input_size": self.input_size, 
            "output_size": self.output_size, 
            "leak": self.leak, 
            "core_parameters": [],
            "core_connections": [],
            "output_buffer": []}

        for core in self.cores:
            core_params = []
            for neuron in core.neurons:
                core_params.append({
                    "threshold": neuron.thresh,
                    "bias": neuron.bias,
                    "weights": neuron.weights
                })
            result["core_parameters"].append(core_params)
        
        for con in self.connections:
            result["core_connections"].append([])
            cur = result["core_connections"][-1]
            for src in con.axon_sources:
                cur.append({
                    "is_input": src.is_input_,
                    "is_off": src.is_off_,
                    "source_core": src.core_,
                    "source_neuron": src.neuron_,
                })
        
        for out in self.output_buffer:
            result["output_buffer"].append({
                "is_input": out.is_input_,
                "is_off": out.is_off_,
                "source_core": out.core_,
                "source_neuron": out.neuron_
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
                        src["is_off"]))

            self.connections.append(Connection(src_list))

        self.cores = []
        for core in data["core_parameters"]:
            neurons = []
            for neuron in core:
                neurons.append(Neuron(neuron["weights"], neuron["threshold"], neuron["bias"]))
            self.cores.append(Core(neurons))

        self.output_buffer = []
        for out in data["output_buffer"]:
            self.output_buffer.append(
                SpikeSource(
                    out["source_core"], 
                    out["source_neuron"],
                    out["is_input"], 
                    out["is_off"]))

        



