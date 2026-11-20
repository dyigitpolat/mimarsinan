#pragma once

#include "simulator/chip/neuron.hpp"
#include "simulator/chip/core.hpp"
#include "simulator/chip/chip.hpp"
#include "simulator/spike_generation/deterministic_spike_generator.hpp"
#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/execution/spiking_execution.hpp"

#include "test_util.hpp"

#include <sstream>

namespace nevresim {
namespace tests {

template <typename Con, typename Span, size_t core_count, size_t in_id>
consteval auto generate_connections_single_core_2x2()
{
    std::array<Con, core_count> cons;

    cons[0].spans_[0] = Span{in_id, 0, 2};
    cons[0].span_count_ = 1;

    return cons;
}

template <typename Src, size_t output_size>
consteval auto generate_outputs_single_core_2x2()
{
    std::array<Src, output_size> outs;    

    outs[0] = Src{0,0};
    outs[1] = Src{0,1};

    return outs;
}

constexpr bool test_single_core()
{
    using weight_t = double;
    constexpr std::size_t axon_count{2};
    constexpr std::size_t neuron_count{2};
    constexpr std::size_t core_count{1};
    constexpr std::size_t input_size{2};
    constexpr std::size_t output_size{2};
    constexpr std::size_t max_spans{1};
    constexpr MembraneLeak<weight_t> leak{0};

    using Src = SpikeSource;
    using Span = SourceSpan;
    using Con = CoreSpanConnection<max_spans>;

    auto chip = generate_test_chip<
        weight_t,
        generate_connections_single_core_2x2<Con, Span, core_count, in>(),
        generate_outputs_single_core_2x2<Src, output_size>(),
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        max_spans,
        leak,
        SpikingCompute<>> ();

    using ChipW = ChipWeights<chip.config_, weight_t, weight_t>;
    using CoreW = CoreWeights<chip.config_, weight_t, weight_t>;
    using NeurW = NeuronWeights<chip.config_, weight_t, weight_t>;
    using Ws = std::array<Weight<weight_t>, axon_count>;
    
    ChipW weights{{
        CoreW{{
            NeurW{Ws{2.0,-2.0}, {}, 1.0}, NeurW{Ws{9.0,-5.0}, 3.0, -2.0}
        }}
    }};
    chip.load_weights(weights);

    std::array<raw_input_t, 2> input{1.0, 1.0};
        
    auto buffer = chip.execute<
        SpikingExecution<3, 1, 2, DeterministicSpikeGenerator, weight_t>>(input);

    chip.reset();
    return buffer == std::array<weight_t, 2>{4, 1};
}

} // namespace nevresim::tests
} // namespace nevresim
