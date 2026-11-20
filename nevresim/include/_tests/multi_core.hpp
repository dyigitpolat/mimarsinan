#pragma once

#include "simulator/chip/neuron.hpp"
#include "simulator/chip/core.hpp"
#include "simulator/chip/chip.hpp"
#include "simulator/spike_generation/deterministic_spike_generator.hpp"
#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/execution/spiking_execution.hpp"
#include "simulator/chip_utilities.hpp"

#include "test_util.hpp"

#include <sstream>

namespace nevresim {
namespace tests {

// Core 0: input[0], input[1]         -> 1 span
// Core 1: core0.neuron[1], off       -> 2 spans
// Core 2: off, core0.neuron[0]       -> 2 spans
// max_spans = 2
template <typename Con, typename Span, size_t core_count,
          size_t in_id, size_t off_id>
consteval auto generate_connections_3_cores_2x2()
{
    std::array<Con, core_count> cons;

    cons[0].spans_[0] = Span{in_id, 0, 2};
    cons[0].span_count_ = 1;

    cons[1].spans_[0] = Span{0, 1, 1};
    cons[1].spans_[1] = Span{off_id, 0, 1};
    cons[1].span_count_ = 2;

    cons[2].spans_[0] = Span{off_id, 0, 1};
    cons[2].spans_[1] = Span{0, 0, 1};
    cons[2].span_count_ = 2;

    return cons;
}

template <typename Src, size_t output_size>
consteval auto generate_outputs_3_cores_2x2()
{
    std::array<Src, output_size> outs;    

    outs[0] = Src{1,0};
    outs[1] = Src{1,1};
    outs[2] = Src{2,0};
    outs[3] = Src{2,1};

    return outs;
}

constexpr bool test_3_core_2x2()
{
    using weight_t = double;
    constexpr std::size_t axon_count{2};
    constexpr std::size_t neuron_count{2};
    constexpr std::size_t core_count{3};
    constexpr std::size_t input_size{2};
    constexpr std::size_t output_size{4};
    constexpr std::size_t max_spans{2};
    constexpr MembraneLeak<weight_t> leak{1};

    using Src = SpikeSource;
    using Span = SourceSpan;
    using Con = CoreSpanConnection<max_spans>;

    auto chip = generate_test_chip<
        weight_t,
        generate_connections_3_cores_2x2<Con, Span, core_count, in, off>(),
        generate_outputs_3_cores_2x2<Src, output_size>(),
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
            NeurW{Ws{2.0,0.0}, 2.0}, NeurW{Ws{0.0,3.0}, 3.0}
        }},
        CoreW{{
            NeurW{Ws{2.0,0.0}, 1.0}, NeurW{Ws{0.0,3.0}, 3.0}
        }},
        CoreW{{
            NeurW{Ws{2.0,0.0}, 2.0}, NeurW{Ws{0.0,3.0}, 2.0}
        }}
    }};
    chip.load_weights(weights);

    std::array<raw_input_t, 2> input{1.0, 1.0};
        
    auto buffer = chip.execute<
        SpikingExecution<9, 3, 4, DeterministicSpikeGenerator, weight_t>>(input);

    chip.reset();
    return buffer == std::array<weight_t, 4>{0, 0, 0, 0};
}

} // namespace nevresim::tests
} // namespace nevresim
