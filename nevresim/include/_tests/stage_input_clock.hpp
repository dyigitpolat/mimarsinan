#pragma once

#include "simulator/chip/chip.hpp"
#include "simulator/compute_policy/compare_policy/strict_compare.hpp"
#include "simulator/compute_policy/fire_policy/lif_fire_policy.hpp"
#include "simulator/compute_policy/reset_policy/zero_reset.hpp"
#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/execution/spiking_execution.hpp"
#include "simulator/spike_generation/front_loaded_spike_generator.hpp"

#include "test_util.hpp"

namespace nevresim {
namespace tests {

// Two one-neuron cores read stage input 0 directly: core 0 starts at latency
// 0, core 1 at latency 2. Each neuron fires on every input spike it reads
// (weight 1, threshold 0.5, zero reset), so its spike count over the run is
// the number of input spikes inside its own window. The input value 0.5 is
// front-loaded over T = 4 cycles: spikes on cycles 0 and 1 only.
//
// Read in the core's local time, core 1 sees cycles 0..3 of the train on
// global cycles 2..5 and fires twice, as core 0 does. Read at the global
// cycle, it would see cycles 2 and 3 (no spike) and fire never.
template <typename Con, typename Span, std::size_t core_count, std::size_t in_id>
consteval auto generate_connections_stage_input_clock()
{
    std::array<Con, core_count> cons;

    cons[0].spans_[0] = Span{in_id, 0, 1};
    cons[0].span_count_ = 1;

    cons[1].spans_[0] = Span{in_id, 0, 1};
    cons[1].span_count_ = 1;

    return cons;
}

template <typename Src, std::size_t output_size>
consteval auto generate_outputs_stage_input_clock()
{
    std::array<Src, output_size> outs;

    outs[0] = Src{0, 0};
    outs[1] = Src{1, 0};

    return outs;
}

constexpr bool test_stage_input_read_in_core_local_time()
{
    using weight_t = double;
    using fire_t = LIFirePolicy<ZeroReset, StrictCompare>;
    constexpr std::size_t axon_count{1};
    constexpr std::size_t neuron_count{1};
    constexpr std::size_t core_count{2};
    constexpr std::size_t input_size{1};
    constexpr std::size_t output_size{2};
    constexpr std::size_t max_spans{1};
    constexpr MembraneLeak<weight_t> leak{0};
    constexpr int T{4};
    constexpr int latency{2};

    using Src = SpikeSource;
    using Span = SourceSpan;
    using Con = CoreSpanConnection<max_spans>;

    auto chip = generate_test_chip<
        weight_t,
        generate_connections_stage_input_clock<Con, Span, core_count, in>(),
        generate_outputs_stage_input_clock<Src, output_size>(),
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        max_spans,
        leak,
        SpikingCompute<fire_t>> ();

    using ChipW = ChipWeights<chip.config_, weight_t, weight_t>;
    using CoreW = CoreWeights<chip.config_, weight_t, weight_t>;
    using NeurW = NeuronWeights<chip.config_, weight_t, weight_t>;
    using Ws = std::array<Weight<weight_t>, axon_count>;

    ChipW weights{{
        CoreW{{NeurW{Ws{1.0}, 0.5}}, 0},
        CoreW{{NeurW{Ws{1.0}, 0.5}}, latency}
    }};
    chip.load_weights(weights);

    std::array<raw_input_t, input_size> input{0.5};

    auto buffer = chip.execute<
        SpikingExecution<T, latency, output_size, FrontLoadedSpikeGenerator,
                         weight_t, fire_t>>(input);

    chip.reset();
    return buffer == std::array<weight_t, output_size>{2, 2};
}

} // namespace nevresim::tests
} // namespace nevresim
