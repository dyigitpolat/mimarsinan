
#pragma once

#include "simulator/chip/neuron.hpp"
#include "simulator/chip/core.hpp"
#include "simulator/chip/chip.hpp"

namespace nevresim::tests
{

// Core 0: input[0..1], off            -> 2 spans
// Core 1: core0[0..2]                 -> 1 span
// Core 2: core1[0..2]                 -> 1 span
// max_spans = 2
template <typename Con, typename Span, size_t core_count,
          size_t in_id, size_t off_id>
consteval auto generate_xor_chip_connections()
{
    std::array<Con, core_count> cons; 
        
    cons[0].spans_[0] = Span{in_id, 0, 2};
    cons[0].spans_[1] = Span{off_id, 0, 1};
    cons[0].span_count_ = 2;

    cons[1].spans_[0] = Span{0, 0, 3};
    cons[1].span_count_ = 1;

    cons[2].spans_[0] = Span{1, 0, 3};
    cons[2].span_count_ = 1;

    return cons;
}

template <typename Src, size_t output_size>
consteval auto generate_xor_chip_outputs()
{
    std::array<Src, output_size> outs;
    
    outs[0] = Src{2,0};
    outs[1] = Src{2,1};

    return outs;
}

template <typename ComputePolicy>
consteval auto generate_xor_chip()
{
    using weight_t = double;
    constexpr std::size_t axon_count{3};
    constexpr std::size_t neuron_count{3};
    constexpr std::size_t core_count{3};
    constexpr std::size_t input_size{2};
    constexpr std::size_t output_size{2};
    constexpr std::size_t max_spans{2};
    constexpr MembraneLeak<weight_t> leak{0};

    using Cfg = ChipConfiguration<
        weight_t,
        weight_t,
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        max_spans,
        leak
    >;

    using Map = Mapping<Cfg>;
    using Src = SpikeSource;
    using Span = SourceSpan;
    using Con = CoreSpanConnection<max_spans>;

    constexpr core_id_t in = k_input_buffer_id;
    constexpr core_id_t off = k_no_connection;

    using Chip = Chip<
        Cfg, 
        Map{
            generate_xor_chip_connections<Con, Span, core_count, in, off>(),
            generate_xor_chip_outputs<Src, output_size>()}, 
        ComputePolicy>;

    constexpr Chip chip{};
    return chip;
}

} // namespace nevresim::tests
