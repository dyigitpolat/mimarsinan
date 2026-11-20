#pragma once

#include "common/types.hpp"

#include <array>

namespace nevresim
{

struct SpikeSource
{
    core_id_t core_{};
    axon_id_t neuron_{};
};

struct SourceSpan
{
    core_id_t core_{};
    axon_id_t start_{};
    std::size_t count_{};
};

template<std::size_t MaxSpans>
struct CoreSpanConnection {
    std::array<SourceSpan, MaxSpans> spans_{};
    std::size_t span_count_{};
};

template <
    typename WeightType,
    typename ThresholdType,
    std::size_t AxonCount,
    std::size_t NeuronCount,
    std::size_t CoreCount,
    std::size_t InputSize,
    std::size_t OutputSize,
    std::size_t MaxSpansPerCore,
    constant<MembraneLeak<WeightType>> LeakAmount
    >
struct ChipConfiguration
{
    static constexpr std::size_t core_count_{CoreCount};
    static constexpr std::size_t neuron_count_{NeuronCount};
    static constexpr std::size_t axon_count_{AxonCount};
    static constexpr std::size_t input_size_{InputSize};
    static constexpr std::size_t output_size_{OutputSize};
    static constexpr std::size_t max_spans_per_core_{MaxSpansPerCore};

    static constexpr MembraneLeak<WeightType> leak_amount_{LeakAmount.value};

    using weight_t = WeightType;
    using threshold_t = ThresholdType;
};

template<typename Config>
struct Mapping {
    std::array<
        CoreSpanConnection<Config::max_spans_per_core_>,
        Config::core_count_>
        core_sources_{};

    std::array<SpikeSource, Config::output_size_> output_sources_{};
};

} // namespace nevresim
