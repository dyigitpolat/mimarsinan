#pragma once

#include "common/constants.hpp"
#include "common/types.hpp"
#include "simulator/spike_generation/spike_io_views.hpp"

#include <array>
#include <ranges>

namespace nevresim {
template <std::size_t InputSize, int SimulationLength, constant<raw_input_t> Threshold = 0.5>
class DeterministicSpikeGenerator
{
public:
    using spike_source_t = std::array<spike_t, InputSize>;

    constexpr
    static spike_source_t generate_spikes(
        const auto& input, int cycle)
    {
        spike_source_t source{};
        if (cycle >= SimulationLength)
        {
            return source;
        }

        const auto in = spike_io::strict_input<InputSize>(input);
        auto out = spike_io::output(source);
        std::ranges::transform(
            in, out.begin(),
            [&](auto item)
            {
                return static_cast<spike_t>(item > Threshold.value);
            });

        return source;
    }
};

} // namespace nevresim
