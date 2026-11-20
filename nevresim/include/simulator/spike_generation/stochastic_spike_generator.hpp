#pragma once

#include "common/constants.hpp"
#include "common/types.hpp"
#include "loaders/input_loader.hpp"
#include "simulator/spike_generation/spike_io_views.hpp"

#include <array>
#include <random>
#include <ranges>

namespace nevresim {

template <std::size_t InputSize, int SimulationLength>
class StochasticSpikeGenerator
{
public:
    using spike_source_t = std::array<spike_t, InputSize>;

    static spike_source_t generate_spikes(
        const auto& input, int cycle)
    {
        spike_source_t source{};
        if (cycle >= SimulationLength)
        {
            return source;
        }
        static std::random_device device;
        static std::mt19937 engine(device());

        return generate_spikes(input, engine);
    }

    static spike_source_t generate_spikes(
        const auto& input, auto& random_engine)
    {
        spike_source_t source{};
        const auto in = spike_io::strict_input<InputSize>(input);
        auto out = spike_io::output(source);
        std::ranges::transform(
            in, out.begin(),
            [&](auto item)
            {
                return static_cast<spike_t>(
                    std::discrete_distribution({1.0 - item, item})(random_engine));
            });

        return source;
    }
};

} // namespace nevresim
