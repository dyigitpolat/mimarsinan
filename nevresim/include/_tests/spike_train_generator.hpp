#pragma once

#include "simulator/spike_generation/spike_train_spike_generator.hpp"
#include "common/types.hpp"

#include <array>

namespace nevresim {
namespace tests {

constexpr bool test_spike_train_generator_replays_flat_train()
{
    constexpr std::size_t input_size{3};
    constexpr int simulation_length{2};

    // cycle 0: [1,0,1], cycle 1: [0,1,0]
    std::array<raw_input_t, input_size * simulation_length> train{
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
    };

    const auto cycle0 =
        SpikeTrainSpikeGenerator<input_size, simulation_length>::generate_spikes(
            train, 0);
    if (cycle0[0] != 1 || cycle0[1] != 0 || cycle0[2] != 1) {
        return false;
    }

    const auto cycle1 =
        SpikeTrainSpikeGenerator<input_size, simulation_length>::generate_spikes(
            train, 1);
    if (cycle1[0] != 0 || cycle1[1] != 1 || cycle1[2] != 0) {
        return false;
    }

    const auto past =
        SpikeTrainSpikeGenerator<input_size, simulation_length>::generate_spikes(
            train, simulation_length);
    return past[0] == 0 && past[1] == 0 && past[2] == 0;
}

static_assert(test_spike_train_generator_replays_flat_train());

} // namespace tests
} // namespace nevresim
