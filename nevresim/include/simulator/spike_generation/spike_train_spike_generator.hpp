#pragma once

#include "common/types.hpp"

#include <array>
#include <cstddef>

namespace nevresim {

/// Replay host-provided per-cycle input spikes (cycle-major layout).
///
/// ``input`` must expose ``data()`` and hold ``InputSize * SimulationLength``
/// values.  Index ``cycle * InputSize + neuron`` is the spike at ``cycle`` for
/// axon ``neuron``.  Cycles ``>= SimulationLength`` emit zero spikes (latency
/// padding handled by ``SpikingExecution`` loop length).
template <std::size_t InputSize, int SimulationLength>
class SpikeTrainSpikeGenerator
{
public:
    using spike_source_t = std::array<spike_t, InputSize>;

    constexpr
    static spike_source_t generate_spikes(
        const auto& input, int cycle)
    {
        spike_source_t source{};
        if (cycle < 0 || cycle >= SimulationLength) {
            return source;
        }

        const auto* data = input.data();
        const std::size_t base =
            static_cast<std::size_t>(cycle) * InputSize;
        for (std::size_t neuron = 0; neuron < InputSize; ++neuron) {
            source[neuron] = static_cast<spike_t>(
                data[base + neuron] != raw_input_t{0});
        }
        return source;
    }
};

} // namespace nevresim
