#pragma once

#include "common/types.hpp"

#include <cstddef>
#include <vector>

namespace nevresim {

/// The stage input of one sample as the T-cycle train every core reads in its
/// own local time.
///
/// A core whose window starts at global cycle W (its latency times the compute
/// policy's latency scale) reads row ``cycle - W`` of the train at global cycle
/// ``cycle``, so cycle 0 of the stage input reaches it on its first active
/// cycle and cycle T-1 on its last: the timing of the fused segment and of the
/// host-side executors. Reading the input at the global cycle instead hands a
/// core at latency L > 0 the train from cycle L on and never its last L cycles.
///
/// The train is generated once per sample, cycles 0..T-1 in order: the same
/// provider calls, in the same order, as a per-cycle generation at the global
/// cycle, so a chip whose every input reader starts at latency 0 runs as it
/// did. A read outside [0, T) is silent.
template <
    std::size_t InputSize,
    int SimulationLength,
    template<auto... Args> class SpikeProvider>
class StageInputTrain
{
    std::vector<spike_t> spikes_;

public:
    constexpr explicit StageInputTrain(const auto& input)
        : spikes_(InputSize * static_cast<std::size_t>(SimulationLength))
    {
        for (int cycle = 0; cycle < SimulationLength; ++cycle)
        {
            const auto row =
                SpikeProvider<InputSize, SimulationLength>::generate_spikes(
                    input, cycle);
            const std::size_t base =
                static_cast<std::size_t>(cycle) * InputSize;
            for (std::size_t i = 0; i < InputSize; ++i)
                spikes_[base + i] = row[i];
        }
    }

    /// Row ``row`` of the train at the reader's local cycle ``local_cycle``.
    constexpr spike_t read(std::size_t row, int local_cycle) const
    {
        if (local_cycle < 0 || local_cycle >= SimulationLength)
            return spike_t{0};
        return spikes_[static_cast<std::size_t>(local_cycle) * InputSize + row];
    }
};

/// One stage-input read: row ``row`` at the reading core's local cycle. A
/// value input (the real-valued, analytical and quantized TTFS policies) has
/// no time axis and is read as it is.
template <typename Input>
constexpr auto read_stage_input(
    const Input& input, std::size_t row, [[maybe_unused]] int local_cycle)
{
    if constexpr (requires { input.read(row, local_cycle); })
        return input.read(row, local_cycle);
    else
        return input[row];
}

} // namespace nevresim
