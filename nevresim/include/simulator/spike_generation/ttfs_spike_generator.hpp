#pragma once

#include "common/types.hpp"
#include "simulator/spike_generation/spike_io_views.hpp"

#include <array>
#include <cmath>
#include <ranges>

namespace nevresim {

/// TTFS (Time-To-First-Spike) spike generator — genuine single-spike.
///
/// Each input fires **exactly once**, at its time-coded cycle; the information
/// is in the timing, not in a held value (a real TTFS axon does not re-send the
/// spike every cycle). The downstream neuron reconstructs the ramp
/// (``Σ w_j·(t−t_j)``) from this single arrival via its ``ramp_current``.
///
///     spike_time = round(SimulationLength * (1 - value))
///
/// - value = 1.0  →  single spike at cycle 0
/// - value > 0.0  →  single spike at cycle spike_time
/// - value = 0.0  →  spike_time = SimulationLength → never fires
///
template <std::size_t InputSize, int SimulationLength>
class TTFSSpikeGenerator
{
public:
    using spike_source_t = std::array<spike_t, InputSize>;

    constexpr
    static spike_source_t generate_spikes(
        const auto& input, int cycle)
    {
        spike_source_t source{};
        const auto in = spike_io::strict_input<InputSize>(input);
        auto out = spike_io::output(source);
        std::ranges::transform(
            in, out.begin(),
            [cycle](auto item)
            {
                auto clamped = item;
                if (clamped < 0) clamped = 0;
                if (clamped > 1) clamped = 1;

                int spike_time = static_cast<int>(
                    std::llround(SimulationLength * (1.0 - clamped)));

                return static_cast<spike_t>(
                    spike_time < SimulationLength && cycle == spike_time);
            });

        return source;
    }
};

} // namespace nevresim
