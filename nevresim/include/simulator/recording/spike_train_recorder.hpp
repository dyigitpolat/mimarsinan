#pragma once

// Per-neuron output spike TRAINS, compiled in ONLY when
// NEVRESIM_RECORD_SPIKE_TRAINS is defined (a dedicated instrumented build,
// like the SPKREC count recorder beside it). The production accuracy path is
// untouched and pays no cost.
//
// Where SPKREC accumulates window COUNTS, this records the per-cycle emission
// history of every neuron in PRODUCER-LOCAL time: index k of a core's train is
// that core's k-th active cycle (cycle - latency*lat_scale), which is the
// convention every carry consumer shares (the HCM executor's carry seam and
// SANA-FE's _compute_seg_output_raster). Emitted to stderr as its own record
// line — `SPKTRN <core> <bits per neuron...>` then `SPKTRN_END` — so the
// existing SPKREC parser is untouched.

#ifdef NEVRESIM_RECORD_SPIKE_TRAINS

#include "common/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>

namespace nevresim {

template <typename Chip, int SimulationLength>
struct SpikeTrainRecorder
{
    static constexpr std::size_t core_count = Chip::config_.core_count_;
    static constexpr std::size_t neuron_count = Chip::config_.neuron_count_;

    std::array<
        std::array<std::array<std::uint8_t, SimulationLength>, neuron_count>,
        core_count> out_{};

    void clear()
    {
        for (auto& core : out_)
            for (auto& train : core)
                train.fill(0);
    }

    // Record one cycle for every core inside its active window
    // [latency*lat_scale, latency*lat_scale + SimulationLength).
    void accumulate(const Chip& chip, int cycle, int lat_scale)
    {
        for (std::size_t c = 0; c < core_count; ++c)
        {
            const int lat = chip.get_cores()[c].get_latency() * lat_scale;
            const int local = cycle - lat;
            if (local < 0 || local >= SimulationLength)
                continue;

            const auto& output = chip.get_cores()[c].get_output();
            for (std::size_t n = 0; n < neuron_count; ++n)
                out_[c][n][local] =
                    static_cast<long>(output[n]) != 0 ? 1 : 0;
        }
    }

    void dump(std::ostream& os) const
    {
        for (std::size_t c = 0; c < core_count; ++c)
        {
            os << "SPKTRN " << c;
            for (std::size_t n = 0; n < neuron_count; ++n)
            {
                os << ' ';
                for (int t = 0; t < SimulationLength; ++t)
                    os << static_cast<char>('0' + out_[c][n][t]);
            }
            os << '\n';
        }
        os << "SPKTRN_END\n";
    }
};

template <typename Chip, int SimulationLength>
inline std::unique_ptr<SpikeTrainRecorder<Chip, SimulationLength>>
make_spike_train_recorder()
{
    return std::make_unique<SpikeTrainRecorder<Chip, SimulationLength>>();
}

} // namespace nevresim

#endif // NEVRESIM_RECORD_SPIKE_TRAINS
