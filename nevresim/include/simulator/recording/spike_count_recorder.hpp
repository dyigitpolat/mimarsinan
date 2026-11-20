#pragma once

// Per-core spike-count recorder, compiled in ONLY when NEVRESIM_RECORD_SPIKES
// is defined (the parity harness builds a dedicated recording binary). The
// production accuracy path is untouched and pays no cost.
//
// For each core it accumulates, over that core's active window
// [latency*lat_scale, latency*lat_scale + s_width):
//   - per-axon INPUT counts  = sum of the exact signals fed to core.compute()
//   - per-neuron OUTPUT counts = sum of the core's emitted output_
// mirroring the HCM reference recorder (`record_in_t += input_signals`,
// `record_out_t += buffers`, gated to [core.latency, core.latency+T)).
// Emitted to stderr so the stdout logit protocol is unchanged.

#ifdef NEVRESIM_RECORD_SPIKES

#include "common/types.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <ostream>

namespace nevresim {

template <typename Chip>
struct SpikeCountRecorder
{
    static constexpr std::size_t core_count = Chip::config_.core_count_;
    static constexpr std::size_t axon_count = Chip::config_.axon_count_;
    static constexpr std::size_t neuron_count = Chip::config_.neuron_count_;

    std::array<std::array<long, axon_count>, core_count> in_{};
    std::array<std::array<long, neuron_count>, core_count> out_{};

    void clear()
    {
        for (auto& row : in_) row.fill(0);
        for (auto& row : out_) row.fill(0);
    }

    // Accumulate one cycle for every core inside its active window. ``latched``
    // (single-spike TTFS) keeps the per-neuron/per-axon value high once seen,
    // matching the host's latched reconstruction; LIF accumulates raw spikes.
    template <typename ComputePolicy>
    void accumulate(
        const Chip& chip, const auto& input_buffer, int cycle,
        int s_width, int lat_scale, bool latched)
    {
        for (std::size_t c = 0; c < core_count; ++c)
        {
            const int lat = chip.get_cores()[c].get_latency() * lat_scale;
            if (cycle < lat || cycle >= lat + s_width)
                continue;

            const auto axons =
                ComputePolicy::get_axon_input(chip, static_cast<core_id_t>(c),
                                              input_buffer, cycle);
            for (std::size_t a = 0; a < axon_count; ++a)
            {
                const long v = static_cast<long>(axons[a]);
                if (latched)
                    in_[c][a] = in_[c][a] ? in_[c][a] : (v ? (cycle - lat + 1) : 0);
                else
                    in_[c][a] += v;
            }

            const auto& output = chip.get_cores()[c].get_output();
            for (std::size_t n = 0; n < neuron_count; ++n)
            {
                const long v = static_cast<long>(output[n]);
                if (latched)
                    out_[c][n] = out_[c][n] ? out_[c][n] + 1 : (v ? 1 : 0);
                else
                    out_[c][n] += v;
            }
        }
    }

    void dump(std::ostream& os) const
    {
        for (std::size_t c = 0; c < core_count; ++c)
        {
            os << "SPKREC " << c << " IN";
            for (std::size_t a = 0; a < axon_count; ++a)
                os << ' ' << in_[c][a];
            os << " OUT";
            for (std::size_t n = 0; n < neuron_count; ++n)
                os << ' ' << out_[c][n];
            os << '\n';
        }
        os << "SPKREC_END\n";
    }
};

template <typename Chip>
inline std::unique_ptr<SpikeCountRecorder<Chip>> make_spike_recorder()
{
    return std::make_unique<SpikeCountRecorder<Chip>>();
}

} // namespace nevresim

#endif // NEVRESIM_RECORD_SPIKES
