#pragma once

// Final-membrane read port (C2), compiled in ONLY when NEVRESIM_EXPORT_MEMBRANE
// is defined — the production accuracy build is untouched and pays no cost.
//
// For every network output sourced from a core neuron it captures
// membrane/threshold (m_T in threshold units) at that core's window-end cycle
// (latency + SimulationLength - 1, after the cycle's compute), matching the
// HCM executor which freezes each core at its own window end. One
// "MEMB v0 ... vN" line per sample goes to stderr; the stdout spike-count
// protocol stays byte-identical to the default build.
//
// The reader is a constexpr-friendly value type: the compile-time chip's
// constexpr self-tests (_tests/all.hpp) evaluate SpikingExecution::execute in
// a constant expression, so capture() must be constexpr and dump() is only
// reachable at runtime (std::is_constant_evaluated() guard at the call site).

#ifdef NEVRESIM_EXPORT_MEMBRANE

#include "common/constants.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <ostream>

namespace nevresim {

template <typename Chip, int SimulationLength>
struct FinalMembraneReader
{
    static constexpr std::size_t output_size = Chip::config_.output_size_;

    std::array<double, output_size> values_{};

    constexpr void capture(const Chip& chip, int cycle)
    {
        for (std::size_t i = 0; i < output_size; ++i)
        {
            const auto& source = chip.get_mapping().output_sources_[i];
            if (source.core_ == k_input_buffer_id
                || source.core_ == k_no_connection
                || source.core_ == k_always_on)
            {
                continue;
            }
            const auto& core = chip.get_cores()[source.core_];
            if (cycle != core.get_latency() + SimulationLength - 1)
            {
                continue;
            }
            const double membrane =
                static_cast<double>(core.get_neuron_membrane(source.neuron_));
            const double threshold =
                static_cast<double>(core.get_neuron_threshold(source.neuron_));
            // Production wire thresholds are >= 1 (quantize scale / 1.0);
            // a zero threshold only appears in constexpr self-test fixtures.
            values_[i] = threshold != 0.0 ? membrane / threshold : membrane;
        }
    }

    void dump(std::ostream& os) const
    {
        const auto previous_precision =
            os.precision(std::numeric_limits<double>::max_digits10);
        os << "MEMB";
        for (const auto value : values_)
        {
            os << ' ' << value;
        }
        os << '\n';
        os.precision(previous_precision);
    }
};

} // namespace nevresim

#endif // NEVRESIM_EXPORT_MEMBRANE
