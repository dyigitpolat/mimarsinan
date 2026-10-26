#pragma once

#include "simulator/chip_utilities.hpp"
#include "simulator/compute_policy/always_on_policy.hpp"

#include "common/constants.hpp"

namespace nevresim
{

template <
    typename Chip,
    typename SignalType,
    typename AlwaysOnPolicy = EveryCycleAlwaysOn,
    bool LatencyGated = true,
    int LatencyScale = 1>
class ComputePolicyBase
{
public:
    // Per-core assembled axon signals for a given cycle. Public so the
    // spike-count recorder (NEVRESIM_RECORD_SPIKES) can mirror exactly what
    // each core integrates, without duplicating the span-gather logic.
    static constexpr
    auto get_axon_input(
        const Chip& chip,
        core_id_t core_id,
        const auto& input_buffer,
        [[maybe_unused]] int cycle = 0)
    {
        std::array<SignalType, Chip::config_.axon_count_> signals{};
        std::size_t dest = 0;

        const auto& conn = chip.get_mapping().core_sources_[core_id];
        for (std::size_t s = 0; s < conn.span_count_; ++s)
        {
            const auto& span = conn.spans_[s];

            if (span.core_ == k_no_connection)
            {
                dest += span.count_;
            }
            else if (span.core_ == k_input_buffer_id)
            {
                for (std::size_t i = 0; i < span.count_; ++i)
                    signals[dest++] = input_buffer[span.start_ + i];
            }
            else if (span.core_ == k_always_on)
            {
                const int window_start =
                    chip.get_cores()[core_id].get_latency() * LatencyScale;
                if (AlwaysOnPolicy::delivers(cycle, window_start))
                    for (std::size_t i = 0; i < span.count_; ++i)
                        signals[dest++] = SignalType{1};
                else
                    dest += span.count_;
            }
            else
            {
                const auto& output =
                    chip.get_cores()[span.core_].get_output();
                for (std::size_t i = 0; i < span.count_; ++i)
                    signals[dest++] = output[span.start_ + i];
            }
        }

        return signals;
    }

    static constexpr
    SignalType retrieve_signal(
        const Chip& chip,
        SpikeSource source,
        const auto& input_buffer,
        [[maybe_unused]] int cycle = 0)
    {
        if (source.core_ == k_input_buffer_id)
            return input_buffer[source.neuron_];
        else if (source.core_ == k_no_connection)
            return SignalType{};
        else if (source.core_ == k_always_on)
        {
            // An always-on read as a network output has window start 0
            // (see TTFSCascadeExecution::win_start).
            return AlwaysOnPolicy::delivers(cycle, /*window_start=*/0)
                ? SignalType{1} : SignalType{};
        }
        return chip.get_cores()[source.core_]
                   .get_output()[source.neuron_];
    }

public:
    static constexpr
    void compute(Chip& chip, const auto& input_buffer, int cycle)
    {
        std::array<
            std::array<SignalType, Chip::config_.axon_count_>, 
            Chip::config_.core_count_> axons{};

        for(int core_id{}; auto& axon : axons)
        {
            axon = get_axon_input(chip, core_id++, input_buffer, cycle);
        }

        for(int core_id{}; auto& core : chip.get_cores())
        {
            if constexpr (LatencyGated)
            {
                if(cycle >= core.get_latency() * LatencyScale)
                {
                    core.compute(axons[core_id]);
                }
            }
            else
            {
                core.compute(axons[core_id]);
            }

            ++core_id;
        }
    }

    static constexpr
    auto read_output_buffer(
        const Chip& chip, const auto& input_buffer, int cycle = 0)
    {
        std::array<SignalType, Chip::config_.output_size_> output{};
        for(std::size_t output_idx{}; output_idx < Chip::config_.output_size_; ++output_idx)
        {
            output[output_idx] = retrieve_signal(
                chip, 
                chip.get_mapping().output_sources_[output_idx], 
                input_buffer,
                cycle);
        }
        return output;
    }
};

} // namespace nevresim
