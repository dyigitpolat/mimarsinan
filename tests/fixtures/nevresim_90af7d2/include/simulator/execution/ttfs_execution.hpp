#pragma once

#include "simulator/compute_policy/ttfs_quantized_compute.hpp"
#include "simulator/compute_policy/ttfs_cascade_compute.hpp"
#include "simulator/compute_policy/compare_policy/inclusive_compare.hpp"
#include "common/types.hpp"
#include "common/constants.hpp"

#include <array>
#include <cstddef>

namespace nevresim {

/// Execution policy for TTFS quantized (cycle-based) mode.
///
/// Structurally mirrors ``SpikingExecution``: the **outer loop** iterates
/// over ``(Latency + 1) * S`` discrete cycles.  On each cycle the compute
/// policy processes every core whose latency-gated window includes that
/// cycle (a core with hop-latency L is active during cycles
/// ``[L*S, (L+1)*S)``).
///
/// Neurons are **stateful** — Phase 1 (initial charge) runs on the first
/// active cycle and Phase 2 (fire-once check + constant ramp) advances
/// one step per subsequent active cycle.
///
/// After all cycles complete the output buffer is read once (output
/// neurons hold their final activation values).
///
template <int S, int Latency, typename ComparePolicy = InclusiveCompare>
class TTFSExecution
{
public:
    using compute_policy_t = TTFSQuantizedCompute<S, ComparePolicy>;

    template <typename ConcreteComputePolicy>
    constexpr static auto execute(
        const auto& input,
        auto& chip)
    {
        constexpr int total_cycles = (Latency + 1) * S;

        for (int cycle = 0; cycle < total_cycles; ++cycle) {
            ConcreteComputePolicy::compute(chip, input, cycle);
        }

        return ConcreteComputePolicy::read_output_buffer(chip, input);
    }
};

/// Execution policy for genuine cascaded (greedy) single-spike TTFS.
///
/// Pipelined like ``SpikingExecution`` (``S + Latency`` cycles, per-cycle spike
/// generation), but TTFS-specific: each neuron emits **one** spike, so the host
/// reconstructs each output's value by latching that single arrival and counting
/// the latch ONLY within that source's own window ``[src_latency, src_latency+S)``
/// → ``count = src_latency + S − fire = value·S`` (the genuine TTFS value). Full-
/// window accumulation would overcount shallow sources by ``(Latency − src_latency)``
/// and saturate everything when ``Latency >> S``. This single-spike windowed decode
/// lives here in the TTFS execution, not in the LIF ``SpikingExecution``.
template <
    int SimulationLength,
    int Latency,
    std::size_t OutputSize,
    template<auto... Args> class SpikeProvider,
    typename WeightType,
    typename ComparePolicy = InclusiveCompare>
class TTFSCascadeExecution
{
public:
    using compute_policy_t = TTFSCascadeCompute<ComparePolicy>;

    template <typename ConcreteComputePolicy, typename Chip>
    constexpr static auto execute(
        const auto& input,
        Chip& chip)
    {
        using weight_t = WeightType;

        // Per-output window start = its source core's hop-latency (0 for input /
        // always-on sources), so the count covers exactly [src_latency, +S).
        const auto& mapping = chip.get_mapping();
        std::array<int, OutputSize> win_start{};
        for (std::size_t k = 0; k < OutputSize; ++k)
        {
            const auto& src = mapping.output_sources_[k];
            if (src.core_ == k_input_buffer_id || src.core_ == k_always_on
                    || src.core_ == k_no_connection)
                win_start[k] = 0;
            else
                win_start[k] = chip.get_cores()[src.core_].get_latency();
        }

        std::array<Weight<weight_t>, OutputSize> buffer{};
        std::array<spike_t, OutputSize> out_latched{};
        for (int i = 0; i < SimulationLength + Latency; ++i)
        {
            const auto& spikes{
                SpikeProvider<(Chip::config_).input_size_, SimulationLength>
                    ::generate_spikes(input, i)};

            ConcreteComputePolicy::compute(chip, spikes, i);

            auto&& out{ConcreteComputePolicy::read_output_buffer(chip, spikes, i)};
            for (std::size_t k = 0; k < OutputSize; ++k)
            {
                if (i < win_start[k] || i >= win_start[k] + SimulationLength)
                    continue;
                if (out[k] != 0)
                    out_latched[k] = 1;
                buffer[k] += out_latched[k];
            }
        }
        return buffer;
    }
};

} // namespace nevresim
