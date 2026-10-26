#pragma once

#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/fire_policy/default_fire.hpp"

#ifdef NEVRESIM_RECORD_SPIKES
#include "simulator/recording/spike_count_recorder.hpp"
#include <iostream>
#endif

#ifdef NEVRESIM_RECORD_SPIKE_TRAINS
#include "simulator/recording/spike_train_recorder.hpp"
#include <iostream>
#endif

#ifdef NEVRESIM_EXPORT_MEMBRANE
#include "simulator/recording/final_membrane_reader.hpp"
#include <iostream>
#include <type_traits>
#endif

#include <algorithm>
#include <cstddef>
#include <functional>

namespace nevresim {

template <
    int SimulationLength,
    int Latency,
    std::size_t OutputSize,
    template<auto... Args> class SpikeProvider,
    typename WeightType,
    typename FirePolicy = DefaultFirePolicy>
class SpikingExecution
{
public:
    using compute_policy_t = SpikingCompute<FirePolicy>;

    template <typename ConcreteComputePolicy, typename Chip>
    constexpr static auto execute(
        const auto& input,
        Chip& chip)
    {
        using weight_t = WeightType;

        std::array<Weight<weight_t>, OutputSize> buffer{};
#ifdef NEVRESIM_RECORD_SPIKES
        auto recorder = make_spike_recorder<Chip>();
#endif
#ifdef NEVRESIM_RECORD_SPIKE_TRAINS
        auto train_recorder =
            make_spike_train_recorder<Chip, SimulationLength>();
#endif
#ifdef NEVRESIM_EXPORT_MEMBRANE
        FinalMembraneReader<Chip, SimulationLength> membrane_reader{};
#endif
        for(int i = 0; i < SimulationLength + Latency; ++i){
            const auto& spikes{
                SpikeProvider<(Chip::config_).input_size_, SimulationLength>
                    ::generate_spikes(input, i)};

            ConcreteComputePolicy::compute(chip, spikes, i);

#ifdef NEVRESIM_RECORD_SPIKES
            recorder->template accumulate<ConcreteComputePolicy>(
                chip, spikes, i, SimulationLength, /*lat_scale=*/1, /*latched=*/false);
#endif
#ifdef NEVRESIM_RECORD_SPIKE_TRAINS
            train_recorder->accumulate(chip, i, /*lat_scale=*/1);
#endif

#ifdef NEVRESIM_EXPORT_MEMBRANE
            membrane_reader.capture(chip, i);
#endif

            auto&& out{ConcreteComputePolicy::read_output_buffer(chip, spikes, i)};
            std::ranges::transform(
                std::ranges::cbegin(out), std::ranges::cend(out),
                std::ranges::cbegin(buffer), std::ranges::cend(buffer),
                std::ranges::begin(buffer),
                std::plus{});
        }

#ifdef NEVRESIM_RECORD_SPIKES
        recorder->dump(std::cerr);
#endif
#ifdef NEVRESIM_RECORD_SPIKE_TRAINS
        train_recorder->dump(std::cerr);
#endif

#ifdef NEVRESIM_EXPORT_MEMBRANE
        // The compile-time chip's constexpr self-tests reach here too; the
        // stderr dump is runtime-only.
        if (!std::is_constant_evaluated())
        {
            membrane_reader.dump(std::cerr);
        }
#endif
        return buffer;
    }
};

} // namespace nevresim
