#pragma once

#include "simulator/compute_policy/neuron/neuron_compute.hpp"
#include "simulator/chip_weights.hpp"
#include "common/types.hpp"

namespace nevresim {

template <typename Config, typename ComputePolicy>
class Neuron
{
    NeuronCompute<Config, ComputePolicy> compute_{};

public:
    constexpr 
    Neuron() {}

    constexpr typename ComputePolicy::signal_t 
    compute(const auto& incoming_signal)
    {
        return compute_(incoming_signal);
    }

    constexpr
    void reset()
    {
        compute_.reset();
    }

    // Instantiated only for compute policies that expose the state (spiking).
    constexpr
    auto get_membrane_potential() const
    {
        return compute_.get_membrane_potential();
    }

    constexpr
    auto get_threshold() const
    {
        return compute_.get_threshold();
    }

    constexpr
    void load_weights(
        const NeuronWeights<Config{}, typename Config::weight_t, typename Config::threshold_t>& weights)
    {
        compute_.load_weights(weights);
    }
};

} // namespace nevresim