#pragma once

#include "simulator/compute_policy/real_valued_compute.hpp"
#include "simulator/chip_weights.hpp"
#include "common/types.hpp"

#include <numeric>

namespace nevresim {

template <typename Config, typename ComputePolicy>
class NeuronCompute;

template <typename Config>
class NeuronCompute<Config, RealValuedCompute>
{
    using weight_t = typename Config::weight_t;

    using weights_array_t = std::array<Weight<weight_t>, Config::axon_count_>;
    using signal_t = typename RealValuedCompute::signal_t;

    weights_array_t weights_{};
    Bias<weight_t> bias_{};

public:
    constexpr RealValuedCompute::signal_t
    operator()(const auto& incoming_signal)
    {
        return std::max(
            signal_t{0}, 
            std::inner_product(
                std::cbegin(weights_), std::cend(weights_),
                std::cbegin(incoming_signal), 
                signal_t{0}
            ) + bias_);
    }

    constexpr void reset() {}

    constexpr
    void load_weights(
        const NeuronWeights<Config{}, typename Config::weight_t, typename Config::threshold_t>& weights)
    {
        weights_ = weights.weights_;
        bias_ = weights.bias_;
    }
};

} // namespace nevresim