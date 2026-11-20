#pragma once

#include "simulator/chip_utilities.hpp"
#include "common/constants.hpp"
#include "common/types.hpp"

#include <array>

namespace nevresim {

template <ChipConfiguration Config, typename WeightType, typename ThresholdType>
class NeuronWeights
{
    using weight_t = WeightType;
    using threshold_t = ThresholdType;

public:
    std::array<weight_t, Config.axon_count_> weights_;
    threshold_t threshold_{};
    Bias<weight_t> bias_{};
};

template <ChipConfiguration Config, typename WeightType, typename ThresholdType>
class CoreWeights
{
public:
    std::array<
        NeuronWeights<Config, WeightType, ThresholdType>,
        Config.neuron_count_> neurons_;
    int latency_{};
};

template <ChipConfiguration Config, typename WeightType, typename ThresholdType>
class ChipWeights
{
public:
    std::array<CoreWeights<Config, WeightType, ThresholdType>, Config.core_count_> cores_;
};

} // namespace nevresim