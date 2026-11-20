#pragma once

#include "simulator/chip_utilities.hpp"
#include "simulator/chip_weights.hpp"
#include "common/constants.hpp"
#include "common/types.hpp"

#include <istream>

namespace nevresim {

template <ChipConfiguration Config, typename WeightType, typename ThresholdType>
class WeightsLoader
{
public:
    using chip_weights_t =
        ChipWeights<Config, WeightType, ThresholdType>;

public:
    chip_weights_t chip_weights_;

    friend std::istream& operator>>(
        std::istream& weights_stream, 
        WeightsLoader& loader)
    {
        
        for(auto& core : loader.chip_weights_.cores_)
        {
            weights_stream >> core.latency_;
            for(auto& neuron : core.neurons_)
            {
                weights_stream >> neuron.threshold_;
                weights_stream >> neuron.bias_;
                for(auto& weight : neuron.weights_)
                {
                    weights_stream >> weight;
                }
            }
        }

        return weights_stream;
    }
};

} // namespace nevresim