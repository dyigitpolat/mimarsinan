#pragma once

#include "common/types.hpp"

#include <istream>
#include <vector>

namespace nevresim {

/// Load a flattened spike train for one sample.
///
/// File format::
///
///     target batch_size input_size simulation_length
///     spike[cycle=0, neuron=0] ... spike[cycle=0, neuron=input_size-1]
///     spike[cycle=1, neuron=0] ...
///     (input_size * simulation_length values total, cycle-major)
class SpikeTrainInputLoader
{
public:
    int target_{};
    int batch_size_{};
    int input_size_{};
    int simulation_length_{};
    std::vector<raw_input_t> spikes_;

    friend std::istream& operator>>(
        std::istream& stream, SpikeTrainInputLoader& loader)
    {
        stream >> loader.target_;
        stream >> loader.batch_size_;
        stream >> loader.input_size_;
        stream >> loader.simulation_length_;

        const std::size_t count =
            static_cast<std::size_t>(loader.input_size_)
            * static_cast<std::size_t>(loader.simulation_length_);
        loader.spikes_.assign(count, raw_input_t{0});
        for (auto& item : loader.spikes_) {
            stream >> item;
        }

        return stream;
    }
};

} // namespace nevresim
