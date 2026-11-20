#pragma once

#include "common/constants.hpp"
#include "common/types.hpp"

#include <vector>
#include <istream>

namespace nevresim {

class InputLoader
{
public:
    int target_;
    int batch_size_;
    int input_size_;
    std::vector<raw_input_t> input_;

    friend std::istream& operator>>(
        std::istream& stream, InputLoader& loader)
    {
        stream >> loader.target_;
        stream >> loader.batch_size_;
        stream >> loader.input_size_;

        loader.input_ = std::vector<raw_input_t>(loader.input_size_);
        for(auto& item : loader.input_)
        {
            stream >> item;
        }

        return stream;
    }
};

} // namespace nevresim