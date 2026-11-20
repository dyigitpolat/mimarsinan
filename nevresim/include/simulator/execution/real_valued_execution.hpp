#pragma once

#include "simulator/compute_policy/real_valued_compute.hpp"

#include <cstddef>

namespace nevresim {

class RealValuedExecution
{
public:
    using compute_policy_t = RealValuedCompute;

    template <typename ConcreteComputePolicy>
    constexpr static auto execute(
        const auto& input, 
        auto& chip)
    {
        for(std::size_t i = 0; i < chip.config_.core_count_; ++i){
            ConcreteComputePolicy::compute(chip, input, i);
        }

        return ConcreteComputePolicy::read_output_buffer(chip, input);
    }
};

} // namespace nevresim