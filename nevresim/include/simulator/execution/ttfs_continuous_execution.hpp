#pragma once

#include "simulator/compute_policy/ttfs_analytical_compute.hpp"

#include <cstddef>

namespace nevresim {

/// Execution policy for continuous (event-based / analytical) TTFS.
///
/// Structurally identical to ``RealValuedExecution``: a single forward
/// sweep through all cores, relying on latency-gating to process them
/// in topological order.  The only difference is the compute policy
/// (``TTFSAnalyticalCompute``), whose neuron specialisation divides the
/// ReLU output by the per-neuron threshold.
class TTFSContinuousExecution
{
public:
    using compute_policy_t = TTFSAnalyticalCompute;

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

