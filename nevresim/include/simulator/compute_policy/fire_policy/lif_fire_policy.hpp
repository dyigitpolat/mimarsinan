#pragma once

#include "common/types.hpp"
#include "simulator/compute_policy/always_on_policy.hpp"

namespace nevresim
{

template <typename ResetPolicy, typename ComparePolicy>
struct LIFirePolicy
{
    using always_on_policy = EveryCycleAlwaysOn;
    static constexpr bool latency_gated = true;

    static constexpr spike_t fire(const auto& threshold, auto& membrane_potential)
    {
        const spike_t spike = ComparePolicy::fires(threshold, membrane_potential);
        if (spike)
        {
            ResetPolicy::apply(threshold, membrane_potential);
        }
        return spike;
    }
};

} // namespace nevresim
