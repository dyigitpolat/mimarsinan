#pragma once

#include "simulator/compute_policy/compute_policy_base.hpp"
namespace nevresim 
{

class RealValuedCompute
{
public: 
    using signal_t = double;

    template <typename Chip>
    using concrete_policy_t = ComputePolicyBase<Chip, signal_t>;
};

} // namespace nevresim