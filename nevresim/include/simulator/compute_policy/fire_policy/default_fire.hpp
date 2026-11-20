#pragma once

#include "simulator/compute_policy/compare_policy/strict_compare.hpp"
#include "simulator/compute_policy/fire_policy/lif_fire_policy.hpp"
#include "simulator/compute_policy/reset_policy/subtractive_reset.hpp"

namespace nevresim
{

using DefaultFirePolicy = LIFirePolicy<SubtractiveReset, StrictCompare>;

} // namespace nevresim
