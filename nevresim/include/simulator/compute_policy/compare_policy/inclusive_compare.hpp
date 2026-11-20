#pragma once

#include "common/types.hpp"

namespace nevresim
{

struct InclusiveCompare
{
    static constexpr spike_t fires(const auto& threshold, const auto& membrane_potential)
    {
        return static_cast<spike_t>(threshold <= membrane_potential);
    }
};

} // namespace nevresim
