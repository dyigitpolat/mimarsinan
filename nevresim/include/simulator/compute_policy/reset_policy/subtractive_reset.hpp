#pragma once

namespace nevresim
{

struct SubtractiveReset
{
    static constexpr void apply(const auto& threshold, auto& membrane_potential)
    {
        membrane_potential -= threshold;
    }
};

} // namespace nevresim
