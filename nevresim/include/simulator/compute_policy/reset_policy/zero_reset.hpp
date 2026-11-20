#pragma once

namespace nevresim
{

struct ZeroReset
{
    static constexpr void apply(const auto&, auto& membrane_potential)
    {
        membrane_potential = 0;
    }
};

} // namespace nevresim
