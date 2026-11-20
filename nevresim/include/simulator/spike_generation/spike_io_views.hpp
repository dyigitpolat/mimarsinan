#pragma once

#include "common/types.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <span>

namespace nevresim::spike_io {

/// Fixed-extent view over a spike output buffer (`std::array<spike_t, N>`).
template<std::size_t N>
constexpr std::span<spike_t, N> output(std::array<spike_t, N>& storage) noexcept
{
    return std::span<spike_t, N>{storage};
}

/// Strict input view: ``input.size()`` must equal ``InputSize`` (checked via assert).
template<std::size_t InputSize, typename Input>
constexpr auto strict_input(const Input& input)
{
    assert(static_cast<std::size_t>(input.size()) == InputSize);
    using value_t = typename Input::value_type;
    return std::span<const value_t, InputSize>{input.data(), InputSize};
}

} // namespace nevresim::spike_io
