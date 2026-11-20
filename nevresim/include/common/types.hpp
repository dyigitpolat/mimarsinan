#pragma once

#include <cstddef>
#include <cstdint>

namespace nevresim {

template<typename T>
struct constant
{
    T value;
    consteval constant(T val) : value(val) {}
};

using raw_input_t = double;
using spike_t = std::int_fast8_t;

using core_id_t = std::size_t;
using axon_id_t = std::size_t;

template <typename WeightType>
using Weight = WeightType;

template <typename WeightType>
using Threshold = WeightType;

template <typename WeightType>
using MembranePotential = WeightType;

template <typename WeightType>
using Bias = WeightType;

template <typename WeightType>
using MembraneLeak = WeightType;

} // namespace nevresim