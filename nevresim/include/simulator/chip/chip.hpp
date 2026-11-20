#pragma once

#include "loaders/weights_loader.hpp"
#include "common/constants.hpp"
#include "common/types.hpp"
#include "simulator/chip/core.hpp"
#include "simulator/chip_utilities.hpp"
#include "simulator/execution/chip_executor.hpp"

#include <array>
#include <utility>
#include <cstddef>
#include <istream>
#include <type_traits>

namespace nevresim {

template<
    typename Config,
    Mapping<Config> mapping,
    typename ComputePolicy
    >
class Chip
{
    using concrete_policy_t = 
        typename ComputePolicy::template concrete_policy_t<Chip>;
    using core_t = Core<Config, ComputePolicy>;
    using cores_array_t = 
        std::array<core_t, Config::core_count_>;

    cores_array_t cores_{};

public:
    using config_t = Config;
    static constexpr Config config_ = Config{};
    static constexpr Mapping mapping_ = mapping;

public:
    constexpr const Mapping<Config>& get_mapping() const { return mapping_; }

    constexpr 
    Chip(cores_array_t cores) : cores_(cores) {}

    constexpr 
    Chip() : cores_{} {}
    
    template <typename ExecutionPolicy>
    constexpr 
    auto execute(const auto& input)
    {
        static_assert(
            std::is_same_v<
                typename ExecutionPolicy::compute_policy_t, ComputePolicy>);

        return ChipExecutor<ExecutionPolicy>::
            template execute<concrete_policy_t>(input, *this);
    }
    
    constexpr
    void reset()
    {
        for(auto& core : cores_)
        {
            core.reset();
        }
    }

    constexpr
    void load_weights(
        const ChipWeights<config_, typename Config::weight_t, typename Config::threshold_t>& weights)
    {
        for(auto core_iter = std::begin(weights.cores_); auto& core : cores_)
        {
            core.load_weights(*core_iter++);
        }
    }

    constexpr
    const cores_array_t& get_cores() const { return cores_; } 

    constexpr
    cores_array_t& get_cores() { return cores_; } 
};

} // namespace nevresim