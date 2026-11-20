#pragma once

#include "simulator/chip/chip.hpp"
#include "loaders/mapping_loader.hpp"

namespace nevresim {

template<typename Config, typename ComputePolicy>
class RuntimeChip
{
    using concrete_policy_t =
        typename ComputePolicy::template concrete_policy_t<RuntimeChip>;
    using core_t = Core<Config, ComputePolicy>;
    using cores_array_t = std::array<core_t, Config::core_count_>;

    cores_array_t cores_{};
    Mapping<Config> mapping_{};

public:
    using config_t = Config;
    static constexpr Config config_ = Config{};

    constexpr RuntimeChip() = default;

    constexpr RuntimeChip(cores_array_t cores, Mapping<Config> mapping)
        : cores_(cores), mapping_(mapping) {}

    constexpr const Mapping<Config>& get_mapping() const { return mapping_; }

    constexpr Mapping<Config>& mapping_mut() { return mapping_; }

    template <typename ExecutionPolicy>
    constexpr auto execute(const auto& input)
    {
        static_assert(
            std::is_same_v<
                typename ExecutionPolicy::compute_policy_t, ComputePolicy>);

        return ChipExecutor<ExecutionPolicy>::
            template execute<concrete_policy_t>(input, *this);
    }

    constexpr void reset()
    {
        for (auto& core : cores_)
            core.reset();
    }

    constexpr void load_weights(
        const ChipWeights<config_, typename Config::weight_t, typename Config::threshold_t>& weights)
    {
        for (auto core_iter = std::begin(weights.cores_); auto& core : cores_)
            core.load_weights(*core_iter++);
    }

    constexpr const cores_array_t& get_cores() const { return cores_; }
    constexpr cores_array_t& get_cores() { return cores_; }
};

} // namespace nevresim
