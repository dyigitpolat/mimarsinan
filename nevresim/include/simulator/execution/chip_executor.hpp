#pragma once

namespace nevresim {

template <typename ExecutePolicy>
class ChipExecutor
{
public:
    template <typename ConcreteComputePolicy>
    constexpr static auto execute(
        const auto& input, 
        auto& chip)
    {
        return ExecutePolicy::
            template execute<ConcreteComputePolicy>(input, chip);
    }
};

} // namespace nevresim