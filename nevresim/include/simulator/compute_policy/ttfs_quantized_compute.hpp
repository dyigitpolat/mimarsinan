#pragma once

#include "simulator/compute_policy/compute_policy_base.hpp"
#include "simulator/compute_policy/compare_policy/inclusive_compare.hpp"

namespace nevresim
{

/// Compute policy for TTFS quantized (cycle-based) mode.
///
/// Each neuron simulates S discrete time steps per the B1-model TTFS
/// dynamics.  The neuron is **stateful**: Phase 1 (initial charge) runs
/// on the first active cycle, and Phase 2 (fire-once check + ramp)
/// advances one step per subsequent cycle.
///
/// Latency gating is **enabled** with ``LatencyScale = S``, so a core
/// with hop-latency L becomes active at cycle ``L * S``.  This matches
/// the Python-side ``for cycle in range(total_cycles)`` outer loop.
///
/// Template parameters: ``S`` is the number of discrete time steps;
/// ``ComparePolicy`` is the config-driven firing comparator (strict ``<`` vs
/// inclusive ``<=``), shared with the LIF / cascaded-TTFS fire policies.
///
template <int S, typename ComparePolicy = InclusiveCompare>
class TTFSQuantizedCompute
{
public:
    using signal_t = double;
    using compare_policy_t = ComparePolicy;
    static constexpr int time_steps = S;

    template <typename Chip>
    using concrete_policy_t =
        ComputePolicyBase<Chip, signal_t, EveryCycleAlwaysOn, true, S>;
};

} // namespace nevresim
