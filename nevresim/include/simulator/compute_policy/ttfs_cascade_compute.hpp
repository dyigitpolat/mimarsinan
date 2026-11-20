#pragma once

#include "simulator/compute_policy/compute_policy_base.hpp"
#include "simulator/compute_policy/compare_policy/inclusive_compare.hpp"
#include "common/types.hpp"

namespace nevresim
{

/// Compute policy for genuine cascaded (greedy) single-spike TTFS.
///
/// Distinct from the LIF ``SpikingCompute`` and the analytical
/// ``TTFSQuantizedCompute``: each neuron fires **exactly once** (binary single
/// spike on the wire), and the integration is a **ramp** reconstructed from the
/// single-spike arrivals by the per-neuron ``NeuronCompute`` specialization.
///
/// Cores are **latency-gated** (active from their hop-latency ``L``), so each
/// neuron's bias/ramp start at its reference time (no premature firing) and it
/// fires within its window ``[L, L+S)`` — the genuine TTFS value is then decoded
/// per-source as ``(L + S - fire)/S`` by ``TTFSCascadeExecution``. An always-on
/// (bias) axon encodes value 1.0 as a single spike at the consuming core's local
/// window start (``WindowStartAlwaysOn``), ramped downstream.
///
/// ``ComparePolicy`` is the config-driven firing comparator (strict ``<`` vs
/// inclusive ``<=``), shared with the LIF / quantized-TTFS fire decisions.
template <typename ComparePolicy = InclusiveCompare>
class TTFSCascadeCompute
{
public:
    using signal_t = spike_t;
    using compare_policy_t = ComparePolicy;

    template <typename Chip>
    using concrete_policy_t =
        ComputePolicyBase<Chip, signal_t, WindowStartAlwaysOn,
                          /*LatencyGated=*/true>;
};

} // namespace nevresim
