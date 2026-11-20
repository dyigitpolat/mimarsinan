#pragma once

#include "simulator/compute_policy/compute_policy_base.hpp"

namespace nevresim 
{

/// Compute policy for the continuous (analytical) TTFS forward pass.
///
/// Semantically identical to ``RealValuedCompute`` at the policy level
/// (real-valued signals, ``EveryCycleAlwaysOn``, ``LatencyGated=true``),
/// but typed as a distinct class so that ``NeuronCompute`` can be specialised
/// to divide by the per-neuron threshold — giving the analytical TTFS
/// ``relu(W @ x + b) / θ`` behaviour.
class TTFSAnalyticalCompute
{
public: 
    using signal_t = double;

    template <typename Chip>
    using concrete_policy_t = ComputePolicyBase<Chip, signal_t>;
};

} // namespace nevresim

