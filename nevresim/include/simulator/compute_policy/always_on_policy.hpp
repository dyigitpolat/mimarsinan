#pragma once

namespace nevresim
{

/// Strategy: when does an always-on (bias) axon deliver its spike to a core?
///
/// Encapsulates the per-mode bias-delivery timing so ``ComputePolicyBase`` stays
/// branch-free. ``window_start`` is the consuming core's local window start in
/// global cycles (``latency * LatencyScale``).

/// Every active cycle — LIF's per-cycle bias spike and the analytical / quantized
/// real-valued constant bias signal. ``window_start`` is irrelevant.
struct EveryCycleAlwaysOn
{
    static constexpr bool delivers(int /*cycle*/, int /*window_start*/) { return true; }
};

/// Once, at the core's local window start — genuine cascaded single-spike TTFS:
/// the bias is one spike, ramp-integrated downstream, so a latency-gated deeper
/// core integrates it (firing at global cycle 0 would drop it).
struct WindowStartAlwaysOn
{
    static constexpr bool delivers(int cycle, int window_start) { return cycle == window_start; }
};

} // namespace nevresim
