#pragma once

#include "simulator/compute_policy/ttfs_cascade_compute.hpp"
#include "simulator/chip_weights.hpp"
#include "common/types.hpp"

#include <numeric>

namespace nevresim {

template <typename Config, typename ComputePolicy>
class NeuronCompute;

/// Genuine cascaded single-spike TTFS neuron.
///
/// Fires **exactly once** (the firing-cycle transition is the only spike emitted;
/// silent forever after — a real TTFS axon does not re-send). The integration is
/// a **ramp**: weighted single-spike arrivals accumulate into a persistent
/// ``ramp_current_``, which is added to the membrane every cycle, so one input
/// spike at ``t_j`` contributes its weight on every later cycle
/// (``membrane(t) = Σ w_j·(t − t_j) + b·t``). No reset. ``reset()`` clears the
/// per-sample state between inputs. Firing uses the config-driven ``ComparePolicy``.
template <typename Config, typename ComparePolicy>
class NeuronCompute<Config, TTFSCascadeCompute<ComparePolicy>>
{
    using weight_t = typename Config::weight_t;
    using threshold_t = typename Config::threshold_t;
    using weights_array_t = std::array<Weight<weight_t>, Config::axon_count_>;

    weights_array_t weights_{};
    Bias<weight_t> bias_{};
    threshold_t threshold_{};
    MembranePotential<weight_t> membrane_potential_{};
    MembranePotential<weight_t> ramp_current_{};
    bool has_fired_{false};

public:
    constexpr typename TTFSCascadeCompute<ComparePolicy>::signal_t
    operator()(const auto& incoming_spikes)
    {
        ramp_current_ +=
            std::inner_product(
                std::cbegin(weights_), std::cend(weights_),
                std::cbegin(incoming_spikes),
                static_cast<MembranePotential<weight_t>>(0));
        membrane_potential_ += ramp_current_ + bias_ - Config::leak_amount_;

        if (has_fired_)
            return 0;  // single spike — already emitted the timing-coded spike

        const spike_t spike =
            ComparePolicy::fires(threshold_, membrane_potential_);
        if (spike)
            has_fired_ = true;
        return spike;
    }

    constexpr void reset()
    {
        membrane_potential_ = 0;
        ramp_current_ = 0;
        has_fired_ = false;
    }

    constexpr
    void load_weights(
        const NeuronWeights<Config{}, typename Config::weight_t, typename Config::threshold_t>& weights)
    {
        threshold_ = weights.threshold_;
        weights_ = weights.weights_;
        bias_ = weights.bias_;
    }
};

} // namespace nevresim
