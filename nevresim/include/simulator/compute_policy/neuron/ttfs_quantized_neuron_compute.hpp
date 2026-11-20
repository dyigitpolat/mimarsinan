#pragma once

#include "simulator/compute_policy/ttfs_quantized_compute.hpp"
#include "simulator/compute_policy/timing_decision.hpp"
#include "simulator/chip_weights.hpp"
#include "common/types.hpp"

#include <numeric>
#include <algorithm>

namespace nevresim {

template <typename Config, typename ComputePolicy>
class NeuronCompute;

/// TTFS quantized neuron — **stateful** cycle-based Phase 1 + Phase 2.
///
/// Called once per active cycle by ``Core::compute``.
///
///   First call  (Phase 1):  V = Σ w_j · a_j + bias   (initial charge), and the
///                           fire step k taken once by ``timing::fire_step``
///   Each  call  (Phase 2):  at step k → latch output (S − k) / S
///   After S calls:          return cached output
///
/// The decision is the one timing decision (``timing_decision.hpp``): on the
/// integer chip's lattice the first step with V + k θ/S reaching θ, decided in
/// integers (the Python executor's rule); every other neuron by the ramp
/// V += θ/S compared with the config-driven comparator, as before.
///
/// The neuron fires **at most once**; after that, the output is latched.
/// ``reset()`` must be called between input samples.
///
template <typename Config, int S, typename ComparePolicy>
class NeuronCompute<Config, TTFSQuantizedCompute<S, ComparePolicy>>
{
    using weight_t = typename Config::weight_t;
    using threshold_t = typename Config::threshold_t;
    using weights_array_t = std::array<Weight<weight_t>, Config::axon_count_>;
    using signal_t = typename TTFSQuantizedCompute<S, ComparePolicy>::signal_t;

    weights_array_t weights_{};
    Bias<weight_t> bias_{};
    threshold_t threshold_{1};
    timing::LatticeNeuron lattice_{};  // the lattice reading's constants, taken when the parameters load

    // --- Per-sample state (cleared by reset()) ---
    signal_t output_{0};
    bool initialized_{false};
    int fire_step_{-1};
    int steps_taken_{0};

public:
    constexpr signal_t
    operator()(const auto& incoming_signal)
    {
        // Phase 1: first active cycle — the membrane (on the lattice, or the
        // initial charge V = Σ w_j · a_j + bias) and the fire step.
        if (!initialized_)
        {
            const signal_t theta = static_cast<signal_t>(threshold_);
            timing::Membrane membrane = timing::lattice_membrane(lattice_, weights_, incoming_signal, S);
            if (!membrane.lattice)
            {
                membrane.v = std::inner_product(
                    std::cbegin(weights_), std::cend(weights_),
                    std::cbegin(incoming_signal),
                    signal_t{0}
                ) + bias_;
            }
            fire_step_ = timing::fire_step(
                membrane, theta, S, timing::strict_compare<ComparePolicy>(), timing::FloatPath::ramp);
            initialized_ = true;
        }

        // All S steps already done — return cached output.
        if (steps_taken_ >= S) return output_;

        // Phase 2: the output latches at the fire step.
        if (steps_taken_ == fire_step_)
        {
            output_ = static_cast<signal_t>(timing::level(S, fire_step_));
        }

        ++steps_taken_;

        return output_;
    }

    constexpr void reset()
    {
        output_ = 0;
        initialized_ = false;
        fire_step_ = -1;
        steps_taken_ = 0;
    }

    constexpr
    void load_weights(
        const NeuronWeights<Config{}, typename Config::weight_t, typename Config::threshold_t>& weights)
    {
        weights_ = weights.weights_;
        bias_ = weights.bias_;
        threshold_ = weights.threshold_;
        lattice_ = timing::lattice_neuron(
            weights_, static_cast<double>(bias_), static_cast<double>(threshold_), S);
    }
};

} // namespace nevresim
