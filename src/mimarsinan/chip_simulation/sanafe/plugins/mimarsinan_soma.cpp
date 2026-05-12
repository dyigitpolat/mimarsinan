// mimarsinan_soma — SANA-FE SomaUnit plugin matching mimarsinan's
// ``SubtractiveLIFReset`` neuron model exactly, with dynamically-sized
// per-neuron state.
//
// Replaces SANA-FE's built-in ``leaky_integrate_fire`` soma (LoihiLifModel),
// which hardcodes ``compartments[loihi_max_compartments]`` (1024) and is
// otherwise Loihi-specific (leak schedule, refractory, noise stream).
// mimarsinan is its own hardware proposal; we want pure subtractive-IF
// semantics with no Loihi shape baked in, and we want the per-core neuron
// count limited only by the user-declared ``max_neurons`` in
// ``platform_constraints.cores``.
//
// Semantics (matches ``mimarsinan.chip_simulation.subtractive_lif``):
//   * du = 1, dv = 0 — no leak, no synaptic persistence.  Incoming current
//     is added to the potential each timestep, then thrown away (the
//     dendrite re-accumulates fresh charge every cycle).
//   * subtractive reset on fire: ``potential -= threshold``.
//   * configurable strict-`<` vs inclusive-`<=` thresholding mode (set via
//     the hardware-level ``thresholding_mode`` attribute).
//   * optional per-neuron ``bias``; bias is added to potential every cycle.
//   * dynamically-sized state, no compile-time per-core cap.
//
// What we deliberately *do not* model (would be wrong for mimarsinan):
//   * Loihi-style multiplicative leak (``leak_decay``).
//   * Refractory periods.
//   * Noise streams.

#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "attribute.hpp"
#include "mapped.hpp"
#include "pipeline.hpp"

class MimarsinanSoma : public sanafe::SomaUnit
{
public:
    MimarsinanSoma()
    {
        // Whitelist the attribute names we accept so SANA-FE doesn't warn
        // about unknown attrs.  These mirror the SubtractiveLIFReset surface.
        register_attributes({"threshold", "bias", "thresholding_mode"});
    }

    sanafe::PipelineResult update(std::size_t neuron_address,
            std::optional<double> current_in,
            long int /*simulation_time*/) override
    {
        ensure_capacity(neuron_address);

        // Integrate input current (du=1: current does not persist, it is
        // delivered fresh from the dendrite each cycle).
        if (current_in.has_value())
        {
            potential_[neuron_address] += current_in.value();
        }
        // Add per-neuron bias every cycle.
        potential_[neuron_address] += bias_[neuron_address];

        // Threshold check.
        const double v = potential_[neuron_address];
        const double vth = threshold_[neuron_address];
        const bool fired = inclusive_threshold_ ? (v >= vth) : (v > vth);

        sanafe::NeuronStatus status = sanafe::idle;
        if (fired)
        {
            // Subtractive reset.
            potential_[neuron_address] -= vth;
            status = sanafe::fired;
        }
        else
        {
            // We always touch the neuron every cycle; signal that.
            status = sanafe::updated;
        }

        sanafe::PipelineResult output;
        output.status = status;
        return output;
    }

    double get_potential(std::size_t neuron_address) override
    {
        if (neuron_address >= potential_.size())
        {
            return 0.0;
        }
        return potential_[neuron_address];
    }

    void reset() override
    {
        potential_.clear();
        threshold_.clear();
        bias_.clear();
    }

    // Hardware-level attributes are set once at arch-construction time.
    // ``thresholding_mode`` selects strict ``<`` (default) vs inclusive ``<=``.
    void set_attribute_hw(const std::string &attribute_name,
            const sanafe::ModelAttribute &param) override
    {
        if (attribute_name == "thresholding_mode")
        {
            const std::string mode = static_cast<std::string>(param);
            inclusive_threshold_ = (mode == "inclusive" || mode == "<=");
        }
    }

    // Per-neuron attributes: each LIF neuron has its own threshold + bias.
    void set_attribute_neuron(std::size_t neuron_address,
            const std::string &attribute_name,
            const sanafe::ModelAttribute &param) override
    {
        ensure_capacity(neuron_address);
        if (attribute_name == "threshold")
        {
            threshold_[neuron_address] = static_cast<double>(param);
        }
        else if (attribute_name == "bias")
        {
            bias_[neuron_address] = static_cast<double>(param);
        }
    }

    // No ``set_attribute_edge`` override — ``SomaUnit`` marks it ``final``
    // because somas don't have synaptic edges.

private:
    // Per-neuron state — all dynamically sized.
    std::vector<double> potential_;
    std::vector<double> threshold_;
    std::vector<double> bias_;
    // Strict ``<`` by default; flips to inclusive ``<=`` via hw attribute.
    bool inclusive_threshold_{false};

    void ensure_capacity(std::size_t neuron_address)
    {
        if (neuron_address >= potential_.size())
        {
            const std::size_t new_size = neuron_address + 1;
            potential_.resize(new_size, 0.0);
            threshold_.resize(new_size, 1.0);   // safe non-zero default
            bias_.resize(new_size, 0.0);
        }
    }
};

extern "C" sanafe::PipelineUnit *create_mimarsinan_soma()
{
    return static_cast<sanafe::PipelineUnit *>(new MimarsinanSoma());
}
