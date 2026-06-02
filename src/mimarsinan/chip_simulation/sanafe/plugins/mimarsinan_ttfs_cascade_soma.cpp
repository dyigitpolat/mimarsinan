// mimarsinan_ttfs_cascade_soma — genuine cascaded (greedy) TTFS soma.
//
// Hardware-faithful single-spike TTFS: each neuron fires **exactly once** (the
// information is in the timing), so only that single spike travels on the wire.
// The integration is a **ramp** reconstructed at the consumer: a single incoming
// spike on an axon at t_j contributes its weight every subsequent cycle. This
// soma realises the ramp via a persistent ``ramp_current`` accumulator:
//
//   ramp_current += current_in   (weighted single-spike arrivals this cycle)
//   membrane     += ramp_current + bias
//   fire once when membrane crosses threshold → single spike, then SILENT.
//
// Mirrors HCM's ``TTFSGreedyCyclePolicy`` (ramp_current + single-spike output)
// and nevresim's single-spike ``SpikingCompute`` neuron. No reset, no preset
// membrane, no per-group window. The firing comparator is config-driven
// (``thresholding_mode``: strict ``<`` vs inclusive ``<=``).

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "attribute.hpp"
#include "mapped.hpp"
#include "pipeline.hpp"

class MimarsinanTtfsCascadeSoma : public sanafe::SomaUnit
{
public:
    MimarsinanTtfsCascadeSoma()
    {
        register_attributes({"threshold", "bias", "thresholding_mode",
                "active_start", "active_length"});
    }

    sanafe::PipelineResult update(std::size_t neuron_address,
            std::optional<double> current_in,
            long int simulation_time) override
    {
        ensure_capacity(neuron_address);

        // Optional per-neuron active-window gate (0-based cycle); disabled when
        // ``active_length == 0`` — cascaded cores integrate from cycle 0.
        const long int cycle = simulation_time - 1;
        const long int start = active_start_[neuron_address];
        const long int len = active_length_[neuron_address];
        if (len > 0 && (cycle < start || cycle >= start + len))
        {
            sanafe::PipelineResult idle_out;
            idle_out.status = sanafe::idle;
            return idle_out;
        }

        // Ramp: accumulate weighted single-spike arrivals into a persistent
        // current, then integrate that ramp (+ bias) into the membrane. One
        // input spike at t_j therefore contributes its weight every subsequent
        // cycle (membrane(t) = Σ w_j·(t − t_j) + b·t).
        if (current_in.has_value())
        {
            ramp_current_[neuron_address] += current_in.value();
        }
        potential_[neuron_address] +=
                ramp_current_[neuron_address] + bias_[neuron_address];

        // Already fired → stay SILENT (single spike; the one already emitted
        // carried the timing). The membrane keeps ramping but no longer fires.
        if (has_fired_[neuron_address])
        {
            sanafe::PipelineResult idle_after_fire;
            idle_after_fire.status = sanafe::updated;
            return idle_after_fire;
        }

        const double v = potential_[neuron_address];
        const double vth = threshold_[neuron_address];
        const bool fired = inclusive_threshold_ ? (v >= vth) : (v > vth);

        sanafe::PipelineResult output;
        if (fired)
        {
            has_fired_[neuron_address] = true;
            output.status = sanafe::fired;  // single spike, on this cycle only
        }
        else
        {
            output.status = sanafe::updated;
        }
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
        ramp_current_.clear();
        threshold_.clear();
        bias_.clear();
        active_start_.clear();
        active_length_.clear();
        has_fired_.clear();
    }

    void set_attribute_hw(const std::string &attribute_name,
            const sanafe::ModelAttribute &param) override
    {
        if (attribute_name == "thresholding_mode")
        {
            const std::string mode = static_cast<std::string>(param);
            inclusive_threshold_ = (mode == "inclusive" || mode == "<=");
        }
    }

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
        else if (attribute_name == "active_start")
        {
            active_start_[neuron_address] =
                    static_cast<long int>(static_cast<int>(param));
        }
        else if (attribute_name == "active_length")
        {
            active_length_[neuron_address] =
                    static_cast<long int>(static_cast<int>(param));
        }
    }

private:
    std::vector<double> potential_;
    std::vector<double> ramp_current_;
    std::vector<double> threshold_;
    std::vector<double> bias_;
    std::vector<long int> active_start_;
    std::vector<long int> active_length_;
    std::vector<bool> has_fired_;
    // TTFS uses inclusive ``<=``; default matches the cascaded comparator.
    bool inclusive_threshold_{true};

    void ensure_capacity(std::size_t neuron_address)
    {
        if (neuron_address >= potential_.size())
        {
            const std::size_t new_size = neuron_address + 1;
            potential_.resize(new_size, 0.0);
            ramp_current_.resize(new_size, 0.0);
            threshold_.resize(new_size, 1.0);
            bias_.resize(new_size, 0.0);
            active_start_.resize(new_size, 0L);
            active_length_.resize(new_size, 0L);
            has_fired_.resize(new_size, false);
        }
    }
};

extern "C" sanafe::PipelineUnit *create_mimarsinan_ttfs_cascade_soma()
{
    return static_cast<sanafe::PipelineUnit *>(
            new MimarsinanTtfsCascadeSoma());
}
