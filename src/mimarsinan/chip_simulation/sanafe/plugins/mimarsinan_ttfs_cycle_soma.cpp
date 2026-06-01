// mimarsinan_ttfs_cycle_soma — genuine single-spike TTFS soma (synchronized schedule).
//
// Unlike mimarsinan_ttfs_quantized_soma (which injects a preset analytical
// membrane and effectively propagates real values), this soma is genuinely
// event-driven: it reconstructs V from the *timings* of incoming single spikes
// and fires exactly once. It is meant to run under the SYNCHRONIZED schedule —
// latency groups execute sequentially, each in its own S-cycle window
// [g*S, (g+1)*S) — so by the time a neuron's window starts, all its inputs have
// arrived and V is complete (causal). No preset_membrane.
//
// Decode-on-arrival: a source neuron in group g' fires one spike at
// (g'*S + k_src); after the input->synapse pipeline delay it is delivered at
// cycle c with (c-1) % S == k_src, carrying activation a = (S - k_src)/S. So
//   V += current_in(c) * (S - ((c-1) mod S)) / S
// accumulated over the input phase reconstructs V = sum_j W_ij * a_j (+ bias).
// At the neuron's window start the fire step k_fire = ceil(S*(1 - V/theta)) is
// computed, and a single spike is emitted at (active_start + k_fire).
//
// NOTE: the exact pipeline-delay offset is validated against the analytical
// reference via the cross-backend parity harness.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "attribute.hpp"
#include "mapped.hpp"
#include "pipeline.hpp"

class MimarsinanTtfsCycleSoma : public sanafe::SomaUnit
{
public:
    MimarsinanTtfsCycleSoma()
    {
        register_attributes({
                "threshold", "bias", "active_start", "active_length",
                "simulation_length",
        });
    }

    sanafe::PipelineResult update(std::size_t neuron_address,
            std::optional<double> current_in,
            long int simulation_time) override
    {
        ensure_capacity(neuron_address);
        const long int cycle = simulation_time - 1;
        const long int start = active_start_[neuron_address];
        const int s = simulation_length_ > 0 ? simulation_length_ : 1;

        if (!initialized_[neuron_address])
        {
            membrane_[neuron_address] = bias_[neuron_address];
            initialized_[neuron_address] = true;
            has_fired_[neuron_address] = false;
            fire_step_[neuron_address] = -1;
            emitted_fire_[neuron_address] = false;
            computed_[neuron_address] = false;
            activation_[neuron_address] = 0.0;
        }

        // Input phase: integrate incoming single spikes, decoding each by its
        // arrival timing back to the source activation (a = (S - k_src)/S).
        // ``cycle <= start`` so an input spike delivered exactly at the window
        // boundary (source k_src = S-1, one cycle of input->synapse delay) is
        // still integrated before V is resolved.
        if (cycle <= start && current_in.has_value())
        {
            const long int local = ((cycle - 1) % s + s) % s;  // k_src
            const double weight = static_cast<double>(s - local) / static_cast<double>(s);
            membrane_[neuron_address] += current_in.value() * weight;
        }

        // At the window start V is complete (all earlier groups done) — resolve
        // the single fire step from the reconstructed membrane.
        if (cycle == start && !computed_[neuron_address])
        {
            apply_fire_step_from_membrane(neuron_address, s);
            computed_[neuron_address] = true;
        }

        sanafe::NeuronStatus status = sanafe::updated;
        if (computed_[neuron_address] && has_fired_[neuron_address]
                && !emitted_fire_[neuron_address] && fire_step_[neuron_address] >= 0
                && (cycle - start) == fire_step_[neuron_address])
        {
            status = sanafe::fired;
            emitted_fire_[neuron_address] = true;
        }

        sanafe::PipelineResult output;
        output.status = status;
        return output;
    }

    double get_potential(std::size_t neuron_address) override
    {
        if (neuron_address >= activation_.size())
        {
            return 0.0;
        }
        return activation_[neuron_address];
    }

    void reset() override
    {
        membrane_.clear();
        activation_.clear();
        threshold_.clear();
        bias_.clear();
        active_start_.clear();
        active_length_.clear();
        initialized_.clear();
        has_fired_.clear();
        fire_step_.clear();
        emitted_fire_.clear();
        computed_.clear();
        simulation_length_ = 0;
    }

    void set_attribute_hw(const std::string &attribute_name,
            const sanafe::ModelAttribute &param) override
    {
        if (attribute_name == "simulation_length")
        {
            simulation_length_ = static_cast<int>(param);
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

    void apply_fire_step_from_membrane(std::size_t neuron_address, int s)
    {
        const double theta = std::max(threshold_[neuron_address], 1e-12);
        const double v = membrane_[neuron_address];
        const double k_fire_raw =
                std::ceil(static_cast<double>(s) * (1.0 - v / theta));
        if (k_fire_raw >= static_cast<double>(s))
        {
            activation_[neuron_address] = 0.0;
            has_fired_[neuron_address] = false;
            fire_step_[neuron_address] = -1;
            return;
        }
        const int k_fire = std::max(
                0, std::min(static_cast<int>(k_fire_raw), s - 1));
        fire_step_[neuron_address] = k_fire;
        activation_[neuron_address] =
                static_cast<double>(s - k_fire) / static_cast<double>(s);
        has_fired_[neuron_address] = activation_[neuron_address] > 0.0;
    }

private:
    std::vector<double> membrane_;
    std::vector<double> activation_;
    std::vector<double> threshold_;
    std::vector<double> bias_;
    std::vector<long int> active_start_;
    std::vector<long int> active_length_;
    std::vector<bool> initialized_;
    std::vector<bool> has_fired_;
    std::vector<int> fire_step_;
    std::vector<bool> emitted_fire_;
    std::vector<bool> computed_;
    int simulation_length_{0};

    void ensure_capacity(std::size_t neuron_address)
    {
        if (neuron_address >= membrane_.size())
        {
            const std::size_t new_size = neuron_address + 1;
            membrane_.resize(new_size, 0.0);
            activation_.resize(new_size, 0.0);
            threshold_.resize(new_size, 1.0);
            bias_.resize(new_size, 0.0);
            active_start_.resize(new_size, 0L);
            active_length_.resize(new_size, 0L);
            initialized_.resize(new_size, false);
            has_fired_.resize(new_size, false);
            fire_step_.resize(new_size, -1);
            emitted_fire_.resize(new_size, false);
            computed_.resize(new_size, false);
        }
    }
};

extern "C" sanafe::PipelineUnit *create_mimarsinan_ttfs_cycle_soma()
{
    return static_cast<sanafe::PipelineUnit *>(new MimarsinanTtfsCycleSoma());
}
