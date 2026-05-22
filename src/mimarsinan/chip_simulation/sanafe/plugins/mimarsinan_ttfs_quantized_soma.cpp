// TTFS quantized: nevresim NeuronCompute<TTFSQuantizedCompute<S>> cycle semantics.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "attribute.hpp"
#include "mapped.hpp"
#include "pipeline.hpp"

class MimarsinanTtfsQuantizedSoma : public sanafe::SomaUnit
{
public:
    MimarsinanTtfsQuantizedSoma()
    {
        register_attributes({
                "threshold", "bias", "active_start", "active_length",
                "simulation_length", "preset_membrane",
        });
    }

    sanafe::PipelineResult update(std::size_t neuron_address,
            std::optional<double> current_in,
            long int simulation_time) override
    {
        ensure_capacity(neuron_address);
        const long int cycle = simulation_time - 1;
        const long int start = active_start_[neuron_address];
        const long int len = active_length_[neuron_address];
        if (len > 0 && (cycle < start || cycle >= start + len))
        {
            sanafe::PipelineResult idle_out;
            idle_out.status = sanafe::idle;
            return idle_out;
        }

        const int s = simulation_length_ > 0 ? simulation_length_ : static_cast<int>(len);
        if (s <= 0)
        {
            sanafe::PipelineResult idle_out;
            idle_out.status = sanafe::idle;
            return idle_out;
        }

        if (!initialized_[neuron_address])
        {
            if (has_preset_[neuron_address])
            {
                membrane_[neuron_address] = preset_membrane_[neuron_address];
            }
            else
            {
                membrane_[neuron_address] = bias_[neuron_address];
                if (current_in.has_value())
                {
                    membrane_[neuron_address] += current_in.value();
                }
            }
            initialized_[neuron_address] = true;
            steps_[neuron_address] = 0;
            has_fired_[neuron_address] = false;
            activation_[neuron_address] = 0.0;
            fire_step_[neuron_address] = -1;
            emitted_fire_[neuron_address] = false;
            if (has_preset_[neuron_address])
            {
                apply_quantized_activation_from_membrane(neuron_address, s);
            }
        }

        if (steps_[neuron_address] >= s)
        {
            sanafe::PipelineResult out;
            out.status = sanafe::updated;
            return out;
        }

        sanafe::NeuronStatus status = sanafe::updated;

        if (has_preset_[neuron_address])
        {
            if (has_fired_[neuron_address] && !emitted_fire_[neuron_address]
                    && fire_step_[neuron_address] >= 0
                    && steps_[neuron_address] == fire_step_[neuron_address])
            {
                status = sanafe::fired;
                emitted_fire_[neuron_address] = true;
            }
        }
        else
        {
            const double theta = std::max(threshold_[neuron_address], 1e-12);
            if (!has_fired_[neuron_address]
                    && membrane_[neuron_address] >= theta)
            {
                fire_step_[neuron_address] = steps_[neuron_address];
                activation_[neuron_address] =
                        static_cast<double>(s - steps_[neuron_address])
                        / static_cast<double>(s);
                has_fired_[neuron_address] = true;
                if (!emitted_fire_[neuron_address])
                {
                    status = sanafe::fired;
                    emitted_fire_[neuron_address] = true;
                }
            }
            membrane_[neuron_address] += theta / static_cast<double>(s);
        }

        ++steps_[neuron_address];

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
        steps_.clear();
        preset_membrane_.clear();
        has_preset_.clear();
        fire_step_.clear();
        emitted_fire_.clear();
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
        else if (attribute_name == "preset_membrane")
        {
            preset_membrane_[neuron_address] = static_cast<double>(param);
            has_preset_[neuron_address] = true;
        }
    }

    void apply_quantized_activation_from_membrane(std::size_t neuron_address, int s)
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
    std::vector<int> steps_;
    std::vector<double> preset_membrane_;
    std::vector<bool> has_preset_;
    std::vector<int> fire_step_;
    std::vector<bool> emitted_fire_;
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
            steps_.resize(new_size, 0);
            preset_membrane_.resize(new_size, 0.0);
            has_preset_.resize(new_size, false);
            fire_step_.resize(new_size, -1);
            emitted_fire_.resize(new_size, false);
        }
    }
};

extern "C" sanafe::PipelineUnit *create_mimarsinan_ttfs_quantized_soma()
{
    return static_cast<sanafe::PipelineUnit *>(new MimarsinanTtfsQuantizedSoma());
}
