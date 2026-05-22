// TTFS continuous (analytical): relu(I + bias) / threshold in one active step.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "attribute.hpp"
#include "mapped.hpp"
#include "pipeline.hpp"

class MimarsinanTtfsContinuousSoma : public sanafe::SomaUnit
{
public:
    MimarsinanTtfsContinuousSoma()
    {
        register_attributes({
                "threshold", "bias", "active_start", "active_length",
                "preset_membrane",
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

        if (!computed_[neuron_address])
        {
            double v;
            if (has_preset_[neuron_address])
            {
                v = preset_membrane_[neuron_address];
            }
            else
            {
                v = bias_[neuron_address];
                if (current_in.has_value())
                {
                    v += current_in.value();
                }
            }
            const double th = std::max(threshold_[neuron_address], 1e-12);
            activation_[neuron_address] =
                    (v > 0.0) ? std::min(1.0, v / th) : 0.0;
            computed_[neuron_address] = true;
        }

        sanafe::PipelineResult output;
        output.status = (activation_[neuron_address] > 0.0)
                ? sanafe::fired : sanafe::updated;
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
        activation_.clear();
        threshold_.clear();
        bias_.clear();
        active_start_.clear();
        active_length_.clear();
        computed_.clear();
        preset_membrane_.clear();
        has_preset_.clear();
    }

    void set_attribute_hw(const std::string &,
            const sanafe::ModelAttribute &) override
    {
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

private:
    std::vector<double> activation_;
    std::vector<double> threshold_;
    std::vector<double> bias_;
    std::vector<long int> active_start_;
    std::vector<long int> active_length_;
    std::vector<bool> computed_;
    std::vector<double> preset_membrane_;
    std::vector<bool> has_preset_;

    void ensure_capacity(std::size_t neuron_address)
    {
        if (neuron_address >= activation_.size())
        {
            const std::size_t new_size = neuron_address + 1;
            activation_.resize(new_size, 0.0);
            threshold_.resize(new_size, 1.0);
            bias_.resize(new_size, 0.0);
            active_start_.resize(new_size, 0L);
            active_length_.resize(new_size, 0L);
            computed_.resize(new_size, false);
            preset_membrane_.resize(new_size, 0.0);
            has_preset_.resize(new_size, false);
        }
    }
};

extern "C" sanafe::PipelineUnit *create_mimarsinan_ttfs_continuous_soma()
{
    return static_cast<sanafe::PipelineUnit *>(new MimarsinanTtfsContinuousSoma());
}
