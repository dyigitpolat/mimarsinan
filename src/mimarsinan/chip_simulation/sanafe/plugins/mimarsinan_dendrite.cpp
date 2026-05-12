// mimarsinan_dendrite — SANA-FE DendriteUnit plugin for mimarsinan.
//
// Replaces SANA-FE's built-in ``accumulator`` dendrite, which hardcodes its
// per-neuron state vectors to ``loihi_max_compartments`` (1024) and therefore
// caps a single core's neuron count at 1024.  mimarsinan supports
// arbitrary user-defined HardCore dimensions via ``platform_constraints.cores``,
// so any Loihi-derived cap is wrong for us.
//
// Semantics (matches the upstream accumulator):
//   * sums incoming synaptic ``current`` per neuron each timestep
//   * resets the per-neuron accumulator at the start of each new timestep
//   * forwards the accumulated current as the dendrite output
//
// Difference: per-neuron state grows on demand (``std::vector::resize``) instead
// of being pre-allocated to a fixed size.  No per-core neuron-count cap.

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "attribute.hpp"
#include "pipeline.hpp"

class MimarsinanDendrite : public sanafe::DendriteUnit
{
public:
    MimarsinanDendrite() = default;

    sanafe::PipelineResult update(std::size_t neuron_address,
            std::optional<double> current,
            std::optional<std::size_t> /*synapse_address*/,
            long int simulation_time) override
    {
        ensure_capacity(neuron_address);

        // Reset accumulator at the start of each new timestep.
        if (timesteps_simulated_[neuron_address] < simulation_time)
        {
            accumulated_charges_[neuron_address] = 0.0;
            timesteps_simulated_[neuron_address] = simulation_time;
        }
        if (current.has_value())
        {
            accumulated_charges_[neuron_address] =
                    accumulated_charges_[neuron_address].value_or(0.0) +
                    current.value();
        }

        sanafe::PipelineResult output;
        output.current = accumulated_charges_[neuron_address];
        return output;
    }

    void reset() override
    {
        accumulated_charges_.clear();
        timesteps_simulated_.clear();
    }

    // The mimarsinan dendrite has no tunable hardware attributes; we still
    // need stubs to satisfy the PipelineUnit interface.
    void set_attribute_hw(const std::string & /*attribute_name*/,
            const sanafe::ModelAttribute & /*param*/) override {}
    void set_attribute_neuron(std::size_t /*neuron_address*/,
            const std::string & /*attribute_name*/,
            const sanafe::ModelAttribute & /*param*/) override {}
    void set_attribute_edge(std::size_t /*synapse_address*/,
            const std::string & /*attribute_name*/,
            const sanafe::ModelAttribute & /*param*/) override {}

private:
    std::vector<std::optional<double>> accumulated_charges_;
    std::vector<long int> timesteps_simulated_;

    void ensure_capacity(std::size_t neuron_address)
    {
        if (neuron_address >= accumulated_charges_.size())
        {
            accumulated_charges_.resize(neuron_address + 1, std::nullopt);
            timesteps_simulated_.resize(neuron_address + 1, 0L);
        }
    }
};

// Factory invoked by ``sanafe::plugin_get_hw`` when the arch YAML names
// ``model: mimarsinan_dendrite`` and points ``plugin:`` at this shared library.
extern "C" sanafe::PipelineUnit *create_mimarsinan_dendrite()
{
    return static_cast<sanafe::PipelineUnit *>(new MimarsinanDendrite());
}
