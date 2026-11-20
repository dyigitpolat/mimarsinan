#pragma once

#include "simulator/spike_generation/stochastic_spike_generator.hpp"
#include "simulator/spike_generation/front_loaded_spike_generator.hpp"
#include "simulator/spike_generation/uniform_spike_generator.hpp"
#include "simulator/execution/spiking_execution.hpp"
#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/real_valued_compute.hpp"
#include "simulator/compute_policy/fire_policy/default_fire.hpp"

#include "_tests/test_util.hpp"
#include "_tests/xor_test/xor_chip.hpp"

#include <iostream> 

namespace nevresim::tests
{

template <typename FirePolicy = DefaultFirePolicy>
bool test_xor()
{
    using weight_t = double;

    static constinit auto chip = 
        generate_xor_chip<SpikingCompute<FirePolicy>>();
    
    using spiking_1000_fl = 
        SpikingExecution<
            999, 1, 2, FrontLoadedSpikeGenerator, weight_t, FirePolicy>;

    using spiking_1000_s = 
        SpikingExecution<
            999, 1, 2, StochasticSpikeGenerator, weight_t, FirePolicy>;
    
    using spiking_1000_u = 
        SpikingExecution<
            999, 1, 2, UniformSpikeGenerator, weight_t, FirePolicy>;

    load_weights<weight_t, weight_t>(
        chip, "include/_tests/xor_test/xor_chip_weights.txt");

    auto [correct, total] = test_on_inputs<spiking_1000_fl>(
        chip,"include/_tests/xor_test/xor_inputs/xor_input_", 4);

    auto [correct_2, total_2] = test_on_inputs<spiking_1000_s>(
        chip,"include/_tests/xor_test/xor_inputs/xor_input_", 4);

    auto [correct_3, total_3] = test_on_inputs<spiking_1000_u>(
        chip,"include/_tests/xor_test/xor_inputs/xor_input_", 4);

    return 
        (correct == total) && 
        (correct_2 == total_2) &&
        (correct_3 == total_3);
}

} // namespace nevresim::tests