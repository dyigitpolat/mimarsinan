main_cpp_template = \
"""
#include "generate_chip.hpp"
#include "simulator/compute_policy/fire_policy/novena_fire.hpp"
#include "simulator/compute_policy/fire_policy/ttfs_fire.hpp"
#include "_tests/all.hpp"

#include "simulator/spike_generation/stochastic_spike_generator.hpp"
#include "simulator/spike_generation/deterministic_spike_generator.hpp"
#include "simulator/spike_generation/front_loaded_spike_generator.hpp"
#include "simulator/spike_generation/uniform_spike_generator.hpp"
#include "simulator/spike_generation/ttfs_spike_generator.hpp"
#include "simulator/execution/spiking_execution.hpp"
#include "simulator/execution/ttfs_execution.hpp"
#include "simulator/execution/ttfs_continuous_execution.hpp"
#include "simulator/execution/real_valued_execution.hpp"
#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/real_valued_compute.hpp"
#include "simulator/compute_policy/ttfs_analytical_compute.hpp"
#include "simulator/compute_policy/ttfs_quantized_compute.hpp"
#include "simulator/compute_policy/fire_policy/novena_fire.hpp"

#include <cstdlib>

namespace nevresim::tests {{
    
void test_main(int start, int end)
{{
    using weight_t = {5};

    {8}

    load_weights<weight_t>(
        chip, "{0}/weights/chip_weights.txt");

    for(int idx = start; idx < end; ++idx)
    {{
        auto [input, target] = load_input_n("{0}/inputs/", idx);
        auto buffer = chip.execute<exec>(input);
        for(auto i : buffer)
        {{
            std::cout << i << ' ';
        }}

        chip.reset();
    }}
}}

}}

int main(int argc, char** argv)
{{
    nevresim::tests::test_main(
        std::atoi(argv[1]), std::atoi(argv[2]));
}}
"""
