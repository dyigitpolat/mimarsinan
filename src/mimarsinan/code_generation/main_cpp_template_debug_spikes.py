main_cpp_template_debug_spikes = \
"""
#include "generate_chip.hpp"
#include "simulator/compute_policy/fire_policy/novena_fire.hpp"
#include "_tests/all.hpp"

#include "simulator/spike_generation/stochastic_spike_generator.hpp"
#include "simulator/execution/spiking_execution.hpp"
#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/real_valued_compute.hpp"
#include "simulator/compute_policy/fire_policy/novena_fire.hpp"

#include <cstdlib>

namespace nevresim::tests {{
    
void test_main(int start, int end)
{{
    static constinit auto chip = 
        generate_chip<SpikingCompute<NovenaFirePolicy>>();
    
    using exec = SpikingExecution<1, StochasticSpikeGenerator, NovenaFirePolicy>;

    load_weights(
        chip, "{0}weights/chip_weights.txt");

    for(int idx = start; idx < end; ++idx)
    {{
        for(int j = 0; j < {2}; ++j)
        {{  
            auto [input, target] = load_input_n("{0}inputs/", idx);
            auto buffer = chip.template execute<exec>(input);
            for(int neuron{{}}; auto i : buffer)
            {{
                if(i) std::cout << neuron << ' ';
                neuron++;
            }}
            std::cout << -1 << ' ';
        }}
    }}
}}

}}

int main(int argc, char** argv)
{{
    nevresim::tests::test_main(
        std::atoi(argv[1]), std::atoi(argv[2]));
}}
"""