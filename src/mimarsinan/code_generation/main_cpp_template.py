main_cpp_template = \
"""
#include "generate_chip.hpp"
#include "_tests/all.hpp"
#include <cstdlib>

namespace nevresim::tests {{
    
void test_main(int start, int end)
{{
    static constinit auto chip = 
        generate_chip<SpikingCompute>();
    
    using exec = SpikingExecution<{2}, StochasticSpikeGenerator>;

    load_weights(
        chip, "{0}weights/chip_weights.txt");

    for(int idx = start; idx < end; ++idx)
    {{
        auto [input, target] = load_input_n("{0}inputs/", idx);
        auto buffer = chip.template execute<exec>(input);
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