main_cpp_template = \
"""
#include "generate_chip.hpp"
#include "_tests/all.hpp"

namespace nevresim::tests {{
    
void test_main()
{{
    static constinit auto chip = 
        generate_chip<SpikingCompute>();
    
    using exec = SpikingExecution<{2}, StochasticSpikeGenerator>;

    load_weights(
        chip, "{0}weights/chip_weights.txt");

    for(int idx = 0; idx < {1}; ++idx)
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

int main()
{{
    nevresim::tests::test_main();
}}
"""