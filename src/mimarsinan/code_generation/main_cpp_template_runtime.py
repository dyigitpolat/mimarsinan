main_cpp_template_runtime = \
"""
#include "generate_chip_config.hpp"
#include "simulator/chip/runtime_chip.hpp"
#include "loaders/mapping_loader.hpp"
#include "simulator/compute_policy/fire_policy/lif_fire_policy.hpp"
#include "simulator/compute_policy/compare_policy/strict_compare.hpp"
#include "simulator/compute_policy/compare_policy/inclusive_compare.hpp"
#include "simulator/compute_policy/reset_policy/subtractive_reset.hpp"
#include "simulator/compute_policy/reset_policy/zero_reset.hpp"
#include "simulator/compute_policy/fire_policy/ttfs_fire.hpp"
#include "simulator/spike_generation/stochastic_spike_generator.hpp"
#include "simulator/spike_generation/deterministic_spike_generator.hpp"
#include "simulator/spike_generation/front_loaded_spike_generator.hpp"
#include "simulator/spike_generation/uniform_spike_generator.hpp"
#include "simulator/spike_generation/ttfs_spike_generator.hpp"
#include "simulator/spike_generation/spike_train_spike_generator.hpp"
#include "simulator/execution/spiking_execution.hpp"
#include "simulator/execution/ttfs_execution.hpp"
#include "simulator/execution/ttfs_continuous_execution.hpp"
#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/ttfs_analytical_compute.hpp"
#include "simulator/compute_policy/ttfs_quantized_compute.hpp"
#include "_tests/test_util.hpp"

#include <cstdlib>
#include <iostream>

namespace nevresim::tests {{

void test_main_runtime(int start, int end)
{{
    using weight_t = {5};
    using threshold_t = {9};
    using cfg_t = RuntimeChipConfig<{6}, weight_t, threshold_t>;
    using chip_t = typename cfg_t::chip_t;

    // Large mappings (many cores x neurons) exceed default stack limits if
    // allocated locally; static storage matches compile-time ``constinit`` chips.
    static chip_t chip{{}};
    static bool chip_initialized = false;
    if (!chip_initialized)
    {{
        chip.mapping_mut() = load_mapping_from_spans_file<typename cfg_t::cfg>(
            "{0}/chip/chip_spans.txt");
        load_weights<weight_t, threshold_t>(chip, "{0}/weights/chip_weights.txt");
        chip_initialized = true;
    }}

    {8}

    for (int idx = start; idx < end; ++idx)
    {{
        {10}
        auto buffer = chip.execute<exec>(input);
        for (auto i : buffer)
            std::cout << i << ' ';
        std::cout << '\\n';
        chip.reset();
    }}
}}

}}

int main(int argc, char** argv)
{{
    nevresim::tests::test_main_runtime(
        std::atoi(argv[1]), std::atoi(argv[2]));
}}
"""
