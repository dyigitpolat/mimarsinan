from mimarsinan.common.file_utils import *
from mimarsinan.code_generation.main_cpp_template import *
from mimarsinan.code_generation.main_cpp_template_real_valued_exec import *


def _build_chip_and_exec_decl(
    *,
    spiking_mode: str,
    firing_mode: str,
    spike_gen_mode: str,
    weight_type: str,
    simulation_length: int,
    latency: int,
    output_count: int,
) -> str:
    """
    Return the C++ lines that declare ``chip`` and ``exec`` inside
    ``test_main``.

    Dispatch is based on ``spiking_mode``:

    * **ttfs** (continuous / analytical): single-pass sweep using
      ``TTFSAnalyticalCompute`` (relu / threshold per neuron).
    * **ttfs_quantized** (cycle-based): ``TTFSExecution`` with
      ``TTFSQuantizedCompute`` (Phase 1 + Phase 2 per neuron).
    * **rate** (Default / Novena): standard ``SpikingExecution``.
    """
    if spiking_mode == "ttfs":
        # Continuous / analytical TTFS — single-pass, no cycles.
        return (
            f"static constinit auto chip = \n"
            f"        generate_chip<TTFSAnalyticalCompute, {weight_type}>();\n"
            f"\n"
            f"    using exec = TTFSContinuousExecution;"
        )
    elif spiking_mode == "ttfs_quantized":
        # Quantised TTFS — single-pass sweep, each neuron internally
        # steps through S discrete time steps (Phase 1 + Phase 2).
        return (
            f"static constinit auto chip = \n"
            f"        generate_chip<TTFSQuantizedCompute<{simulation_length}>, {weight_type}>();\n"
            f"\n"
            f"    using exec = TTFSExecution<{simulation_length}>;"
        )
    else:
        # Rate-coded modes (Default, Novena).
        return (
            f"static constinit auto chip = \n"
            f"        generate_chip<SpikingCompute<{firing_mode}FirePolicy>, {weight_type}>();\n"
            f"\n"
            f"    using exec = SpikingExecution<"
            f"{simulation_length}, {latency}, {output_count}, "
            f"{spike_gen_mode}SpikeGenerator, {weight_type}, "
            f"{firing_mode}FirePolicy>;"
        )


def get_config(spike_gen_mode="Stochastic", firing_mode="Default", weight_type="double", spiking_mode="rate"):
    return {
        "spike_gen_mode": spike_gen_mode,
        "firing_mode": firing_mode,
        "weight_type": weight_type,
        "spiking_mode": spiking_mode,
    }


def generate_main_function(
    generated_files_path,
    input_count,
    output_count,
    simulation_length,
    latency,
    cpp_code_template=main_cpp_template,
    simulation_config=get_config(),
):
    print("Generating main function code...")

    chip_exec_decl = _build_chip_and_exec_decl(
        spiking_mode=simulation_config["spiking_mode"],
        firing_mode=simulation_config["firing_mode"],
        spike_gen_mode=simulation_config["spike_gen_mode"],
        weight_type=simulation_config["weight_type"],
        simulation_length=simulation_length,
        latency=latency,
        output_count=output_count,
    )

    main_cpp_code = cpp_code_template.format(
        generated_files_path,   # {0}
        input_count,            # {1}
        simulation_length,      # {2}
        simulation_config["spike_gen_mode"],   # {3}
        simulation_config["firing_mode"],      # {4}
        simulation_config["weight_type"],      # {5}
        output_count,           # {6}
        latency,                # {7}
        chip_exec_decl,         # {8}  ← chip + exec declarations
    )

    main_cpp_filename = "{}/main/main.cpp".format(generated_files_path)

    prepare_containing_directory(main_cpp_filename)
    f = open(main_cpp_filename, "w")
    f.write(main_cpp_code)
    f.close()


def generate_main_function_for_real_valued_exec(
    generated_files_path,
    cpp_code_template=main_cpp_template_real_valued_exec,
):
    main_cpp_code = cpp_code_template.format(generated_files_path)

    main_cpp_filename = "{}/main/main.cpp".format(generated_files_path)

    prepare_containing_directory(main_cpp_filename)
    f = open(main_cpp_filename, "w")
    f.write(main_cpp_code)
    f.close()
