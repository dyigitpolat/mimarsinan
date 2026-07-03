from mimarsinan.common.file_utils import *
from mimarsinan.code_generation.main_cpp_template import *
from mimarsinan.code_generation.main_cpp_template_real_valued_exec import *
from mimarsinan.code_generation.main_cpp_template_runtime import main_cpp_template_runtime
from mimarsinan.chip_simulation.spiking_mode_policy import (
    ExecPolicySpec,
    NevresimExecParams,
    policy_for_spiking_mode,
)


def resolve_compare_policy(thresholding_mode: str) -> str:
    """Map ``thresholding_mode`` to the nevresim compare policy type."""
    return "InclusiveCompare" if thresholding_mode == "<=" else "StrictCompare"


def resolve_lif_fire_policy(firing_mode: str, thresholding_mode: str) -> str:
    reset = "ZeroReset" if firing_mode == "Novena" else "SubtractiveReset"
    return f"LIFirePolicy<{reset}, {resolve_compare_policy(thresholding_mode)}>"


def _input_load_statement(spike_gen_mode: str, generated_files_path: str) -> str:
    if spike_gen_mode == "SpikeTrain":
        return (
            f'auto [input, target] = load_spike_train_input_n('
            f'"{generated_files_path}/inputs/", idx);'
        )
    return (
        f'auto [input, target] = load_input_n('
        f'"{generated_files_path}/inputs/", idx);'
    )


def resolve_exec_policy(
    *,
    spiking_mode: str,
    firing_mode: str,
    thresholding_mode: str,
    spike_gen_mode: str,
    weight_type: str,
    simulation_length: int,
    latency: int,
    output_count: int,
) -> ExecPolicySpec:
    """Return the (ComputePolicy, Execution) C++ types for nevresim main.

    Delegates the firing × sync → codegen choice to ``SpikingModePolicy`` and
    selects the comparator and LIF reset/fire strings as the thresholding/firing SSOT.
    """
    params = NevresimExecParams(
        compare=resolve_compare_policy(thresholding_mode),
        lif_fire_policy=resolve_lif_fire_policy(firing_mode, thresholding_mode),
        spike_gen_mode=spike_gen_mode,
        weight_type=weight_type,
        simulation_length=simulation_length,
        latency=latency,
        output_count=output_count,
    )
    return policy_for_spiking_mode(spiking_mode).nevresim_exec_policy(params)


def _build_chip_and_exec_decl(
    *,
    spiking_mode: str,
    firing_mode: str,
    thresholding_mode: str,
    spike_gen_mode: str,
    weight_type: str,
    threshold_type: str,
    simulation_length: int,
    latency: int,
    output_count: int,
) -> str:
    """Return C++ lines declaring ``chip`` and ``exec`` for compile-time connectivity."""
    spec = resolve_exec_policy(
        spiking_mode=spiking_mode,
        firing_mode=firing_mode,
        thresholding_mode=thresholding_mode,
        spike_gen_mode=spike_gen_mode,
        weight_type=weight_type,
        simulation_length=simulation_length,
        latency=latency,
        output_count=output_count,
    )
    return (
        f"static constinit auto chip = \n"
        f"        generate_chip<{spec.compute_policy}, {weight_type}, {threshold_type}>();\n"
        f"\n"
        f"    {spec.exec_decl}"
    )


def get_config(
    spike_gen_mode="Stochastic",
    firing_mode="Default",
    weight_type="double",
    spiking_mode="lif",
    threshold_type=None,
    thresholding_mode="<=",
):
    if threshold_type is None:
        threshold_type = weight_type
    return {
        "spike_gen_mode": spike_gen_mode,
        "firing_mode": firing_mode,
        "thresholding_mode": thresholding_mode,
        "weight_type": weight_type,
        "threshold_type": threshold_type,
        "spiking_mode": spiking_mode,
    }


def _build_runtime_exec_decl(
    *,
    spiking_mode: str,
    firing_mode: str,
    thresholding_mode: str,
    spike_gen_mode: str,
    weight_type: str,
    simulation_length: int,
    latency: int,
    output_count: int,
) -> tuple[str, str]:
    """Return (compute_policy_type, exec_type_decl) for runtime-chip main."""
    spec = resolve_exec_policy(
        spiking_mode=spiking_mode,
        firing_mode=firing_mode,
        thresholding_mode=thresholding_mode,
        spike_gen_mode=spike_gen_mode,
        weight_type=weight_type,
        simulation_length=simulation_length,
        latency=latency,
        output_count=output_count,
    )
    return spec.compute_policy, spec.exec_decl


def generate_main_function(
    generated_files_path,
    input_count,
    output_count,
    simulation_length,
    latency,
    cpp_code_template=main_cpp_template,
    simulation_config=get_config(),
    verbose=True,
):
    if verbose:
        print("Generating main function code...")

    chip_exec_decl = _build_chip_and_exec_decl(
        spiking_mode=simulation_config["spiking_mode"],
        firing_mode=simulation_config["firing_mode"],
        thresholding_mode=simulation_config.get("thresholding_mode", "<="),
        spike_gen_mode=simulation_config["spike_gen_mode"],
        weight_type=simulation_config["weight_type"],
        threshold_type=simulation_config["threshold_type"],
        simulation_length=simulation_length,
        latency=latency,
        output_count=output_count,
    )

    main_cpp_code = cpp_code_template.format(
        generated_files_path,
        input_count,
        simulation_length,
        simulation_config["spike_gen_mode"],
        simulation_config["firing_mode"],
        simulation_config["weight_type"],
        output_count,
        latency,
        chip_exec_decl,
        simulation_config["threshold_type"],
        _input_load_statement(
            simulation_config["spike_gen_mode"], generated_files_path,
        ),
    )

    main_cpp_filename = "{}/main/main.cpp".format(generated_files_path)

    prepare_containing_directory(main_cpp_filename)
    f = open(main_cpp_filename, "w")
    f.write(main_cpp_code)
    f.close()


def generate_main_function_runtime(
    generated_files_path,
    input_count,
    output_count,
    simulation_length,
    latency,
    simulation_config=get_config(),
    verbose=True,
):
    if verbose:
        print("Generating runtime main function code...")

    compute_policy, exec_decl = _build_runtime_exec_decl(
        spiking_mode=simulation_config["spiking_mode"],
        firing_mode=simulation_config["firing_mode"],
        thresholding_mode=simulation_config.get("thresholding_mode", "<="),
        spike_gen_mode=simulation_config["spike_gen_mode"],
        weight_type=simulation_config["weight_type"],
        simulation_length=simulation_length,
        latency=latency,
        output_count=output_count,
    )

    main_cpp_code = main_cpp_template_runtime.format(
        generated_files_path,
        input_count,
        simulation_length,
        simulation_config["spike_gen_mode"],
        simulation_config["firing_mode"],
        simulation_config["weight_type"],
        compute_policy,
        latency,
        exec_decl,
        simulation_config["threshold_type"],
        _input_load_statement(
            simulation_config["spike_gen_mode"], generated_files_path,
        ),
    )

    main_cpp_filename = "{}/main/main.cpp".format(generated_files_path)
    prepare_containing_directory(main_cpp_filename)
    with open(main_cpp_filename, "w") as f:
        f.write(main_cpp_code)


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
