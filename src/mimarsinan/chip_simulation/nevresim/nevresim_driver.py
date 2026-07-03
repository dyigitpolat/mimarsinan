from mimarsinan.mapping.export.chip_export import hard_cores_to_chip
from mimarsinan.common.file_utils import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.chip_simulation.nevresim.compile_nevresim import *
from mimarsinan.chip_simulation.nevresim.execute_nevresim import *
from mimarsinan.chip_simulation.nevresim.segment_execute import run_binary_raw
from mimarsinan.chip_simulation.nevresim.compile_cache import (
    NevresimCompileCache,
    cache_key,
    mapping_connectivity_hash,
    policy_hash,
)
from mimarsinan.chip_simulation.nevresim.connectivity import (
    ConnectivityMode,
    default_nevresim_connectivity_mode,
)

import numpy as np
import shutil


def _python_to_cpp_type_name(py_type) -> str:
    """Map a Python numeric type to its C++ template-argument string."""
    if py_type is float:
        return "double"
    return py_type.__name__


class NevresimDriver:
    nevresim_path: str | None = None

    def __init__(
        self, input_buffer_size, hard_core_mapping, generated_files_path, weight_type,
        spike_generation_mode="Stochastic", firing_mode="Default", spiking_mode="lif",
        thresholding_mode="<=",
        threshold_type=None, verbose=True,
        connectivity_mode: ConnectivityMode | None = None,
        compile_cache_dir: str | None = None,
    ):
        assert NevresimDriver.nevresim_path is not None, "nevresim path is not set."

        self.spike_generation_mode = spike_generation_mode
        self.firing_mode = firing_mode
        self.thresholding_mode = thresholding_mode
        self.spiking_mode = spiking_mode
        self.weight_type = weight_type
        self.threshold_type = threshold_type if threshold_type is not None else weight_type
        self.connectivity_mode = (
            connectivity_mode
            if connectivity_mode is not None
            else default_nevresim_connectivity_mode()
        )
        self.compile_cache_dir = compile_cache_dir
        self.hard_core_mapping = hard_core_mapping

        self.chip = hard_cores_to_chip(
            input_buffer_size,
            hard_core_mapping,
            hard_core_mapping.axons_per_core,
            hard_core_mapping.neurons_per_core,
            leak=0,
            weight_type=self.weight_type,
            threshold_type=self.threshold_type)

        chip_json = self.chip.get_chip_json()
        with open(generated_files_path + "/chip.json", "w") as f:
            f.write(chip_json)

        self.generated_files_path = generated_files_path
        wt_cpp = _python_to_cpp_type_name(self.weight_type)
        tt_cpp = _python_to_cpp_type_name(self.threshold_type)
        save_weights_and_chip_code(
            self.chip,
            generated_files_path,
            verbose=verbose,
            connectivity_mode=self.connectivity_mode,
            weight_cpp=wt_cpp,
            threshold_cpp=tt_cpp,
        )

    def _simulator_output_to_predictions(self, simulator_output, number_of_classes):
        total_spikes = sum(simulator_output)
        print("  Total spikes: {}".format(total_spikes))

        prediction_count = int(len(simulator_output) / number_of_classes)
        output_array = np.array(simulator_output).reshape((prediction_count, number_of_classes))

        predictions = np.zeros(prediction_count, dtype=int)
        for i in range(prediction_count):
            predictions[i] = np.argmax(output_array[i])
        return predictions

    def emit_main(self, max_input_count, simulation_length, latency, verbose=True):
        """Generate main.cpp only (no compilation)."""
        if verbose:
            print(f"spiking mode: {self.spiking_mode}, firing mode: {self.firing_mode}")
        wt_cpp = _python_to_cpp_type_name(self.weight_type)
        tt_cpp = _python_to_cpp_type_name(self.threshold_type)
        cfg = get_config(
            self.spike_generation_mode, self.firing_mode,
            wt_cpp, self.spiking_mode, threshold_type=tt_cpp,
            thresholding_mode=self.thresholding_mode,
        )
        if self.connectivity_mode == "runtime":
            generate_main_function_runtime(
                self.generated_files_path, max_input_count, self.chip.output_size,
                simulation_length, latency, simulation_config=cfg, verbose=verbose,
            )
        else:
            generate_main_function(
                self.generated_files_path, max_input_count, self.chip.output_size,
                simulation_length, latency, main_cpp_template, cfg, verbose=verbose,
            )

    def _compile_cache_lookup(
        self, simulation_length: int, latency: int, output_path=None,
    ) -> str | None:
        if not self.compile_cache_dir:
            return None
        m_hash = mapping_connectivity_hash(self.hard_core_mapping)
        wt_cpp = _python_to_cpp_type_name(self.weight_type)
        tt_cpp = _python_to_cpp_type_name(self.threshold_type)
        p_hash = policy_hash(
            spiking_mode=self.spiking_mode,
            spike_generation_mode=self.spike_generation_mode,
            firing_mode=self.firing_mode,
            thresholding_mode=self.thresholding_mode,
            weight_type_name=wt_cpp,
            threshold_type_name=tt_cpp,
            simulation_length=int(simulation_length),
            latency=int(latency),
            connectivity_mode=self.connectivity_mode,
        )
        cache = NevresimCompileCache(self.compile_cache_dir)
        cached = cache.get_binary(cache_key(m_hash, p_hash))
        if cached is None:
            return None
        if output_path is not None:
            prepare_containing_directory(output_path)
            shutil.copy2(cached, output_path)
            return output_path
        return str(cached)

    def _compile_cache_store(
        self, binary_path: str, simulation_length: int, latency: int,
    ) -> None:
        if not self.compile_cache_dir or binary_path is None:
            return
        m_hash = mapping_connectivity_hash(self.hard_core_mapping)
        wt_cpp = _python_to_cpp_type_name(self.weight_type)
        tt_cpp = _python_to_cpp_type_name(self.threshold_type)
        p_hash = policy_hash(
            spiking_mode=self.spiking_mode,
            spike_generation_mode=self.spike_generation_mode,
            firing_mode=self.firing_mode,
            thresholding_mode=self.thresholding_mode,
            weight_type_name=wt_cpp,
            threshold_type_name=tt_cpp,
            simulation_length=int(simulation_length),
            latency=int(latency),
            connectivity_mode=self.connectivity_mode,
        )
        NevresimCompileCache(self.compile_cache_dir).store_binary(
            cache_key(m_hash, p_hash), binary_path,
        )

    def emit_main_and_compile(
        self, max_input_count, simulation_length, latency, output_path=None
    ) -> str:
        """Generate main.cpp and compile. Returns binary path; raises on failure."""
        cached = self._compile_cache_lookup(simulation_length, latency, output_path)
        if cached is not None:
            self.emit_main(max_input_count, simulation_length, latency, verbose=False)
            return cached

        self.emit_main(max_input_count, simulation_length, latency)
        simulator_filename = compile_simulator(
            self.generated_files_path, NevresimDriver.nevresim_path, output_path=output_path
        )
        if simulator_filename is None:
            raise Exception("Compilation failed.")
        self._compile_cache_store(simulator_filename, simulation_length, latency)
        return simulator_filename

    def _run_simulator(
        self, input_loader, simulation_length, latency, max_input_count=None, num_proc=0,
        simulator_filename: str | None = None,
    ):
        """Save inputs, optionally compile, run, return raw float list."""
        if max_input_count is None:
            max_input_count = len(input_loader)
            print("  Max input count:", max_input_count)

        if simulator_filename is None:
            simulator_filename = self.emit_main_and_compile(
                max_input_count, simulation_length, latency
            )

        raw = run_binary_raw(
            binary_path=simulator_filename,
            work_dir=self.generated_files_path,
            input_loader=input_loader,
            output_size=self.chip.output_size,
            simulation_length=int(simulation_length),
            input_size=self.chip.input_size,
            spike_generation_mode=self.spike_generation_mode,
            max_input_count=max_input_count,
            num_proc=num_proc,
        )
        return raw.reshape(-1).tolist(), max_input_count

    def predict_spiking(self, input_loader, simulation_length, latency, max_input_count=None, num_proc=0):
        simulator_output, _ = self._run_simulator(
            input_loader, simulation_length, latency, max_input_count, num_proc)
        return self._simulator_output_to_predictions(simulator_output, self.chip.output_size)

    def predict_spiking_raw(self, input_loader, simulation_length, latency, max_input_count=None, num_proc=0):
        """Run simulation and return raw spike counts as (num_samples, num_outputs)."""
        simulator_output, max_input_count = self._run_simulator(
            input_loader, simulation_length, latency, max_input_count, num_proc)
        num_outputs = self.chip.output_size
        prediction_count = int(len(simulator_output) / num_outputs)
        return np.array(simulator_output).reshape((prediction_count, num_outputs))

    def emit_main_and_compile_recording(
        self, max_input_count, simulation_length, latency, output_path=None,
    ) -> str:
        """Compile a dedicated ``NEVRESIM_RECORD_SPIKES`` binary (per-core spike
        counts on stderr). Never cached — keep it off the production binary path."""
        self.emit_main(max_input_count, simulation_length, latency, verbose=False)
        binary = compile_simulator(
            self.generated_files_path, NevresimDriver.nevresim_path,
            output_path=output_path, verbose=False,
            extra_flags=["-DNEVRESIM_RECORD_SPIKES"],
        )
        if binary is None:
            raise Exception("Recording-build compilation failed.")
        return binary

    def predict_spiking_raw_with_records(
        self, input_loader, simulation_length, latency,
        max_input_count=None, num_proc=1,
    ):
        """Run a recording build and return ``(raw, spike_records)`` — per-sample
        ``{core: {"in","out"}}`` counts windowed to ``[lat, lat+T)``, the nevresim
        analogue of HCM ``CoreSpikeCounts`` (single-process to keep sample order)."""
        if max_input_count is None:
            max_input_count = len(input_loader)
        binary = self.emit_main_and_compile_recording(
            max_input_count, simulation_length, latency,
        )
        raw, spike_records = run_binary_raw(
            binary_path=binary,
            work_dir=self.generated_files_path,
            input_loader=input_loader,
            output_size=self.chip.output_size,
            simulation_length=int(simulation_length),
            input_size=self.chip.input_size,
            spike_generation_mode=self.spike_generation_mode,
            max_input_count=max_input_count,
            num_proc=num_proc,
            record_spikes=True,
        )
        return raw, spike_records

    def predict_spiking_raw_from_binary(
        self,
        simulator_filename: str,
        input_loader,
        simulation_length,
        latency,
        max_input_count=None,
        num_proc=0,
    ) -> np.ndarray:
        """Run simulation using a pre-compiled binary. Does not compile."""
        simulator_output, max_input_count = self._run_simulator(
            input_loader, simulation_length, latency, max_input_count, num_proc,
            simulator_filename=simulator_filename,
        )
        num_outputs = self.chip.output_size
        prediction_count = int(len(simulator_output) / num_outputs)
        return np.array(simulator_output).reshape((prediction_count, num_outputs))
