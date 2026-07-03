import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from mimarsinan.common.file_utils import prepare_containing_directory
from mimarsinan.common.build_utils import find_cpp20_compiler


@dataclass
class CompileResult:
    binary_path: str | None
    compile_s: float
    compiler: str
    compiler_family: str
    success: bool
    trace_json: str | None = None


def compile_simulator(
    generated_files_path,
    nevresim_path,
    output_path=None,
    verbose=True,
    *,
    optimization: str = "-O3",
    time_trace: bool = False,
    trace_output_dir: str | Path | None = None,
    extra_flags: list[str] | None = None,
    return_timing: bool = False,
) -> CompileResult | str | None:
    """Compile the nevresim simulator.

    Returns CompileResult when *return_timing* or *time_trace* is set;
    otherwise returns binary path string or None for backward compatibility.
    """
    if verbose:
        print("Compiling nevresim for mapped chip...")

    cc_command, family = find_cpp20_compiler()
    if cc_command is None:
        print("No C++20-capable compiler found.")
        return None if not (time_trace or return_timing) else CompileResult(None, 0.0, "", "", False)

    if verbose:
        print(f"  Using compiler: {cc_command} (family={family})")

    step_limit = 256 * 256 * 10000
    if family == "gcc":
        step_limit_option = f"-fconstexpr-ops-limit={step_limit}"
    else:
        step_limit_option = f"-fconstexpr-steps={step_limit}"

    # Default into the per-run tree: a shared binary path lets a concurrent run overwrite the binary another is executing (ETXTBSY / wrong simulator).
    if output_path is None:
        output_path = str(Path(generated_files_path) / "bin" / "simulator")
    simulator_filename = output_path
    prepare_containing_directory(simulator_filename)

    cmd = [
        cc_command,
        optimization,
        step_limit_option,
        "{}/main/main.cpp".format(generated_files_path),
        "-I", "{}/include".format(nevresim_path),
        "-I", "{}/chip".format(generated_files_path),
        "-std=c++20",
    ]

    if family == "clang":
        cmd.append("-stdlib=libc++")

    if time_trace and family == "clang":
        cmd.append("-ftime-trace")

    if extra_flags:
        cmd.extend(extra_flags)

    cmd += ["-o", simulator_filename]

    t0 = time.perf_counter()
    p = subprocess.Popen(cmd)
    rc = p.wait()
    compile_s = time.perf_counter() - t0
    success = rc == 0

    trace_json = None
    if time_trace and family == "clang" and success:
        search_dir = Path(generated_files_path) / "main"
        traces = sorted(search_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        trace_json = str(traces[0]) if traces else None

    if success:
        if verbose:
            print("Compilation outcome:", simulator_filename)
        result = CompileResult(
            str(simulator_filename), compile_s, cc_command, family, True, trace_json,
        )
        return result if (time_trace or return_timing) else result.binary_path
    print("Compilation failed.")
    result = CompileResult(None, compile_s, cc_command, family, False, trace_json)
    return result if (time_trace or return_timing) else None
