import os
import subprocess
import time


def _parse_stdout_tokens(stdout: str) -> list[float]:
    if not stdout or not stdout.strip():
        return []
    return [float(val) for val in stdout.split()]


def execute_simulator(
    simulator_filename,
    input_count,
    num_proc=0,
    *,
    expected_values: int | None = None,
):
    """Run the nevresim binary. Supports both absolute and relative paths.

    When *num_proc* is 0 (default), uses ``os.cpu_count() // 2`` workers
    (at least 1).

    Binary stdout protocol: for each sample in ``[start, end)``, emit
    ``output_size`` whitespace-separated numbers followed by a newline.

    When *expected_values* is set (typically ``input_count * output_size``),
    raises ``RuntimeError`` if the parsed token count does not match.
    """
    if num_proc <= 0:
        num_proc = max(1, (os.cpu_count() or 2) // 2)
    num_proc = min(num_proc, input_count) if input_count > 0 else 1
    print(f"Executing simulator ({num_proc} processes)...")

    start_time = time.time()

    executable = (
        simulator_filename
        if os.path.isabs(simulator_filename)
        else "./{}".format(simulator_filename)
    )

    workers: list[tuple[int, int, subprocess.Popen]] = []
    for i in range(num_proc):
        start = i * input_count // num_proc
        end = (i + 1) * input_count // num_proc
        if i == num_proc - 1:
            end = input_count

        cmd = [executable, str(start), str(end)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        workers.append((start, end, proc))

    output_values: list[float] = []
    for start, end, proc in workers:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            err = (stderr or "").strip()
            raise RuntimeError(
                f"nevresim simulator failed for samples [{start}, {end}) "
                f"(exit {proc.returncode})"
                + (f": {err}" if err else "")
            )
        output_values.extend(_parse_stdout_tokens(stdout))

    end_time = time.time()
    print("  Simulation time:", end_time - start_time)

    if expected_values is not None and len(output_values) != expected_values:
        raise RuntimeError(
            f"nevresim output size mismatch: expected {expected_values} values, "
            f"got {len(output_values)}"
        )

    return output_values
