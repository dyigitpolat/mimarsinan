import os
import subprocess
import time


def _parse_stdout_tokens(stdout: str) -> list[float]:
    if not stdout or not stdout.strip():
        return []
    return [float(val) for val in stdout.split()]


def parse_spike_records(stderr: str) -> list[dict[int, dict[str, list[int]]]]:
    """Parse ``SPKREC`` lines (NEVRESIM_RECORD_SPIKES build) into per-sample records.

    One ``SPKREC <core> IN ... OUT ...`` line per core; ``SPKREC_END`` closes a
    sample. Full per-axon/neuron arrays; trimming to used spans happens at projection."""
    samples: list[dict[int, dict[str, list[int]]]] = []
    current: dict[int, dict[str, list[int]]] = {}
    for line in (stderr or "").splitlines():
        line = line.strip()
        if not line.startswith("SPKREC"):
            continue
        if line == "SPKREC_END":
            samples.append(current)
            current = {}
            continue
        toks = line.split()
        core = int(toks[1])
        in_idx = toks.index("IN")
        out_idx = toks.index("OUT")
        in_counts = [int(v) for v in toks[in_idx + 1:out_idx]]
        out_counts = [int(v) for v in toks[out_idx + 1:]]
        current[core] = {"in": in_counts, "out": out_counts}
    return samples


def execute_simulator(
    simulator_filename,
    input_count,
    num_proc=0,
    *,
    expected_values: int | None = None,
    record_spikes: bool = False,
):
    """Run the nevresim binary across ``num_proc`` workers (0 ⇒ ``cpu_count() // 2``).

    Stdout protocol: per sample in ``[start, end)``, ``output_size`` whitespace-
    separated numbers then a newline; a mismatch vs ``expected_values`` raises."""
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
    spike_records: list[dict] = []
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
        if record_spikes:
            spike_records.extend(parse_spike_records(stderr))

    end_time = time.time()
    print("  Simulation time:", end_time - start_time)

    if expected_values is not None and len(output_values) != expected_values:
        raise RuntimeError(
            f"nevresim output size mismatch: expected {expected_values} values, "
            f"got {len(output_values)}"
        )

    if record_spikes:
        return output_values, spike_records
    return output_values
