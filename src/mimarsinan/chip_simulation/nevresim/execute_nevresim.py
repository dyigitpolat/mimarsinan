"""Bounded multi-process execution of a compiled nevresim binary."""

import os
import subprocess
import time

from mimarsinan.models.spiking.serial.refusals import EMISSION_COUNT_CEILING
from mimarsinan.chip_simulation.execution_bounds import (
    SimulationTimeoutError,
    kill_process_group,
    resolve_simulation_step_timeout_s,
    retry_once_on_timeout,
)


def _parse_stdout_tokens(stdout: str) -> list[float]:
    if not stdout or not stdout.strip():
        return []
    return [float(val) for val in stdout.split()]


def parse_membrane_records(stderr: str | None) -> list[list[float]]:
    """Parse ``MEMB`` lines (NEVRESIM_EXPORT_MEMBRANE build) into per-sample rows.

    One line per sample: ``m_T / theta`` per network output, captured at each
    source core's window-end cycle."""
    rows: list[list[float]] = []
    for line in (stderr or "").splitlines():
        line = line.strip()
        if not line.startswith("MEMB"):
            continue
        rows.append([float(v) for v in line.split()[1:]])
    return rows


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


def _parse_binary_train(core: int, token: str) -> list[int]:
    """One neuron's SPKTRN bitstring. A per-cycle law emits at most one spike
    per neuron, so anything but 0/1 here is a protocol violation, not a count:
    the counted wire has its OWN versioned line."""
    if set(token) - {"0", "1"}:
        raise ValueError(f"SPKTRN core {core}: non-binary train symbol")
    return [int(char) for char in token]


def _parse_counted_train(core: int, token: str) -> list[int]:
    """One neuron's SPKTRN2 comma-joined per-cycle counts."""
    counts = []
    for field in token.split(","):
        if not field.isdigit():
            raise ValueError(
                f"SPKTRN2 core {core}: {field!r} is not a per-cycle count"
            )
        count = int(field)
        if count > EMISSION_COUNT_CEILING:
            raise ValueError(
                f"SPKTRN2 core {core}: per-cycle count {count} exceeds the "
                f"count currency's ceiling {EMISSION_COUNT_CEILING}"
            )
        counts.append(count)
    return counts


def parse_spike_trains(stderr: str) -> list[dict[int, list[list[int]]]]:
    """Parse the spike-train record lines (NEVRESIM_RECORD_SPIKE_TRAINS build)
    into per-sample trains: ``{core: [per-cycle counts per neuron]}`` in
    PRODUCER-LOCAL time. ``SPKTRN_END`` closes a sample.

    TWO wire versions, both closed by the same terminator: ``SPKTRN`` carries a
    bit raster (one char per cycle) and ``SPKTRN2`` carries per-cycle COUNTS
    (comma-joined per neuron), which is what a law that can fire more than once
    per cycle actually produces. The binary version stays STRICT — a non-binary
    symbol on a ``SPKTRN`` line is a violated protocol, not a count.

    A ragged core (neuron trains of unequal length) fails loud — that is a
    protocol violation, not a short train."""
    samples: list[dict[int, list[list[int]]]] = []
    current: dict[int, list[list[int]]] = {}
    for line in (stderr or "").splitlines():
        line = line.strip()
        if not line.startswith("SPKTRN"):
            continue
        if line == "SPKTRN_END":
            samples.append(current)
            current = {}
            continue
        toks = line.split()
        counted = toks[0] == "SPKTRN2"
        parse_train = _parse_counted_train if counted else _parse_binary_train
        core = int(toks[1])
        trains = [parse_train(core, token) for token in toks[2:]]
        widths = {len(t) for t in trains}
        if len(widths) > 1:
            raise ValueError(
                f"{toks[0]} core {core}: ragged trains (lengths "
                f"{sorted(widths)}) — every neuron of a core shares one window"
            )
        current[core] = trains
    return samples


def _launch_workers(
    executable: str, input_count: int, num_proc: int,
) -> list[tuple[int, int, subprocess.Popen]]:
    """Spawn one simulator process per sample range, each in its own process group."""
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
            start_new_session=True,
        )
        workers.append((start, end, proc))
    return workers


def _reap_workers(workers: list[tuple[int, int, subprocess.Popen]]) -> None:
    """Kill every worker's process group and drain/close its pipes."""
    for _start, _end, proc in workers:
        kill_process_group(proc.pid)
    for _start, _end, proc in workers:
        try:
            proc.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            pass


def _collect_worker_outputs(
    workers: list[tuple[int, int, subprocess.Popen]],
    *,
    deadline: float,
    timeout_s: float,
    record_spikes: bool,
    export_membrane: bool = False,
    record_spike_trains: bool = False,
) -> tuple[list[float], list[dict], list[dict], list[list[float]]]:
    output_values: list[float] = []
    spike_records: list[dict] = []
    spike_trains: list[dict] = []
    membrane_records: list[list[float]] = []
    for start, end, proc in workers:
        remaining = deadline - time.monotonic()
        try:
            stdout, stderr = proc.communicate(timeout=max(remaining, 0.001))
        except subprocess.TimeoutExpired:
            _reap_workers(workers)
            raise SimulationTimeoutError(
                f"nevresim simulator exceeded the {timeout_s:.0f}s wall cap "
                f"waiting on samples [{start}, {end}); "
                f"killed {len(workers)} worker(s)"
            ) from None
        if proc.returncode != 0:
            _reap_workers(workers)
            err = (stderr or "").strip()
            raise RuntimeError(
                f"nevresim simulator failed for samples [{start}, {end}) "
                f"(exit {proc.returncode})"
                + (f": {err}" if err else "")
            )
        output_values.extend(_parse_stdout_tokens(stdout))
        if record_spikes:
            spike_records.extend(parse_spike_records(stderr))
        if record_spike_trains:
            spike_trains.extend(parse_spike_trains(stderr))
        if export_membrane:
            membrane_records.extend(parse_membrane_records(stderr))
    return output_values, spike_records, spike_trains, membrane_records


def execute_simulator_full(
    simulator_filename,
    input_count,
    num_proc=0,
    *,
    expected_values: int | None = None,
    record_spikes: bool = False,
    export_membrane: bool = False,
    record_spike_trains: bool = False,
    timeout_s: float | None = None,
) -> tuple[list[float], list[dict], list[dict], list[list[float]]]:
    """Run the nevresim binary across ``num_proc`` workers (0 ⇒ ``cpu_count() // 2``).

    Returns ``(output_values, spike_records, membrane_records)``; the record
    lists are empty unless their flag is set (dedicated builds).
    Stdout protocol: per sample in ``[start, end)``, ``output_size`` whitespace-
    separated numbers then a newline; a mismatch vs ``expected_values`` raises.
    ``export_membrane`` (a ``NEVRESIM_EXPORT_MEMBRANE`` build) additionally
    parses per-sample ``MEMB`` stderr rows in sample order.
    The whole worker batch runs under one wall cap (``timeout_s``, else the
    resolved step timeout): on expiry the workers' process groups are killed
    and the batch retries once before failing loud."""
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
    cap = resolve_simulation_step_timeout_s(timeout_s)

    def attempt(
        _attempt_index: int,
    ) -> tuple[list[float], list[dict], list[dict], list[list[float]]]:
        workers = _launch_workers(executable, input_count, num_proc)
        return _collect_worker_outputs(
            workers,
            deadline=time.monotonic() + cap,
            timeout_s=cap,
            record_spikes=record_spikes,
            export_membrane=export_membrane,
            record_spike_trains=record_spike_trains,
        )

    output_values, spike_records, spike_trains, membrane_records = retry_once_on_timeout(
        attempt, description=f"nevresim execute ({executable})",
    )

    end_time = time.time()
    print("  Simulation time:", end_time - start_time)

    if expected_values is not None and len(output_values) != expected_values:
        raise RuntimeError(
            f"nevresim output size mismatch: expected {expected_values} values, "
            f"got {len(output_values)}"
        )

    return output_values, spike_records, spike_trains, membrane_records


def execute_simulator(
    simulator_filename,
    input_count,
    num_proc=0,
    *,
    expected_values: int | None = None,
    record_spikes: bool = False,
    timeout_s: float | None = None,
):
    """Narrowing wrapper over ``execute_simulator_full`` for existing callers:
    returns ``output_values``, or ``(output_values, spike_records)`` with
    ``record_spikes``."""
    output_values, spike_records, _trains, _membranes = execute_simulator_full(
        simulator_filename,
        input_count,
        num_proc,
        expected_values=expected_values,
        record_spikes=record_spikes,
        timeout_s=timeout_s,
    )
    if record_spikes:
        return output_values, spike_records
    return output_values
