#!/usr/bin/env python3
"""AUDIT one sealed deployment bundle from its own BYTES, adversarially.

    env/bin/python scripts/hacc/selftest/deployment_envelope_audit.py BUNDLE
        [--samples N] [--json OUT]

Everything else in this tree checks a bundle against the objects that produced
it. This checks it against NOTHING but itself. It decodes the program stream the
board will execute, rebuilds the crossbar out of those words, and asks two
questions the producer is not allowed to answer:

  1. THE ENVELOPE. Is every decoded weight inside the representable cell of the
     fabric this bundle's own claims name, every decoded threshold inside its
     membrane register, every axon address inside its crossbar, and every
     per-cycle emission inside the 127 COUNT CURRENCY -- which no crossbar width
     lifts, because it is what a segment boundary carries and not what a core is?

  2. THE FROZEN TRUTH. Does the cycle-accurate twin, run on the network
     RECONSTRUCTED FROM THE SHIPPED BYTES and driven by the shipped entry
     rasters, reproduce the per-pass counts the bundle froze?

A bundle that passes both is one whose evidence survives having its producer
taken away. Exit 0 on a clean audit, 1 on a finding, 2 on a bundle this auditor
cannot decode (and it says which, rather than reporting an audit it never ran).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from mimarsinan.chip_simulation import odin_deployment_bundle as bundle  # noqa: E402
from mimarsinan.chip_simulation import odin_deployment_encoding as aer  # noqa: E402
from mimarsinan.chip_simulation.odin_fpga.chip_selection import (  # noqa: E402
    chip_config_of_bundle,
)
from mimarsinan.chip_simulation.odin_rtl.reference import simulate_cycles  # noqa: E402
from mimarsinan.chip_simulation.odin_rtl.stimulus import (  # noqa: E402
    OP_AER,
    OP_PROG,
    decode_ops,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw  # noqa: E402
from mimarsinan.code_generation.cpp_chip_model import SpikeSource  # noqa: E402
from mimarsinan.mapping.export.odin.feasibility import (  # noqa: E402
    EMISSION_CEILING,
    propagate_emission_bounds,
)
from mimarsinan.mapping.export.odin_gen.packer import (  # noqa: E402
    PROG_SEL_MEMBRANE,
    PROG_SEL_REGISTER,
    PROG_SEL_SYNAPSE,
    PROG_SEL_THRESHOLD,
    unpack_synapse_words,
)
from mimarsinan.mapping.latency.chip import ChipLatency  # noqa: E402
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping  # noqa: E402


class AuditFinding(AssertionError):
    """The bundle's own bytes contradict what it claims about itself."""


class AuditUndecodable(RuntimeError):
    """This auditor cannot read this fabric's program; it says so, and stops."""


def words_of(payload: bytes) -> Tuple[int, ...]:
    return struct.unpack(f"<{len(payload) // 4}I", payload)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFinding(message)


# --- 1. the envelope, read off the decoded program ---------------------------


def decode_variant_core(plan: Dict[str, Any], spec: Any) -> Dict[str, Any]:
    """One generated core's crossbar, thresholds and membranes, from its bytes."""
    ops = decode_ops(list(words_of(bundle.decode_payload(
        plan["program"], what=f"core {plan['core']}/program"))))
    codes = {op.code for op in ops}
    check(codes == {OP_PROG},
          f"core {plan['core']}: a generated core is programmed through its "
          f"configuration port alone; the decoded program carries {sorted(codes)}")
    check(all(op.code == OP_PROG for op in decode_ops(
              [int(word) for word in plan["clear_prefix"]] + [0])),
          f"core {plan['core']}: the per-sample CLEAR writes something other "
          f"than the generated core's configuration port")
    # The membranes are NOT in the program: they are the per-sample CLEAR the
    # reader replays before every sample, so they are decoded from there.
    clear = decode_ops([int(word) for word in plan["clear_prefix"]] + [0])
    synapse: Dict[int, int] = {}
    thresholds: Dict[int, int] = {}
    membranes: Dict[int, int] = {}
    gates: List[int] = []
    for op in list(ops) + list(clear):
        _core, sel, addr, data = op.args
        if sel == PROG_SEL_SYNAPSE:
            synapse[int(addr)] = int(data)
        elif sel == PROG_SEL_THRESHOLD:
            thresholds[int(addr)] = int(data)
        elif sel == PROG_SEL_MEMBRANE:
            membranes[int(addr)] = int(data)
        elif sel == PROG_SEL_REGISTER:
            gates.append(int(data))
        else:
            raise AuditFinding(
                f"core {plan['core']}: configuration selector {sel} is not one "
                f"the generated core declares")
    check(len(synapse) == spec.synapse_depth,
          f"core {plan['core']}: {len(synapse)} synapse word(s) against the "
          f"declared depth {spec.synapse_depth}")
    check(gates and gates[0] == 1 and gates[-1] == 0 and len(gates) == 4,
          f"core {plan['core']}: the program does not gate activity while it "
          f"writes the memories (gate writes {gates})")
    grid = unpack_synapse_words(
        [synapse[address] for address in range(spec.synapse_depth)], spec=spec)
    return {"grid": grid, "thresholds": thresholds, "membranes": membranes}


def audit_envelope(document: Dict[str, Any], config: Any,
                   decoded: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Every decoded number against the envelope of the fabric it names."""
    low, high = config.weight_range
    ceiling = config.theta_ceiling
    peak_weight, peak_theta, slots = 0, 0, 0
    for plan, core in zip(document["cores"], decoded):
        index = int(plan["core"])
        used = int(plan["used_neurons"])
        width = len(plan["routes"])
        slots = max(slots, width)
        check(width <= config.effective_max_axons,
              f"core {index}: {width} axon slot(s) against the fabric's "
              f"effective fan-in {config.effective_max_axons}")
        for slot in range(width):
            for neuron in range(used):
                weight = int(core["grid"][slot][neuron])
                check(low <= weight <= high,
                      f"core {index} slot {slot} neuron {neuron}: decoded "
                      f"weight {weight} outside the cell's [{low}, {high}]")
                peak_weight = max(peak_weight, abs(weight))
        for neuron in range(used):
            theta = int(core["thresholds"][neuron])
            check(1 <= theta <= ceiling,
                  f"core {index} neuron {neuron}: decoded threshold {theta} "
                  f"outside [1, {ceiling}]")
            peak_theta = max(peak_theta, theta)
            check(int(core["membranes"][neuron]) < theta,
                  f"core {index} neuron {neuron}: decoded membrane init is not "
                  f"below its own threshold")
    return {"peak_abs_weight": peak_weight, "weight_range": [low, high],
            "peak_theta": peak_theta, "theta_ceiling": ceiling,
            "widest_core_slots": slots,
            "effective_max_axons": config.effective_max_axons}


def audit_stimulus_words(document: Dict[str, Any], encoding: Any) -> Dict[str, Any]:
    """Every AER word of every frozen stimulus, against this fabric's wording."""
    reserved = 0 if encoding.emits_tref else encoding.tref_word
    seen, foreign = 0, 0
    for entry in document["samples"]:
        payload = bundle.decode_payload(
            entry["stimulus"], what=f"sample {entry['index']}/stimulus")
        for op in decode_ops(list(words_of(payload))):
            if op.code != OP_AER:
                continue
            seen += 1
            word = int(op.args[1])
            if encoding.emits_tref and word == encoding.tref_word:
                continue
            check(word < (1 << encoding.address_bits),
                  f"sample {entry['index']}: AER word {word} is outside the "
                  f"{encoding.address_bits}-bit address space of the "
                  f"{encoding.name} fabric")
            if reserved and word == reserved:
                foreign += 1
    check(foreign == 0,
          f"{foreign} frozen event(s) carry this fabric's RESERVED time "
          f"reference word as an axon address")
    return {"aer_events": seen, "wording": encoding.name,
            "address_bits": encoding.address_bits}


# --- 2. the frozen truth, re-derived from the reconstructed network ----------


def source_of(route) -> SpikeSource:
    kind, index = int(route[0]), int(route[1])
    if kind == bundle.SOURCE_OFF:
        return SpikeSource(0, 0, is_off=True)
    if kind == bundle.SOURCE_INPUT:
        return SpikeSource(-2, index, is_input=True)
    if kind == bundle.SOURCE_ALWAYS_ON:
        return SpikeSource(-3, 0, is_always_on=True)
    return SpikeSource(kind, index)


def rebuild_mapping(document: Dict[str, Any],
                    decoded: List[Dict[str, Any]]) -> HardCoreMapping:
    """The network the shipped PROGRAM BYTES describe, and nothing else."""
    cores: List[HardCore] = []
    for plan, image in zip(document["cores"], decoded):
        width = len(plan["routes"])
        neurons = int(plan["neurons"])
        used = int(plan["used_neurons"])
        matrix = np.asarray(
            [[int(image["grid"][slot][neuron]) for neuron in range(neurons)]
             for slot in range(width)], dtype=np.float64)
        thetas = {int(image["thresholds"][neuron]) for neuron in range(used)}
        check(len(thetas) == 1,
              f"core {plan['core']}: {len(thetas)} distinct threshold(s) over "
              f"its used neurons; one core carries one theta")
        core = HardCore(axons_per_core=width, neurons_per_core=neurons,
                        has_bias_capability=False)
        core.core_matrix = matrix
        core.axon_sources = [source_of(route) for route in plan["routes"]]
        core.threshold = float(thetas.pop())
        core.available_axons = 0
        core.available_neurons = neurons - used
        cores.append(core)
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = cores
    readout = int(document["readout"]["core"])
    mapping.output_sources = np.asarray(
        [SpikeSource(readout, neuron)
         for neuron in document["readout"]["neurons"]], dtype=object)
    return mapping


def audit_frozen_truth(document: Dict[str, Any], mapping: HardCoreMapping,
                       wanted: int) -> Dict[str, Any]:
    """Re-run the twin on the rebuilt network and compare the frozen windows."""
    block = document["chip_config"]
    law = SomaLaw(
        firing_mode=str(block["soma_law"]["firing_mode"]),
        thresholding_mode=str(block["soma_law"]["thresholding_mode"]),
        firing_granularity=str(block["soma_law"]["firing_granularity"]),
        membrane_arithmetic=str(block["soma_law"]["membrane_arithmetic"]),
        membrane_bits=int(block["soma_law"]["membrane_bits"]),
        bias_slot=str(block["soma_law"]["bias_slot"]),
    )
    latency = int(ChipLatency(mapping).calculate())
    check(latency == int(block["chip_latency"]),
          f"the rebuilt network's chip latency is {latency}; the bundle "
          f"declares {block['chip_latency']}")
    rasters = {int(entry["index"]): entry["entry_raster"]
               for entry in document["samples"]}
    certified = list(bundle.certification_samples(document))[:max(int(wanted), 1)]
    check(len(certified) >= 1, "the bundle certifies no sample at all")
    scored = 0
    for index in certified:
        trace = simulate_cycles(
            mapping, soma_law=law, input_counts=rasters[index],
            simulation_length=int(block["simulation_length"]),
            membrane_init=int(block["membrane_init"]), chip_latency=latency)
        frozen = bundle.expected_pass_windows(document, index)
        windows = trace.window_counts()
        for core, rows in frozen.items():
            got = [int(value) for value in windows[int(core)]]
            check(got == list(rows[0]),
                  f"sample {index} core {core}: the twin re-derived from the "
                  f"shipped bytes says {got[:8]}..., the bundle froze "
                  f"{list(rows[0])[:8]}...")
        scores = bundle.readout_scores(
            document, {c: list(w) for c, w in enumerate(windows)})
        expected = document["expected"]["final"][index]
        check(list(scores) == list(expected["scores"]),
              f"sample {index}: re-derived scores {scores} against the frozen "
              f"{expected['scores']}")
        check(bundle.predicted_label(document, scores) == int(expected["predicted"]),
              f"sample {index}: the re-derived prediction is not the frozen one")
        scored += 1
    bounds = propagate_emission_bounds(mapping, ceiling=EMISSION_CEILING)
    return {"samples_re_derived": scored,
            "certified_samples": len(bundle.certification_samples(document)),
            "peak_emission_bound": max(bounds.values(), default=0),
            "count_currency": EMISSION_CEILING}


def audit(path: Path, *, samples: int) -> Dict[str, Any]:
    document = bundle.load_bundle(str(path))
    config = chip_config_of_bundle(document)
    encoding = aer.encoding_of_bundle(document)
    if encoding.name != aer.AER_VARIANT:
        raise AuditUndecodable(
            f"{path.name} names the {config.name} fabric, whose program is an "
            f"SPI byte stream over the vendored crossbar's own memory map. This "
            f"auditor rebuilds a network out of a GENERATED core's "
            f"configuration-port words; it will not report an audit of the "
            f"stock fabric that it did not run")
    decoded = [decode_variant_core(plan, config.core_spec)
               for plan in document["cores"]]
    report = {
        "bundle": document["name"],
        "self_hash": document["self_hash"],
        "chip": config.name,
        "cores": len(document["cores"]),
        "stimulus": audit_stimulus_words(document, encoding),
        "envelope": audit_envelope(document, config, decoded),
    }
    report["frozen_truth"] = audit_frozen_truth(
        document, rebuild_mapping(document, decoded), samples)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle")
    parser.add_argument("--samples", type=int, default=3,
                        help="how many certified samples to re-derive (>= 3)")
    parser.add_argument("--json", default=None)
    options = parser.parse_args(argv)
    try:
        report = audit(Path(options.bundle), samples=options.samples)
    except AuditUndecodable as exc:
        print(f"UNDECODABLE: {exc}", file=sys.stderr)
        return 2
    except AuditFinding as exc:
        print(f"FINDING: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2, sort_keys=True)
    if options.json:
        Path(options.json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
