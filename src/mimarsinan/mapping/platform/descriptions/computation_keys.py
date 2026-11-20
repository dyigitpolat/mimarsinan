"""[U-CHIPKEYS] The one loader rule for the keys that change the computation: a chip description states each from its
source, with the source's quote and location, or names it unstated (never a silent default).

The keys are the paper's chip-description list (Definition 1, "changes the computation"): the core semantics and
neuron model, the weight grid, the on-chip bias; for value cores the activation grid, the converter law at core
outputs with its width, the supported operation classes and the run-time writable matrix (operand cores); for
spiking cores the neuron's family, variant, window, levels, firing granularity, reset law, membrane register and
threshold comparison (``CHIP_COMPUTATION_KEYS``). Every other function-bearing key of the chip groups is the
deployment's choice (``DEPLOYMENT_CHOICES``, each with why); a function-bearing key a description sets is stated too.

A bundle's ``computation`` block holds one statement per required key path:

- ``{"stated": {value, source, location, quote, reads[, derivation][, assumed_fields][, note]}}``: the source states
  the value; the bundle sets the key to it (its function-bearing fields, for an object value), ``reads`` is the span
  of the quote that states it (one span per field for an object, where the fields are read apart), and the value is
  read there by the reading rule (``readings``: one integer token per integer field or a unit-anchored one, a named
  ``derivation``'s input the same way, a closed string never against another spelled option); fields of an object
  value the source does not state are named in ``assumed_fields`` with why.
- ``{"unstated": {note[, assumed]}}``: the source states no value in the key's meaning; the note says why and who
  supplies it. Without ``assumed`` the bundle sets nothing (the deployment declares the key); with it, the bundle
  sets ``assumed``, a named assumption, never a silent default.

[U-CHIPKEYS2] The weight grid has one definition for every description: the composed weight one synapse represents,
on the toolchain's symmetric signed grid; its statement carries ``composition`` beside it (the cells one weight's value
is bit-sliced over and the bits of one cell, each stated or unstated, and where both are stated the named law that
composes them; [U-CHIPKEYS3] the slices named apart from the core field ``cells_per_weight``, whose count holds whole
copies of them).

[U-CHIPKEYS3] m-4: a third form, ``{"departs": {computation, source, location, quote, reads, assumed, reason[,
note]}}``, states that the source's computation is one the toolchain's key cannot hold: the source's computation in
words with its quote, the value assumed in its place (the bundle sets it) and why the key cannot hold the source's;
the description's faithfulness label prints the departure (``departures``), so its class is never exact.
[U-CHIPKEYS5/6] The departure is written by the label grammar (``label_clauses``), its computation and the value
assumed judged by their words at load and printed as quoted fields.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from mimarsinan.config_schema.registry import REGISTRY, Category
from mimarsinan.config_schema.registry.build import declared_paths
from mimarsinan.config_schema.registry.decision_classes import (
    COMPOSITE, FUNCTION_BEARING, DecisionClassError, _path_value, class_of_path)
from mimarsinan.mapping.platform.descriptions import label_clauses
from mimarsinan.mapping.platform.descriptions.readings import (
    COMPOSITION_READINGS, COMPOSITION_TIES, composition_errors, judged_reading, read_spans, value_reading_errors)

VALUE, SPIKING = "mvm", "spiking"
#: Definition 1's chip properties that change the computation: key path -> (the core semantics it applies to, None
#: for every chip; what it is).
CHIP_COMPUTATION_KEYS: Mapping[str, Tuple[Optional[str], str]] = {
    "core_semantics": (None, "the integer semantics and neuron model: value or spiking cores"),
    "weight_bits": (None, "the weight grid: the composed weight one synapse represents, its cells per weight and "
                          "cell precision stated beside it"),
    "cores.has_bias": (None, "the on-chip bias of the core types"),
    "activation_bits": (VALUE, "the activation grid at the host-chip boundaries"),
    "readout_law": (VALUE, "the converter law at core outputs and its width"),
    "onchip_arithmetic": (VALUE, "the operation classes the cores execute"),
    "cores.operand": (VALUE, "the run-time writable matrix (operand cores)"),
    "spiking_family": (SPIKING, "the neuron model's family"),
    "spiking_variant": (SPIKING, "the neuron model's variant"),
    "simulation_steps": (SPIKING, "the spike window"),
    "target_tq": (SPIKING, "the activation levels"),
    "firing_granularity": (SPIKING, "when the threshold is evaluated"),
    "firing_mode": (SPIKING, "the reset law"),
    "membrane_bits": (SPIKING, "the membrane register's width"),
    "membrane_signed": (SPIKING, "the membrane register's sign"),
    "thresholding_mode": (SPIKING, "the threshold comparison"),
}
#: The registry groups that declare the chip and its neuron dynamics: their function-bearing keys are swept.
CHIP_GROUPS = ("hardware", "spiking")
#: Function-bearing keys of the chip groups that a chip key above covers (the key a description states instead).
COVERED_BY = {"cores": "cores.has_bias, cores.operand and cores.weight_bits (its function-bearing fields)",
              "has_bias": "cores.has_bias (the flat key is the bias of core types that state none)"}
#: Function-bearing keys of the chip groups that are the deployment's choice, not the chip's.
DEPLOYMENT_CHOICES: Mapping[str, str] = {
    "bias_resolution": "how the adaptation holds a bias (the weight grid or the law's resolution)",
    "bias_row_splitting": "how a bias rides always-on rows on cores without an on-chip bias (the mapping's choice)",
    "bias_rows_per_perceptron": "the declared row count of a fixed bias split (the mapping's choice)",
    "comparator_half_step": "where the lowering places the per-cycle compare levels (exact, zero bit cost)",
    "hw_config_mode": "whether the chip is declared or searched (how a chip is chosen, not what it computes)",
    "layer_core_types": "which core type each layer is assigned (the deployment's placement of core types)",
    "lif_membrane_init": "the window-start membrane guard of the LIF lowering",
    "lif_membrane_readout": "how the lowering decodes output cores (counts with the membrane)",
    "lif_window_firing": "the firing of a synchronized LIF hop in its window (both forms declared, option A)",
    "membrane_arithmetic": "derived from membrane_bits and membrane_signed (declaring the width declares it)",
    "quantizer_state": "the deployment's calibrated ranges and entry grids",
    "readout_ranges": "how the deployment calibrates the law's output ranges (per perceptron or consumer group)",
    "spike_generation_mode": "the deployment's input spike encoding",
    "spike_phase_dither": "the deployment's input comb rotation",
    "weight_scale_granularity": "one weight scale per core or per neuron (the adaptation's choice)",
    "weight_scale_rule": "the adaptation's weight scale rule",
}
#: Named derivations a stated integer may take from the number its quote reads.
DERIVATIONS: Mapping[str, Tuple[str, Any]] = {
    "sign_and_magnitude": ("a magnitude of n bits with its own sign spans 2^n - 1 codes each side: the symmetric "
                           "signed grid of n + 1 bits", lambda n: n + 1),
    "ieee_significand": ("an IEEE 754 binary format of n bits scales with a significand of 11, 24 or 53 bits (half, "
                         "single and double precision, the implicit bit counted)", {16: 11, 32: 24, 64: 53}.get),
}
_STATED = frozenset({"value", "source", "location", "quote", "reads", "derivation", "assumed_fields", "note"})
_UNSTATED = frozenset({"note", "assumed"})
#: [U-CHIPKEYS3] The departing form: its fields and its non-empty text fields (``assumed`` is required beside them).
_DEPARTS = frozenset({"computation", "source", "location", "quote", "reads", "assumed", "reason", "note"})
_DEPARTS_TEXTS = ("computation", "source", "location", "quote", "reads", "reason")
#: [U-CHIPKEYS3] The faithfulness label's separators, which a departure's computation never holds.
_LABEL_SEPARATORS = label_clauses.SEPARATORS
MISSING = object()


def _semantics_keys(semantics: Any) -> List[str]:
    return [path for path, (applies, _) in CHIP_COMPUTATION_KEYS.items() if applies in (None, semantics)]


def _function_bearing(path: str) -> bool:
    try:
        return class_of_path(REGISTRY, path) == FUNCTION_BEARING
    except DecisionClassError:
        return False


def required_paths(values: Mapping[str, Any]) -> Tuple[str, ...]:
    """The key paths a description's ``values`` must state or name unstated: the chip keys of its core semantics and
    every other function-bearing key or core field it sets."""
    out = _semantics_keys(values.get("core_semantics"))
    for path in declared_paths({k: v for k, v in values.items() if k != "cores"}):
        key = path.split(".")[0]
        if _function_bearing(key) and key not in out:
            out.append(key)
    for field in sorted({f for core in values.get("cores") or () for f in core}):
        if _function_bearing(f"cores.{field}") and f"cores.{field}" not in out:
            out.append(f"cores.{field}")
    return tuple(out)


def key_value(values: Mapping[str, Any], path: str) -> Any:
    """The function-bearing part of ``values`` at ``path`` (``MISSING`` when unset): a core field's one value over
    the core types (their list when they differ), an object key's function-bearing fields, else the value."""
    key, _, field = path.partition(".")
    if key == "cores" and field:
        stated = [core.get(field, MISSING) for core in values.get("cores") or ()]
        if all(v is MISSING for v in stated):
            return MISSING
        return stated[0] if all(_same(v, stated[0]) for v in stated) else [None if v is MISSING else v
                                                                             for v in stated]
    if key not in values:
        return MISSING
    value = values[key]
    entry = REGISTRY.get(key)
    if isinstance(value, Mapping) and entry is not None and entry.value_schema is not None \
            and entry.value_schema.fields:
        return {f: v for f, v in value.items() if _function_bearing(f"{key}.{f}")}
    return value


def _same(first: Any, second: Any) -> bool:
    return json.dumps(first, sort_keys=True, default=repr) == json.dumps(second, sort_keys=True, default=repr)


def _options(path: str, field: Optional[str]) -> Tuple[Any, ...]:
    """The closed legal values of a key (``field`` None) or of one field of an object key, () where none."""
    key, _, sub = path.partition(".")
    entry = REGISTRY.get(key)
    if entry is None or sub:
        return ()
    if field is None:
        return tuple(entry.options) if isinstance(entry.options, (tuple, list)) else ()
    schema = entry.value_schema.fields.get(field) if entry.value_schema is not None else None
    if schema is None:
        return ()
    return tuple(schema.options or (schema.item.options if schema.item is not None else None) or ())


def _departs_errors(path: str, body: Mapping[str, Any]) -> List[str]:
    """[U-CHIPKEYS3] Why a departing statement does not hold beside its value check: its value assumed in the source's
    place, a reads span of its quote, and a computation the label can print (no label separator; [U-CHIPKEYS6/7] the
    computation and each text of the value assumed, read raw, judged as typed text: the alphabet and no clause)."""
    errors = [] if "assumed" in body else [f"{path}: departs needs the value assumed in the source's place"]
    errors += [f"{path}: {e}" for e in read_spans(body, [None])[1]]
    if set(str(body["computation"])) & _LABEL_SEPARATORS:
        errors.append(f"{path}: a departing computation is printed in the label, so it holds no ':' or ';'")
    errors += [f"{path}: {e}" for slot, text in (("computation", body["computation"]), *(("assumed", t) for t in
               label_clauses.texts_of(body.get("assumed")))) if (e := label_clauses.text_error(slot, text))]
    return errors


def _judge_one(path: str, entry: Any, values: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    beside = frozenset({"composition"}) if path in COMPOSITION_READINGS else frozenset()
    kind, body, errors = judged_reading(path, entry, _STATED | beside, _UNSTATED | beside,
                                        {"departs": (_DEPARTS | beside, _DEPARTS_TEXTS)})
    if kind is None or errors:
        return None, errors
    have = key_value(values, path)
    if kind == "stated":
        if "value" not in body or have is MISSING or not _same(have, body["value"]):
            errors.append(f"{path}: stated {body.get('value')!r}, but the description sets "
                          f"{'nothing' if have is MISSING else repr(have)}")
        errors += value_reading_errors(path, body, lambda field: _options(path, field), DERIVATIONS)
    else:
        if kind == "departs":
            errors += _departs_errors(path, body)
        if "assumed" in body and (have is MISSING or not _same(have, body["assumed"])):
            errors.append(f"{path}: assumed {body['assumed']!r}, but the description sets "
                          f"{'nothing' if have is MISSING else repr(have)}")
        elif kind == "unstated" and "assumed" not in body and have is not MISSING:
            errors.append(f"{path}: named unstated without its assumed value, but the description sets {have!r}")
    if beside and "composition" not in body:
        errors.append(f"{path}: states its composition beside it ({list(COMPOSITION_READINGS[path])}, each stated "
                      "or unstated with a note)")
    elif beside:
        reading, field = COMPOSITION_TIES[path]
        counts = tuple(core[field] for core in values.get("cores") or () if field in core)
        errors += composition_errors(f"{path}.composition", body["composition"], None if have is MISSING else have,
                                     COMPOSITION_READINGS[path], DERIVATIONS, (reading, field, counts))
    return {"kind": kind, **json.loads(json.dumps(body))}, errors


def departures(statements: Mapping[str, Mapping[str, Any]]) -> Tuple[str, ...]:
    """[U-CHIPKEYS3] The departures a description's statements state (each ``departs`` statement, in key order), as
    its faithfulness label prints them."""
    return tuple(label_clauses.clause("departs", key=path, computation=s["computation"],
                                      assumed=json.dumps(s["assumed"]))
                 for path, s in sorted(statements.items()) if s.get("kind") == "departs")


def statement_errors(block: Any, values: Mapping[str, Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...], Dict]:
    """``(unstated, errors, statements)``: the required keys with no statement (each with what it is), every other
    defect of the block, and the canonical statements by path."""
    if values.get("core_semantics") is None:
        return (), ("core_semantics: a chip description states its core semantics or assumes one by name (the "
                    "registry's default is no chip's statement)",), {}
    block = {} if block is None else block
    if not isinstance(block, Mapping):
        return (), (f"computation must be an object, got {type(block).__name__}",), {}
    required = required_paths(values)
    unstated = tuple(f"{path} ({CHIP_COMPUTATION_KEYS[path][1]})" if path in CHIP_COMPUTATION_KEYS else path
                     for path in required if path not in block)
    errors = [f"{path}: not a computation key of this {values.get('core_semantics')} description (its keys are "
              f"{list(required)})" for path in block if path not in required]
    statements = {}
    for path in required:
        if path in block:
            statement, problems = _judge_one(path, block[path], values)
            errors += problems
            if statement is not None:
                statements[path] = statement
    return unstated, tuple(errors), statements


def skeleton(values: Mapping[str, Any], note: str) -> Dict[str, Any]:
    """A statement block naming every required key of ``values`` unstated with ``note``, ``assumed`` where ``values``
    sets it: a description's starting form, each entry to be replaced by its source's statement (and the form of a
    hand-made test chip, whose every key is the test's)."""
    if values.get("core_semantics") is None:
        raise ValueError("a skeleton needs the description's core semantics")
    out: Dict[str, Dict[str, Dict[str, Any]]] = {
        path: {"unstated": {"note": note} if (have := key_value(values, path)) is MISSING else
               {"note": note, "assumed": json.loads(json.dumps(have))}} for path in required_paths(values)}
    for path, names in COMPOSITION_READINGS.items():  # the readings beside a key, each the test's too
        if path in out:
            out[path]["unstated"]["composition"] = {name: {"unstated": {"note": note}} for name in names}
    return out


def function_bearing_differences(first: Any, second: Any) -> Tuple[str, ...]:
    """The function-bearing key paths whose values differ between two bundles (absent meanings included)."""
    paths = set(declared_paths(first.values)) | set(declared_paths(second.values))
    return tuple(path for path in sorted(paths)
                 if class_of_path(REGISTRY, path) == FUNCTION_BEARING
                 and _path_value(first.values, path) != _path_value(second.values, path))


def unsettable_reason(key: str) -> Optional[str]:
    """Why no chip description may set ``key`` (None when it may): not a registry key, not declarable, or a composite
    or top-level key. The loader judges every bundle key by it, and the sweep reads the keys it admits."""
    entry = REGISTRY.get(key)
    if entry is None:
        return f"{key} is not a registry key"
    if entry.category is Category.RUNTIME or (entry.category is Category.DERIVED and not entry.declarable):
        return f"{key} is not declarable"
    if entry.decision_class == COMPOSITE or entry.section not in ("deployment_parameters", "platform_constraints"):
        return f"{key} cannot be set by a bundle (a composite or top-level key)"
    return None


def swept_keys(groups: Iterable[str] = CHIP_GROUPS) -> Tuple[str, ...]:
    """The function-bearing registry keys of the chip groups a description may set, each a chip key, covered by one,
    or a deployment choice (the class sweep's sites)."""
    return tuple(sorted(key for key, entry in REGISTRY.items()
                        if entry.group in tuple(groups) and entry.decision_class == FUNCTION_BEARING
                        and unsettable_reason(key) is None))
