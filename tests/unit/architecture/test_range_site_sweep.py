"""[U-BRANGE2 stage 1] The sweep of the range class, kept closed: one rule from one source at every integer-path site.

The class: a range, scale or clamp level chosen by a statistic or a coupling that does not count the error the grid
causes, so that values a model relies on are clamped, or a grid coarsened, without a refusal. Every module of the
toolchain's source that computes such a statistic (a quantile, an order statistic, a top fraction) is listed here with
its disposition: the rule itself, its primitive, a reading that deploys no range, or a site of the spiking route
(stage 2 by the rulings of 2026-11-11, not this patch). A new module computing one, or a listed module no other module
imports (a dead clamp, the review's m-4: ``weight_clipping.py`` is removed), fails here. Every integer-path site reads
the rule from its one source (``range_rule``; the drive site ``drive_rule``), and every union of grids read together
is one site of it (``reader_grids.read_together``).

[U-BRANGE2 stage 1b] The former quantile is no longer a reading in src (the verify's m-5): the test programs calibrate
their readouts by the step's own calibration, so ``seam_intervals``, ``measure_seam_quantiles`` and the dead
``quantile`` and ``candidates`` parameters left; the restore tests read the former quantile through the tests' own
reference. Every exception class the rule's modules and its integer-path sites define is a refusal by name (the
verify's m-7: ``RangeSearchError`` was a bare ``RuntimeError``).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
_STATISTIC = re.compile(r"safe_quantile\(|torch\.quantile\(|\.quantile\(|kthvalue\(|\.topk\(|torch\.topk\(|percentile")

#: Every module computing a statistic of the class, with its disposition.
DISPOSITION = {
    "mapping/support/tensor_stats.py": "PRIMITIVE: an exact quantile at every size (F-3)",
    "config_schema/registry/entries_tuning.py": "STAGE 2: the declared theta policies' names and documentation",
    "deployment_record/platform_physics/host_calibration.py": "OUT: host wall statistics, no grid deployed",
    "pipelining/pipeline_steps/activation_utils.py": "READING: S24, a median for the report",
    "spiking/distribution_matching.py": "STAGE 2: S19, the fan-in statistic of the spiking route",
    "spiking/seam_audit.py": "READING: S23, an audit statistic (a probe script's)",
    "transformations/activation_scale_policy.py": "STAGE 2: S16, theta of the spiking route",
    "tuning/orchestration/install_resolution/capture.py": "READING: S22, the install-resolution capture",
    "tuning/orchestration/signed_seam_install.py": "STAGE 2: S20, the signed seam covers of the spiking route",
    "tuning/orchestration/lif_entry_encode.py": "STAGE 2: the host-fed segment entry gauge of the spiking route (U-LIF D1, D2), a calibration quantile",
}

#: The integer value path's range sites and the one source each reads (the review's S1 to S7, S25 and the unions).
INTEGER_SITES = {
    "pipelining/pipeline_steps/quantization/boundary_quantization_step.py": "mapping.support.range_rule",
    "pipelining/pipeline_steps/quantization/readout_law_step.py": "mapping.support.range_rule",
    "mapping/lanes/model_lanes.py": "mapping.support.range_rule",
    "mapping/support/bias_drive.py": "mapping.support.drive_rule",
    "pipelining/pipeline_steps/quantization/bias_row_drive_route.py": "mapping.support.drive_rule",
}

#: The rule's sources: every module they import from the toolchain's support package is a module of the rule.
SOURCES = ("mapping.support.range_rule", "mapping.support.drive_rule")


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        yield path.relative_to(SRC).as_posix(), path.read_text(encoding="utf-8")


def _imports(text: str):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def test_every_module_computing_a_statistic_of_the_class_is_listed_with_its_disposition():
    found = {rel for rel, text in _modules() if _STATISTIC.search(text)}
    assert found == set(DISPOSITION), {"unlisted": sorted(found - set(DISPOSITION)),
                                       "listed, no statistic": sorted(set(DISPOSITION) - found)}


def test_no_module_computing_a_deployed_range_is_dead():
    importers = {}
    for rel, text in _modules():
        for module in _imports(text):
            importers.setdefault(module, set()).add(rel)
    for rel, disposition in DISPOSITION.items():
        if disposition.startswith(("OUT:", "READING:")):  # a reading or out of the class: it deploys no range
            continue
        dotted = "mimarsinan." + rel[:-3].replace("/", ".")
        users = {u for m, users in importers.items() if m == dotted or m.startswith(dotted + ".") for u in users}
        assert users - {rel}, f"{rel}: no module imports it (a dead range site)"


def test_the_dead_clamp_is_removed():
    assert not (SRC / "transformations" / "weight_clipping.py").exists()


def test_every_integer_path_site_reads_its_one_source():
    texts = dict(_modules())
    for rel, source in INTEGER_SITES.items():
        assert "mimarsinan." + source in set(_imports(texts[rel])), f"{rel} does not read {source}"
    readout = set(_imports(texts["pipelining/pipeline_steps/quantization/readout_law_step.py"]))
    assert "mimarsinan.mapping.support.reader_grids" in readout, "the readout sites read no union of grids"


def test_the_search_lives_in_one_place():
    """No module but the rule's own defines a range search over candidates (the engine's entry points)."""
    owners = {rel for rel, text in _modules()
              if re.search(r"^def (search_ranges|least_squares_ranges|least_squares_pooled|least_squares_recorded)\(",
                           text, re.M)}
    assert owners == {"mapping/support/range_search.py", "mapping/support/range_rule.py"}, sorted(owners)


def _rule_modules():
    """The rule's modules: its sources, every module they import from ``mimarsinan.mapping.support`` (transitively),
    and every integer-path site."""
    texts = dict(_modules())
    seen, todo = set(), ["mimarsinan." + s for s in SOURCES]
    while todo:
        dotted = todo.pop()
        if dotted in seen:
            continue
        seen.add(dotted)
        rel = dotted[len("mimarsinan."):].replace(".", "/") + ".py"
        todo += [m for m in _imports(texts[rel]) if m.startswith("mimarsinan.mapping.support.")
                 and m[len("mimarsinan."):].replace(".", "/") + ".py" in texts]
    return sorted(seen | {"mimarsinan." + rel[:-3].replace("/", ".") for rel in INTEGER_SITES})


def test_every_exception_class_of_the_rule_s_modules_is_a_refusal_by_name():
    """[U-BRANGE2 stage 1b, the verify's m-7] A class of exception defined in a module of the rule or of an
    integer-path site is a ``ToolchainRefusal`` keeping a builtin base, so a refusal of the rule is told apart from a
    crash by its type (the rulings' 'The Extended Fuzz's One Error Class', item 3)."""
    import importlib
    import inspect

    from mimarsinan.common.refusal import ToolchainRefusal

    modules = _rule_modules()
    assert {"mimarsinan.mapping.support.range_search", "mimarsinan.mapping.support.range_calls"} <= set(modules)
    defined = []
    for dotted in modules:
        module = importlib.import_module(dotted)
        defined += [(dotted, obj) for _name, obj in inspect.getmembers(module, inspect.isclass)
                    if obj.__module__ == dotted and issubclass(obj, BaseException)]
    assert defined, "fixture: the rule's modules define their refusals"
    bare = [f"{dotted}.{cls.__name__}" for dotted, cls in defined if not issubclass(cls, ToolchainRefusal)]
    assert not bare, bare
    unbased = [cls.__name__ for _d, cls in defined if cls.__bases__ == (ToolchainRefusal,)]
    assert not unbased, unbased


def test_no_quantile_reading_of_a_range_is_left_in_src():
    """[U-BRANGE2 stage 1b, the verify's m-5] No module of src keeps the former quantile as a reading of a range site,
    and the sources' callers pass no quantile: the rule's entry points take none."""
    import inspect

    import mimarsinan.mapping.lanes.model_lanes as lanes
    import mimarsinan.mapping.support.range_rule as range_rule
    import mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step as boundary_step

    left = [name for module, names in ((range_rule, ("seam_intervals", "RANGE_QUANTILE")),
                                       (boundary_step, ("measure_seam_quantiles", "CALIBRATION_QUANTILE")))
            for name in names if hasattr(module, name)]
    assert not left, left
    for function in (range_rule.least_squares_ranges, boundary_step.calibrate_boundary_scales,
                     lanes.measure_lane_ranges):
        params = set(inspect.signature(function).parameters)
        assert not params & {"quantile", "candidates"}, (function.__name__, sorted(params))
