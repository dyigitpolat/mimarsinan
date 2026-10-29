"""[U-C4, B3 intake open item 4] Cells per stored weight in the chip description: capacity, shipped values, refusals.

A core type declares ``cells_per_weight`` (absent: 1), the array cells (devices) one stored weight occupies in the unit
its chip description's per-cell rows are declared for, and ``weight_cell_rows``, how many of them lie on adjacent rows
of the weight's column (the rest on adjacent columns of its row): a weight is a block of cells. ``max_axons`` x
``max_neurons`` keep counting the weights a core holds; a chip description (a target bundle) whose geometry counts cells
derives its cores' weight capacity from it, and refuses by name a block whose cells hold no whole weights.

- NeuRRAM: "Each neural-network weight is encoded by the differential conductance between two RRAM cells on adjacent
  rows along the same column." (wan2022a Methods, PDF p.10), so its 256 x 256 RRAM cells (``neurram_like``) hold
  128 x 256 weights, as the paper maps a matrix: "split it into 2 segments, each with dimension 128 × 256, and program
  the two segments to two cores using the differential-row weight mapping" (PDF p.14). Two cells, on two rows.
- ISAAC-like: "We therefore represent one 16-bit synaptic weight with 16/w w-bit cells located in the same row."
  with w = 2 (shafiee2016isaac, Section V 'Synaptic Weights and ADCs', p.5): the 8 its conversion model already
  declares, on one row; its 128 x 128 two-bit cells hold 128 x 16 weights.
- PRIME [targeted M1]: "Finally, we employ two crossbar arrays store positive and negative weights, respectively, and
  allow them to share the same input port." (chi2016prime Section III-A1, p.30; the tree's card
  ``docs/research/physics/prime_constants_research.md``), so at its bound 4-bit weights a signed weight takes two cells
  on one input row, one in each array of the pair: its 256 x 256 cell mats hold 256 x 128 signed weights each.
- Every other shipped chip description stores one weight per cell of its geometry (HERMES-64: "a PCM crossbar array
  capable of storing a 256 × 256 weight matrix", gallo2022a p.2).
"""

import copy
import json
import os
from pathlib import Path

import pytest

import mimarsinan
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
from mimarsinan.config_schema.registry.entries_write import CORE_TYPE_SCHEMA
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.mapping.platform.imc_platforms import get_imc_platform
from mimarsinan.mapping.platform.target_bundles import TargetBundleError, load_bundle, target_bundles
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline, merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
from mimarsinan.pipelining.session import parse_deployment_config

TARGETS_DIR = Path(mimarsinan.__file__).parent / "mapping" / "platform" / "targets"
FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "fixtures", "deployment_configs")
PC = "platform_constraints"
#: The chip descriptions whose stored weight takes several cells: (cells per weight, cells on rows, registered geometry).
SEVERAL = {"neurram": (2, 2, "neurram_like"), "isaac_like": (8, 1, "isaac_like"),
           "prime_mat_256x256": (2, 1, "prime_mat_256x256")}


def _file(name):
    return json.loads((TARGETS_DIR / f"{name}.json").read_text())


def _core(name):
    core, = target_bundles()[name].values["cores"]
    return core


def _pcfg(name, **platform):
    return build_platform_constraints_resolved(merge_pipeline_config({}, {"target": name, **platform}))


def _bundle_with(tmp_path, name, **fields):
    data = _file(name)
    data["core_fields"] = {**data.get("core_fields", {}), **fields}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data))
    return path


class TestNeuRRAMCapacity:
    def test_a_core_holds_128_by_256_weights_of_two_cells_each(self):
        assert _core("neurram") == {"max_axons": 128, "max_neurons": 256, "count": 48, "has_bias": False,
                                    "write_granularity": "cell", "write_units_in_parallel": 1,
                                    "cells_per_weight": 2, "weight_cell_rows": 2}

    def test_the_weight_capacity_is_half_the_chip_s_cells(self):
        core = _core("neurram")
        weights = core["max_axons"] * core["max_neurons"]
        assert weights == 32_768 and weights * core["count"] == 1_572_864
        cells, = get_imc_platform("neurram_like").cores
        assert weights * core["cells_per_weight"] * core["count"] == cells["max_axons"] * cells["max_neurons"] * 48

    def test_the_array_is_the_registered_256_by_256_cells(self):
        from mimarsinan.mapping.platform.weight_cells import array_extent
        registered, = get_imc_platform("neurram_like").cores
        assert array_extent(_core("neurram")) == (registered["max_axons"], registered["max_neurons"]) == (256, 256)

    def test_a_resolved_run_maps_onto_the_weight_capacity(self):
        core, = _pcfg("neurram", activation_bits=4)["cores"]
        assert (core["max_axons"], core["max_neurons"], core["cells_per_weight"]) == (128, 256, 2)
        merged = merge_pipeline_config({}, {"target": "neurram", "activation_bits": 4})
        assert (merged["max_axons"], merged["max_neurons"]) == (128, 256)  # the flat maxima the mapping tiles at

    def test_the_bundle_cites_the_differential_row_mapping(self):
        notes = _file("neurram")["notes"]
        for quote in ("Each neural-network weight is encoded by the differential conductance between two RRAM cells "
                      "on adjacent rows along the same column.", "PDF p.10", "each with dimension 128 × 256",
                      "using the differential-row weight mapping", "PDF p.14", "python/nn_layers.py:33-37"):
            assert quote in notes, quote


class TestIsaacLike:
    def test_eight_two_bit_cells_along_neurons(self):
        core = _core("isaac_like")
        assert (core["max_axons"], core["max_neurons"], core["count"]) == (128, 16, 16128)
        assert (core["cells_per_weight"], core["weight_cell_rows"]) == (8, 1)

    def test_the_array_is_the_registered_128_by_128_cells(self):
        from mimarsinan.mapping.platform.weight_cells import array_extent
        registered, = get_imc_platform("isaac_like").cores
        assert array_extent(_core("isaac_like")) == (registered["max_axons"], registered["max_neurons"])

    def test_the_count_is_the_one_its_conversion_model_declares(self):
        model = get_platform_physics("isaac_like").conversion_model
        assert model["cells_per_weight"] == _core("isaac_like")["cells_per_weight"] == 8

    def test_the_bundle_cites_the_bit_slicing(self):
        notes = _file("isaac_like")["notes"]
        for quote in ("We therefore represent one 16-bit synaptic weight with 16/w w-bit cells located in the same "
                      "row.", "w = 2", "Synaptic Weights and ADCs", "p.5"):
            assert quote in notes, quote


class TestPrime:
    """[targeted M1] PRIME's signed weights: "Finally, we employ two crossbar arrays store positive and negative
    weights, respectively, and allow them to share the same input port." (chi2016prime Section III-A1, p.30). At the
    geometry's bound 4-bit weights (one 4-bit MLC cell each, Section V-A, p.36) a signed weight is two cells on one
    input row, one in each crossbar of the pair; the tree's card reads the registered 256 x 256 mats so ("the ± split
    puts the negative half in a second mat", prime_constants_research.md section 3.3(b))."""

    def test_a_mat_holds_256_by_128_signed_weights_of_two_cells_each(self):
        assert _core("prime_mat_256x256") == {"max_axons": 256, "max_neurons": 128, "count": 128, "has_bias": False,
                                              "cells_per_weight": 2, "weight_cell_rows": 1}

    def test_the_weight_capacity_is_half_the_chip_s_cells(self):
        core = _core("prime_mat_256x256")
        cells, = get_imc_platform("prime_mat_256x256").cores
        assert core["max_axons"] * core["max_neurons"] * core["count"] == 4_194_304
        assert 2 * 4_194_304 == cells["max_axons"] * cells["max_neurons"] * cells["count"]

    def test_the_array_is_the_registered_256_by_256_cells(self):
        from mimarsinan.mapping.platform.weight_cells import array_extent
        registered, = get_imc_platform("prime_mat_256x256").cores
        assert array_extent(_core("prime_mat_256x256")) == (registered["max_axons"], registered["max_neurons"])

    def test_the_count_is_the_one_at_the_bound_four_bit_weights(self):
        assert target_bundles()["prime_mat_256x256"].values["weight_bits"] == 4

    def test_the_bundle_cites_the_signed_pair_and_the_card(self):
        notes = _file("prime_mat_256x256")["notes"]
        for quote in ("Finally, we employ two crossbar arrays store positive and negative weights, respectively, and "
                      "allow them to share the same input port.", "Section III-A1", "p.30",
                      "docs/research/physics/prime_constants_research.md",
                      "two 4-bit cells to represent one 8-bit synaptic weight"):
            assert quote in notes, quote


class TestEveryOtherChipDescription:
    @pytest.mark.parametrize("name", sorted(set(target_bundles()) - set(SEVERAL)))
    def test_one_cell_per_weight(self, name):
        for core in target_bundles()[name].values["cores"]:
            assert "cells_per_weight" not in core and "weight_cell_rows" not in core

    def test_these_counts_are_the_only_ones_declared(self):
        declared = {name: (c["cells_per_weight"], c["weight_cell_rows"])
                    for name, bundle in target_bundles().items() for c in bundle.values["cores"]
                    if "cells_per_weight" in c}
        assert declared == {name: (n, rows) for name, (n, rows, _) in SEVERAL.items()}

    def test_hermes_64_s_geometry_counts_weights(self):
        assert "capable of storing a 256 × 256 weight matrix" in get_imc_platform("hermes_64").provenance
        assert "cells_per_weight" not in _core("hermes_64")


class TestContradictingDeclarations:
    """A block the geometry's cells cannot hold as whole weights, a count without its rows, or rows that do not divide
    it, is refused by name."""

    @pytest.mark.parametrize("fields, named", [
        ({"cells_per_weight": 3, "weight_cell_rows": 3}, "cells_per_weight 3 in blocks of 3 x 1 cells"),
        ({"cells_per_weight": 3, "weight_cell_rows": 3}, "its 256 cell rows hold no whole number of weights"),
        ({"cells_per_weight": 512, "weight_cell_rows": 1}, "its 256 cell columns hold no whole number of weights"),
        ({"cells_per_weight": 2}, "cells_per_weight 2 states no weight_cell_rows"),
        ({"cells_per_weight": 4, "weight_cell_rows": 3}, "weight_cell_rows 3 does not divide cells_per_weight 4"),
    ], ids=["not_whole_rows", "names_the_rows", "not_whole_columns", "no_rows", "rows_not_dividing"])
    def test_a_declaration_that_contradicts_the_geometry_refuses_by_name(self, tmp_path, fields, named):
        data = _file("neurram")
        data["core_fields"] = {k: v for k, v in data["core_fields"].items()
                               if k not in ("cells_per_weight", "weight_cell_rows")}
        data["core_fields"].update(fields)
        path = tmp_path / "neurram.json"
        path.write_text(json.dumps(data))
        with pytest.raises(TargetBundleError, match=named):
            load_bundle(path)

    @pytest.mark.parametrize("fields, named", [
        ({"cells_per_weight": 0, "weight_cell_rows": 1}, r"cells_per_weight=0 is below the minimum 1"),
        ({"cells_per_weight": 2, "weight_cell_rows": 0}, r"weight_cell_rows=0 is below the minimum 1"),
        ({"cells_per_weight": 2, "weight_cell_rows": "two"}, r"core_fields.weight_cell_rows must be"),
    ])
    def test_a_value_the_registry_refuses_is_refused(self, tmp_path, fields, named):
        with pytest.raises(TargetBundleError, match=named):
            load_bundle(_bundle_with(tmp_path, "hermes_64", **fields))

    def test_one_cell_per_weight_keeps_the_geometry(self, tmp_path):
        bundle = load_bundle(_bundle_with(tmp_path, "hermes_64", cells_per_weight=1, weight_cell_rows=1))
        core, = bundle.values["cores"]
        assert (core["max_axons"], core["max_neurons"], core["count"]) == (256, 256, 64)

    def test_a_block_of_two_rows_by_two_columns(self, tmp_path):
        """W38's arm W form: an 8-bit weight on two 4-bit differential pairs, bit slices across adjacent columns
        (4 cells per weight): NeuRRAM's 256 x 256 cells hold 128 x 128 such weights."""
        from mimarsinan.mapping.platform.weight_cells import array_extent
        data = _file("neurram")
        data["core_fields"].update(cells_per_weight=4, weight_cell_rows=2)
        path = tmp_path / "neurram.json"
        path.write_text(json.dumps(data))
        core, = load_bundle(path).values["cores"]
        assert (core["max_axons"], core["max_neurons"], core["count"]) == (128, 128, 48)
        assert array_extent(core) == (256, 256)


def _load(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def _value_document(cores, **platform):
    doc = _load("mvm_aq_lenet5.json")
    pc = {k: v for k, v in doc[PC].items() if k not in ("cores", "max_axons", "max_neurons")}
    doc[PC] = {**pc, "cores": copy.deepcopy(cores), **platform}
    return doc


def _run(doc, tmp_path):
    parsed = parse_deployment_config({**copy.deepcopy(doc), "_working_directory": str(tmp_path)},
                                     data_provider_factory=object())
    dp = dict(parsed.deployment_parameters)
    DeploymentPipeline.apply_preset(parsed.pipeline_mode, dp)
    flat = merge_pipeline_config(dp, copy.deepcopy(parsed.platform_constraints))
    return build_platform_constraints_resolved(flat)


def _verdicts(doc, tmp_path):
    document = validate_deployment_config(copy.deepcopy(doc))
    wizard = [row["message"] for row in resolve_draft(copy.deepcopy(doc)).errors]
    try:
        _run(doc, tmp_path)
        run = None
    except ValueError as exc:
        run = exc
    return document, wizard, run


PAIR = {"max_axons": 64, "max_neurons": 64, "count": 4, "has_bias": True, "cells_per_weight": 2}


class TestDoors:
    def test_a_count_without_its_rows_refuses_by_name_at_every_door(self, tmp_path):
        document, wizard, run = _verdicts(_value_document([PAIR]), tmp_path)
        named = "cores[0].cells_per_weight 2 states no weight_cell_rows"
        assert any(named in m for m in document), document
        assert any(named in m for m in wizard), wizard
        assert isinstance(run, ToolchainRefusal) and named in str(run), repr(run)

    def test_rows_that_do_not_divide_the_count_refuse_by_name_at_every_door(self, tmp_path):
        document, wizard, run = _verdicts(_value_document([{**PAIR, "weight_cell_rows": 3}]), tmp_path)
        named = "cores[0].weight_cell_rows 3 does not divide cells_per_weight 2"
        assert any(named in m for m in document), document
        assert any(named in m for m in wizard), wizard
        assert isinstance(run, ToolchainRefusal) and named in str(run), repr(run)

    def test_a_count_with_its_rows_is_admitted_at_every_door(self, tmp_path):
        document, wizard, run = _verdicts(_value_document([{**PAIR, "weight_cell_rows": 2}]), tmp_path)
        assert not [m for m in document if "cells_per_weight" in m or "weight_cell_rows" in m], document
        assert not [m for m in wizard if "cells_per_weight" in m or "weight_cell_rows" in m], wizard
        assert run is None, repr(run)

    def test_a_count_the_conversion_model_contradicts_refuses_at_the_run_door(self, tmp_path):
        """The bit-sliced conversion model's ``cells_per_weight`` is the same quantity (ISAAC-like declares 8): a core
        type stating another count beside it refuses by name where the run resolves its physics."""
        doc = _value_document([{**PAIR, "cells_per_weight": 4, "weight_cell_rows": 1}],
                              platform_physics_profile="isaac_like")
        with pytest.raises(ToolchainRefusal, match=r"cells_per_weight 4.*conversion model.*8"):
            _run(doc, tmp_path)

    def test_an_agreeing_count_beside_the_conversion_model_resolves(self, tmp_path):
        doc = _value_document([{**PAIR, "cells_per_weight": 8, "weight_cell_rows": 1}],
                              platform_physics_profile="isaac_like")
        assert _run(doc, tmp_path)["cores"][0]["cells_per_weight"] == 8


class TestRegistry:
    @pytest.mark.parametrize("field", ["cells_per_weight", "weight_cell_rows"])
    def test_a_cost_only_positive_int_core_type_field(self, field):
        schema = CORE_TYPE_SCHEMA.fields[field]
        assert schema.decision_class == COST_ONLY and schema.low == 1
        assert schema.doc and schema.absent_means

    def test_the_count_is_absent_one_and_its_rows_required_above_it(self):
        assert CORE_TYPE_SCHEMA.fields["cells_per_weight"].absent_means.startswith("1")
        assert "required where cells_per_weight exceeds 1" in CORE_TYPE_SCHEMA.fields["weight_cell_rows"].absent_means
