"""The census runner's acceptance checker and its scout pinning (``scripts/requirement_census.py``)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

import mimarsinan
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping import capability
from mimarsinan.mapping.capability.census import ladder_census
from mimarsinan.mapping.capability.ladder import LADDER, PLACEMENT_LABEL, payload_writes
from mimarsinan.mapping.capability.mac_split import (
    COUNTERS_DISAGREE,
    ESTIMATOR_BELOW_CORE_FLOOR,
    UNCOUNTED_PRODUCT,
    MacCounters,
)
from mimarsinan.mapping.capability.member import (
    AGREEMENT,
    INCOMPLETE,
    REFUSED,
    VERDICT_COLUMN,
    ExactnessVerdict,
    MemberCensus,
)
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import probe_stages

from .ir_fixtures import grouped_conv3d_graph, grouped_graph, lane_graph, seed_mac_gap

REQUIRED_STAGES = probe_stages(MVM_PACKAGING)
SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "requirement_census.py"
GROUPED_BOUNDARY = "grouped convolution on the host until N5"
#: The provenance every census record carries: the runner's sha, the capability package's and the source tree's.
DIGESTS = ("script_sha256_16", "capability_sha256_16", "mimarsinan_src_sha256_16")
HEADER = dict(zip(DIGESTS, ("0" * 16, "1" * 16, "2" * 16)))


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("requirement_census_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rows(c0_extra=None):
    rows = []
    for level in LADDER:
        for write in payload_writes(level):
            row = {"level": level.name, "payload_write": write, "label": None if level.name == "C0" else PLACEMENT_LABEL}
            if level.name == "C0":
                row.update(c0_extra or {})
            rows.append(row)
    return rows


def _member(status="fp64 agreement", **verdict):
    agreeing = status == "fp64 agreement"
    base = {"column": VERDICT_COLUMN, "status": status, "relative_error": 1e-16 if agreeing else None,
            "stages": [[stage, 1e-16] for stage in REQUIRED_STAGES] if agreeing else [],
            "refusal": None, "refusal_is_toolchain_refusal": None, "root_refusal": None,
            "root_refusal_is_toolchain_refusal": None, "value_mismatch": False, "missing_stages": []}
    base.update(verdict)
    return {"verdict": base, "rows": _rows({VERDICT_COLUMN: base["relative_error"]}),
            "mac_counters": {"has_multi_input_op": False, "agree": True}}


def _refused(name="RepresentabilityError", **verdict):
    return _member("refused", **{"refusal": name, "refusal_is_toolchain_refusal": True, "root_refusal": name,
                                 "root_refusal_is_toolchain_refusal": True, **verdict})


def _with_c0(member, **fields):
    return {**member, "rows": [{**r, **fields} if r["level"] == "C0" else r for r in member["rows"]]}


def _check(runner, tmp_path, members, **header):
    """``header`` overrides a provenance field; None leaves it out."""
    path = tmp_path / "census.json"
    doc = {**HEADER, **header, "members": members}
    path.write_text(json.dumps({key: value for key, value in doc.items() if value is not None}))
    return runner.check([str(path)])


def _shape(side):
    return [3, side, side]


def _note(declared, side):
    return "declared" if side == declared else f"reduced from {declared} to {side}"


def _zoo(declared=8, census=8, verdict=None, *, census_note=None, verdict_note=None, member_mark=True,
         row_mark=True, declared_shape=True):
    """A zoo member judged at ``verdict`` (default: the census side), its rows at ``census``, both against
    ``declared``; the notes and marks are written by hand here, each overridable to seed a breach."""
    verdict = census if verdict is None else verdict
    member = _member()
    member.update({"input_shape": _shape(census),
                   "resolution": {"census": census_note or _note(declared, census),
                                  "verdict": verdict_note or _note(declared, verdict)}})
    if declared_shape:
        member["declared_input_shape"] = _shape(declared)
    if verdict != census:
        if member_mark:
            member["verdict_input_shape"] = _shape(verdict)
        if row_mark:
            member = _with_c0(member, verdict_input_shape=_shape(verdict))
    return member


def _boundary(**fields):
    return {"boundary": GROUPED_BOUNDARY, **fields}


def test_a_clean_census_passes(runner, tmp_path):
    wrapped = _refused("ConversionValueParityError", root_refusal="NonTensorHostValueError")
    stated = _with_c0(_member(), core_mac_floor=10, quantities={"onchip_macs": 10, "host_macs": 0})
    shared = _with_c0(_member(), stated_boundaries=[_boundary(host_macs=8, host_mac_share=0.04)])
    unshared = _with_c0(_member(), mac_total_withheld=ESTIMATOR_BELOW_CORE_FLOOR,
                        stated_boundaries=[_boundary(host_macs=8, withheld=ESTIMATOR_BELOW_CORE_FLOOR)])
    zoo = {"z_declared": _zoo(8, 8), "z_verdict_reduced": _zoo(8, 8, 4), "z_both_reduced": _zoo(8, 4, 4)}
    assert _check(runner, tmp_path, {"a": _member(), "b": _refused(), "c": wrapped, "d": stated, "e": shared,
                                     "f": unshared, **zoo}) == 0


@pytest.mark.parametrize("member", [
    {"loud_failure": "RuntimeError: boom"},
    _member("refused", refusal="RuntimeError", refusal_is_toolchain_refusal=False),
    {**_member(), "census_refusal": "UnknownOpKindError: x"},
    {**_member(), "mac_counters": {"has_multi_input_op": False, "agree": False}},
    # CL4: since G1's input_shapes rule the two counters agree on every row, a multi-input op's included.
    {**_member(), "mac_counters": {"has_multi_input_op": True, "agree": False}},
    {**_member(), "rows": _rows()[:3]},
    {**_member(), "rows": [{**r, VERDICT_COLUMN: 0.0} if r["level"] == "C2" else r for r in _rows()]},
    {**_member(), "rows": [{**r, "label": None} for r in _rows()]},
    _member(stages=[["converted model", 0.0]]),
    _member("value mismatch", relative_error=1e-3, value_mismatch=True, refusal="ConversionValueParityError",
            refusal_is_toolchain_refusal=True, root_refusal="ConversionValueParityError",
            root_refusal_is_toolchain_refusal=True),
    _refused(value_mismatch=True),
    _refused(root_refusal="KeyError", root_refusal_is_toolchain_refusal=False),
    _member("incomplete verdict", missing_stages=["identity value program"]),
    _with_c0(_member(), core_mac_floor=10, quantities={"onchip_macs": 5}),
    _with_c0({**_member(), "mac_counters": {"has_multi_input_op": True, "agree": False}}, quantities={"host_macs": 0}),
    # A stated boundary's share over a withheld total, a boundary with neither a share nor a reason, a share
    # without the host MACs it divides.
    _with_c0(_member(), mac_total_withheld=ESTIMATOR_BELOW_CORE_FLOOR,
             stated_boundaries=[_boundary(host_macs=8, host_mac_share=0.8)]),
    _with_c0(_member(), stated_boundaries=[_boundary(host_macs=8)]),
    _with_c0(_member(), stated_boundaries=[_boundary(host_mac_share=0.04)]),
    # Resolution marks: a reduced verdict without its shape, a "declared" verdict at another shape, a member mark the
    # C0 row lacks, a resolution without the declared shape, a "declared" census at another shape, a note naming
    # another side, a verdict shape that only repeats the census shape.
    _zoo(8, 8, 4, member_mark=False, row_mark=False),
    _zoo(8, 8, 4, verdict_note="declared"),
    _zoo(8, 8, 4, row_mark=False),
    _zoo(8, 8, 4, declared_shape=False),
    _zoo(8, 4, 4, census_note="declared"),
    _zoo(8, 8, 4, verdict_note="reduced from 8 to 2"),
    {**_zoo(8, 8), "verdict_input_shape": _shape(8)},
    # A C0 row marked although its member is not (scout-shaped, and zoo-shaped with notes that fit an unmarked
    # member), and a mark that repeats the census shape on both member and C0 row: each is caught by its own guard.
    _with_c0({**_member(), "input_shape": _shape(8)}, verdict_input_shape=_shape(4)),
    _zoo(8, 8, 4, member_mark=False, verdict_note="declared"),
    _with_c0({**_zoo(8, 8), "verdict_input_shape": _shape(8)}, verdict_input_shape=_shape(8)),
    _with_c0({**_member(), "input_shape": _shape(8), "verdict_input_shape": _shape(8)},
             verdict_input_shape=_shape(8)),
])
def test_each_acceptance_breach_fails_the_check(runner, tmp_path, member):
    assert _check(runner, tmp_path, {"a": member}) == 1


@pytest.mark.parametrize("key", DIGESTS)
def test_a_record_without_its_provenance_fails_the_check(runner, tmp_path, key):
    assert _check(runner, tmp_path, {"a": _member()}, **{key: None}) == 1


@pytest.mark.parametrize("key", DIGESTS)
def test_records_from_two_code_states_fail_the_check(runner, tmp_path, key):
    paths = []
    for i, value in enumerate(("a" * 16, "b" * 16)):
        path = tmp_path / f"census{i}.json"
        path.write_text(json.dumps({**HEADER, key: value, "members": {f"m{i}": _member()}}))
        paths.append(str(path))
    assert runner.check(paths[:1]) == 0
    assert runner.check(paths) == 1, "one census, one code state"


def test_a_record_pins_the_capability_package_and_the_source_tree(runner, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    (tmp_path / "notes.txt").write_text("not source")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "c.py").write_text("z = 3\n")
    (tmp_path / "pkg" / "notes.txt").write_text("not source either")
    digest = runner.source_digest(str(tmp_path))
    recipe = b"a.py\0x = 1\n\0b.py\0y = 2\n\0pkg/c.py\0z = 3\n\0"
    assert digest == hashlib.sha256(recipe).hexdigest()[:16], "the documented recipe, nested packages included"
    (tmp_path / "pkg" / "c.py").write_text("z = 4\n")
    assert runner.source_digest(str(tmp_path)) != digest, "an edit to a nested module alone changes the digest"
    (tmp_path / "pkg" / "c.py").write_text("z = 3\n")
    assert runner.source_digest(str(tmp_path)) == digest
    (tmp_path / "b.py").write_text("y = 3\n")
    assert runner.source_digest(str(tmp_path)) != digest, "an edit to the source alone changes the digest"
    header = runner._header("zoo")
    assert header["capability_sha256_16"] == runner.source_digest(str(Path(capability.__file__).parent))
    assert header["mimarsinan_src_sha256_16"] == runner.source_digest(str(Path(mimarsinan.__file__).parent))
    assert set(DIGESTS) <= set(header)


def test_the_scout_instances_are_imported_only_from_the_pinned_files(runner, tmp_path):
    (tmp_path / "req_census.py").write_text("def members():\n    return []\n")
    (tmp_path / "net_models.py").write_text("")
    with pytest.raises(SystemExit, match="sha256"):
        runner.scout_members(str(tmp_path))


def test_the_tv_zoo_column_is_the_scout_subset(runner):
    assert len(runner.TV_ZOO_23) == 23 == len(set(runner.TV_ZOO_23))
    assert runner.ZOO_MAX_PARAMS == 200_000_000


def _agreement():
    return ExactnessVerdict(AGREEMENT, relative_error=0.0, bound=1e-12,
                            stages=tuple((stage, 0.0) for stage in REQUIRED_STAGES))


def test_a_row_withholds_every_mac_field_an_undercounting_total_cannot_back(runner):
    rows = ladder_census(lane_graph(), value_bits=8)
    census = MemberCensus(_agreement(), MacCounters(0, 0, 55, False), rows)
    for row in rows:
        data = runner._row_json(row, census)
        assert "onchip_mac_share" not in data and "onchip_macs" not in data["quantities"]
        assert data["mac_total_withheld"] == ESTIMATOR_BELOW_CORE_FLOOR == data["withheld"]["onchip_macs"]
        assert data["withheld"]["onchip_mac_share"] == ESTIMATOR_BELOW_CORE_FLOOR
        assert data["quantities"]["host_macs"] == data["host_macs"] == 0 and data["core_mac_floor"] == 56
    stated = runner._row_json(rows[0], MemberCensus(_agreement(), MacCounters(0, 0, 100, False), rows))
    assert stated["onchip_mac_share"] == 1.0 and stated["quantities"]["onchip_macs"] == 100
    assert "mac_total_withheld" not in stated and stated["withheld"] == {}


def test_a_row_of_a_member_whose_counters_disagree_names_the_reason_and_both_numbers(runner):
    rows = ladder_census(lane_graph(), value_bits=8)
    census = MemberCensus(_agreement(), MacCounters(0, 40, 500, True), rows)
    for row in rows:
        data = runner._row_json(row, census)
        assert "host_macs" not in data and "host_macs" not in data["quantities"]
        assert "onchip_mac_share" not in data and "onchip_macs" not in data["quantities"]
        assert data["mac_counters_disagree"] == {
            "reason": COUNTERS_DISAGREE, "count_host_ops": 0, "estimate_onchip_fraction_host": 40}
        assert data["withheld"]["host_macs"] == COUNTERS_DISAGREE


def test_a_stated_boundary_carries_its_host_share_only_over_a_backed_total(runner, tmp_path):
    rows = ladder_census(grouped_graph(), value_bits=8)
    c0, c2 = rows[0], rows[2]
    assert (c0.level, c2.level) == ("C0", "C2")
    short = MemberCensus(_agreement(), MacCounters(8, 8, 10, False), rows)
    withheld = runner._row_json(c0, short)
    assert withheld["mac_total_withheld"] == ESTIMATOR_BELOW_CORE_FLOOR
    assert withheld["stated_boundaries"] == [_boundary(host_macs=8, withheld=ESTIMATOR_BELOW_CORE_FLOOR)]
    backed = MemberCensus(_agreement(), MacCounters(8, 8, 200, False), rows)
    stated = runner._row_json(c0, backed)
    assert "mac_total_withheld" not in stated
    assert stated["stated_boundaries"] == [_boundary(host_macs=8, host_mac_share=0.04)]
    assert runner._row_json(c2, backed)["stated_boundaries"] == [], "C2 declares G: no host boundary"
    for census in (short, backed):
        member = {**_member(), "rows": [runner._row_json(row, census) for row in rows]}
        assert _check(runner, tmp_path, {"grouped": member}) == 0


def test_a_stated_boundary_whose_product_the_mac_ssot_counts_zero_states_no_host_macs(runner, monkeypatch):
    seed_mac_gap(monkeypatch, "Conv3d")
    rows = ladder_census(grouped_conv3d_graph(), value_bits=8)
    data = runner._row_json(rows[0], MemberCensus(_agreement(), MacCounters(0, 0, 200, False), rows))
    assert "mac_total_withheld" not in data, "the total is backed; only the boundary's own count is missing"
    assert data["stated_boundaries"] == [_boundary(withheld=f"{UNCOUNTED_PRODUCT}: Conv3d")]


def test_a_stated_boundary_share_names_every_reason_it_is_withheld(runner, tmp_path, monkeypatch):
    """Host MACs uncounted and the total withheld: the entry names both, numerator first, as the row's share does."""
    seed_mac_gap(monkeypatch, "Conv3d")
    rows = ladder_census(grouped_conv3d_graph(), value_bits=8)
    assert rows[0].core_mac_floor == 80
    census = MemberCensus(_agreement(), MacCounters(0, 0, 10, False), rows)
    data = runner._row_json(rows[0], census)
    both = f"{UNCOUNTED_PRODUCT}: Conv3d; {ESTIMATOR_BELOW_CORE_FLOOR}"
    assert data["mac_total_withheld"] == ESTIMATOR_BELOW_CORE_FLOOR
    assert data["stated_boundaries"] == [_boundary(withheld=both)], "no host_macs, and both reasons named"
    assert data["withheld"]["onchip_mac_share"] == both, "the row's own share joins the same two reasons"
    member = {**_member(), "rows": [runner._row_json(row, census) for row in rows]}
    assert _check(runner, tmp_path, {"grouped_conv3d": member}) == 0


def test_a_counted_grouped_conv3d_states_its_boundary_host_macs(runner):
    rows = ladder_census(grouped_conv3d_graph(), value_bits=8)
    data = runner._row_json(rows[0], MemberCensus(_agreement(), MacCounters(8, 8, 200, False), rows))
    assert data["stated_boundaries"] == [_boundary(host_macs=8, host_mac_share=0.04)]


def _refused_verdict():
    return ExactnessVerdict(REFUSED, refusal="RepresentabilityError", root_refusal="RepresentabilityError")


def test_a_member_without_rows_marks_the_verdict_input_on_its_c0_row_only(runner):
    census = MemberCensus(_refused_verdict(), None, ())
    rows = runner._rowless(census, (3, 4, 4))
    assert [(r["level"], r["payload_write"]) for r in rows] == [
        (level.name, write) for level in LADDER for write in payload_writes(level)]
    assert rows[0]["level"] == "C0" and rows[0]["verdict_input_shape"] == [3, 4, 4]
    assert all("verdict_input_shape" not in row for row in rows[1:])
    assert all(row["refused"] == "RepresentabilityError" for row in rows)
    assert all("verdict_input_shape" not in row for row in runner._rowless(census))


class _SharedLinear(torch.nn.Module):
    """One Linear reached from two call sites: refused by name until call-site weight sharing (N7) exists."""

    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(8, 8), torch.nn.Linear(8, 3)

    def forward(self, x):
        return self.b(torch.relu(self.a(torch.relu(self.a(x)))))


def test_a_refused_member_judged_on_another_input_is_marked_on_its_c0_row(runner, tmp_path):
    torch.manual_seed(0)
    shared = _SharedLinear().eval()
    record = runner.census_record(shared, (2, 8), 3, verdict_shape=(1, 8))
    assert record["verdict"]["status"] == REFUSED and record["verdict"]["root_refusal_is_toolchain_refusal"]
    assert record["verdict"]["root_refusal"] == "RepresentabilityError"
    assert record["verdict_input_shape"] == record["rows"][0]["verdict_input_shape"] == [1, 8]
    assert all("verdict_input_shape" not in row for row in record["rows"][1:])
    assert _check(runner, tmp_path, {"shared": {"input_shape": [2, 8], **record}}) == 0


def test_an_incomplete_verdict_exports_the_stages_it_misses(runner, tmp_path, capsys):
    verdict = ExactnessVerdict(INCOMPLETE, stages=((REQUIRED_STAGES[0], 0.0),), missing_stages=REQUIRED_STAGES[1:])
    census = MemberCensus(verdict, None, ())
    data = runner._verdict_json(census)
    assert data["status"] == INCOMPLETE and data["missing_stages"] == list(REQUIRED_STAGES[1:]) != []
    assert _check(runner, tmp_path, {"a": {"verdict": data, "rows": runner._rowless(census)}}) == 1
    assert f"missing {list(REQUIRED_STAGES[1:])}" in capsys.readouterr().out


def _tiny_conv(channels):
    def build():
        torch.manual_seed(0)
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels, 2, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(), torch.nn.Linear(2, 3)).eval()

    return build


def test_the_verdict_and_the_census_resolutions_are_recorded_separately(runner):
    model = _tiny_conv(1)()
    record = runner.census_record(model, (1, 8, 8), 3, verdict_shape=(1, 4, 4))
    assert record["verdict"]["status"] == AGREEMENT
    assert record["verdict_input_shape"] == [1, 4, 4]
    assert record["rows"][0]["verdict_input_shape"] == [1, 4, 4], "the C0 row names the input its verdict judged"
    assert all("verdict_input_shape" not in row for row in record["rows"][1:])
    assert record["rows"][0]["boundary_elements"]["entry"] == 64, "the census ran on the 8 x 8 input"
    assert set(record["peak_rss_gb"]) == {"after_verdict", "after_census"}
    same = runner.census_record(model, (1, 4, 4), 3)
    assert "verdict_input_shape" not in same and all("verdict_input_shape" not in row for row in same["rows"])
    assert runner.resolution_note(224, 224) == "declared"
    assert runner.resolution_note(224, 112) == "reduced from 224 to 112"


@pytest.mark.parametrize("declared, side, verdict, expected", [
    (224, None, None, ((3, 224, 224), (3, 224, 224), {"census": "declared", "verdict": "declared"})),
    (224, None, 112, ((3, 224, 224), (3, 112, 112), {"census": "declared", "verdict": "reduced from 224 to 112"})),
    (299, None, 150, ((3, 299, 299), (3, 150, 150), {"census": "declared", "verdict": "reduced from 299 to 150"})),
    (224, 112, None, ((3, 112, 112), (3, 112, 112),
                      {"census": "reduced from 224 to 112", "verdict": "reduced from 224 to 112"})),
    (224, 112, 224, ((3, 112, 112), (3, 224, 224), {"census": "reduced from 224 to 112", "verdict": "declared"})),
])
def test_the_zoo_side_choice_marks_each_resolution_against_the_declared_side(runner, declared, side, verdict,
                                                                             expected):
    assert runner.zoo_sides(declared, side, verdict) == expected


def test_the_zoo_runner_judges_the_verdict_at_the_verdict_side_and_the_rows_at_the_census_side(
        runner, tmp_path, monkeypatch):
    judged = []
    real = runner.c0_verdict

    def spy(model, shape, **kwargs):
        judged.append(tuple(shape))
        return real(model, shape, **kwargs)

    monkeypatch.setattr(runner, "zoo_builder", lambda name: _tiny_conv(3))
    monkeypatch.setattr(runner, "ZOO_INPUT_SIDE", {"tiny": 8})
    monkeypatch.setattr(runner, "ZOO_CLASSES", 3)
    monkeypatch.setattr(runner, "c0_verdict", spy)
    out = tmp_path / "zoo.json"
    runner.run_zoo(str(out), ["tiny"], verdict_side=4)
    record = json.loads(out.read_text())["members"]["tiny"]
    assert judged == [(3, 4, 4)]
    assert record["input_shape"] == record["declared_input_shape"] == [3, 8, 8]
    assert record["verdict_input_shape"] == record["rows"][0]["verdict_input_shape"] == [3, 4, 4]
    assert record["resolution"] == {"census": "declared", "verdict": "reduced from 8 to 4"}
    assert record["rows"][0]["boundary_elements"]["entry"] == 3 * 8 * 8, "the census ran at the declared side"
    assert runner.check([str(out)]) == 0


def _pins(tmp_path, pins):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"members": pins}))
    return str(path)


def _check_pinned(runner, tmp_path, members, pins):
    path = tmp_path / "census.json"
    path.write_text(json.dumps({**HEADER, "members": members}))
    return runner.check([str(path)], pins=_pins(tmp_path, pins))


def test_each_zoo_reading_equals_its_pinned_name(runner, tmp_path, capsys):
    """CL4 re-pins the zoo's per-member readings (plan §11.3 acceptance 5): an agreement or the root refusal name."""
    members = {"a": _member(), "b": _refused("RepresentabilityError")}
    assert _check_pinned(runner, tmp_path, members, {"a": "fp64 agreement", "b": "RepresentabilityError"}) == 0
    assert _check_pinned(runner, tmp_path, members, {"a": "fp64 agreement", "b": "GenericOpArgumentError"}) == 1
    assert "b: pinned GenericOpArgumentError, reads RepresentabilityError" in capsys.readouterr().out


def test_a_member_without_a_pin_and_a_pin_without_a_member_both_fail(runner, tmp_path, capsys):
    assert _check_pinned(runner, tmp_path, {"a": _member()}, {}) == 1
    assert "a: not pinned" in capsys.readouterr().out
    assert _check_pinned(runner, tmp_path, {"a": _member()}, {"a": "fp64 agreement", "gone": "fp64 agreement"}) == 1
    assert "gone: pinned but absent" in capsys.readouterr().out


ZOO_PINS = Path(__file__).resolve().parents[3] / "fixtures" / "requirement_census" / "zoo_pins.json"
SRC = Path(mimarsinan.__file__).resolve().parent


def _refusal_class(name: str):
    """The toolchain class ``name``, found by its definition in the source tree and imported from its module."""
    for path in SRC.rglob("*.py"):
        if f"class {name}(" in path.read_text(encoding="utf-8"):
            module = ".".join(("mimarsinan",) + path.relative_to(SRC).with_suffix("").parts)
            return getattr(importlib.import_module(module), name)
    return None


def test_every_zoo_pin_is_an_agreement_or_a_refusal_the_toolchain_defines():
    """CL4's zoo re-pin (plan §11.3 acceptance 5): each zoo member's pinned reading, which ``check --pins`` compares
    with its census record, is the fp64 agreement or the name of a ToolchainRefusal subclass of this toolchain (never
    a value mismatch, which is an exactness failure to fix, and never an anonymous error)."""
    pins = json.loads(ZOO_PINS.read_text(encoding="utf-8"))["members"]
    assert len(pins) >= 70, "the whole in-zoo roster is pinned"
    for member, reading in sorted(pins.items()):
        if reading == AGREEMENT:
            continue
        refusal = _refusal_class(reading)
        assert refusal is not None and issubclass(refusal, ToolchainRefusal), (member, reading)
