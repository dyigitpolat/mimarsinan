"""[U-J J-1] The edge host description: a Jetson-class module as the deployment host, every constant sourced.

Owner rule of 2026-10-28: host work is not free, and its latency, energy and power (the host-chip boundary I/O
included) are modelled from a low-power Jetson-class device, never measured on the experiment machines. The host is
declared as data (``platform_physics/hosts/<name>.json`` beside its ``.md``): its published power modes, a mode
chosen by a declared rule, and per row a class (published, derived, estimated) and a band. Each constant here is
recomputed from its source's printed numbers (``RA/paper_pdfs/fetch_20261028_host/SOURCES.md``), so a changed value
or a lost source fails by name. A host block names its host on every row; one deployment has one host, so a block
that mixes a measured server row with the edge host is refused by name, and an identified host replaces the profile's
host rows whole."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mimarsinan.deployment_record.platform_physics import vocabulary
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    HOST_IDENTITY_FIELD, REQUIRED_HOST_ROWS, HostBlockMixError, HostDescriptionError, HostModeError,
    HostModeIncompleteError, host_description, host_description_block, host_descriptions, select_mode,
)
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.units import ENERGY, POWER, RATE, TIME

NAME = "jetson_orin_nano_8gb"
HOSTS = Path(vocabulary.__file__).resolve().parent / "hosts"

# --- the published inputs, as their sources print them -------------------------------------------------------
#: NVIDIA Jetson Linux r36.5, Supported Modes, table "Jetson Orin Nano 8GB": (online CPUs, CPU max MHz, memory MHz).
MODES = {"15W": (6, 1510.4, 2133.0, 15.0, 0), "7W": (4, 960.0, 2133.0, 7.0, 1)}
FMLA_PER_CYCLE, FP32_LANES = 2, 4                    # Arm Cortex-A78AE SWOG Issue 6.0 sec. 3.17 p. 40 (128-bit Q form)
ET_CYCLES = (75, 360, 390)                           # nachin2026executorch Table 1: ExecuTorch framework, mul, add
MI_CYCLES = (324_399, 7_976, 8_493)                  # PyTorch Mobile Interpreter, same columns
ACTIVE_W, IDLE_W = (11.01, 11.12), (5.69, 5.68)      # cao2026vlaulap Table 4 (15 W mode, VDD_IN)
PEER_IDLE_W = 4.0                                    # [U-J2 K-3] tsuyuki2026efficient Table 5 (15 W mode, supply input)
EMOP_PJ = {0.7: 444.9, 2.1: 174.7, 3.2: 141.4}       # prashanthis2025pagoda p. 25
BW_AGX = (103.9, 204.8)                              # p. 22: 103.9 GB/s at 2.1 GHz; rating 204.8 GB/s at 3.2 GHz
GEN3_GTS, LANES, TLP_SHARE, MWR_HDR = 8.0, 4, 57.88 / 62.96, 24   # neugebauer2018understanding pp. 3-4
DMA_READ_NS, PCIE_RT_NS = 529.0, 900.0               # neugebauer2018understanding pp. 8, 3
TX_PJ_PER_BIT = 1.93                                 # bae2021a abstract


def _expected():
    """(low, nominal, high, unit, class) of every row, per mode and common, from the inputs above."""
    out = {}
    per_op = lambda cycles: sum(cycles) / 2  # noqa: E731 (a two-operator model)
    peak = 16 * 2 * 2133.0 * 1e6 / 1e9
    sustained = peak * BW_AGX[0] / (BW_AGX[1] * 2.1 / 3.2)
    for mode, (cores, mhz, _emc, budget, _mid) in MODES.items():
        one = FMLA_PER_CYCLE * FP32_LANES * mhz / 1e3
        rows = {"host_macs_per_s": (one, one, one * cores, "G/s", "derived"),
                "t_host_op_overhead": (per_op(ET_CYCLES) / mhz, per_op(ET_CYCLES) / mhz, per_op(MI_CYCLES) / mhz,
                                       "us", "derived"),
                "t_host_mem_per_byte": (1 / peak, 1 / sustained, 1 / sustained, "ns", "derived")}
        if mode == "15W":
            # [U-J2 K-3, amended] the low ends rest on the peer-reviewed idle, the high end of p_host on the vendor's
            # VDD_IN average-power budget of the mode, the nominals on the preprint's one-setup pair
            rows["p_host"] = (PEER_IDLE_W, ACTIVE_W[0], budget, "W", "published")
            rows["p_host_idle"] = (PEER_IDLE_W, sum(IDLE_W) / 2, max(IDLE_W), "W", "published")
        out[mode] = rows
    phys = GEN3_GTS * 128 / 130 * LANES / 8
    rate = {m: phys * TLP_SHARE * m / (m + MWR_HDR) for m in (512, 256, 128)}
    bits = {m: 8 * 130 / 128 / (TLP_SHARE * m / (m + MWR_HDR)) for m in (512, 256, 128)}
    out["common"] = {
        # [targeted M3] Orin AGX's LPDDR5 read as this module's: a cross-module proxy, so derived
        "e_host_mem_per_byte": (EMOP_PJ[3.2], EMOP_PJ[2.1], EMOP_PJ[2.1], "pJ", "derived"),
        "e_link_per_byte": (TX_PJ_PER_BIT * bits[512], TX_PJ_PER_BIT * bits[256], 2 * TX_PJ_PER_BIT * bits[128],
                            "pJ", "derived"),
        "t_link_per_byte": (1 / rate[512], 1 / rate[256], 1 / rate[128], "ns", "derived"),
        "t_link_transfer": (DMA_READ_NS / 1e3, PCIE_RT_NS / 1e3, 2 * PCIE_RT_NS / 1e3, "us", "derived"),
    }
    return out


#: Each row's sources, by the id its citation must name.
SOURCES = {"host_macs_per_s": ("Cortex-A78AE", "r36.5"), "t_host_op_overhead": ("nachin2026executorch", "r36.5"),
           "t_host_mem_per_byte": ("prashanthis2025pagoda", "r36.5"),
           "p_host": ("cao2026vlaulap", "r36.5", "VDD_IN Average Power: 15W", "tsuyuki2026efficient"),
           "p_host_idle": ("cao2026vlaulap", "tsuyuki2026efficient"), "e_host_mem_per_byte": ("prashanthis2025pagoda",),
           "e_link_per_byte": ("bae2021a", "neugebauer2018understanding"),
           "t_link_per_byte": ("neugebauer2018understanding",), "t_link_transfer": ("neugebauer2018understanding",)}


class TestTheDescription:
    def test_it_is_shipped_beside_its_description_file(self):
        assert NAME in host_descriptions()
        description = host_description(NAME)
        assert (HOSTS / description.description_file).is_file()
        assert description.default_mode == "15W"
        assert set(description.modes) == set(MODES)

    def test_each_mode_states_its_published_power_mode(self):
        description = host_description(NAME)
        for mode, (cores, mhz, emc, budget, mode_id) in MODES.items():
            facts = description.modes[mode]
            assert (facts["online_cpu"], facts["cpu_max_mhz"], facts["memory_max_mhz"], facts["power_budget_w"],
                    facts["mode_id"]) == (cores, mhz, emc, budget, mode_id)

    @pytest.mark.parametrize("mode", ["15W", "7W", "common"])
    def test_each_constant_is_its_sources_arithmetic(self, mode):
        description = host_description(NAME)
        declared = description.constants if mode == "common" else description.modes[mode]["constants"]
        expected = _expected()[mode]
        assert set(declared) == set(expected)
        for key, (low, nominal, high, unit, kind) in expected.items():
            row = declared[key]
            assert row["unit"] == unit and row["evidence_kind"] == kind, key
            assert (row["low"], row["nominal"], row["high"]) == pytest.approx((low, nominal, high), rel=1e-5), key
            assert all(source in row["citation"] for source in SOURCES[key]), (key, row["citation"])

    def test_no_constant_comes_from_the_experiment_machines(self):
        text = (HOSTS / f"{NAME}.json").read_text() + (HOSTS / f"{NAME}.md").read_text()
        for word in ("sura", "EPYC", "RTX", "Blackwell", "calibrate_host"):
            assert word not in text, word
        description = host_description(NAME)
        rows = [*description.constants.values()] + [r for m in description.modes.values()
                                                    for r in m["constants"].values()]
        assert rows and all(row["evidence_kind"] in ("published", "datasheet", "derived", "estimated") for row in rows)


class TestTheModeRule:
    def test_the_default_rule_takes_the_documented_default_mode(self):
        description = host_description(NAME)
        assert select_mode(description, "default") == "15W"
        assert select_mode(description, "7W") == "7W"

    def test_an_unknown_mode_refuses_by_name(self):
        with pytest.raises(HostModeError, match="9W.*15W.*7W|9W.*7W.*15W"):
            select_mode(host_description(NAME), "9W")

    def test_an_unknown_description_refuses_by_name(self):
        with pytest.raises(HostDescriptionError, match=NAME):
            host_description("no_such_host")

    def test_a_mode_without_its_power_rows_refuses_by_name(self):
        with pytest.raises(HostModeIncompleteError) as refused:
            host_description_block(NAME, "7W")
        assert "p_host" in str(refused.value) and "p_host_idle" in str(refused.value)
        assert "no published measurement" in str(refused.value)


class TestTheHostBlock:
    def test_the_block_is_every_host_row_of_the_mode_with_its_identity(self):
        block = host_description_block(NAME)
        assert set(block) == set(REQUIRED_HOST_ROWS)
        assert {row[HOST_IDENTITY_FIELD] for row in block.values()} == {f"{NAME} 15W"}
        expected = {**_expected()["15W"], **_expected()["common"]}
        for key, row in block.items():
            assert (row["low"], row["nominal"], row["high"]) == pytest.approx(expected[key][:3], rel=1e-5), key

    def test_every_block_row_is_a_host_vocabulary_constant(self):
        groups = {row[0]: row[1] for row in vocabulary.ROWS}
        assert all(groups[key] == vocabulary.HOST for key in host_description_block(NAME))
        physics = resolve_platform_physics(None, host_description_block(NAME))
        assert set(REQUIRED_HOST_ROWS) <= set(physics.constants)

    def test_the_new_host_rows_and_their_dimensions(self):
        rows = {row[0]: row for row in vocabulary.ROWS}
        for key, dimension in (("p_host_idle", POWER), ("e_host_mem_per_byte", ENERGY), ("t_host_mem_per_byte", TIME),
                               ("e_link_per_byte", ENERGY), ("t_link_per_byte", TIME), ("t_link_transfer", TIME),
                               ("host_macs_per_s", RATE)):
            assert rows[key][1] == vocabulary.HOST and rows[key][2] == dimension, key

    def test_an_identified_host_replaces_the_profiles_host_rows_whole(self):
        shipped = resolve_platform_physics("study_compass", None)
        assert shipped.has("host_compute_rate")
        edge = resolve_platform_physics("study_compass", merge_override_blocks({"host": host_description_block(NAME)}))
        assert not edge.has("host_compute_rate")
        assert edge.band("p_host_idle").nominal == pytest.approx(5.685)
        assert edge.has("e_dma_per_byte")  # the chip's own rows stay


#: A server row as the host calibration seals it (evidence measured, the machine named in its note).
SERVER_ROW = {"low": 13.37, "nominal": 13.48, "high": 13.64, "unit": "us", "evidence_kind": "measured",
              "note": "Per-invocation host ComputeOp dispatch wall measured on the experiment server."}


class TestOneHostPerDeployment:
    def test_a_measured_server_row_inside_the_edge_block_refuses_by_name(self):
        block = host_description_block(NAME)
        block["t_host_op_overhead"] = dict(SERVER_ROW)
        with pytest.raises(HostBlockMixError, match="t_host_op_overhead") as refused:
            resolve_platform_physics("study_compass", merge_override_blocks({"host": block}))
        assert NAME in str(refused.value)

    def test_a_server_block_beside_the_edge_block_refuses_by_name(self):
        server = {"host_compute_rate": {"nominal": 1.0, "unit": "fraction", "evidence_kind": "measured",
                                        "note": "a server's own rate"}}
        merged = merge_override_blocks({"host": host_description_block(NAME), "server_host": server})
        with pytest.raises(HostBlockMixError, match="host_compute_rate"):
            resolve_platform_physics("study_compass", merged)

    def test_an_identified_row_may_not_be_a_measurement(self):
        block = host_description_block(NAME)
        block["p_host"] = {**block["p_host"], "evidence_kind": "measured", "note": "measured here"}
        with pytest.raises(HostBlockMixError, match="p_host.*measured"):
            resolve_platform_physics("study_compass", block)

    def test_two_host_descriptions_refuse_by_name(self):
        block = host_description_block(NAME)
        other = copy.deepcopy(block)
        for row in other.values():
            row[HOST_IDENTITY_FIELD] = "another_host 10W"
        mixed = {**{k: v for k, v in block.items() if k.startswith("t_")},
                 **{k: v for k, v in other.items() if not k.startswith("t_")}}
        with pytest.raises(HostBlockMixError, match="another_host"):
            resolve_platform_physics("study_compass", mixed)

    def test_the_server_host_alone_stays_a_resolvable_sensitivity(self):
        server = {"t_host_op_overhead": dict(SERVER_ROW),
                  "host_macs_per_s": {"nominal": 22.0, "unit": "G/s", "evidence_kind": "measured", "note": "n"},
                  "p_host": {"low": 4.98, "nominal": 37.34, "high": 280.0, "unit": "W", "evidence_kind": "published",
                             "citation": "a band over two hosts"}}
        physics = resolve_platform_physics("study_compass", merge_override_blocks({"host": server}))
        assert physics.has("host_compute_rate") and physics.band("p_host").nominal == pytest.approx(37.34)
        assert not physics.has("p_host_idle")


#: [targeted M3] The rows whose source departs from the owner's rule (constants from published documentation or
#: peer-reviewed measurements) and what each departure states: the power rows' source is an arXiv preprint measured
#: on the Super developer kit (Jetson Linux r36.5, table "Jetson Orin Nano 8GB Super": its 15W mode runs 6 CPUs at
#: 1497.6 MHz, its default mode is 25W); the memory rows read the arXiv version of a peer-reviewed paper whose
#: reviewed version (ICPP 2026) is paywalled, and the memory energy is another module's (Orin AGX).
SUPER_15W_CPU_MHZ, SUPER_DEFAULT = "1497.6", "25W"
DEPARTURES = {
    "p_host": ("cao2026vlaulap", "preprint", "Super", SUPER_15W_CPU_MHZ, SUPER_DEFAULT),
    "p_host_idle": ("cao2026vlaulap", "preprint", "Super", SUPER_15W_CPU_MHZ, SUPER_DEFAULT),
    "e_host_mem_per_byte": ("prashanthis2025pagoda", "arXiv", "k2026pagoda", "Orin AGX"),
    "t_host_mem_per_byte": ("prashanthis2025pagoda", "arXiv", "k2026pagoda", "Orin AGX"),
}


class TestTheDeparturesAreNamed:
    def test_the_description_names_each_departure_and_no_other(self):
        departures = host_description(NAME).departures
        assert set(departures) == set(DEPARTURES)
        for key, words in DEPARTURES.items():
            assert all(word in departures[key] for word in words), (key, departures[key])

    def test_each_departing_row_carries_its_departure_in_its_note(self):
        departures = host_description(NAME).departures
        block = host_description_block(NAME)
        for key in DEPARTURES:
            assert departures[key] in block[key]["note"], key
        assert all("departure" not in row["note"].lower() for key, row in block.items() if key not in DEPARTURES)

    def test_the_memory_energy_is_a_derived_proxy_with_its_arithmetic(self):
        row = host_description(NAME).constants["e_host_mem_per_byte"]
        assert row["evidence_kind"] == "derived" and "Orin AGX" in row["derivation"]

    def test_the_prose_names_the_departures(self):
        prose = (HOSTS / f"{NAME}.md").read_text()
        for word in ("preprint", "Super", SUPER_15W_CPU_MHZ, SUPER_DEFAULT, "k2026pagoda", "departure"):
            assert word in prose, word

    def test_a_departure_naming_no_row_refuses_by_name(self, tmp_path, monkeypatch):
        import mimarsinan.deployment_record.platform_physics.hosts.description as described

        data = json.loads((HOSTS / f"{NAME}.json").read_text())
        data["departures"]["no_such_row"] = "a departure of a row the description does not declare"
        (tmp_path / f"{NAME}.json").write_text(json.dumps(data))
        (tmp_path / data["description_file"]).write_text((HOSTS / data["description_file"]).read_text())
        monkeypatch.setattr(described, "_DIR", tmp_path)
        with pytest.raises(HostDescriptionError, match="no_such_row"):
            host_description(NAME)


def test_the_description_file_is_valid_json_with_one_format():
    data = json.loads((HOSTS / f"{NAME}.json").read_text())
    assert data["format_version"] == 1 and data["name"] == NAME
