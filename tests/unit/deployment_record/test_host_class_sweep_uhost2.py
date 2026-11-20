"""[U-HOST2] The class swept again on the value: every site where a host row enters a physics or is read, one test each.

The defect class: a host price taken from a label (a host's name stamped on another machine's value) rather than from
the identified host's published rows. A value enters a physics only at the loader's entries, so each entry refuses a
relabelled value by name (E: a run's overrides, a profile, a sealed physics read back at every reader, a pipeline
config, a calibration artifact); and each consumer of a host row reads exactly the admitted value (C: its output moves
with a declared derivation of the rows it reads, and its seal prices alike). The host-wall consumers are swept in
``test_host_wall_uhost.py``, ``test_host_wall_clauses_uhost2.py`` and ``test_host_ratchet_uhost2.py``. No
description, chip or mode is named here: the first read description, its default mode and the registry's profiles.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.compute_time import producer_walls, store_walls
from mimarsinan.deployment_record.platform_physics import available_profiles, get_platform_physics
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.admission import host_derivation_block
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    HOST_IDENTITY_FIELD, host_description_block, host_descriptions,
)
from mimarsinan.deployment_record.platform_physics.hosts.link_ends import link_end
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.schema.provenance import Band

from unit.deployment_record.g3_fixtures import quantities

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
NAME = host_descriptions()[0]
CORNERS = ("low", "nominal", "high")


def _block():
    return host_description_block(NAME)


def _chip(store=False):
    """The first registry profile that takes the identified block (``store``: one that declares a backing store)."""
    for name in available_profiles():
        if store and get_platform_physics(name).backing_store is None:
            continue
        try:
            resolve_platform_physics(name, merge_override_blocks({"host": _block()}))
        except ToolchainRefusal:
            continue
        return name
    raise AssertionError("no registry profile takes the identified host block")


def _physics(rows=None, store=False):
    blocks = {"host": rows or _block()}
    if store:
        blocks["store"] = ST.store_block(ST.HOST, "a host-owned variant (test)")
    return resolve_platform_physics(_chip(store), merge_override_blocks(blocks))


def _relabelled(key, factor=2.0):
    """The identified block with ``key``'s band moved by ``factor`` under the host's name (no derivation declared)."""
    block = copy.deepcopy(_block())
    block[key].update({end: block[key][end] * factor for end in CORNERS})
    return block


def _sealed_relabelled(key):
    payload = _physics().to_dict()
    payload["constants"][key].update({end: payload["constants"][key][end] * 2.0 for end in CORNERS})
    return payload


def _refused(call):
    with pytest.raises(ToolchainRefusal) as refused:
        call()
    return str(refused.value)


class TestEEveryEntryRefusesARelabelledValue:
    @pytest.mark.parametrize("key", sorted(_block()))
    def test_a_runs_override_rows(self, key):
        assert key in _refused(lambda: _physics(_relabelled(key)))

    def test_a_profiles_own_rows(self):
        """A profile declares no host row (it cannot name a host: a declaration's fields are closed)."""
        from mimarsinan.deployment_record.platform_physics.loader import profile_from_dict

        data = {"format_version": 1, "name": "t", "display_name": "T", "description_file": "t.md",
                "validity": {"measurement_kind": "mixed"},
                "constants": {key: {k: v for k, v in row.items() if k in ("low", "nominal", "high", "unit",
                                                                          "evidence_kind", "citation", "derivation",
                                                                          "note")}
                              for key, row in _block().items()}}
        why = _refused(lambda: profile_from_dict(data))
        assert all(key in why for key in _block())

    @pytest.mark.parametrize("key", sorted(_block()))
    def test_a_sealed_physics(self, key):
        assert key in _refused(lambda: PlatformPhysics.from_dict(_sealed_relabelled(key)))

    def test_every_reader_of_a_sealed_physics_reads_it_through_the_judged_entry(self):
        """Every reader of a sealed physics in src calls PlatformPhysics.from_dict (the judged entry), and no module
        builds a physics from a payload's rows any other way (the construction census of test_host_class_sweep_uhost
        names the direct constructions)."""
        readers = []
        for path in sorted(SRC.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Attribute) and node.attr == "from_dict" and getattr(
                        node.value, "id", "") == "PhysicsConstantValue":
                    readers.append(path.relative_to(SRC).as_posix())
        assert sorted(set(readers)) == ["deployment_record/platform_physics/profile.py"]

    def test_a_platforms_declared_overrides(self):
        """A platform that carries its profile and overrides unresolved (mapping/support/operand_price.py
        declared_physics; the pipeline's resolver, the search problem and the fidelity emission resolve the same
        pair through resolve_platform_physics) reads them through the same loader."""
        from mimarsinan.mapping.support.operand_price import declared_physics

        key = sorted(_block())[0]
        platform = {"platform_physics_profile": _chip(), "platform_physics_overrides": merge_override_blocks(
            {"host": _relabelled(key)})}
        assert key in _refused(lambda: declared_physics(platform))

    def test_a_calibration_artifact(self, monkeypatch, tmp_path):
        """A development machine's calibration, its rows stamped with the identified host's name, refuses."""
        import mimarsinan.deployment_record.platform_physics.host_calibration as calibration

        monkeypatch.setattr(calibration, "measure_host_mac_rates", lambda **kw: {(1, 784, 500): 2.0e9})
        monkeypatch.setattr(calibration, "measure_host_op_overhead_s", lambda: (1e-5, 2e-5, 3e-5))
        monkeypatch.setattr(calibration, "measure_p_host_w", lambda: 20.0)
        rows = calibration.run_calibration()["diagnostic"]["rows"]
        identity = _block()[sorted(_block())[0]][HOST_IDENTITY_FIELD]
        stamped = {key: {**row, HOST_IDENTITY_FIELD: identity, "evidence_kind": "estimated",
                         "note": row.get("note") or "a test row"} for key, row in rows.items()}
        why = _refused(lambda: _physics(stamped))
        assert all(key in why for key in rows)


VIEW = dict(synaptic_events=1e6, host_macs=1e6, compute_op_count=3, cores_physical=8, latency_steps=32, timesteps=4,
            neurons_used=100, segment_cores=1, host_memory_bound_bytes=4096, segment_in_bytes=1024,
            segment_out_bytes=64, host_link_transfers=4, reprogrammed_bytes=2048, reprogrammed_cores=1,
            connectivity_entries=8)


def _terms(names):
    def read(physics):
        terms = {t.name: t for t in price_absolute(quantities(**VIEW), physics).terms}
        return tuple((terms[n].band.low, terms[n].band.nominal, terms[n].band.high) for n in names)
    return read


def _census():
    passes = (SimpleNamespace(payload_bytes=4096, written_cells_producer=512),
              SimpleNamespace(payload_bytes=0, written_cells_producer=0))
    return SimpleNamespace(passes=passes)


def _store_walls(physics):
    walls = store_walls(None, _census(), physics)
    return tuple((w.low, w.nominal, w.high) for w in walls)


def _producer_walls(physics):
    policy = SimpleNamespace(producer_write_source="host", payload_bits=8)
    step = Band(1e-9, 1e-9, 1e-9, "one write step (test)")
    band = producer_walls(step, [1, 0], _census(), physics, policy)
    return (band.low, band.nominal, band.high)


def _slower_port(physics):
    band = ST.slower_port(physics, Band(1e-12, 1e-12, 1e-12, "a fast port (test)"))
    return (band.low, band.nominal, band.high)


def _store_byte_energy(physics):
    band = ST.store_byte_energy(physics)
    return (band.low, band.nominal, band.high)


def _link_end(physics):
    return tuple(link_end(physics, key) for key in sorted(physics.link_ends))


#: (C) Each consumer of a host row: (the rows it reads, whether it needs a host-owned store, its output).
CONSUMERS = {
    "cost/absolute/context.py host_time": (("host_macs_per_s", "t_host_op_overhead"), False,
                                           _terms(("latency_host_s",))),
    "cost/host_terms.py host_memory_time": (("t_host_mem_per_byte",), False, _terms(("latency_host_s",))),
    "cost/host_terms.py host_link_time": (("t_link_per_byte", "t_link_transfer"), False,
                                          _terms(("latency_host_link_s",))),
    "cost/host_terms.py host_side_energy": (("p_host", "p_host_idle", "e_link_per_byte", "e_host_mem_per_byte"),
                                            False, _terms(("energy_host_work_mj", "energy_host_idle_mj",
                                                           "energy_host_link_mj", "energy_host_mem_mj"))),
    "cost/host_terms.py price_system_energy": (("p_host_idle",), False, _terms(("energy_host_system_mj",))),
    "cost/absolute/energy.py host energy": (("p_host",), False, _terms(("energy_host_mj",))),
    "cost/absolute/metrics.py e2e latency": (("host_macs_per_s",), False, _terms(("e2e_latency_s",))),
    "cost/absolute/formulas.py energy components": (("e_link_per_byte",), False, _terms(("energy_per_inference_mj",))),
    "cost/compute_time.py store_walls": (("t_link_transfer", "t_link_per_byte"), True, _store_walls),
    "cost/compute_time.py producer_walls": (("t_link_transfer", "t_link_per_byte"), True, _producer_walls),
    "platform_physics/hosts/store.py slower_port": (("t_link_per_byte",), True, _slower_port),
    "platform_physics/hosts/store.py store_byte_energy": (("e_link_per_byte", "e_host_mem_per_byte"), True,
                                                          _store_byte_energy),
}


class TestCEveryConsumerReadsTheAdmittedValue:
    @pytest.mark.parametrize("site", sorted(CONSUMERS))
    def test_the_output_moves_with_a_declared_derivation_and_prices_alike_from_its_seal(self, site):
        rows, store, output = CONSUMERS[site]
        physics = _physics(store=store)
        # a floor row alone moves down (a system-level floor stays under the power it lies beneath); others move up
        factor = 0.5 if all(key in physics.system_energy_rows for key in rows) else 2.0
        derived = _physics({**_block(), **host_derivation_block(NAME, "default", {key: factor for key in rows})},
                           store=store)
        assert output(derived) != output(physics), site
        assert output(PlatformPhysics.from_dict(derived.to_dict())) == output(derived), site

    def test_link_end_reads_the_ends_the_description_states(self):
        """hosts/link_ends.py link_end: the energy term's boundary text follows the admitted row's declared ends."""
        physics = _physics()
        assert physics.link_ends and all("boundary" in text for text in _link_end(physics))
        assert _link_end(PlatformPhysics.from_dict(physics.to_dict())) == _link_end(physics)
