"""[U-HOST] The class swept: every site where a host row can enter a physics, and every consumer of a host row.

A host row reaches a pricer only through a physics; a physics is built at the loader's entries (a profile, a run's
overrides, a sealed payload read back) or derived from one by a site that adds rows of its own group. So the sweep
has three parts, one test per site: (S1) every reader of a sealed physics refuses a sealed host row that no
identified description declares; (S2) every construction or derivation site is a loader entry or adds no host row,
and the census of those sites is closed (a new site fails here until it is swept); (S3) every pricing consumer of a
host row prices the identified host's rows identically when its physics is read back from its seal (the identity
survives the seal at every consumer). The host-wall consumers are swept in ``test_host_wall_uhost.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block, host_descriptions
from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics

from unit.deployment_record.g3_fixtures import quantities

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
NAME = host_descriptions()[0]
CHIP = "generic_estimated_22nm"


def _unidentified_payload():
    """A sealed physics carrying a host row that names no host (a server's rate, as a pre-rule toolchain sealed it)."""
    payload = get_platform_physics(CHIP).to_dict()
    payload["constants"]["host_macs_per_s"] = {
        "key": "host_macs_per_s", "low": 22.0, "nominal": 22.0, "high": 22.0, "unit": "G/s",
        "evidence_kind": "measured", "citation": "", "derivation": "", "note": "a server's rate (test)",
        "overridden": True}
    return payload


def _declared_physics_of(payload):
    from mimarsinan.deployment_record.cost.absolute.report import declared_physics_of

    return declared_physics_of(SimpleNamespace(identity=SimpleNamespace(platform={"platform_physics_resolved": payload})))


def _prices_per_event(payload):
    from mimarsinan.deployment_record.cost.compute_time import prices_per_event

    return prices_per_event(payload)


def _host_link_context(payload):
    from mimarsinan.deployment_record.objectives.host_link import host_link_context

    return host_link_context(SimpleNamespace(), {"platform_physics_resolved": payload}, None, None)


def _candidate_fragments(payload):
    from mimarsinan.search.problems.joint.candidate_fragments import candidate_fragments

    return candidate_fragments({"platform_physics_resolved": payload})


def _candidate_physics(payload):
    from mimarsinan.search.problems.joint.problem import JointArchHwProblem

    return JointArchHwProblem.candidate_physics.fget(
        SimpleNamespace(fixed_platform_constraints={"platform_physics_resolved": payload}))


def _operand_price(payload):
    from mimarsinan.mapping.support.operand_price import declared_physics

    return declared_physics({"platform_physics_resolved": payload})


def _write_physics(payload):
    from mimarsinan.deployment_record.platform_physics.resolve import write_physics_profile

    return write_physics_profile(CHIP, {CHIP: payload})


def _compute_physics(payload):
    from mimarsinan.deployment_record.platform_physics.resolve import compute_physics_profile

    return compute_physics_profile(CHIP, {CHIP: payload})


#: (S1) The readers of a sealed physics, each by its own entry (file: the site's PlatformPhysics.from_dict call).
READERS = {
    "deployment_record/cost/absolute/report.py (declared_physics_of)": _declared_physics_of,
    "deployment_record/cost/compute_time.py (prices_per_event)": _prices_per_event,
    "deployment_record/objectives/host_link.py (host_link_context)": _host_link_context,
    "search/problems/joint/candidate_fragments.py (candidate_fragments)": _candidate_fragments,
    "search/problems/joint/problem.py (candidate_physics)": _candidate_physics,
    "mapping/support/operand_price.py (declared_physics)": _operand_price,
    "deployment_record/platform_physics/resolve.py (write_physics_profile)": _write_physics,
    "deployment_record/platform_physics/resolve.py (compute_physics_profile)": _compute_physics,
}


class TestS1EveryReaderOfASealedPhysics:
    @pytest.mark.parametrize("site", sorted(READERS))
    def test_a_sealed_host_row_no_identified_description_declares_refuses_by_name(self, site):
        with pytest.raises(ToolchainRefusal) as refused:
            READERS[site](_unidentified_payload())
        assert "host_macs_per_s" in str(refused.value) and "identified host description" in str(refused.value)

    def test_a_reference_case_can_restate_no_host_row(self):
        """correlation/run.py's reader: a case restates its profile's constants only, and no profile declares a host
        row, so a case cannot bring one."""
        from mimarsinan.deployment_record.correlation.run import _with_overrides

        with pytest.raises(KeyError, match="p_host"):
            _with_overrides(get_platform_physics(CHIP), {"p_host": 1.0}, "a case (test)")


def _calls(predicate):
    out = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and predicate(node):
                out.append(str(path.relative_to(SRC)))
    return sorted(out)


def _is_from_dict(node):
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "from_dict" and getattr(func.value, "id", "") == (
        "PlatformPhysics")


def _is_direct(node):
    return getattr(node.func, "id", "") == "PlatformPhysics"


class TestS2TheConstructionSitesAreClosed:
    def test_the_sealed_physics_readers_are_the_swept_ones(self):
        """Every PlatformPhysics.from_dict call in the toolchain is a swept reader (S1), or the correlation reader,
        or a scaling-ladder or quantity-technology reader whose payload is a sealed physics as well."""
        swept = sorted(site.split(" ")[0] for site in READERS) + [
            "deployment_record/correlation/run.py",
            "deployment_record/cost/absolute/walls/measured.py",  # [composition: U-ENERGY] the measured static leg
            "deployment_record/introspection/scaling_ladder.py",
            "deployment_record/quantities/compute_technology.py", "deployment_record/quantities/write_technology.py"]
        assert _calls(_is_from_dict) == sorted(swept)

    def test_the_direct_constructions_are_the_loaders_the_custom_profile_and_the_probe(self):
        """A physics built directly: the loader's two entries, the empty custom profile, and the capability probe (an
        availability probe, never a target's priced physics)."""
        assert _calls(_is_direct) == sorted([
            "deployment_record/platform_physics/loader.py", "deployment_record/platform_physics/loader.py",
            "deployment_record/platform_physics/probe.py", "deployment_record/platform_physics/resolve.py"])

    @pytest.mark.parametrize("site,keys", [
        ("resolve.with_write_technology (a core type's write rows)", "WRITE_TECHNOLOGY_KEYS"),
        ("resolve.COMPUTE_TECHNOLOGY_KEYS (a core type's compute rows)", "COMPUTE_TECHNOLOGY_KEYS"),
        ("profile.PlatformPhysics.at_carry_buffer (the carry buffer's rows)", "CARRY_BUFFER_KEYS"),
        ("converter (the converter's rows at a resolution)", "CONVERTER_KEYS"),
        ("resolve.resolve_platform_physics (a lane block's rows)", "LANE_ROWS"),
    ])
    def test_every_derivation_site_adds_no_host_row(self, site, keys):
        from mimarsinan.deployment_record.platform_physics import converter, profile, resolve
        from mimarsinan.deployment_record.platform_physics import lane_blocks

        rows = next(getattr(m, keys) for m in (resolve, profile, converter, lane_blocks) if hasattr(m, keys))
        assert rows and not [key for key in rows if spec_for(key).group == HOST], site


#: The edge physics every host-row consumer reads: the chip and the identified host's block.
def _edge():
    return resolve_platform_physics(CHIP, merge_override_blocks({"host": host_description_block(NAME)}))


#: A view that exercises every host term: host MACs and dispatch, memory-bound host bytes, the boundary's crossing
#: bytes and transfers, a program load (no pass carry: the chip states no backing store, which a carry needs).
VIEW = dict(synaptic_events=1e6, host_macs=1e6, compute_op_count=3, cores_physical=8, latency_steps=32, timesteps=4,
            neurons_used=100, segment_cores=1, host_memory_bound_bytes=4096, segment_in_bytes=1024,
            segment_out_bytes=64, host_link_transfers=4, reprogrammed_bytes=2048, reprogrammed_cores=1,
            connectivity_entries=8)

#: (S3) The pricing consumers of a host row, each by the terms it writes (file: the reading site).
CONSUMERS = {
    "cost/absolute/context.py (host_time: host_macs_per_s, t_host_op_overhead)": ("latency_host_s",),
    "cost/host_terms.py (host_memory_time: t_host_mem_per_byte)": ("latency_host_s",),
    "cost/host_terms.py (host_link_time: t_link_per_byte, t_host_mem_per_byte, t_link_transfer)": (
        "latency_host_link_s",),
    "cost/host_terms.py (the energy planes: p_host, p_host_idle, e_link_per_byte, e_host_mem_per_byte)": (
        "energy_host_work_mj", "energy_host_idle_mj", "energy_host_link_mj", "energy_host_mem_mj"),
    "cost/absolute/energy.py (p_host over the host time)": ("energy_host_mj", "energy_per_inference_mj"),
    "cost/absolute/metrics.py (the host time in the latency)": ("e2e_latency_s", "throughput_inferences_s"),
    "cost/absolute/formulas.py (the link and host-memory energy components)": ("energy_per_inference_mj",),
}


class TestS3EveryConsumerKeepsTheIdentifiedHostThroughItsSeal:
    @pytest.mark.parametrize("site", sorted(CONSUMERS))
    def test_the_consumer_prices_alike_from_the_physics_and_from_its_seal(self, site):
        physics = _edge()
        sealed = PlatformPhysics.from_dict(physics.to_dict())
        fresh = {t.name: t for t in price_absolute(quantities(**VIEW), physics).terms}
        again = {t.name: t for t in price_absolute(quantities(**VIEW), sealed).terms}
        for name in CONSUMERS[site]:
            assert name in fresh, (site, name, sorted(fresh))
            assert (fresh[name].band, fresh[name].source) == (again[name].band, again[name].source), (site, name)
        assert {k: v.host for k, v in sealed.constants.items() if spec_for(k).group == HOST} == {
            k: v.host for k, v in physics.constants.items() if spec_for(k).group == HOST}
