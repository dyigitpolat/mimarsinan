"""Every declared capability reaches the layout answer, at every forwarding site.

``permission_kwargs()`` forwarded THREE of the declared capability fields and
dropped ``schedule_policy`` — which the 12 literature IMC presets set — so search
and the wizard evaluated a different scheduler than deployment runs. These pin
the two surfaces: ``capability_bits()`` is the complete declaration (derived from
the dataclass, so a new field can never be silently unforwarded), and
``layout_kwargs()`` is the subset the layout helpers take — checked against their
real signatures and against the four call sites that splat it.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

from mimarsinan.mapping.layout.layout_plan import build_layout_plan
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.verification.layout_verification_scheduling import (
    compute_mapping_stats,
)
from mimarsinan.mapping.verification.verifier import verify_hardware_config

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"

# The four sites that turn a declared platform into a layout answer. The agent
# introspection surface is one of them THROUGH the record's candidate view — the
# compilagent backend no longer computes a layout answer of its own (pinned by
# tests/unit/search/optimizers/compilagent/test_introspection_import_direction).
FORWARDING_SITES = (
    "gui/server/routes_layout.py",
    "mapping/verification/wizard_layout_verify.py",
    "search/problems/joint/layout_hook.py",
    "deployment_record/introspection/views.py",
)


class TestCapabilityBits:
    def test_every_declared_field_is_served(self):
        caps = ChipCapabilities(
            max_axons=256, max_neurons=128, hardware_bias=True,
            allow_coalescing=True, allow_neuron_splitting=True,
            allow_scheduling=True, allow_per_layer_s=True,
            schedule_policy="bank_clustered", max_schedule_passes=4,
        )
        bits = caps.capability_bits()
        assert set(bits) == {f.name for f in fields(ChipCapabilities)}
        assert bits == {
            "max_axons": 256, "max_neurons": 128, "hardware_bias": True,
            "allow_coalescing": True, "allow_neuron_splitting": True,
            "allow_scheduling": True, "allow_per_layer_s": True,
            "schedule_policy": "bank_clustered", "max_schedule_passes": 4,
        }

    def test_schedule_policy_and_budget_are_read_from_the_platform(self):
        caps = ChipCapabilities.from_platform_constraints(
            {"schedule_policy": "bank_clustered", "max_schedule_passes": 3}
        )
        assert caps.schedule_policy == "bank_clustered"
        assert caps.max_schedule_passes == 3
        # Absent ⇒ the resolver's documented defaults.
        default = ChipCapabilities.from_platform_constraints({})
        assert (default.schedule_policy, default.max_schedule_passes) == ("pool", 8)


class TestTheGeometryIsPartOfTheDeclaration:
    """A served bit must be the platform's answer, never an unread default.

    ``capability_bits()`` is a TRUST channel (the agent's ``inspect_capabilities``
    tool and the record's capability payload read it), so every field it derives
    from the dataclass has to be populated from the declaration — serving
    ``max_axons: None`` for a chip that declares 256-axon cores is worse than
    omitting the key.
    """

    PLATFORM = {
        "cores": [
            {"max_axons": 256, "max_neurons": 128, "count": 4, "has_bias": True},
            {"max_axons": 128, "max_neurons": 64, "count": 8, "has_bias": True},
        ],
        "allow_scheduling": True,
        "schedule_policy": "bank_clustered",
    }

    def test_no_served_bit_is_an_unread_default_on_a_declared_platform(self):
        bits = ChipCapabilities.from_platform_constraints(self.PLATFORM).capability_bits()
        unread = [
            name for name, value in bits.items() if value is None
        ]
        assert not unread, unread
        assert bits["max_axons"] == 256
        assert bits["max_neurons"] == 128
        assert bits["hardware_bias"] is True

    def test_it_reads_the_wizard_bodys_core_types_key_too(self):
        """The GUI verify route posts ``core_types``; a platform posts ``cores``."""
        body = ChipCapabilities.from_platform_constraints(
            {"core_types": [{"max_axons": 32, "max_neurons": 16, "count": 2}]}
        )
        assert (body.max_axons, body.max_neurons) == (32, 16)

    def test_a_bias_less_grid_reports_the_axon_the_bias_consumes(self):
        """``has_bias=False`` cores spend one axon on the always-on row."""
        caps = ChipCapabilities.from_platform_constraints(
            {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 1,
                        "has_bias": False}]}
        )
        assert caps.hardware_bias is False
        assert caps.max_axons == 255
        assert caps.max_neurons == 256

    def test_the_resolved_geometry_is_the_layout_mappers_own(self):
        """Same SSOT, so a strategy resolved here tiles as the mapper tiles."""
        from mimarsinan.mapping.platform.platform_constraints import (
            resolve_platform_mapping_params,
        )
        caps = ChipCapabilities.from_platform_constraints(self.PLATFORM)
        pmap = resolve_platform_mapping_params(
            self.PLATFORM["cores"], allow_coalescing=False,
        )
        assert (caps.max_axons, caps.max_neurons, caps.hardware_bias) == (
            pmap.effective_max_axons, pmap.effective_max_neurons, pmap.hardware_bias,
        )

    def test_an_undeclared_grid_stays_unset_rather_than_invented(self):
        caps = ChipCapabilities.from_platform_constraints({"allow_scheduling": True})
        assert (caps.max_axons, caps.max_neurons) == (None, None)
        assert caps.hardware_bias is False


class TestLayoutKwargs:
    def test_carries_the_scheduling_declaration_the_builder_consumes(self):
        caps = ChipCapabilities(
            allow_scheduling=True, schedule_policy="bank_clustered",
            max_schedule_passes=2,
        )
        assert caps.layout_kwargs() == {
            "allow_neuron_splitting": False,
            "allow_coalescing": False,
            "allow_scheduling": True,
            "schedule_policy": "bank_clustered",
            "max_schedule_passes": 2,
        }

    def test_every_key_is_a_real_parameter_of_every_layout_helper(self):
        keys = set(ChipCapabilities().layout_kwargs())
        for helper in (compute_mapping_stats, verify_hardware_config, build_layout_plan):
            params = set(inspect.signature(helper).parameters)
            assert keys <= params, (helper.__name__, sorted(keys - params))

    def test_it_is_a_subset_of_the_full_declaration(self):
        caps = ChipCapabilities(schedule_policy="bank_clustered")
        bits = caps.capability_bits()
        assert all(bits[k] == v for k, v in caps.layout_kwargs().items())

    def test_the_strategy_serves_the_same_answer(self):
        caps = ChipCapabilities(allow_scheduling=True, max_schedule_passes=5)
        strategy = MappingStrategy.resolve(caps)
        assert strategy.layout_kwargs() == caps.layout_kwargs()
        assert strategy.max_schedule_passes == 5


def _splatted_calls(path: Path) -> set[str]:
    """Names of methods splatted as ``**caps.<name>()`` anywhere in the file."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg is not None:
                continue
            value = keyword.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                found.add(value.func.attr)
    return found


class TestForwardingSites:
    def test_all_four_sites_forward_the_full_layout_declaration(self):
        offenders = []
        for site in FORWARDING_SITES:
            splatted = _splatted_calls(SRC / site)
            if "layout_kwargs" not in splatted:
                offenders.append(f"{site}: splats {sorted(splatted)}")
        assert not offenders, (
            "these sites still forward a partial capability set: " + str(offenders)
        )

    def test_no_site_still_splats_the_three_bit_subset(self):
        offenders = [
            site
            for site in FORWARDING_SITES
            if "permission_kwargs" in _splatted_calls(SRC / site)
        ]
        assert not offenders, offenders
