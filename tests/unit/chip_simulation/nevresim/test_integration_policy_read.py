"""[U-H seam] nevresim executes a non-default integration policy only where the tree it compiles against carries it.

The event-serial fold (``EventSerialIntegrate``) and the sync-fire register (``WholeVectorSaturatingSigned``) are the
ODIN program's change to nevresim's sources. nevresim is a nested repository whose commit of that change waits for
its owner, so a checkout can hold a tree without it; there a program naming the policy failed only after a full
build, with no name ("Recording build compilation failed."). The driver now asks one probe of the tree before it
emits such a program, with the W33 probe's two checks (declared sites, then the tree's own self-test constant-evaluated
by nevresim's compiler), and refuses the point by name. The default policy is every tree's template default: it is
never probed, so no default program changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mimarsinan.chip_simulation.nevresim import integration_policy_read as P
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.nevresim_policy_types import (
    EVENT_SERIAL_INTEGRATE,
    WHOLE_VECTOR_INTEGRATE,
    WHOLE_VECTOR_SATURATING_SIGNED,
    NevresimPolicyTypeError,
    nevresim_integration_policy,
)
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.common.refusal import ToolchainRefusal

from unit.chip_simulation import nevresim_trees as NT

_PER_EVENT_8 = SomaLaw.resolve({"spiking_family": "lif", "spiking_variant": "streamed", "firing_mode": "Novena",
                                "firing_granularity": "per_event", "membrane_bits": 8})


def _tree_with_every_site(tmp_path: Path, name: str) -> Path:
    """This checkout's include tree with every declared site's text present (each site file gains its marker)."""
    root = NT._copy(tmp_path, name)
    for sites, _ in P.POLICY_SITES.values():
        for path, marker, _replaced in sites:
            file = root / "include" / path
            file.parent.mkdir(parents=True, exist_ok=True)
            text = file.read_text() if file.is_file() else ""
            for replaced in _replaced:
                text = text.replace(replaced, "")
            file.write_text(text + f"\n// {marker}\n")
    (root / "include" / "probe_test_marker.txt").write_text(str(root))   # a tree no other test probed
    return root


class TestTheDefaultPolicyIsNeverProbed:
    def test_the_default_point_resolves_to_the_template_default(self):
        assert nevresim_integration_policy(DEFAULT_SOMA_LAW) == WHOLE_VECTOR_INTEGRATE

    def test_the_default_policy_is_established_on_any_tree(self, tmp_path, monkeypatch):
        monkeypatch.setattr(P, "self_test_problem", lambda *a: pytest.fail("the default policy was probed"))
        read = P.integration_policy_read(WHOLE_VECTOR_INTEGRATE, tmp_path / "absent")
        assert read.established and "template default" in read.evidence
        P.require_integration_policy(WHOLE_VECTOR_INTEGRATE, tmp_path / "absent")


class TestTheDeclaration:
    def test_every_non_default_policy_the_codegen_emits_is_declared(self):
        assert set(P.POLICY_SITES) == {EVENT_SERIAL_INTEGRATE, WHOLE_VECTOR_SATURATING_SIGNED}

    def test_each_policy_declares_its_header_and_its_self_test(self):
        for policy, (sites, (header, function)) in P.POLICY_SITES.items():
            paths = {path for path, _, _ in sites}
            assert any(policy in marker for _, marker, _ in sites), policy
            assert header in paths and function.startswith("nevresim::tests::"), policy

    def test_a_width_argument_reads_the_policy_template(self, tmp_path):
        assert not P.integration_policy_read(f"{EVENT_SERIAL_INTEGRATE}<8>", tmp_path / "absent").established
        assert "no nevresim include tree" in P.integration_policy_read(
            f"{EVENT_SERIAL_INTEGRATE}<8>", tmp_path / "absent").evidence

    def test_an_undeclared_policy_is_not_established(self, tmp_path):
        read = P.integration_policy_read("FuturePolicy<4>", tmp_path)
        assert not read.established and "FuturePolicy" in read.evidence


class TestATreeWithoutTheChangeIsRefusedByName:
    def test_the_release_base_tree_lacks_the_event_serial_sites(self, tmp_path):
        root = NT._copy(tmp_path, "nevresim_without")
        (root / "include" / "simulator" / "compute_policy" / "integration_policy").mkdir(parents=True, exist_ok=True)
        header = root / "include" / P.POLICY_SITES[EVENT_SERIAL_INTEGRATE][1][0]
        header.unlink(missing_ok=True)
        read = P.integration_policy_read(f"{EVENT_SERIAL_INTEGRATE}<8>", root)
        assert not read.established and "event_serial_integrate.hpp is missing" in read.evidence

    def test_the_refusal_names_the_policy_the_point_and_the_tree(self, tmp_path):
        root = NT._copy(tmp_path, "nevresim_without_any")
        for sites, _ in P.POLICY_SITES.values():
            for path, _, _ in sites:
                (root / "include" / path).unlink(missing_ok=True)
        with pytest.raises(P.IntegrationPolicyRefusal, match=EVENT_SERIAL_INTEGRATE) as refusal:
            P.require_integration_policy(f"{EVENT_SERIAL_INTEGRATE}<8>", root)
        assert str(root / "include") in str(refusal.value)
        assert isinstance(refusal.value, ToolchainRefusal) and isinstance(refusal.value, NevresimPolicyTypeError)

    def test_a_tree_whose_self_test_does_not_hold_is_refused(self, tmp_path, monkeypatch):
        root = _tree_with_every_site(tmp_path, "nevresim_sites_only")
        monkeypatch.setattr(P, "self_test_problem", lambda *a: "the self-test X does not hold (stub)")
        read = P.integration_policy_read(f"{EVENT_SERIAL_INTEGRATE}<8>", root)
        assert not read.established and "does not hold" in read.evidence

    def test_a_tree_with_every_site_and_a_holding_self_test_is_established(self, tmp_path, monkeypatch):
        root = _tree_with_every_site(tmp_path, "nevresim_sites_and_test")
        monkeypatch.setattr(P, "self_test_problem", lambda *a: None)
        for policy in P.POLICY_SITES:
            read = P.integration_policy_read(f"{policy}<16>", root)
            assert read.established and "self-test" in read.evidence, policy

    def test_the_probe_runs_once_per_tree_fingerprint(self, tmp_path, monkeypatch):
        root = _tree_with_every_site(tmp_path, "nevresim_fingerprint")
        calls = []
        monkeypatch.setattr(P, "self_test_problem", lambda *a: calls.append(a) or None)
        P.integration_policy_read(f"{EVENT_SERIAL_INTEGRATE}<8>", root)
        P.integration_policy_read(f"{EVENT_SERIAL_INTEGRATE}<16>", root)
        assert len(calls) == 1
        header = root / "include" / P.POLICY_SITES[EVENT_SERIAL_INTEGRATE][1][0]
        header.write_text(header.read_text() + "\n// edited\n")
        P.integration_policy_read(f"{EVENT_SERIAL_INTEGRATE}<8>", root)
        assert len(calls) == 2


class TestThisCheckout:
    def test_the_verdict_on_this_checkout_follows_its_sources(self):
        """Either tree may be checked out: the read is established exactly where every declared site holds."""
        root = NT.checkout_tree()
        sites, _ = P.POLICY_SITES[EVENT_SERIAL_INTEGRATE]
        present = not P.site_problems(root / "include", sites)
        read = P.integration_policy_read(f"{EVENT_SERIAL_INTEGRATE}<8>", root)
        assert read.established <= present
        if not present:
            assert "lack" in read.evidence


class TestTheDriverRefusesBeforeItEmits:
    def test_emit_main_refuses_a_policy_its_tree_lacks_by_name(self, tmp_path, monkeypatch):
        root = NT._copy(tmp_path, "nevresim_for_driver")
        for sites, _ in P.POLICY_SITES.values():
            for path, _, _ in sites:
                (root / "include" / path).unlink(missing_ok=True)
        monkeypatch.setattr(NevresimDriver, "nevresim_path", str(root))
        driver = NevresimDriver.__new__(NevresimDriver)
        driver.soma_law = _PER_EVENT_8
        driver.spiking_mode, driver.firing_mode, driver.spike_generation_mode = "lif", "Novena", "Uniform"
        driver.thresholding_mode, driver.weight_type, driver.threshold_type = "<=", int, int
        driver.generated_files_path = str(tmp_path / "generated")
        with pytest.raises(P.IntegrationPolicyRefusal, match="EventSerialIntegrate<8>"):
            driver.emit_main(1, 4, 1, verbose=False)
        assert not (tmp_path / "generated").exists()


def test_the_probe_reuses_the_w33_checks():
    """One implementation of the two checks serves both source-tree probes."""
    from mimarsinan.chip_simulation import stage_input_clock as S

    assert P.site_problems is S.site_problems
    assert P.self_test_problem is S.self_test_problem
