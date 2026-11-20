"""[W12] Lemma A (admission): every program the toolchain admits meets the program and executor conditions, or the
toolchain refuses it by name. The integrated screen's residue (CL9 of pass 2, 122 of 862 hypothesis-on seeds) refused
without a name, every one by ``WideFanInUnsupportedError`` (a ``ValueError``); the hypothesis table's other guard sites
refused by ``AssertionError`` (``CoreResidencyViolation``, H1-theta's co-resident singletons) and ``RuntimeError``
(``ChipLatency``'s cyclic core graph). Each is a refusal by name now (a ``ToolchainRefusal``), keeping its builtin
base so every ``except ValueError``, ``AssertionError`` or ``RuntimeError`` still catches it; the packer's wrapper keeps
the name of any refusal it wraps; and a screen seed that refused without a name refuses by name."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mimarsinan.common.refusal import ToolchainRefusal, is_toolchain_refusal
from mimarsinan.mapping.platform.mapping_structure import WideFanInUnsupportedError, compute_fc_tiling_mode


def test_a_wide_fan_in_without_coalescing_refuses_by_name():
    with pytest.raises(WideFanInUnsupportedError) as caught:
        compute_fc_tiling_mode(in_features=25, out_features=4, max_axons=8, max_neurons=16, has_bias=False,
                               hardware_bias=False, allow_coalescing=False)
    assert is_toolchain_refusal(caught.value) and isinstance(caught.value, ValueError)


def test_co_resident_singletons_that_disagree_refuse_by_name():
    from mimarsinan.mapping.platform.core_residency import CoreResidencyViolation, adopt_or_check

    def soft(i, theta):
        return SimpleNamespace(id=i, threshold=theta, get_output_count=lambda: 1)

    hard = SimpleNamespace(threshold=None)
    adopt_or_check(hard, soft(0, 1.0), constrained=("threshold",))
    with pytest.raises(CoreResidencyViolation) as caught:
        adopt_or_check(hard, soft(1, 2.0), constrained=("threshold",))
    assert is_toolchain_refusal(caught.value) and isinstance(caught.value, AssertionError)


def _source(core):
    return SimpleNamespace(core_=core, neuron_=0, is_off_=False, is_input_=False, is_always_on_=False)


def test_a_cyclic_core_graph_refuses_by_name():
    from mimarsinan.mapping.latency.chip import ChipLatency

    cores = [SimpleNamespace(axon_sources=[_source(1)], get_core_matrix=lambda: np.ones((1, 1)), latency=None),
             SimpleNamespace(axon_sources=[_source(0)], get_core_matrix=lambda: np.ones((1, 1)), latency=None)]
    with pytest.raises(RuntimeError) as caught:
        ChipLatency(SimpleNamespace(cores=cores, output_sources=[_source(0)])).calculate()
    assert is_toolchain_refusal(caught.value) and "cyclic core dependency" in str(caught.value)


def test_the_packer_wrapper_keeps_the_name_of_any_refusal_it_wraps(monkeypatch):
    """``Hard-core packing failed``: a refusal by name raised inside the packer reaches the caller under its own class
    (CoreCapacityRefusal was the only one kept); an anonymous failure stays anonymous (a crash, never a refusal)."""
    from mimarsinan.mapping.packing import hybrid_segment

    class _NamedInside(ToolchainRefusal, RuntimeError):
        pass

    for inner, named in ((_NamedInside("declared pool cannot hold it"), True), (RuntimeError("a bug"), False)):
        with pytest.raises(RuntimeError) as caught:
            hybrid_segment.reraise_packing_failure(inner, "diagnostic text")
        assert is_toolchain_refusal(caught.value) is named
        assert type(caught.value) is (type(inner) if named else RuntimeError)
        assert "diagnostic text" in str(caught.value) and caught.value.__cause__ is inner


def test_a_wrapped_refusal_whose_constructor_takes_more_keeps_its_own_name():
    """A refusal class whose constructor needs more than a message is re-raised as itself, never renamed."""
    from mimarsinan.mapping.packing import hybrid_segment

    class _TwoArguments(ToolchainRefusal, RuntimeError):
        def __init__(self, what, why):
            super().__init__(f"{what}: {why}")

    inner = _TwoArguments("pool", "too small")
    with pytest.raises(_TwoArguments) as caught:
        hybrid_segment.reraise_packing_failure(inner, "diagnostic text")
    assert caught.value is inner


@pytest.mark.parametrize("arm, seed", [("on", 2), ("on", 13), ("on_scheduled", 3), ("on_values_classes", 23)])
def test_screen_seeds_that_refused_without_a_name_refuse_by_name(arm, seed, monkeypatch):
    """Seeds of the CL9 screen (``11_wave2/integration2/cl9/gates/screen/<arm>.jsonl``) whose outcome was
    ``refused_unnamed`` by ``WideFanInUnsupportedError``. [U-TGAUGE5] A draw whose run reads the window is mapped
    through the mapping's window pass, so its identity program runs and the deployment's own refusal is read (seed 13's
    relay-free identity program is refused by the window rule first, ``test_the_window_rule_refuses_first``)."""
    from mimarsinan.certification.randomized.arms import ARMS
    from mimarsinan.certification.randomized.runner import REFUSED, instance_for, run_instance

    from unit.certification.x3a_window import map_through_the_window, reads_the_window

    if reads_the_window(instance_for(seed, ARMS[arm]).semantics):
        map_through_the_window(monkeypatch)

    verdict = run_instance(seed, arm)
    assert verdict.outcome == REFUSED and verdict.detail.startswith("WideFanInUnsupportedError: ")


def test_the_window_rule_refuses_first():
    """[U-TGAUGE5] This file's refusal test: seed 13's relay-free identity program (a per-cycle cascade holding a live
    edge at a latency gap other than one) is refused by name before the deployment's own refusal is reached."""
    from mimarsinan.certification.randomized.runner import REFERENCE_ERROR, run_instance

    verdict = run_instance(13, "on")
    assert verdict.outcome == REFERENCE_ERROR and verdict.detail.startswith("UnbalancedFanInError: "), verdict.detail
