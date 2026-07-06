"""verify_hardware_config memo: identical layout questions answered once."""

from __future__ import annotations

import copy

from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.verification.verifier import mapping_verifier_hw as vhw


def _softcores(n=6, base=32):
    return [
        LayoutSoftCoreSpec(
            input_count=base + i, output_count=base - i,
            threshold_group_id=i % 2, latency_tag=i % 3, name=f"sc{i}",
        )
        for i in range(n)
    ]


def _core_types():
    return [{"max_axons": 128, "max_neurons": 128, "count": 16}]


class TestVerifyHardwareConfigMemo:
    def setup_method(self):
        vhw.clear_verify_hardware_config_memo()

    def test_identical_inputs_hit_the_memo(self, monkeypatch):
        calls = {"n": 0}
        real = vhw._verify_hardware_config_uncached

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(vhw, "_verify_hardware_config_uncached", counting)
        a = vhw.verify_hardware_config(_softcores(), _core_types())
        b = vhw.verify_hardware_config(_softcores(), _core_types())
        assert calls["n"] == 1
        assert a == b

    def test_memo_returns_equal_but_independent_results(self):
        a = vhw.verify_hardware_config(_softcores(), _core_types())
        b = vhw.verify_hardware_config(_softcores(), _core_types())
        assert a == b
        assert a is not b
        b["stats"]["extra_poison"] = True
        c = vhw.verify_hardware_config(_softcores(), _core_types())
        assert "extra_poison" not in c["stats"]

    def test_flags_and_shapes_key_the_memo(self, monkeypatch):
        calls = {"n": 0}
        real = vhw._verify_hardware_config_uncached

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(vhw, "_verify_hardware_config_uncached", counting)
        vhw.verify_hardware_config(_softcores(), _core_types())
        vhw.verify_hardware_config(_softcores(), _core_types(), allow_neuron_splitting=True)
        vhw.verify_hardware_config(_softcores(n=7), _core_types())
        vhw.verify_hardware_config(
            _softcores(), [{"max_axons": 64, "max_neurons": 64, "count": 32}]
        )
        assert calls["n"] == 4

    def test_memoized_result_matches_uncached(self):
        soft = _softcores(n=10)
        expected = vhw._verify_hardware_config_uncached(
            copy.deepcopy(soft), _core_types(), allow_neuron_splitting=True,
            allow_coalescing=False, allow_scheduling=False,
        )
        vhw.verify_hardware_config(soft, _core_types(), allow_neuron_splitting=True)
        got = vhw.verify_hardware_config(soft, _core_types(), allow_neuron_splitting=True)
        assert got["feasible"] == expected["feasible"]
        assert got["stats"] == expected["stats"]
        assert got["errors"] == expected["errors"]
