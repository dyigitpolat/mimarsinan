"""Reliability check: the mimarsinan plugins must not perturb SANA-FE's
energy / latency / counter accounting compared to the built-in models.

Energy and latency in SANA-FE come from per-event constants declared in
the arch YAML, multiplied at runtime by the *count* of events
(spikes processed, dendrite updates, soma accesses, soma updates, soma
spike-outs, NoC hops).  The plugins replace ``accumulator`` and
``leaky_integrate_fire`` purely to lift the Loihi-derived 1024-neurons-
per-core cap, **not** to change the per-event accounting.  This file
proves that claim.

We compare two configurations side-by-side:

  * **built-in** — ``accumulator`` dendrite + ``leaky_integrate_fire``
    soma, configured to approximate ``SubtractiveLIFReset`` (``leak_decay
    = 1.0``, ``input_decay = 0.0``, ``reset_mode = soft``,
    ``force_update = True``);
  * **plugin** — ``mimarsinan_dendrite`` + ``mimarsinan_soma``.

The built-in still applies its baked-in ``loihi_quantize`` (1/64 fixed
point), which is a Loihi-specific quirk we deliberately drop in our
plugin (mimarsinan is its own hardware proposal).  To keep the test
fair, we use INTEGER weights and thresholds so quantization is a no-op
on the values the soma sees.  Under those constraints every field
SANA-FE reports must match bit-exactly between the two backends.

Skipped when:
  * SANA-FE is not installed (``pip install sanafe``);
  * the mimarsinan plugins haven't been built
    (``scripts/bootstrap_sanafe.sh``).
"""
from __future__ import annotations

import math
import os
import tempfile

import pytest


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SO_DEND = os.path.join(
    PROJ_ROOT, "build", "mimarsinan_sanafe_plugins", "libmimarsinan_dendrite.so",
)
SO_SOMA = os.path.join(
    PROJ_ROOT, "build", "mimarsinan_sanafe_plugins", "libmimarsinan_soma.so",
)


def _have_sanafe() -> bool:
    try:
        import sanafe  # noqa: F401
        return True
    except Exception:
        return False


def _have_plugins() -> bool:
    return os.path.isfile(SO_DEND) and os.path.isfile(SO_SOMA)


pytestmark = [
    pytest.mark.skipif(not _have_sanafe(),
                       reason="SANA-FE not installed (scripts/bootstrap_sanafe.sh)"),
    pytest.mark.skipif(not _have_plugins(),
                       reason="mimarsinan plugins not built "
                              "(scripts/bootstrap_sanafe.sh)"),
    pytest.mark.slow,
    pytest.mark.integration,
]


# ---------------------------------------------------------------------------
# Arch templates — identical per-event energies in both versions; only the
# dendrite + soma model identity differs.
# ---------------------------------------------------------------------------


def _make_arch_yaml(*, use_plugin: bool) -> str:
    if use_plugin:
        dendrite = f"plugin: {SO_DEND}\n                model: mimarsinan_dendrite"
        soma = (f"plugin: {SO_SOMA}\n                model: mimarsinan_soma"
                f"\n                thresholding_mode: strict")
    else:
        dendrite = "model: accumulator"
        soma = "model: leaky_integrate_fire"
    one_core = f"""          attributes: {{buffer_position: soma, buffer_inside_unit: false, max_neurons_supported: 8192}}
          axon_in: [{{name: ai, attributes: {{energy_message_in: 0.0, latency_message_in: 1.0e-9}}}}]
          synapse: [{{name: dense_syn, attributes: {{model: current_based, energy_process_spike: 3.55e-11, latency_process_spike: 3.8e-09}}}}]
          dendrite:
            - name: d
              attributes:
                {dendrite}
                update_every_timestep: true
                energy_update: 1.3e-12
                latency_update: 1.0e-9
          soma:
            - name: lif
              attributes:
                {soma}
                energy_access_neuron: 5.12e-11
                latency_access_neuron: 6.0e-9
                energy_update_neuron: 2.16e-11
                latency_update_neuron: 3.7e-9
                energy_spike_out: 6.93e-11
                latency_spike_out: 3.0e-8
            - name: inputs[0..63]
              attributes: {{model: input, energy_access_neuron: 0.0, latency_access_neuron: 0.0, energy_update_neuron: 0.0, latency_update_neuron: 0.0, energy_spike_out: 0.0, latency_spike_out: 0.0}}
          axon_out: [{{name: ao, attributes: {{energy_message_out: 1.11e-10, latency_message_out: 5.1e-09}}}}]
"""
    return f"""architecture:
  name: cmp
  attributes: {{topology: mesh, width: 2, height: 1, link_buffer_size: 16, sync_model: fixed, latency_sync: 0.0}}
  tile:
    - name: t0
      attributes: {{energy_north_hop: 3.5e-12, latency_north_hop: 5.0e-9, energy_east_hop: 3.5e-12, latency_east_hop: 5.0e-9, energy_south_hop: 3.5e-12, latency_south_hop: 5.0e-9, energy_west_hop: 3.5e-12, latency_west_hop: 5.0e-9}}
      core:
        - name: c0
{one_core}        - name: c1
{one_core}"""


def _load_arch(yaml: str):
    import sanafe
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml)
        path = f.name
    return sanafe.load_arch(path)


# ---------------------------------------------------------------------------
# Per-test network builders
# ---------------------------------------------------------------------------


def _builtin_extra() -> dict:
    """Knobs that bring SANA-FE's built-in LIF as close to SubtractiveLIFReset
    as it can go without losing the fixed-point quantize step."""
    return {"leak_decay": 1.0, "input_decay": 0.0, "reset_mode": "soft",
            "reset": 0.0, "force_update": True}


def _idle_network(arch, *, use_plugin: bool):
    """Two cores × 5 LIF neurons.  Threshold is huge → nobody fires.
    With ``force_update``+``update_every_timestep`` every neuron is still
    touched every cycle so the per-cycle accounting is exercised.
    """
    import sanafe
    net = sanafe.Network()
    attrs_extra = {} if use_plugin else _builtin_extra()
    for core_idx in range(2):
        g = net.create_neuron_group(f"core{core_idx}", 5, model_attributes={})
        for i in range(5):
            attrs = dict(attrs_extra)
            attrs["threshold"] = 1e9
            attrs["bias"] = 0.0
            g[i].set_attributes(
                soma_hw_name="lif",
                log_spikes=True, log_potential=True,
                model_attributes=attrs,
            )
            g[i].map_to_core(arch.tiles[0].cores[core_idx])
    return net


def _deterministic_fire_network(arch, *, use_plugin: bool):
    """Inputs + LIF chain with INTEGER weights and INTEGER thresholds so
    SANA-FE's 1/64 quantization of the potential is a no-op (only
    integer values cross the firing boundary).  Both backends therefore
    produce identical spike behaviour and identical event counts.
    """
    import sanafe
    net = sanafe.Network()
    attrs_extra = {} if use_plugin else _builtin_extra()

    inp = net.create_neuron_group("in", 2)
    inp[0].set_attributes(soma_hw_name="inputs[0]",
                          model_attributes={"spikes": [1] * 12})
    inp[1].set_attributes(soma_hw_name="inputs[1]",
                          model_attributes={"spikes": [1, 0] * 6})
    inp[0].map_to_core(arch.tiles[0].cores[0])
    inp[1].map_to_core(arch.tiles[0].cores[0])

    lif0 = net.create_neuron_group("lif0", 3, model_attributes={})
    for i in range(3):
        attrs = dict(attrs_extra)
        attrs["threshold"] = 2.0
        attrs["bias"] = 0.0
        lif0[i].set_attributes(soma_hw_name="lif",
                               log_spikes=True, log_potential=True,
                               model_attributes=attrs)
        lif0[i].map_to_core(arch.tiles[0].cores[0])

    lif1 = net.create_neuron_group("lif1", 2, model_attributes={})
    for i in range(2):
        attrs = dict(attrs_extra)
        attrs["threshold"] = 3.0
        attrs["bias"] = 0.0
        lif1[i].set_attributes(soma_hw_name="lif",
                               log_spikes=True, log_potential=True,
                               model_attributes=attrs)
        lif1[i].map_to_core(arch.tiles[0].cores[1])

    for src in inp:
        for dst in lif0:
            src.connect_to_neuron(dst, {"weight": 1.0, "synapse_hw_name": "dense_syn"})
    for src in lif0:
        for dst in lif1:
            src.connect_to_neuron(dst, {"weight": 1.0, "synapse_hw_name": "dense_syn"})
    return net


def _run(yaml_text: str, build_network, T: int = 12) -> dict:
    import sanafe
    arch = _load_arch(yaml_text)
    net = build_network(arch)
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    return chip.sim(timesteps=T)


def _close(a, b, *, rtol=1e-12, atol=1e-18) -> bool:
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


_COUNTER_FIELDS = ("timesteps_executed", "spikes", "packets_sent",
                   "neurons_updated", "neurons_fired")
_FLOAT_FIELDS = ("sim_time",)
_ENERGY_KEYS = ("synapse", "dendrite", "soma", "network", "total")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_idle_network_all_counters_and_energy_match():
    """No firing — exercises pure per-cycle accounting.  Must match exactly."""
    res_b = _run(_make_arch_yaml(use_plugin=False),
                 lambda arch: _idle_network(arch, use_plugin=False))
    res_p = _run(_make_arch_yaml(use_plugin=True),
                 lambda arch: _idle_network(arch, use_plugin=True))

    for k in _COUNTER_FIELDS:
        assert res_b[k] == res_p[k], (
            f"idle: {k} differs ({res_b[k]} vs {res_p[k]})"
        )
    for k in _FLOAT_FIELDS:
        assert _close(res_b[k], res_p[k]), (
            f"idle: {k} differs ({res_b[k]} vs {res_p[k]})"
        )
    for k in _ENERGY_KEYS:
        a, b = res_b["energy"][k], res_p["energy"][k]
        assert _close(a, b), (
            f"idle: energy.{k} differs ({a} vs {b}; Δ={b - a})"
        )


def test_deterministic_fire_all_counters_and_energy_match():
    """Integer weights → 1/64 quantize is a no-op → identical spike behaviour
    → identical event counts → identical energies / latencies."""
    res_b = _run(_make_arch_yaml(use_plugin=False),
                 lambda arch: _deterministic_fire_network(arch, use_plugin=False))
    res_p = _run(_make_arch_yaml(use_plugin=True),
                 lambda arch: _deterministic_fire_network(arch, use_plugin=True))

    # Non-zero spike activity in this configuration.
    assert res_b["spikes"] > 0
    assert res_b["neurons_fired"] > 0

    for k in _COUNTER_FIELDS:
        assert res_b[k] == res_p[k], (
            f"fire: {k} differs ({res_b[k]} vs {res_p[k]})"
        )
    for k in _FLOAT_FIELDS:
        assert _close(res_b[k], res_p[k]), (
            f"fire: {k} differs ({res_b[k]} vs {res_p[k]})"
        )
    for k in _ENERGY_KEYS:
        a, b = res_b["energy"][k], res_p["energy"][k]
        assert _close(a, b), (
            f"fire: energy.{k} differs ({a} vs {b}; Δ={b - a})"
        )
