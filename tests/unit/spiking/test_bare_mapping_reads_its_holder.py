"""[U-TGAUGE4] The bare packed mapping's read (the U-TGAUGE3 verify's m-2 and its W17 gap): each output neuron's gauge
is read through its holder's declaration of the threshold register, and a read the declaration does not place refuses
by name, never an index error and never a count ranking.

A hard core declares its activation scale per core (one value, or one vector its residents share) or per neuron range
(a target that stores it per neuron: the register outside its ``constrained_properties``, each resident's value over its
own neurons, which only the program's source table places).
The read took ``core.activation_scale`` by its size at the hard core's neuron index: a per-neuron declaration read as no
gauge (the counts ranked), and two residents sharing one vector raised numpy's IndexError at the second one's first
neuron.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.hybrid_run.output_decode import OutputGaugeError, bare_mapping_output_gauge
from mimarsinan.code_generation.cpp_chip_model_types import SpikeSource
from mimarsinan.mapping.packing.softcore.hard_core import HardCore
from mimarsinan.mapping.platform.core_residency import ALL_SINGLETON_NAMES, adopt_or_check


def _resident(width, scale):
    return types.SimpleNamespace(activation_scale=scale, threshold=1.0, parameter_scale=None,
                                 input_activation_scale=None, boundary_grid=None, get_output_count=lambda: width)


def _core(constrained, scales, width=3, total=8):
    core = HardCore(axons_per_core=8, neurons_per_core=total)
    core.constrained_properties = frozenset(constrained)
    for scale in scales:
        core.available_neurons -= width
        adopt_or_check(core, _resident(width, scale), constrained=core.constrained_properties)
    return core


def _mapping(core, n_out):
    return types.SimpleNamespace(cores=[core],
                                 output_sources=np.array([SpikeSource(0, i) for i in range(n_out)], dtype=object))


@pytest.mark.parametrize("gauges", [(0.5, 4.0), (1.5, 1.5)], ids=["two_gauges", "one_gauge"])
def test_a_per_neuron_declaration_refuses_by_name(gauges):
    """Two residents recorded per neuron (a target that stores the activation scale per neuron range): the read found
    no ``core.activation_scale`` and ranked the counts. Each resident's gauge sits over its own neurons, which only the
    program's source table places, so the bare read refuses by name, at one gauge as at two."""
    core = _core(ALL_SINGLETON_NAMES - {"activation_scale"}, [torch.tensor(g) for g in gauges])
    with pytest.raises(OutputGaugeError, match="per neuron range"):
        bare_mapping_output_gauge(_mapping(core, 6))


def test_two_residents_sharing_one_vector_refuse_by_name_not_by_an_index_error():
    """One per-core vector (0.5, 1.0, 4.0) adopted by two residents of three neurons each: which resident's channel
    neuron 3 holds is not declared on this core, so the read refuses by name (it raised IndexError at neuron 3)."""
    vector = torch.tensor([0.5, 1.0, 4.0])
    core = _core(ALL_SINGLETON_NAMES, [vector, vector.clone()])
    with pytest.raises(OutputGaugeError, match="is not declared"):
        bare_mapping_output_gauge(_mapping(core, 6))


def test_one_resident_s_vector_over_the_core_s_used_neurons_is_read_at_each_neuron():
    """A per-core vector that spans the core's used neurons places each neuron: at three gauges the read refuses (the
    counts do not rank), at one it ranks them."""
    three = _core(ALL_SINGLETON_NAMES, [torch.tensor([0.5, 1.0, 4.0])])
    with pytest.raises(OutputGaugeError, match="no gauge table"):
        bare_mapping_output_gauge(_mapping(three, 3))
    one = _core(ALL_SINGLETON_NAMES, [torch.tensor([2.0, 2.0, 2.0])])
    assert bare_mapping_output_gauge(_mapping(one, 3)) is None


@pytest.mark.parametrize("gauges", [(0.5, 4.0, 4.0), (0.5, 0.5, 4.0)], ids=["two_gauges", "two_gauges_reversed"])
def test_a_bare_mapping_at_two_gauges_refuses(gauges):
    """The verify's W17 gap: the refusal's test read three distinct gauges, so a refusal only at three or more passed."""
    core = types.SimpleNamespace(activation_scale=torch.tensor(gauges, dtype=torch.float32),
                                 neurons_per_core=len(gauges))
    with pytest.raises(OutputGaugeError, match="no gauge table"):
        bare_mapping_output_gauge(_mapping(core, len(gauges)))
