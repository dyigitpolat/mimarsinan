"""[U-DOOR] The event-serial fold of a weight-quantized program decides its ties on the integer chip lattice.

The NF slot folds the effective weight times the integer threshold (``W * theta_int``, float32), the deployed core folds
the integer weights themselves. On the unbounded accumulator the fold snapped onto no lattice (only a saturating
register's LSB), so a true threshold tie on the integer lattice was decided by float dust on one twin and by the exact
value on the other (the fixture forms at input data scales 1 and 2, 8 seeds: 127 of 208 cases, 57,168 of 1,514,496
counts differed on R + U-SC3, in both planes; ``final_units/udoor/probes/out/base``).
The fold now snaps, inside the measurement plane, onto the weight-quantized program's integer chip lattice: the
half-unit quantum ``CHIP_LATTICE_SCALE`` that the per-cycle policy and the NF's lattice node already snap onto. A
saturating register keeps its own LSB lattice; a float program has no lattice. Both twins carry the program's lattice:
the deployed policy from ``membrane_integer_lattice`` (the Weight Quantization step), the NF slot from its LIF's armed
lattice.

The NF slot folds in the deployed core's float64 (``COMPUTE_DTYPE``), not in the effective weight's float32: a float
program's near tie the deployed sum decides exactly was split by float32 dust (``test_exactness_door``: the event-serial
law with float weights on a chain of convolution cores, ``chain_2``, under the dithered encode: seed 5, 11 counts).

Every refusal of the event-serial and register machinery is a refusal by name (``ToolchainRefusal``): a rail of a
register that asserts its rails unreachable, the count currency's emission bound, the window-start membrane, and every
soma-law refusal. An admitted program either reads exact or refuses by name.
"""

from __future__ import annotations

import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal, is_toolchain_refusal
from mimarsinan.models.nn.lif_kernels import CHIP_LATTICE_SCALE, MembraneRailTouchedError, measurement_plane
from mimarsinan.models.spiking.cycle_policy import LIFCyclePolicy, cycle_neuron_policy
from mimarsinan.models.nn.activations.lif_serial import SerialFoldSlot
from mimarsinan.models.spiking.serial import lif_serial_fold
from mimarsinan.models.spiking.serial import refusals as serial_refusals
from mimarsinan.models.spiking.serial.policy import SerialLIFCyclePolicy, serial_policy_for

_PER_EVENT = SomaLaw.resolve({"firing_granularity": "per_event"})
_PER_EVENT_SAT8 = SomaLaw.resolve({"firing_granularity": "per_event", "membrane_bits": 8})


def _dusty_weight() -> tuple[int, int, torch.Tensor]:
    """``(q, theta_int, w)``: an integer weight ``q`` whose NF-slot form ``float32(q / theta_int) * theta_int`` lands
    just BELOW ``q`` (float dust), as the NF slot computes it from a quantized effective weight."""
    for theta_int in range(3, 64):
        for q in range(1, 8):
            w = torch.tensor([q / theta_int], dtype=torch.float32) * float(theta_int)
            if float(w) < q:
                return q, theta_int, w
    raise AssertionError("no float32 dust witness in range")


def _fold(weight: torch.Tensor, theta: float, *, law: SomaLaw, integer_lattice: bool) -> torch.Tensor:
    """Two occurrences of one slot against a threshold of exactly twice the integer weight: a true tie."""
    memb = torch.zeros(1, 1, dtype=torch.float32)
    events = torch.tensor([[2.0]], dtype=torch.float32)
    return lif_serial_fold(memb, weight.reshape(1, 1), events, torch.tensor(theta, dtype=torch.float32),
                           soma_law=law, integer_lattice=integer_lattice)


def test_the_witness_is_a_true_tie_the_dust_decides_against_the_integer_fold():
    q, _theta_int, w = _dusty_weight()
    with measurement_plane():
        exact = _fold(torch.tensor([float(q)]), 2.0 * q, law=_PER_EVENT, integer_lattice=True)
        dusty = _fold(w, 2.0 * q, law=_PER_EVENT, integer_lattice=False)
    assert float(exact) == 1.0 and float(dusty) == 0.0


def test_a_quantized_program_decides_the_tie_on_its_integer_lattice_inside_the_measurement_plane():
    q, _theta_int, w = _dusty_weight()
    with measurement_plane():
        assert float(_fold(w, 2.0 * q, law=_PER_EVENT, integer_lattice=True)) == 1.0


def test_outside_the_measurement_plane_the_membrane_stays_continuous():
    """The lattice is the measurement plane's (the tuning plane keeps the continuous membrane), as everywhere else."""
    q, _theta_int, w = _dusty_weight()
    assert float(_fold(w, 2.0 * q, law=_PER_EVENT, integer_lattice=True)) == 0.0


def test_a_saturating_register_keeps_its_own_lsb_lattice():
    """The register's LSB lattice is unchanged by the program lattice (ODIN's native pairing, byte for byte)."""
    q, _theta_int, w = _dusty_weight()
    with measurement_plane():
        armed = _fold(w, 2.0 * q, law=_PER_EVENT_SAT8, integer_lattice=True)
        unarmed = _fold(w, 2.0 * q, law=_PER_EVENT_SAT8, integer_lattice=False)
    assert float(armed) == float(unarmed) == 1.0


def test_the_integer_lattice_is_one_constant():
    assert LIFCyclePolicy._CHIP_LATTICE_SCALE == CHIP_LATTICE_SCALE == 2.0


def test_the_deployed_policy_carries_the_program_lattice_into_the_fold():
    policy = cycle_neuron_policy("lif", "cascaded", "Novena", integer_lattice=True, soma_law=_PER_EVENT)
    assert isinstance(policy, SerialLIFCyclePolicy) and policy.integer_lattice is True
    assert serial_policy_for(_PER_EVENT).integer_lattice is False
    q, _theta_int, w = _dusty_weight()
    state = policy.make_state(1, 1, torch.device("cpu"), torch.float32)
    with measurement_plane():
        counts = policy.step(state, w.reshape(1, 1), torch.tensor([[2.0]]),
                             torch.tensor(2.0 * q, dtype=torch.float32), hw_bias=None, thresholding_mode="<=")
    assert float(counts) == 1.0


def test_a_rail_of_a_register_that_asserts_it_unreachable_is_a_refusal_by_name():
    assert issubclass(MembraneRailTouchedError, ToolchainRefusal)
    assert issubclass(MembraneRailTouchedError, ValueError)
    assert is_toolchain_refusal(MembraneRailTouchedError("rail"))


def _float32_near_tie() -> tuple[torch.Tensor, float]:
    """``(w, theta)``: two float32 weights whose float32 sum reaches ``theta`` while their float64 sum (the deployed
    core's) stays below it, so float32 folding fires where the deployed core does not."""
    for a in (0.1, 0.2, 0.3, 0.7):
        w = torch.tensor([[a, 1.0 - a]], dtype=torch.float32)
        s32 = (w[0, 0] + w[0, 1]).item()
        s64 = float(w[0, 0].double() + w[0, 1].double())
        if s32 >= 1.0 > s64:
            return w, 1.0
    raise AssertionError("no float32 near-tie witness")


def test_the_nf_slot_folds_a_float_program_in_the_deployed_core_float64():
    w, theta = _float32_near_tie()
    events = torch.tensor([[1.0, 1.0]])
    slot = SerialFoldSlot(soma_law=_PER_EVENT, weight=w, bias=None, theta=theta, membrane_init=0.0)
    slot.feed(events)
    with measurement_plane():
        nf = slot.run_cycle(torch.nn.functional.linear(events, w), 1.0)
        deployed = lif_serial_fold(torch.zeros(1, 1, dtype=torch.float64), w.double(), events.double(),
                                   torch.tensor(theta, dtype=torch.float64), soma_law=_PER_EVENT)
    assert slot.weight.dtype == torch.float64
    assert float(nf) == float(deployed) == 0.0


def test_every_refusal_of_the_event_serial_and_register_machinery_is_a_refusal_by_name():
    named = [getattr(serial_refusals, name) for name in serial_refusals.__all__
             if name.endswith("Error") and isinstance(getattr(serial_refusals, name), type)]
    assert len(named) >= 10
    assert all(issubclass(cls, ToolchainRefusal) for cls in named), [c.__name__ for c in named
                                                                      if not issubclass(c, ToolchainRefusal)]
