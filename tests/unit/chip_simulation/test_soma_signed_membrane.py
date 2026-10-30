"""[ODIN P6] ``membrane_arithmetic='saturating_signed'``: the sync-fire register.

The third value on an EXISTING axis, added by the same additive protocol P1
used: the vocabulary grows by one, signedness is declared beside the width it
qualifies (platform ``membrane_signed``), the arithmetic falls out of the PAIR,
and every configuration that declares no width resolves byte-identically —
which the golden snapshot's zero-deletion diff proves once for the whole
surface.

What makes the value real rather than cosmetic: a per-CYCLE window whose net
charge goes negative floors at zero on an unsigned register and therefore
CANNOT hold the number the unbounded accumulator holds. The two's-complement
register can, and only while its rails stay unreached — so every implementation
asserts the rails instead of clamping silently.
"""

import pytest
import torch

from mimarsinan.chip_simulation.certification import CertificationCell
from mimarsinan.chip_simulation.nevresim_policy_types import (
    NevresimPolicyTypeError,
    counts_on_the_wire,
    emits_integration_policy,
    nevresim_integration_policy,
)
from mimarsinan.chip_simulation.soma_axes import (
    MEMBRANE_ARITHMETICS,
    SATURATING_SIGNED_MEMBRANE,
    SATURATING_UNSIGNED_MEMBRANE,
    UNBOUNDED_MEMBRANE,
    derived_membrane_arithmetic,
    legal_membrane_arithmetics,
    resolved_membrane_signed,
)
from mimarsinan.chip_simulation.soma_capability import (
    BackendSomaLawError,
    require_soma_law_supported,
    supports_soma_law,
)
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.config_schema.derivation.soma import (
    enforce_soma_axes_contract,
    soma_contract_error_rows,
)
from mimarsinan.config_schema.recipe_fold import (
    _soma_law_denies_membrane_readout,
)
from mimarsinan.models.nn.lif_kernels import MembraneRailTouchedError
from mimarsinan.models.spiking.cycle_policy import cycle_neuron_policy
from mimarsinan.models.spiking.serial.refusals import (
    SaturatingMembraneRefusalError,
    refuse_saturating_membrane,
)

_LIF = {"spiking_family": "lif", "spiking_variant": "streamed"}
_SYNC_FIRE = {**_LIF, "membrane_bits": 16, "membrane_signed": True}
_STOCK_ODIN = {**_LIF, "firing_mode": "Novena", "firing_granularity": "per_event",
               "membrane_bits": 8}

_EXECUTING_BACKENDS = ("hcm", "unified", "hybrid", "nevresim", "odin_rtl")
_REFUSING_BACKENDS = ("sanafe", "lava", "loihi", "training")


class TestTheVocabularyGrewByExactlyOneValue:
    def test_the_axis_appends_the_signed_register_and_keeps_its_order(self):
        assert MEMBRANE_ARITHMETICS == (
            UNBOUNDED_MEMBRANE, SATURATING_UNSIGNED_MEMBRANE,
            SATURATING_SIGNED_MEMBRANE,
        )

    @pytest.mark.parametrize("variant", ["analytical", "cascaded"])
    def test_a_saturating_register_stays_a_lif_family_law(self, variant):
        assert SATURATING_SIGNED_MEMBRANE in legal_membrane_arithmetics(_LIF)
        assert legal_membrane_arithmetics(
            {"spiking_family": "ttfs", "spiking_variant": variant}
        ) == (UNBOUNDED_MEMBRANE,)

    def test_a_value_core_rules_out_every_saturating_register(self):
        assert legal_membrane_arithmetics(
            {"core_semantics": "mvm"}) == (UNBOUNDED_MEMBRANE,)

    @pytest.mark.parametrize("cfg", [
        {}, {"spiking_family": "nonsense"}, {"cores": "not a grid"},
        {"membrane_signed": "yes"}, {"membrane_signed": None},
    ])
    def test_every_predicate_is_total_over_partial_and_invalid_configs(self, cfg):
        assert derived_membrane_arithmetic(cfg) in MEMBRANE_ARITHMETICS
        assert isinstance(resolved_membrane_signed(cfg), bool)


class TestSignednessIsDeclaredBesideTheWidth:
    def test_a_declared_width_alone_still_derives_the_unsigned_register(self):
        assert derived_membrane_arithmetic(
            {**_LIF, "membrane_bits": 8}) == SATURATING_UNSIGNED_MEMBRANE

    def test_the_pair_derives_the_signed_register(self):
        assert derived_membrane_arithmetic(_SYNC_FIRE) == SATURATING_SIGNED_MEMBRANE

    def test_signedness_without_a_width_declares_nothing(self):
        assert derived_membrane_arithmetic(
            {**_LIF, "membrane_signed": True}) == UNBOUNDED_MEMBRANE

    def test_only_the_literal_true_declares_a_signed_register(self):
        """A truthy string is a malformed declaration; the registry's own type
        error is the single truth, and reading it as 'signed' would silently
        deploy a different register."""
        assert resolved_membrane_signed({"membrane_signed": "true"}) is False
        assert resolved_membrane_signed({"membrane_signed": 1}) is False
        assert resolved_membrane_signed({"membrane_signed": True}) is True


class TestTheCrossKeyContractIsKeyedAndRemediable:
    def _rows(self, cfg):
        return soma_contract_error_rows(cfg, {})

    def test_the_default_point_pays_nothing(self):
        enforce_soma_axes_contract({})
        assert self._rows({}) == []

    def test_the_sync_fire_point_is_admitted(self):
        enforce_soma_axes_contract({**_SYNC_FIRE, "cores": [
            {"max_axons": 256, "max_neurons": 256, "count": 1, "has_bias": True}]})

    def test_an_unsigned_declaration_against_a_signed_register_is_keyed(self):
        cfg = {**_SYNC_FIRE,
               "membrane_arithmetic": SATURATING_UNSIGNED_MEMBRANE}
        rows = self._rows(cfg)
        assert [row["key"] for row in rows] == ["membrane_arithmetic"]
        assert "membrane_signed=True" in rows[0]["message"]
        assert {"membrane_signed"} <= {
            remedy["key"] for remedy in rows[0]["remedies"]}
        with pytest.raises(ValueError, match="membrane_signed"):
            enforce_soma_axes_contract(cfg)

    def test_a_signed_declaration_against_an_unsigned_register_is_keyed(self):
        cfg = {**_LIF, "membrane_bits": 16,
               "membrane_arithmetic": SATURATING_SIGNED_MEMBRANE}
        rows = self._rows(cfg)
        assert [row["key"] for row in rows] == ["membrane_arithmetic"]
        assert "membrane_signed=False" in rows[0]["message"]

    def test_a_signed_register_with_no_width_is_keyed(self):
        cfg = {**_LIF, "membrane_arithmetic": SATURATING_SIGNED_MEMBRANE}
        rows = self._rows(cfg)
        assert rows and rows[0]["key"] == "membrane_arithmetic"
        assert "membrane_bits" in rows[0]["message"]

    def test_the_signed_register_is_refused_under_the_event_serial_law(self):
        """The row-pair realization's zero-magnitude member is a no-op only
        against a register that FLOORS at zero, and nothing folds events on a
        two's-complement membrane — so the pair is refused by key, not run."""
        cfg = {**_STOCK_ODIN, "membrane_bits": 16, "membrane_signed": True,
               "cores": [{"max_axons": 128, "max_neurons": 256, "count": 1,
                          "has_bias": False}]}
        rows = self._rows(cfg)
        assert "membrane_arithmetic" in [row["key"] for row in rows]
        message = next(r["message"] for r in rows
                       if r["key"] == "membrane_arithmetic")
        assert "per_event" in message and SATURATING_SIGNED_MEMBRANE in message

    def test_the_stock_odin_point_is_untouched_by_the_new_value(self):
        enforce_soma_axes_contract({**_STOCK_ODIN, "cores": [
            {"max_axons": 128, "max_neurons": 256, "count": 1,
             "has_bias": False}]})


class TestTheResolvedLawCarriesTheSignedInterval:
    def test_the_register_interval_is_two_s_complement(self):
        law = SomaLaw.resolve(_SYNC_FIRE)
        assert law.membrane_arithmetic == SATURATING_SIGNED_MEMBRANE
        assert law.membrane_bounds == (-32768.0, 32767.0)
        assert law.is_signed_membrane is True
        assert law.saturates is True

    def test_the_unsigned_register_is_unchanged(self):
        law = SomaLaw.resolve(_STOCK_ODIN)
        assert law.membrane_bounds == (0.0, 255.0)
        assert law.is_signed_membrane is False
        assert law.asserts_no_saturation is False

    def test_only_the_signed_register_asserts_its_rails(self):
        assert SomaLaw.resolve(_SYNC_FIRE).asserts_no_saturation is True
        assert DEFAULT_SOMA_LAW.asserts_no_saturation is False

    def test_the_default_point_is_untouched(self):
        assert DEFAULT_SOMA_LAW.membrane_bounds is None
        assert DEFAULT_SOMA_LAW.point_tag() is None
        assert DEFAULT_SOMA_LAW.is_default_point is True


class TestCellIdentityDiscriminatesTheSignedRegister:
    def test_the_point_tag_separates_signed_from_unsigned_at_the_same_width(self):
        signed = SomaLaw.resolve({**_LIF, "membrane_bits": 16,
                                  "membrane_signed": True})
        unsigned = SomaLaw.resolve({**_LIF, "membrane_bits": 16})
        assert signed.point_tag() == "ssat16"
        assert unsigned.point_tag() == "sat16"
        assert signed.point_tag() != unsigned.point_tag()

    def test_the_cell_key_carries_it_and_round_trips(self):
        law = SomaLaw.resolve(_SYNC_FIRE)
        cell = CertificationCell.from_mode_policy(
            policy_for_spiking_mode("lif", soma_law=law), backend="hcm")
        assert cell.cell_key == "lif@hcm#ssat16"
        assert CertificationCell.from_key(cell.cell_key) == cell


class TestTheCapabilityGuardOpensExactlyWhereAnExecutorLanded:
    @pytest.mark.parametrize("backend", _EXECUTING_BACKENDS)
    def test_the_executing_backends_admit_the_signed_register(self, backend):
        law = SomaLaw.resolve(_SYNC_FIRE)
        assert supports_soma_law(backend, law) is True
        require_soma_law_supported(law, backend=backend, context="ctx")

    @pytest.mark.parametrize("backend", _REFUSING_BACKENDS)
    def test_every_other_backend_refuses_it_by_name(self, backend):
        law = SomaLaw.resolve(_SYNC_FIRE)
        assert supports_soma_law(backend, law) is False
        with pytest.raises(BackendSomaLawError) as exc:
            require_soma_law_supported(law, backend=backend, context="ctx")
        message = str(exc.value)
        assert SATURATING_SIGNED_MEMBRANE in message
        assert SATURATING_UNSIGNED_MEMBRANE not in message
        assert backend in message and "ctx" in message

    def test_the_unsigned_register_keeps_its_own_refusal_text(self):
        """A signed register is also a saturating one, so the two refusals must
        stay distinguishable: reporting the unsigned value for a signed
        declaration would name a law the config never wrote."""
        law = SomaLaw.resolve({**_LIF, "membrane_bits": 8})
        with pytest.raises(BackendSomaLawError) as exc:
            require_soma_law_supported(law, backend="loihi", context="ctx")
        assert SATURATING_UNSIGNED_MEMBRANE in str(exc.value)
        assert SATURATING_SIGNED_MEMBRANE not in str(exc.value)


class TestTheNevresimPolicyString:
    def test_the_sync_fire_point_names_the_signed_whole_vector_policy(self):
        assert nevresim_integration_policy(SomaLaw.resolve(_SYNC_FIRE)) == (
            "WholeVectorSaturatingSigned<16>")

    def test_the_default_point_still_emits_nothing(self):
        policy = nevresim_integration_policy(DEFAULT_SOMA_LAW)
        assert policy == "WholeVectorIntegrate"
        assert emits_integration_policy(policy) is False

    def test_the_sync_fire_policy_is_named_but_keeps_a_binary_wire(self):
        """One compare per cycle means at most one spike: the counted raster
        must NOT arm, or the carry seam changes format for a law that never
        produces a multiplicity."""
        policy = nevresim_integration_policy(SomaLaw.resolve(_SYNC_FIRE))
        assert emits_integration_policy(policy) is True
        assert counts_on_the_wire(policy) is False

    def test_the_event_serial_policy_still_counts(self):
        policy = nevresim_integration_policy(SomaLaw.resolve(_STOCK_ODIN))
        assert policy == "EventSerialIntegrate<8>"
        assert counts_on_the_wire(policy) is True
        assert emits_integration_policy(policy) is True

    def test_a_signed_per_event_point_refuses_by_name(self):
        law = SomaLaw(
            firing_mode="Novena", thresholding_mode="<=",
            firing_granularity="per_event",
            membrane_arithmetic=SATURATING_SIGNED_MEMBRANE, membrane_bits=16,
        )
        with pytest.raises(NevresimPolicyTypeError, match="per_event"):
            nevresim_integration_policy(law)


def _run_cycles(policy, weights, cycles, threshold):
    """Drive one neuron through ``cycles`` of per-slot counts; return counts."""
    state = policy.make_state(1, 1, torch.device("cpu"), torch.float64)
    weight = torch.tensor(weights, dtype=torch.float64).reshape(1, -1)
    emitted = []
    for events in cycles:
        spikes = policy.step(
            state, weight, torch.tensor([events], dtype=torch.float64),
            torch.tensor(float(threshold), dtype=torch.float64),
            hw_bias=None, thresholding_mode="<=",
        )
        emitted.append(int(spikes.sum().item()))
    return emitted, float(state["memb"].item())


class TestTheTorchPerCycleKernelExecutesTheSignedRegister:
    def _policy(self, cfg):
        return cycle_neuron_policy(
            "lif", "", "Novena", soma_law=SomaLaw.resolve(cfg))

    def test_a_net_negative_cycle_matches_the_unbounded_law_exactly(self):
        """THE case the value exists for: an unsigned register floors the
        negative cycle at 0 and then fires on the next one; the signed register
        carries the debt and reproduces the unbounded accumulator's counts."""
        weights, cycles, theta = [3.0, -7.0], [[1, 1], [1, 0]], 5.0
        signed = self._policy(_SYNC_FIRE)
        exact = self._policy({**_LIF})
        unsigned = self._policy({**_LIF, "membrane_bits": 16})
        assert _run_cycles(signed, weights, cycles, theta) == \
            _run_cycles(exact, weights, cycles, theta)
        assert _run_cycles(signed, weights, cycles, theta)[1] < 0
        assert _run_cycles(unsigned, weights, cycles, theta) != \
            _run_cycles(exact, weights, cycles, theta)

    def test_the_signed_register_still_fires_at_most_once_per_cycle(self):
        counts, _ = _run_cycles(
            self._policy(_SYNC_FIRE), [5.0, 5.0], [[3, 3], [0, 0]], 4.0)
        assert counts == [1, 0]

    def test_the_policy_carries_the_interval_and_the_assertion(self):
        policy = self._policy(_SYNC_FIRE)
        assert policy.membrane_bounds == (-32768.0, 32767.0)
        assert policy.membrane_rail_assert is True

    def test_the_unsigned_register_never_arms_the_assertion(self):
        policy = self._policy({**_LIF, "membrane_bits": 8})
        assert policy.membrane_bounds == (0.0, 255.0)
        assert policy.membrane_rail_assert is False
        # Its saturation IS the modelled substrate: it clamps, it does not raise.
        counts, membrane = _run_cycles(policy, [200.0], [[1], [1]], 1000.0)
        assert counts == [0, 0] and membrane == 255.0

    def test_the_default_point_stays_unbounded_and_unasserted(self):
        policy = self._policy({**_LIF})
        assert policy.membrane_bounds is None
        assert policy.membrane_rail_assert is False


class TestTheRailIsRefusedNeverClamped:
    def _policy(self, bits):
        return cycle_neuron_policy(
            "lif", "", "Novena",
            soma_law=SomaLaw.resolve(
                {**_LIF, "membrane_bits": bits, "membrane_signed": True}))

    def test_the_positive_rail_raises(self):
        with pytest.raises(MembraneRailTouchedError, match="127"):
            _run_cycles(self._policy(8), [127.0], [[1]], 1000.0)

    def test_the_negative_rail_raises(self):
        with pytest.raises(MembraneRailTouchedError, match="-128"):
            _run_cycles(self._policy(8), [-128.0], [[1]], 1000.0)

    def test_a_value_one_short_of_the_rail_is_admitted(self):
        counts, membrane = _run_cycles(self._policy(8), [126.0], [[1]], 1000.0)
        assert counts == [0] and membrane == 126.0

    def test_the_message_names_the_interval_and_the_remedy(self):
        with pytest.raises(MembraneRailTouchedError) as exc:
            _run_cycles(self._policy(8), [127.0], [[1]], 1000.0)
        message = str(exc.value)
        assert "membrane_bits" in message
        assert "never clamped" in message


class TestTheLosslessAccumulatorRefusalsNameTheDeclaredRegister:
    """A refusal that names ``saturating_unsigned`` and ``[0, 2**bits-1]`` for a
    two's-complement declaration reports a law the config never wrote."""

    def test_the_membrane_readout_fold_refuses_the_signed_register_by_name(self):
        reason = _soma_law_denies_membrane_readout(SomaLaw.resolve(_SYNC_FIRE))
        assert reason is not None
        assert SATURATING_SIGNED_MEMBRANE in reason
        assert "[-32768, 32767]" in reason
        assert _soma_law_denies_membrane_readout(DEFAULT_SOMA_LAW) is None

    def test_the_cycle_atomic_refusal_names_the_signed_interval(self):
        with pytest.raises(SaturatingMembraneRefusalError) as exc:
            refuse_saturating_membrane(
                SomaLaw.resolve(_SYNC_FIRE), mechanism="a decode",
                identity="it assumes a lossless accumulator.")
        message = str(exc.value)
        assert SATURATING_SIGNED_MEMBRANE in message
        assert "[-32768, 32767]" in message
        refuse_saturating_membrane(
            DEFAULT_SOMA_LAW, mechanism="a decode", identity="x")

    def test_the_unsigned_register_keeps_naming_its_own_interval(self):
        with pytest.raises(SaturatingMembraneRefusalError) as exc:
            refuse_saturating_membrane(
                SomaLaw.resolve(_STOCK_ODIN), mechanism="a decode",
                identity="it assumes a lossless accumulator.")
        assert "[0, 255]" in str(exc.value)
        assert SATURATING_UNSIGNED_MEMBRANE in str(exc.value)
