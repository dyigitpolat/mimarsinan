"""Conversion models: how many ADC conversions a target's dataflow really needs."""

import pytest

from mimarsinan.deployment_record.platform_physics.conversion import (
    CONVERSION_MODELS,
    ConversionModel,
    conversion_model_for,
)
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue


def _quantities(**values):
    return Quantities({
        key: QuantityValue(float(v), "static") for key, v in values.items()
    })


class TestTheRegistry:
    def test_the_digital_model_is_the_default(self):
        assert conversion_model_for(None).name == "digital"
        assert conversion_model_for({}).name == "digital"

    def test_an_unknown_model_raises_naming_the_known_ones(self):
        with pytest.raises(KeyError) as excinfo:
            conversion_model_for({"model": "quantum_hand_waving"})
        for name in CONVERSION_MODELS:
            assert name in str(excinfo.value)

    def test_every_registered_model_documents_itself(self):
        for name, model in CONVERSION_MODELS.items():
            assert model.name == name
            assert len(model.doc.strip()) >= 40


class TestTheDigitalModel:
    """A digital target converts nothing — and says zero rather than nothing."""

    def test_it_declares_zero_conversions(self):
        model = conversion_model_for(None)
        derived = model.derive(_quantities(macs=1e6, cells_physical=65536))
        assert derived["adc_conversions"].value == 0.0
        assert derived["adc_count"].value == 0.0

    def test_zero_is_a_fact_not_an_absence(self):
        """The distinction the whole program rests on: a digital chip HAS no ADC,
        which is different from a target that has not said."""
        derived = conversion_model_for(None).derive(_quantities(macs=1e6))
        assert derived["adc_conversions"].value == 0.0
        assert "adc_conversions" in derived

    def test_it_needs_no_quantities_at_all(self):
        assert conversion_model_for(None).derive(_quantities())["adc_count"].value == 0.0


class TestTheBitSlicedCrossbarModel:
    def _model(self, **params):
        spec = {
            "model": "bit_sliced_crossbar",
            "array_rows": 128, "array_cols": 128,
            "adc_sharing_factor": 8, "input_bits": 16,
            **params,
        }
        return conversion_model_for(spec)

    def test_conversions_are_column_integrations_times_input_slices(self):
        """Each group of array_rows MACs is one column integration; a bit-serial
        input converts once per slice."""
        derived = self._model().derive(_quantities(macs=128 * 100))
        assert derived["adc_conversions"].value == pytest.approx(100 * 16)

    def test_a_partial_column_still_costs_a_conversion(self):
        """Half a column is still one integration and one conversion — rounding it
        down would price a read the chip performs at zero."""
        derived = self._model().derive(_quantities(macs=64))
        assert derived["adc_conversions"].value == pytest.approx(1 * 16)

    def test_the_converter_count_follows_the_sharing_factor(self):
        """One ADC per `adc_sharing_factor` columns, per array."""
        derived = self._model().derive(_quantities(cells_physical=128 * 128 * 4))
        assert derived["adc_count"].value == pytest.approx(4 * (128 / 8))

    def test_fewer_shared_columns_means_more_converters(self):
        cells = _quantities(cells_physical=128 * 128)
        many = self._model(adc_sharing_factor=1).derive(cells)["adc_count"].value
        few = self._model(adc_sharing_factor=16).derive(cells)["adc_count"].value
        assert many == 16 * few

    def test_more_input_bits_means_more_conversions(self):
        macs = _quantities(macs=128 * 10)
        low = self._model(input_bits=1).derive(macs)["adc_conversions"].value
        high = self._model(input_bits=8).derive(macs)["adc_conversions"].value
        assert high == 8 * low

    def test_it_is_modeled_not_measured(self):
        """A conversion count rests on the target's DECLARED dataflow, so it must
        travel as modeled — never as something the run measured."""
        derived = self._model().derive(_quantities(macs=1e6, cells_physical=1e6))
        assert all(v.provenance == "modeled" for v in derived.values())

    def test_a_missing_multiplicand_yields_no_claim(self):
        """No MAC census means no conversion count — not a zero, which would price
        an analog chip's conversions at nothing."""
        derived = self._model().derive(_quantities(cells_physical=16384))
        assert "adc_conversions" not in derived
        assert "adc_count" in derived

    def test_an_incomplete_declaration_raises_at_construction(self):
        with pytest.raises(ValueError, match="array_rows"):
            conversion_model_for({"model": "bit_sliced_crossbar", "array_cols": 128})

    def test_a_nonsense_parameter_raises(self):
        with pytest.raises(ValueError, match="positive"):
            conversion_model_for({
                "model": "bit_sliced_crossbar", "array_rows": 0, "array_cols": 128,
                "adc_sharing_factor": 8, "input_bits": 16,
            })

    def test_an_unknown_parameter_raises(self):
        with pytest.raises(ValueError, match="surprise"):
            conversion_model_for({
                "model": "bit_sliced_crossbar", "array_rows": 128, "array_cols": 128,
                "adc_sharing_factor": 8, "input_bits": 16, "surprise": 1,
            })


class TestTheModelIsPartOfTheProfile:
    def test_a_shipped_digital_profile_converts_nothing(self):
        from mimarsinan.deployment_record.platform_physics import get_platform_physics

        for name in ("truenorth", "loihi"):
            model = conversion_model_for(get_platform_physics(name).conversion_model)
            assert model.name == "digital"

    def test_the_model_rides_the_profile_round_trip(self):
        from mimarsinan.deployment_record.platform_physics import profile_from_dict

        physics = profile_from_dict({
            "format_version": 1, "name": "t", "display_name": "T",
            "description_file": "t.md",
            "validity": {"measurement_kind": "projection"},
            "conversion_model": {
                "model": "bit_sliced_crossbar", "array_rows": 128,
                "array_cols": 128, "adc_sharing_factor": 8, "input_bits": 16,
            },
            "constants": {},
        })
        assert physics.conversion_model["model"] == "bit_sliced_crossbar"
        restored = profile_from_dict(physics.to_dict())
        assert restored == physics


class TestIsolation:
    def test_a_model_returns_only_the_quantities_it_derives(self):
        model = conversion_model_for({
            "model": "bit_sliced_crossbar", "array_rows": 64, "array_cols": 64,
            "adc_sharing_factor": 4, "input_bits": 8,
        })
        derived = model.derive(_quantities(macs=1e5, cells_physical=1e5, timesteps=32))
        assert set(derived) <= {"adc_conversions", "adc_count"}

    def test_a_model_never_mutates_the_quantities_it_reads(self):
        model = conversion_model_for(None)
        quantities = _quantities(macs=1.0)
        model.derive(quantities)
        assert list(quantities.keys()) == ["macs"]
