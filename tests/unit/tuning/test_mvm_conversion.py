"""The mvm (value-domain) conversion recipe: WQ machinery only, no spiking knobs."""

from mimarsinan.tuning.orchestration.conversion_policy import (
    OPTIMIZATION_DRIVER_FAST,
    ConversionRecipe,
)
from mimarsinan.tuning.orchestration.mvm_conversion import derive_mvm_recipe


class TestMvmRecipe:
    def test_shape(self):
        recipe = derive_mvm_recipe()
        assert isinstance(recipe, ConversionRecipe)
        assert recipe.driver == OPTIMIZATION_DRIVER_FAST
        assert recipe.special_case == "mvm_value_domain"
        assert recipe.rationale

    def test_knobs_are_exactly_the_wq_family(self):
        knobs = derive_mvm_recipe().knobs
        assert knobs == {
            "wq_fast_rates": [0.5, 1.0],
            "wq_fast_steps_per_rate": 0,
            "wq_endpoint_recovery_steps": 600,
        }

    def test_all_spiking_backends_disabled(self):
        assert derive_mvm_recipe().sim_enables == {
            "enable_nevresim_simulation": False,
            "enable_sanafe_simulation": False,
            "enable_loihi_simulation": False,
        }

    def test_recipe_is_deterministic(self):
        assert derive_mvm_recipe() == derive_mvm_recipe()
