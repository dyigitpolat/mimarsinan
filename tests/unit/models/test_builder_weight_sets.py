"""Builders DECLARE their pretrained weight sets; the framework never enumerates.

The torchvision facts (dataset, class count, native geometry, top-1, licence,
parameter count, recipe) are READ FROM the torchvision weight enum, so no number
is hand-copied and none can drift. Every registered set must be loadable by the
builder that declares it.
"""

import pytest

torchvision = pytest.importorskip("torchvision")

from mimarsinan.common.workload_profile import PretrainedWeightSet  # noqa: E402
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry  # noqa: E402

# model_type -> the weight-set ids its builder registers, in declaration order.
EXPECTED_SETS = {
    "torch_resnet50": ("imagenet1k_v1", "imagenet1k_v2"),
    "torch_vit": ("imagenet1k_v1", "imagenet1k_swag_linear_v1"),
    "torch_squeezenet11": ("imagenet1k_v1",),
    "torch_vgg16": ("imagenet1k_v1",),
}


def _sets(model_type):
    profile = ModelRegistry.get_workload_profile(model_type)
    assert profile is not None, model_type
    return profile.pretrained_weight_sets


class TestBuilderRegistrations:
    @pytest.mark.parametrize("model_type,ids", sorted(EXPECTED_SETS.items()))
    def test_builder_registers_its_weight_sets(self, model_type, ids):
        assert tuple(s.id for s in _sets(model_type)) == ids

    @pytest.mark.parametrize("model_type", sorted(EXPECTED_SETS))
    def test_every_set_declares_the_full_contract(self, model_type):
        for weight_set in _sets(model_type):
            assert isinstance(weight_set, PretrainedWeightSet)
            assert weight_set.task and weight_set.dataset and weight_set.label
            assert weight_set.source == "torchvision"
            assert weight_set.num_classes == 1000
            assert len(weight_set.input_shape) == 3
            assert 0.0 < (weight_set.expected_accuracy or 0.0) < 1.0
            assert weight_set.license
            assert (weight_set.num_parameters or 0) > 0
            assert weight_set.preprocessing

    @pytest.mark.parametrize("model_type", sorted(EXPECTED_SETS))
    def test_torchvision_sets_declare_the_adaptation_they_implement(self, model_type):
        """These builders project conv weights onto the provider's channel count
        and let the strategy skip the shape-mismatched head, so a geometry or
        class-count difference is an adaptation, not an incompatibility."""
        for weight_set in _sets(model_type):
            assert weight_set.adapts_input_shape is True
            assert weight_set.adapts_num_classes is True

    def test_facts_are_read_from_the_torchvision_enum_not_hand_copied(self):
        import torchvision.models as models

        by_id = {s.id: s for s in _sets("torch_resnet50")}
        for name in ("IMAGENET1K_V1", "IMAGENET1K_V2"):
            member = models.ResNet50_Weights[name]
            metrics = member.meta["_metrics"]["ImageNet-1K"]
            weight_set = by_id[name.lower()]
            assert weight_set.expected_accuracy == pytest.approx(metrics["acc@1"] / 100.0)
            assert weight_set.num_parameters == member.meta["num_params"]
            assert weight_set.dataset == "ImageNet-1K"
            crop = int(member.transforms().crop_size[0])
            assert weight_set.input_shape == (3, crop, crop)

    def test_a_builder_without_pretrained_weights_registers_an_empty_set(self):
        profile = ModelRegistry.get_workload_profile("lenet5")
        assert profile is not None, "a known builder always has a registration"
        assert profile.pretrained_weight_sets == ()
        assert profile.config_updates()["pretrained_weight_sets"] == []

    def test_unknown_model_type_has_no_registration(self):
        assert ModelRegistry.get_workload_profile("nope_not_a_builder") is None

    def test_the_first_registered_set_is_the_builders_default(self):
        """`weight_source: torchvision` with no pinned set keeps loading the set
        the tier-1/2 corpus has always loaded."""
        assert _sets("torch_resnet50")[0].id == "imagenet1k_v1"
        assert _sets("torch_vit")[0].id == "imagenet1k_v1"


class TestPretrainedFactorySelectsTheSet:
    @pytest.mark.parametrize("model_type,cls_attr", [
        ("torch_resnet50", "ResNet50_Weights"),
        ("torch_vit", "ViT_B_16_Weights"),
        ("torch_squeezenet11", "SqueezeNet1_1_Weights"),
        ("torch_vgg16", "VGG16_BN_Weights"),
    ])
    def test_every_registered_id_names_a_real_torchvision_member(self, model_type, cls_attr):
        import torchvision.models as models

        enum = getattr(models, cls_attr)
        for weight_set in _sets(model_type):
            assert weight_set.id.upper() in enum.__members__

    def test_factory_rejects_an_unregistered_set_id(self):
        builder = ModelRegistry.get_builder_cls("torch_squeezenet11")(
            device="cpu", input_shape=(3, 32, 32), num_classes=10, pipeline_config={},
        )
        with pytest.raises(ValueError, match="not registered"):
            builder.get_pretrained_factory("imagenet1k_v99")
