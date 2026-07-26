"""PackagingContract SSOT: what may be packaged onto a core, per deployment family."""

import dataclasses

import pytest

from mimarsinan.mapping.platform.packaging_contract import (
    MVM_PACKAGING,
    SPIKING_PACKAGING,
    PackagingContract,
    packaging_contract_for,
)


class _Plan:
    def __init__(self, is_mvm):
        self.is_mvm = is_mvm


class TestContracts:
    def test_spiking_contract_reproduces_todays_rule(self):
        c = SPIKING_PACKAGING
        assert c.kinds == frozenset({"perceptron"})
        assert c.absorb_normalization is True
        assert c.absorb_activation is True
        assert c.require_activation is True
        assert c.boundary.domain == "event"
        assert c.boundary.signed is False

    def test_mvm_contract_is_affine_only(self):
        c = MVM_PACKAGING
        assert c.kinds == frozenset({"affine"})
        assert c.absorb_normalization is True
        assert c.absorb_activation is False
        assert c.require_activation is False
        assert c.boundary.domain == "value"
        assert c.boundary.signed is True
        # Deferred quantized-I/O cell: the seam exists, v1 float passthrough.
        assert c.boundary.io_quantization == "none"

    def test_contracts_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            SPIKING_PACKAGING.require_activation = False  # type: ignore[misc]


class TestDerivation:
    def test_spiking_plan_gets_spiking_contract(self):
        assert packaging_contract_for(_Plan(is_mvm=False)) is SPIKING_PACKAGING

    def test_mvm_plan_gets_mvm_contract(self):
        assert packaging_contract_for(_Plan(is_mvm=True)) is MVM_PACKAGING

    def test_real_plans_dispatch(self):
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

        spiking = DeploymentPlan.resolve({})
        mvm = DeploymentPlan.resolve({"core_semantics": "mvm"})
        assert packaging_contract_for(spiking) is SPIKING_PACKAGING
        assert packaging_contract_for(mvm) is MVM_PACKAGING

    def test_value_boundary_is_the_mvm_signature(self):
        assert PackagingContract.__dataclass_fields__["boundary"] is not None
        assert MVM_PACKAGING.is_value_domain
        assert not SPIKING_PACKAGING.is_value_domain


class TestDeferredSurfaces:
    """[mvm W4] the deferred features' seams exist and are typed NOW.

    Quantized-I/O rides ``BoundarySpec.io_quantization``; chip-native op
    kinds ride the ``kinds`` frozenset (consumed by code with the first
    non-affine kind — until then this pins the declared surface)."""

    def test_package_kind_constants_are_the_registry_seam(self):
        from mimarsinan.mapping.platform.packaging_contract import (
            PACKAGE_KIND_AFFINE,
            PACKAGE_KIND_PERCEPTRON,
        )

        assert PACKAGE_KIND_PERCEPTRON in SPIKING_PACKAGING.kinds
        assert PACKAGE_KIND_AFFINE in MVM_PACKAGING.kinds

    def test_io_quantization_seam_is_declared(self):
        from mimarsinan.mapping.platform.packaging_contract import BoundarySpec

        grid = BoundarySpec(domain="value", signed=True, io_quantization="grid")
        assert grid.io_quantization == "grid"
        assert MVM_PACKAGING.boundary.io_quantization == "none"
