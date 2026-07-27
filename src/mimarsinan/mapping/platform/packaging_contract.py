"""Packaging-contract SSOT: what a target's cores accept and what package boundaries mean."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

PACKAGE_KIND_PERCEPTRON = "perceptron"  # MM (+BN) + activation, event I/O
PACKAGE_KIND_AFFINE = "affine"          # MM (+BN) only, value I/O

BOUNDARY_DOMAIN_EVENT = "event"
BOUNDARY_DOMAIN_VALUE = "value"


@dataclass(frozen=True)
class BoundarySpec:
    """The value exchange at a package boundary.

    ``io_quantization`` is the deferred quantized-I/O seam: ``"none"`` = float
    passthrough (v1), ``"grid"`` arms boundary quantizers + the int-exact twin.
    """

    domain: str
    signed: bool
    io_quantization: str = "none"


@dataclass(frozen=True)
class PackagingContract:
    """The declarative packaging rule a converter/lowering pass consults.

    Spiking targets must pair every package with a realizable activation
    (``require_activation``); value-domain MVM targets package any
    weight-stationary affine op and keep activations on the host.
    """

    kinds: FrozenSet[str]
    absorb_normalization: bool
    absorb_activation: bool
    require_activation: bool
    boundary: BoundarySpec

    @property
    def is_value_domain(self) -> bool:
        return self.boundary.domain == BOUNDARY_DOMAIN_VALUE


SPIKING_PACKAGING = PackagingContract(
    kinds=frozenset({PACKAGE_KIND_PERCEPTRON}),
    absorb_normalization=True,
    absorb_activation=True,
    require_activation=True,
    boundary=BoundarySpec(domain=BOUNDARY_DOMAIN_EVENT, signed=False),
)

MVM_PACKAGING = PackagingContract(
    kinds=frozenset({PACKAGE_KIND_AFFINE}),
    absorb_normalization=True,
    absorb_activation=False,
    require_activation=False,
    boundary=BoundarySpec(domain=BOUNDARY_DOMAIN_VALUE, signed=True),
)


def packaging_contract_for(plan) -> PackagingContract:
    """THE plan → packaging-contract dispatch (domain-first, no mode ladders)."""
    return MVM_PACKAGING if plan.is_mvm else SPIKING_PACKAGING
