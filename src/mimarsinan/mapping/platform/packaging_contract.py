"""Packaging-contract SSOT: what a target's cores accept and what package boundaries mean, and the run's conversion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, FrozenSet, Mapping

PACKAGE_KIND_PERCEPTRON = "perceptron"  # MM (+BN) + activation, event I/O
PACKAGE_KIND_AFFINE = "affine"          # MM (+BN) only, value I/O

BOUNDARY_DOMAIN_EVENT = "event"
BOUNDARY_DOMAIN_VALUE = "value"

# How the boundary exchanges values — and therefore which certificate class
# judges the model↔program edge: a float boundary is a continuous relation
# (numeric tolerance), a grid boundary is PIECEWISE CONSTANT (grid units +
# exact decisions; a scalar atol would measure grid-edge chaos instead).
BOUNDARY_IO_FLOAT = "none"
BOUNDARY_IO_GRID = "grid"


@dataclass(frozen=True)
class BoundarySpec:
    """The value exchange at a package boundary.

    ``io_quantization`` selects the exchange: ``BOUNDARY_IO_FLOAT`` = float
    passthrough, ``BOUNDARY_IO_GRID`` = calibrated value-grid quantizers at
    every host→chip entry (armed by a platform's ``activation_bits``).
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

    @property
    def boundary_is_gridded(self) -> bool:
        """Whether the boundary quantizes — THE certificate-class question."""
        return self.boundary.io_quantization == BOUNDARY_IO_GRID


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
    """THE plan → packaging-contract dispatch (domain-first, no mode ladders).

    A value-domain platform that declares ``activation_bits`` realizes the
    gridded boundary; everything downstream (executor snap, certificate
    class) reads that from the contract rather than re-deriving it.
    """
    if not plan.is_mvm:
        return SPIKING_PACKAGING
    if not plan.activation_quantization:
        return MVM_PACKAGING
    return replace(
        MVM_PACKAGING,
        boundary=replace(MVM_PACKAGING.boundary, io_quantization=BOUNDARY_IO_GRID),
    )


@dataclass(frozen=True)
class DeployedConversion:
    """How the deployment converts the run's native model: the plan's packaging contract and its leaf decomposition.

    The ONE object every conversion of a run's native model reads: the Torch Mapping step's conversion, the static
    on-chip estimators that twin it (the Model Building gate and its span report) and the search route's candidates
    (sign-off M10), so a static reading measures the program the pipeline deploys, never another contract's."""

    packaging: PackagingContract = SPIKING_PACKAGING
    decompose_leaf_modules: bool = False


def deployed_conversion_of(plan, config: Mapping[str, Any]) -> DeployedConversion:
    """The run's conversion, resolved once from its plan (``packaging_contract_for``) and its config's leaf
    decomposition (absent: false)."""
    return DeployedConversion(packaging=packaging_contract_for(plan),
                              decompose_leaf_modules=bool(config.get("decompose_leaf_modules", False)))
