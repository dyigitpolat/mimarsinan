"""WHICH FABRIC a declared envelope or a sealed bundle NAMES.

The chip table (``chip_configs.py``) says what each fabric IS; this module is
the one place that answers which one a set of claims identifies.
``ChipConfig.bundle_claims`` / ``claims_of_bundle`` is that predicate — a weight
width, a sign granularity, an effective fan-in and a membrane width together
name at most one configuration this build can produce — and it is what lets a
DECLARED platform and a SEALED bundle reach the same fabric without either of
them carrying a name to be trusted.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from mimarsinan.chip_simulation.odin_fpga.chip_configs import (
    ChipConfig,
    ChipConfigError,
    chip_configs,
)


def chip_config_of_claims(claims: Mapping[str, Any]) -> ChipConfig:
    """The fabric whose IDENTITY is exactly these claims, or a refusal."""
    for config in chip_configs():
        if config.bundle_claims() == dict(claims):
            return config
    raise ChipConfigError(
        f"no chip configuration this build carries has the identity "
        f"{dict(claims)}; the fabrics are "
        + "; ".join(f"{c.name} {c.bundle_claims()}" for c in chip_configs())
        + ". Declare a platform one of them can be, or add a profile to "
          "chip_configs.py (and scripts/hacc/chips.sh) for the fabric you mean.")


def chip_config_for(*, weight_bits: int, weight_sign_granularity: str,
                    effective_max_axons: int, membrane_bits: int) -> ChipConfig:
    """The fabric a DECLARED platform envelope resolves to."""
    return chip_config_of_claims({
        "weight_bits": int(weight_bits),
        "weight_sign_granularity": str(weight_sign_granularity),
        "effective_max_axons": int(effective_max_axons),
        "membrane_bits": int(membrane_bits),
    })


def chip_config_of_bundle(document: Mapping[str, Any]) -> ChipConfig:
    """The fabric one SEALED bundle was mapped against, read off its claims."""
    return chip_config_of_claims(
        ChipConfig.claims_of_bundle(document["chip_config"]))


def named_chip_of_bundle(document: Mapping[str, Any]) -> Optional[str]:
    """The fabric a bundle NAMES, or ``None`` when its claims name none.

    A bundle mapped against a platform narrower than any built fabric (the
    micro fixtures are) is perfectly executable on a wider one, so an unnamed
    fabric is a fact to report rather than a refusal. Packaging, which has to
    choose a bitstream to build, is where it becomes one.
    """
    try:
        return chip_config_of_bundle(document).name
    except ChipConfigError:
        return None
