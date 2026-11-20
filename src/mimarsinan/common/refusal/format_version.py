"""[U-HOST3] The version rule's one reading: a format_version is an exact integer version (an int, never a bool, a float
or a string), else it refuses by name.

Every format the toolchain seals or ships states its version, and "a wrong version is refused, never tolerated
silently" (deployment_record_schema.md section 1). A reader that takes ``int(version)`` or compares ``version ==
expected`` reads 1.5, 1.0, "1" or True as version 1, so a record or file of another version is read under the wrong
rule. Each reader of a version reads it through ``exact_version`` (refusing in this rule's words) or ``is_version``
(for a reader that refuses in its own words).
"""

from __future__ import annotations

from typing import Any

from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal


class FormatVersionError(ToolchainRefusal, ValueError):
    """A format_version that is no exact integer version, or not the version its reader reads, refused by name."""


def exact_version(version: Any, where: str) -> int:
    """``version`` as the version rule reads it: an int, never a bool, a float or a string; else refused by name."""
    if type(version) is not int:
        raise FormatVersionError(f"{where} format_version {version!r} is not an integer version; the version rule "
                                 "reads exact integer versions only")
    return version


def is_version(version: Any, expected: int) -> bool:
    """Whether ``version`` is exactly the integer version ``expected``."""
    return type(version) is int and version == expected
