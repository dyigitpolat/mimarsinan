"""Named physics profiles, discovered as data: a vendor adds one by dropping in files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from mimarsinan.deployment_record.platform_physics.loader import load_profile
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics

#: Profiles ship beside the package, so a profile resolves identically from any cwd.
_PROFILES_DIRNAME = "profiles"

_CACHE: Dict[Path, Dict[str, PlatformPhysics]] = {}


def profiles_dir() -> Path:
    """The directory holding the shipped ``<name>.json`` / ``<name>.md`` pairs."""
    return Path(__file__).with_name(_PROFILES_DIRNAME)


def profiles_in(directory: Path) -> Tuple[str, ...]:
    """Profile names available in ``directory``, sorted."""
    return tuple(sorted(path.stem for path in Path(directory).glob("*.json")))


def available_profiles() -> Tuple[str, ...]:
    """Every shipped profile name, sorted."""
    return profiles_in(profiles_dir())


def profile_path(name: str) -> Path:
    """The profile JSON for ``name``."""
    return profiles_dir() / f"{name}.json"


def profile_description_path(name: str) -> Path:
    """The prose description file for ``name`` — citations, derivations, rationales."""
    return profiles_dir() / get_platform_physics(name).description_file


def load_profiles_in(directory: Path) -> Dict[str, PlatformPhysics]:
    """Every profile in ``directory``, loaded and validated, cached per directory."""
    directory = Path(directory)
    if directory not in _CACHE:
        _CACHE[directory] = {
            name: load_profile(directory / f"{name}.json")
            for name in profiles_in(directory)
        }
    return _CACHE[directory]


def get_platform_physics(name: str) -> PlatformPhysics:
    """The named profile, or a loud error naming every profile that ships."""
    profiles = load_profiles_in(profiles_dir())
    try:
        return profiles[name]
    except KeyError:
        raise KeyError(
            f"unknown platform physics profile {name!r}; the shipped profiles are "
            f"{sorted(profiles)}"
        ) from None
