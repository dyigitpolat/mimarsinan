"""[U-HOST2] The host descriptions the toolchain reads, and the one contract a description is added by. [U-HOST3] The
held store a sealed physics' description origin is judged against.

A host description is ``<name>.json`` beside ``<name>.md`` (``hosts.description``). The shipped ones sit in this
package's directory; any other enters through ``register_host_description`` (a vendor's description, or a test's own
description of test numbers), which is the same contract: the file pair, judged whole by ``host_description`` before
it is admitted, its name unique among the shipped and the registered descriptions. ``unregister_host_description``
takes a registered one out again (a shipped one cannot be). The admission rule (``hosts.admission``) reads the rows of
exactly these descriptions, so a host row is admitted only as a row some read description declares.

A description registered while a process runs is read at once, never after a restart. [U-HOST3] A reader keeps what it
has read by digest, so nothing it keeps outlives the bytes it was read from (revision 2's registration counter is
withdrawn: the digest names the bytes).

[U-HOST3] A description's origin (``HostOrigin``) is its name, the path it was read from, the SHA-256 of its JSON
bytes (the file that states every row, mode, energy plane and link end; the ``.md`` beside it is prose and prices
nothing) and its kind, ``shipped`` or ``registered``. A physics whose rows name a host seals the origin of the
description they were judged against, and a sealed physics is judged at read-back against the description of that
digest, held by digest (``held_description``), never against whatever description holds the name now. One rule for
both kinds, held in two places: every version of a shipped description sits in this package's ``held/`` directory as
``<sha256>.json``, byte for byte (content-addressed; a held file is never removed), and a registered description is
held by the registry while it stays registered, its file read again at every read and refused by name once its bytes
are no longer the ones it was registered with. A description not held refuses by name. A description's name is one
token (an identity ``'<description> <mode>'`` is split at its first space), so a name no identity can name refuses.
[U-HOST4] A description is identified by its digest (under its kind and name), never by its path: origins compare,
collect and count by kind, name and digest, so one digest held at two paths is one description; the path stays in
the record as stated text, where the description was read.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal

_SHIPPED = Path(__file__).resolve().parent
#: [U-HOST3] The shipped descriptions' held store: every version shipped, as ``<sha256>.json``.
_HELD = _SHIPPED / "held"
#: [U-HOST3] Where a shipped description is stated to be read from: its directory under the source root, so a sealed
#: record's bytes never depend on where a checkout lies.
SHIPPED_DIRECTORY = _SHIPPED.relative_to(_SHIPPED.parents[3]).as_posix()
SHIPPED, REGISTERED = "shipped", "registered"
ORIGIN_KINDS = (SHIPPED, REGISTERED)
#: [U-HOST3] The closed fields of a sealed origin.
ORIGIN_FIELDS = ("kind", "name", "path", "sha256")
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: Each registered description: its file and the SHA-256 of the bytes judged at registration.
_REGISTERED: Dict[str, Tuple[Path, str]] = {}
#: The shipped directory's listing, keyed by the directory and its modification stamp.
_LISTING: Dict[Tuple[str, int], Dict[str, Path]] = {}
#: Each read file's bytes and digest, keyed by its path and stamp (a file is hashed again whenever it changes).
_READ: Dict[Tuple[str, int, int], Tuple[bytes, str]] = {}


class HostRegistryError(ToolchainRefusal, ValueError):
    """A registration that does not stand: no file pair, a name already read, an unknown name
    to take out, or [U-HOST3] a registered file whose bytes are no longer the ones registered."""


class HostOriginError(ToolchainRefusal, ValueError):
    """[U-HOST3] A sealed description origin outside its form, or one whose description is not held, refused by
    name."""


@dataclass(frozen=True)
class HostOrigin:
    """[U-HOST3] Which description a physics' host rows were judged against: name, path, digest and kind. [U-HOST4]
    Its identity is its kind, name and digest; the path is a statement, never compared."""

    name: str
    path: str = field(compare=False)
    sha256: str
    kind: str

    @property
    def label(self) -> str:
        return f"{self.kind} host description {self.name!r} ({self.path}, sha256 {self.sha256})"

    def to_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "name": self.name, "path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: Any) -> "HostOrigin":
        """A sealed origin, judged on its form (closed fields, a kind, a digest, a path, a name an identity names)."""
        if not isinstance(data, Mapping) or sorted(data) != list(ORIGIN_FIELDS):
            raise HostOriginError(f"a sealed host origin states exactly {list(ORIGIN_FIELDS)}, got {data!r}")
        if data["kind"] not in ORIGIN_KINDS:
            raise HostOriginError(f"a sealed host origin's kind is one of {list(ORIGIN_KINDS)}, got {data['kind']!r}")
        if not isinstance(data["sha256"], str) or not _SHA256.fullmatch(data["sha256"]):
            raise HostOriginError(f"a sealed host origin's sha256 is 64 lowercase hex digits, got {data['sha256']!r}")
        if not isinstance(data["path"], str) or not data["path"]:
            raise HostOriginError(f"a sealed host origin states the path it was read from, got {data['path']!r}")
        if (why := name_refusal(data["name"])) is not None:
            raise HostOriginError(f"a sealed host origin: {why}")
        return cls(name=data["name"], path=data["path"], sha256=data["sha256"], kind=data["kind"])


def name_refusal(name: Any) -> Optional[str]:
    """[U-HOST3] Why ``name`` cannot name a host description, or None: one non-empty token, no whitespace."""
    if not isinstance(name, str) or name.split() != [name]:  # "" splits to [], never [""]
        return (f"a host description's name is one token (an identity '<description> <mode>' is split at its first "
                f"space), got {name!r}")
    return None


def _read(path: Path) -> Tuple[bytes, str]:
    """``path``'s bytes and their SHA-256, hashed again whenever the file's stamp changes."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _READ:
        data = path.read_bytes()
        _READ[key] = (data, hashlib.sha256(data).hexdigest())
    return _READ[key]


def description_paths() -> Dict[str, Path]:
    """Every readable description, by name: the shipped ones (listed again whenever the directory changes, so a file
    added while a process runs is read) and the registered ones."""
    stamp = (str(_SHIPPED), _SHIPPED.stat().st_mtime_ns)
    if _LISTING.get(stamp) is None:
        _LISTING.clear()
        _LISTING[stamp] = {path.stem: path for path in sorted(_SHIPPED.glob("*.json"))}
    return {**_LISTING[stamp], **{name: path for name, (path, _digest) in _REGISTERED.items()}}


def description_path(name: str) -> Optional[Path]:
    """The description file of ``name``, or None when no read description has that name."""
    return description_paths().get(str(name))


def _registered_bytes(name: str) -> bytes:
    """[U-HOST3] A registered description's bytes, which must still be the ones it was registered with."""
    path, digest = _REGISTERED[name]
    if not path.is_file():
        raise HostRegistryError(f"registered host description {name!r}: its file {path} is gone (registered with "
                                f"sha256 {digest}); a registered description is read from the file it was registered "
                                "with")
    data, now = _read(path)
    if now != digest:
        raise HostRegistryError(f"registered host description {name!r}: its file {path} changed after registration "
                                f"(sha256 {now}, registered {digest}); register it again")
    return data


def read_description(name: str) -> Optional[Tuple[Path, bytes, HostOrigin]]:
    """[U-HOST3] The description readable now by ``name``: its file, its bytes and its origin (None: no read
    description has that name); a registered one whose file no longer holds its registered bytes refuses by name."""
    name = str(name)
    if name in _REGISTERED:
        path, digest = _REGISTERED[name]
        return path, _registered_bytes(name), HostOrigin(name, str(path), digest, REGISTERED)
    path = description_path(name)
    if path is None or not path.is_file():
        return None
    data, digest = _read(path)
    return path, data, HostOrigin(name, f"{SHIPPED_DIRECTORY}/{path.name}", digest, SHIPPED)


def held_description(origin: HostOrigin) -> bytes:
    """[U-HOST3] The JSON bytes held under ``origin``'s digest for its kind and name: a shipped one from the held
    store, a registered one from the registry while it stays registered; a description not held refuses by name."""
    if origin.kind == REGISTERED:
        entry = _REGISTERED.get(origin.name)
        if entry is None or entry[1] != origin.sha256:
            raise HostOriginError(f"{origin.label} is not held: no description of that name and digest is registered "
                                  f"in this process (registered: {sorted(_REGISTERED)}); register the description "
                                  "it was sealed under")
        return _registered_bytes(origin.name)
    held = _HELD / f"{origin.sha256}.json"
    if not held.is_file():
        raise HostOriginError(f"{origin.label} is not held: the shipped held store has no {held.name} (every "
                              "version of a shipped description is held there, byte for byte)")
    data, digest = _read(held)
    if digest != origin.sha256:
        raise HostOriginError(f"the held file {held.name} hashes to {digest}, not its name")
    return data


def register_host_description(path: Path, judge: Callable[[str], object]) -> str:
    """Register the description ``path`` (``<name>.json`` beside the ``.md`` it names) and return its name, after
    ``judge`` (``hosts.description.host_description``) has read it whole; a refusal of the judge leaves it unregistered.
    A name the shipped or registered descriptions already use refuses by name; [U-HOST3] so does one no identity can
    name (the judge's one rule for every description, ``name_refusal``). The bytes judged are held by their digest
    while the description stays registered."""
    path = Path(path).resolve()
    if path.suffix != ".json" or not path.is_file():
        raise HostRegistryError(f"a host description is a <name>.json file beside its <name>.md; got {str(path)!r}")
    name = path.stem
    if name in description_paths():
        raise HostRegistryError(f"host description {name!r} is already read (from {description_paths()[name]}); a "
                                "name names one description")
    _REGISTERED[name] = (path, _read(path)[1])
    judged = False
    try:
        judge(name)
        judged = True
    finally:
        if not judged:  # the judge refused (its refusal propagates): the description stays unregistered
            del _REGISTERED[name]
    return name


def unregister_host_description(name: str) -> None:
    """Take the registered description ``name`` out; a shipped or unknown name refuses by name."""
    if name not in _REGISTERED:
        raise HostRegistryError(f"no registered host description {name!r}; registered: {sorted(_REGISTERED)} (a "
                                "shipped description cannot be taken out)")
    del _REGISTERED[name]
