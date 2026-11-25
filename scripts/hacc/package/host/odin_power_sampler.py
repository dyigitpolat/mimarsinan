#!/usr/bin/env python3
"""Card power and temperature telemetry for the ODIN power protocol (H0).

WHAT IT READS. The card's own electrical and thermal reports, the ones XRT
builds in ``core/common/sensor.cpp`` (XRT branch 2024.2, the runtime HACC
ships as 2.18.179): ``read_electrical`` fills ``power_consumption_watts``,
``power_consumption_max_watts``, ``power_consumption_warning`` and a
``power_rails`` array of ``{id, description, voltage{volts, is_present},
current{amps, is_present}}``; ``read_thermals`` fills a ``thermals`` array of
``{location_id, description, temp_C, is_present}``. Every value arrives as a
JSON STRING (boost's ``write_json``), ``"N/A"`` where the card gave none, and
the whole tree is ``{"msg": "No sensors present"}`` when there are no sensors.

TWO ROUTES, ONE PARSER. Both reach the same trees, so both feed the same
``parse_report``:

  * route A, ``xrt-smi`` (or the deprecated ``xbutil``): one
    ``examine -d <BDF> -r electrical thermal -f JSON -o <file> --force`` per
    reading, a sidecar process that holds no device handle between reads.
    ``produce_reports`` (``core/tools/common/XBHelpMenus.cpp``) wraps the trees
    as ``{"schema_version": ..., "devices": [{"interface_type": "pcie",
    "device_id": <BDF>, "electrical": {...}, "thermals": [...]}]}``;
  * route B, pyxrt ``device.get_info(xrt_info_device.electrical)`` and
    ``.thermal`` (``core/common/api/xrt_device.cpp``): the same two trees as
    JSON text, unwrapped; ``xrt_info_device.bdf`` names the card.

TOLERANT OF SCHEMA CHANGE, NEVER OF A MISSING READING. The parser walks the
tree instead of indexing fixed paths, so a wrapper added or removed, a value
sent as a number instead of a string, an extra field, a missing rail table or
a missing thermal table all parse. What it REFUSES, by a typed error, is a
report without a board power reading: no power field anywhere, a field that
carries ``"N/A"`` or any non-number, a non-positive value, or one beyond what a
PCIe card can draw. A refused reading is recorded as a refusal, never as zero
and never as an interpolated value.

THE E64 ACCEPTANCE, AS THIS SAMPLER READS IT. E64 asks that fake mode "parses
recorded electrical fixtures and refuses a missing rail". Read here: the board
power reading is required (a report without one is refused, as above, and it
never falls back to the power limit or to a rail sum); rails and thermals are
optional, recorded absent, never zero (no rail table gives ``rails == {}``; a
rail or sensor whose ``is_present`` is false gives ``None`` or no entry, never
0). The parser's fixture is constructed from the XRT 2024.2 sources, not
recorded on a card; the recorded fixture exists only after the pilot, when the
first raw reports of ``probe/`` and ``raw/`` are committed beside it (run_plan.md,
After the Allocation, step 2).

STANDALONE BY CONSTRUCTION, like the driver: standard library only, plus pyxrt
on route B (which ships with XRT). Python 3.8, the board node's interpreter.

    python3 host/odin_power_sampler.py bdf --device-index 0
    python3 host/odin_power_sampler.py probe --bdf 0000:81:00.1 --out results/power/probe
    python3 host/odin_power_sampler.py sample --source xrt-smi --bdf 0000:81:00.1 \\
        --interval-s 0.25 --out results/power/samples.jsonl
    python3 host/odin_power_sampler.py parse <saved report.json> ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

SAMPLES_SCHEMA = "odin_power_samples/1"
PROBE_SCHEMA = "odin_power_probe/1"

#: The board power field XRT 2024.2 writes, then the renamings a later XRT may
#: make. Matched as WHOLE keys, case-insensitive, in this order; the key that
#: was actually read is recorded with every sample. ``power_consumption_max_watts``
#: is the card's power LIMIT and is deliberately not among them.
POWER_ALIASES = (
    "power_consumption_watts", "power_watts", "board_power_watts",
    "total_power_watts",
)
POWER_MAX_ALIASES = ("power_consumption_max_watts", "max_power_watts")

#: Text XRT uses where a sensor gave no value (``"N/A"`` throughout sensor.cpp).
NOT_A_READING = frozenset(("", "n/a", "na", "none", "null", "nan", "-",
                           "unknown", "not available", "not supported"))

#: The most a PCIe accelerator card can draw (slot plus two auxiliary
#: connectors, 75 + 2 x 150 W) with margin. A larger number is a unit or
#: parsing fault, not a reading.
MAX_PLAUSIBLE_W = 400.0

TEMP_KEYS = ("temp_c", "temperature_c", "celsius")
LOCATION_KEYS = ("location_id", "id", "description", "name")

#: XRT's command line, route A. ``{bdf}`` and ``{out}`` are filled per read; the
#: pilot reads ``xrt-smi examine --help`` and can override the whole list with
#: ODIN_SMI_EXAMINE (a JSON list) without touching this file.
SMI_EXAMINE = ("examine", "-d", "{bdf}", "-r", "electrical", "thermal",
               "-f", "JSON", "-o", "{out}", "--force")
SMI_EXAMINE_ENV = "ODIN_SMI_EXAMINE"
SMI_NAMES = ("xrt-smi", "xbutil")

#: Raw reports kept per sampler run: the first good one, and one per schema
#: signature change, so a drifted schema is visible without keeping every read.
RAW_KEEP = 16

#: The clock every sample and every protocol marker is stamped with. On Linux
#: CLOCK_MONOTONIC is one clock for every process on the host, which is what
#: lets a sidecar process's samples be placed inside the executor's windows.
CLOCK = "time.monotonic"

_BDF_CHARS = set("0123456789abcdefABCDEF:.")


class OdinTelemetryError(RuntimeError):
    """The telemetry path refused; the message says what and how to fix it."""


class OdinTelemetryMissing(OdinTelemetryError):
    """The report carries no board power READING (absent, N/A or implausible)."""


class OdinTelemetryMismatch(OdinTelemetryError):
    """The report is about a different card than the one this run uses."""


class OdinTelemetrySourceError(OdinTelemetryError):
    """The reader itself failed: no binary, a non-zero exit, a timeout."""


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def to_number(value: Any) -> Optional[float]:
    """A JSON value as a float, or None when it is not a number at all."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.lower() in NOT_A_READING:
            return None
        # XRT never appends units, but a later schema might ("23.1 W").
        for suffix in (" watts", " watt", " w", "w", " c", "c"):
            if text.lower().endswith(suffix) and text[:-len(suffix)].strip():
                text = text[:-len(suffix)].strip()
                break
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("false", "0", "no", "")


def _walk(node: Any, path: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], Any, Any]]:
    """Every (path, key, value) under ``node``; list items carry the key '[]'."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key, value
            yield from _walk(value, path + (str(key),))
    elif isinstance(node, list):
        for item in node:
            yield path, "[]", item
            yield from _walk(item, path + ("[]",))


def schema_signature(document: Any) -> str:
    """A short digest of the report's key structure (values ignored)."""
    paths = sorted({".".join(path + (str(key),))
                    for path, key, value in _walk(document)
                    if not isinstance(value, (dict, list))})
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()[:12]


def _device_nodes(document: Any) -> List[Tuple[Optional[str], Any]]:
    """The per-device subtrees, each with the BDF it declares (or None)."""
    found: List[Tuple[Optional[str], Any]] = []
    if isinstance(document, dict):
        devices = document.get("devices")
        if isinstance(devices, list) and devices:
            for entry in devices:
                if isinstance(entry, dict):
                    found.append((_bdf_of(entry), entry))
            if found:
                return found
    return [(_bdf_of(document) if isinstance(document, dict) else None, document)]


def _bdf_of(node: Dict[str, Any]) -> Optional[str]:
    for key in ("device_id", "bdf", "device_bdf"):
        value = node.get(key)
        if isinstance(value, str) and looks_like_bdf(value):
            return normalize_bdf(value)
    return None


def looks_like_bdf(text: str) -> bool:
    text = text.strip().strip("[]")
    parts = text.split(":")
    return (len(parts) in (2, 3) and "." in parts[-1]
            and set(text) <= _BDF_CHARS)


def normalize_bdf(text: str) -> str:
    """'0000:81:00.1' in lower case; a BDF without a domain gets '0000:'."""
    text = text.strip().strip("[]").lower()
    if text.count(":") == 1:
        text = "0000:" + text
    return text


def _messages(node: Any) -> List[str]:
    return [str(value) for _path, key, value in _walk(node)
            if str(key).lower() in ("msg", "error_msg", "message")
            and isinstance(value, (str, int, float))]


def _scalar_entries(node: Any, aliases: Sequence[str]) -> List[Tuple[int, int, str, Any]]:
    """(alias rank, depth, dotted path, value) for every key matching an alias."""
    wanted = [alias.lower() for alias in aliases]
    hits = []
    for path, key, value in _walk(node):
        name = str(key).lower()
        if name in wanted and not isinstance(value, (dict, list)):
            hits.append((wanted.index(name), len(path),
                         ".".join(path + (str(key),)), value))
    hits.sort(key=lambda hit: (hit[0], hit[1], hit[2]))
    return hits


def _rails(node: Any) -> Dict[str, Dict[str, Optional[float]]]:
    rails: Dict[str, Dict[str, Optional[float]]] = {}
    for _path, _key, value in _walk(node):
        if not isinstance(value, dict):
            continue
        voltage, current = value.get("voltage"), value.get("current")
        if not (isinstance(voltage, dict) or isinstance(current, dict)):
            continue
        name = next((str(value[key]) for key in ("id", "description", "name")
                     if isinstance(value.get(key), str)), None)
        if name is None:
            continue
        volts = amps = None
        if isinstance(voltage, dict) and _truthy(voltage.get("is_present", True)):
            volts = to_number(voltage.get("volts", voltage.get("value")))
        if isinstance(current, dict) and _truthy(current.get("is_present", True)):
            amps = to_number(current.get("amps", current.get("value")))
        rails[name] = {
            "volts": volts, "amps": amps,
            "watts": (volts * amps) if volts is not None and amps is not None else None,
        }
    return rails


def _temps(node: Any) -> Dict[str, float]:
    temps: Dict[str, float] = {}
    for _path, _key, value in _walk(node):
        if not isinstance(value, dict):
            continue
        lowered = {str(key).lower(): key for key in value}
        key = next((lowered[name] for name in TEMP_KEYS if name in lowered), None)
        if key is None or not _truthy(value.get("is_present", True)):
            continue
        reading = to_number(value[key])
        if reading is None:
            continue
        name = next((str(value[k]) for k in LOCATION_KEYS
                     if isinstance(value.get(k), str)), f"sensor{len(temps)}")
        temps[name] = reading
    return temps


def parse_report(document: Any, *, bdf: Optional[str] = None) -> Dict[str, Any]:
    """One reading out of an electrical/thermal report, or a typed refusal.

    ``document`` is the report as JSON text or as the decoded tree, in any of
    the shapes the module docstring lists. ``bdf`` pins the card: a report that
    names other devices only is refused, so the sampler can never read the
    second U55C the board VM lists.
    """
    if isinstance(document, (bytes, bytearray)):
        document = document.decode("utf-8", errors="replace")
    if isinstance(document, str):
        try:
            document = json.loads(document)
        except ValueError as exc:
            raise OdinTelemetryMissing(
                f"the report is not JSON ({exc}); the reader returned text "
                f"that is not an electrical report") from exc
    signature = schema_signature(document)
    nodes = _device_nodes(document)
    want = normalize_bdf(bdf) if bdf else None
    if want is not None:
        named = [(found, node) for found, node in nodes if found is not None]
        matching = [(found, node) for found, node in named if found == want]
        if named and not matching:
            raise OdinTelemetryMismatch(
                f"the report describes {sorted(found for found, _ in named)}, "
                f"not {want}: the sampler would be reading another card")
        if matching:
            nodes = matching
    if len(nodes) > 1:
        raise OdinTelemetryMismatch(
            f"the report describes {len(nodes)} devices and no BDF pins one; "
            f"pass the BDF of the card this run uses")
    device_bdf, node = nodes[0]
    hits = _scalar_entries(node, POWER_ALIASES)
    if not hits:
        notes = _messages(node)
        raise OdinTelemetryMissing(
            f"no board power field ({' / '.join(POWER_ALIASES)}) anywhere in "
            f"the report"
            + (f"; the report says {notes}" if notes else "")
            + ". A report without the field is not a reading of zero watts")
    _rank, _depth, power_path, raw = hits[0]
    watts = to_number(raw)
    if watts is None:
        raise OdinTelemetryMissing(
            f"{power_path} carries {raw!r}, which is not a number: the card "
            f"gave no power reading")
    if not 0.0 < watts <= MAX_PLAUSIBLE_W:
        raise OdinTelemetryMissing(
            f"{power_path} = {watts!r} W is outside (0, {MAX_PLAUSIBLE_W:g}] W; "
            f"a powered card cannot read that, so it is a unit or sensor fault")
    limit_hits = _scalar_entries(node, POWER_MAX_ALIASES)
    return {
        "power_w": watts,
        "power_field": power_path,
        "power_max_w": to_number(limit_hits[0][3]) if limit_hits else None,
        "rails": _rails(node),
        "temps_c": _temps(node),
        "bdf": device_bdf,
        "signature": signature,
    }


# ---------------------------------------------------------------------------
# The two routes
# ---------------------------------------------------------------------------


def resolve_smi(explicit: Optional[str] = None) -> str:
    """The XRT command-line tool: xrt-smi first, the deprecated xbutil second."""
    if explicit:
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        raise OdinTelemetrySourceError(f"{explicit} is not an executable")
    roots = [os.environ.get("XILINX_XRT", ""), "/opt/xilinx/xrt"]
    for name in SMI_NAMES:
        for root in roots:
            if root:
                candidate = os.path.join(root, "bin", name)
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return candidate
        found = shutil.which(name)
        if found:
            return found
    raise OdinTelemetrySourceError(
        "neither xrt-smi nor xbutil is on this node ($XILINX_XRT/bin, "
        "/opt/xilinx/xrt/bin, PATH); source /opt/xilinx/xrt/setup.sh first")


def smi_examine_template() -> List[str]:
    declared = os.environ.get(SMI_EXAMINE_ENV)
    if not declared:
        return list(SMI_EXAMINE)
    try:
        template = json.loads(declared)
    except ValueError as exc:
        raise OdinTelemetrySourceError(
            f"{SMI_EXAMINE_ENV} is not a JSON list: {exc}") from exc
    if not (isinstance(template, list) and all(isinstance(x, str) for x in template)):
        raise OdinTelemetrySourceError(f"{SMI_EXAMINE_ENV} must be a JSON list of strings")
    return template


class SmiSource:
    """Route A: one ``xrt-smi examine`` per reading, no handle kept open."""

    name = "xrt-smi"

    def __init__(self, *, bdf: str, scratch: str, binary: Optional[str] = None,
                 timeout_s: float = 30.0) -> None:
        if not bdf:
            raise OdinTelemetrySourceError(
                "route A needs the BDF of the run's card (-d); resolve it with "
                "the 'bdf' command first")
        self.bdf = normalize_bdf(bdf)
        self.binary = resolve_smi(binary)
        self.timeout_s = float(timeout_s)
        os.makedirs(scratch, exist_ok=True)
        self.out = os.path.join(scratch, f"examine_{os.getpid()}.json")
        self.template = smi_examine_template()

    def command(self) -> List[str]:
        return [self.binary] + [part.replace("{bdf}", self.bdf).replace("{out}", self.out)
                                for part in self.template]

    def describe(self) -> Dict[str, Any]:
        return {"source": self.name, "binary": self.binary, "bdf": self.bdf,
                "command": self.command(), "timeout_s": self.timeout_s}

    def read(self) -> str:
        try:
            os.remove(self.out)
        except OSError:
            pass
        try:
            completed = subprocess.run(
                self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise OdinTelemetrySourceError(
                f"{os.path.basename(self.binary)} examine did not answer within "
                f"{self.timeout_s:g} s") from exc
        except OSError as exc:
            raise OdinTelemetrySourceError(f"{self.binary}: {exc}") from exc
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip()[-400:]
            raise OdinTelemetrySourceError(
                f"{os.path.basename(self.binary)} examine exited "
                f"{completed.returncode}: {tail}")
        if os.path.isfile(self.out):
            with open(self.out, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()
        # Some XRT builds print the JSON instead of writing -o; take it if it is JSON.
        text = completed.stdout or ""
        start = text.find("{")
        if start >= 0:
            return text[start:]
        raise OdinTelemetrySourceError(
            f"{os.path.basename(self.binary)} examine wrote no {self.out} and "
            f"printed no JSON")

    def close(self) -> None:
        try:
            os.remove(self.out)
        except OSError:
            pass


class PyxrtSource:
    """Route B: ``device.get_info`` on the electrical and thermal keys."""

    name = "pyxrt"

    def __init__(self, *, device_index: int = 0, bdf: Optional[str] = None,
                 fake_pyxrt: Optional[str] = None,
                 plant: Optional[Dict[str, Any]] = None) -> None:
        import odin_board_driver as driver  # the one pyxrt loader, fakes included
        try:
            self._xrt = driver.load_pyxrt(fake_pyxrt)
        except driver.OdinDriverError as exc:
            raise OdinTelemetrySourceError(str(exc)) from exc
        if plant is not None and hasattr(self._xrt, "arm"):
            self._xrt.arm(plant=plant)
        self.fake = bool(fake_pyxrt)
        self.device_index = int(device_index)
        self._device: Any = None
        try:
            self._info = self._xrt.xrt_info_device
            self._device = self._xrt.device(self.device_index)
            self.bdf = normalize_bdf(str(self._device.get_info(self._info.bdf)))
        except Exception as exc:  # the binding's own errors are not ours to type
            raise OdinTelemetrySourceError(
                f"pyxrt device {self.device_index}: {type(exc).__name__}: {exc}") from exc
        if bdf and normalize_bdf(bdf) != self.bdf:
            raise OdinTelemetryMismatch(
                f"pyxrt device index {self.device_index} is {self.bdf}, not the "
                f"pinned {normalize_bdf(bdf)}")

    def describe(self) -> Dict[str, Any]:
        return {"source": self.name, "device_index": self.device_index,
                "bdf": self.bdf, "fake": self.fake}

    def read(self) -> str:
        try:
            electrical = self._device.get_info(self._info.electrical)
            thermal = self._device.get_info(self._info.thermal)
        except Exception as exc:
            raise OdinTelemetrySourceError(
                f"pyxrt get_info: {type(exc).__name__}: {exc}") from exc
        document: Dict[str, Any] = {"device_id": self.bdf}
        for key, text in (("electrical", electrical), ("thermal", thermal)):
            try:
                document[key] = json.loads(str(text))
            except ValueError:
                document[key] = {"msg": str(text)}
        return json.dumps(document, sort_keys=True)

    def close(self) -> None:
        self._device = None


def open_source(name: str, *, bdf: Optional[str], device_index: int,
                scratch: str, fake_pyxrt: Optional[str] = None,
                plant: Optional[Dict[str, Any]] = None,
                smi_binary: Optional[str] = None) -> Any:
    if name in ("xrt-smi", "xbutil", "smi"):
        return SmiSource(bdf=bdf or "", scratch=scratch, binary=smi_binary)
    if name == "pyxrt":
        return PyxrtSource(device_index=device_index, bdf=bdf,
                           fake_pyxrt=fake_pyxrt, plant=plant)
    raise OdinTelemetrySourceError(
        f"unknown telemetry source {name!r}; known: xrt-smi, pyxrt")


def resolve_bdf(*, device_index: int, fake_pyxrt: Optional[str] = None) -> str:
    """The BDF behind an XRT device index, read through pyxrt."""
    return PyxrtSource(device_index=device_index, fake_pyxrt=fake_pyxrt).bdf


# ---------------------------------------------------------------------------
# One reading, the probe, the sampler loop
# ---------------------------------------------------------------------------


def host_stamp() -> Dict[str, Any]:
    return {"hostname": socket.gethostname(), "python": platform.python_version(),
            "platform": platform.platform(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def read_once(source: Any, *, bdf: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    """(row, raw text): a parsed reading or a typed refusal, both stamped."""
    started = time.monotonic()
    utc = time.time()
    raw: Optional[str] = None
    row: Dict[str, Any] = {}
    try:
        raw = source.read()
        row.update(parse_report(raw, bdf=bdf))
        row["ok"] = True
    except OdinTelemetryError as exc:
        row.update({"ok": False, "refusal": type(exc).__name__, "reason": str(exc)})
        if raw is not None:
            try:
                row["signature"] = schema_signature(json.loads(raw))
            except ValueError:
                pass
    ended = time.monotonic()
    row.update({"t_start": started, "t_end": ended,
                "t_mono": 0.5 * (started + ended), "latency_s": ended - started,
                "utc_epoch": utc})
    return row, raw


def probe(sources: Sequence[Any], *, bdf: Optional[str], reads: int,
          out_dir: str) -> Dict[str, Any]:
    """A few reads per route: which one answers with power, and how fast."""
    os.makedirs(out_dir, exist_ok=True)
    report: Dict[str, Any] = {"schema": PROBE_SCHEMA, "host": host_stamp(),
                              "bdf": bdf, "clock": CLOCK, "routes": []}
    for source in sources:
        rows = []
        for index in range(max(1, int(reads))):
            row, raw = read_once(source, bdf=bdf)
            rows.append(row)
            if raw is not None and index == 0:
                with open(os.path.join(out_dir, f"raw_{source.name}.json"), "w",
                          encoding="utf-8") as handle:
                    handle.write(raw)
        good = [row for row in rows if row["ok"]]
        report["routes"].append({
            "describe": source.describe(),
            "reads": len(rows), "ok": len(good),
            "latency_s_median": statistics.median(r["latency_s"] for r in rows),
            "power_w": [r["power_w"] for r in good],
            "temps_c_first": good[0]["temps_c"] if good else {},
            "refusals": sorted({r.get("refusal", "") + ": " + r.get("reason", "")
                                for r in rows if not r["ok"]}),
            "signature": good[0]["signature"] if good else None,
        })
    usable = [route for route in report["routes"] if route["ok"] == route["reads"]]
    report["chosen"] = (min(usable, key=lambda r: r["latency_s_median"])
                        ["describe"]["source"] if usable else None)
    with open(os.path.join(out_dir, "probe.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


class _Stop:
    def __init__(self) -> None:
        self.requested = False

    def __call__(self, *_args: Any) -> None:
        self.requested = True


def sample(source: Any, *, out_path: str, bdf: Optional[str], interval_s: float,
           duration_s: Optional[float] = None, max_samples: Optional[int] = None,
           stop_file: Optional[str] = None, raw_dir: Optional[str] = None) -> int:
    """Read until told to stop, one JSON line per reading, flushed each time.

    A killed job keeps every line written before it died. Stops on SIGTERM or
    SIGINT, on ``stop_file`` appearing, after ``duration_s`` or after
    ``max_samples`` readings, whichever comes first.
    """
    stop = _Stop()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    raw_dir = raw_dir or os.path.join(os.path.dirname(os.path.abspath(out_path)), "raw")
    os.makedirs(raw_dir, exist_ok=True)
    seen_signatures: set = set()
    kept = 0
    began = time.monotonic()
    count = 0
    with open(out_path, "a", encoding="utf-8") as handle:
        header = {"schema": SAMPLES_SCHEMA, "header": True, "host": host_stamp(),
                  "pid": os.getpid(), "clock": CLOCK, "t_mono": began,
                  "interval_s": float(interval_s), "bdf": bdf,
                  "source": source.describe()}
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        handle.flush()
        next_at = began
        while not stop.requested:
            row, raw = read_once(source, bdf=bdf)
            row["seq"] = count
            signature = row.get("signature")
            if raw is not None and kept < RAW_KEEP and signature not in seen_signatures:
                seen_signatures.add(signature)
                name = f"raw_{count:06d}_{signature or 'unparsed'}.json"
                with open(os.path.join(raw_dir, name), "w", encoding="utf-8") as raw_handle:
                    raw_handle.write(raw)
                row["raw_kept"] = name
                kept += 1
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            count += 1
            if max_samples is not None and count >= max_samples:
                break
            if duration_s is not None and time.monotonic() - began >= duration_s:
                break
            if stop_file and os.path.exists(stop_file):
                break
            next_at += float(interval_s)
            pause = next_at - time.monotonic()
            if pause > 0:
                time.sleep(pause)
            else:
                next_at = time.monotonic()
    source.close()
    return count


def load_samples(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(headers, rows) of a samples file; a torn last line is skipped, not fatal."""
    headers: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            (headers if record.get("header") else rows).append(record)
    return headers, rows


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _plant(text: Optional[str]) -> Optional[Dict[str, Any]]:
    return json.loads(text) if text else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device-index", type=int, default=0)
        p.add_argument("--bdf", default="")
        p.add_argument("--fake-pyxrt", default=None,
                       help="import this module instead of pyxrt (no hardware)")
        p.add_argument("--fake-plant", default=None,
                       help="JSON: the synthetic power plant a FAKE pyxrt answers with")
        p.add_argument("--smi", default=None, help="the xrt-smi/xbutil binary to use")

    bdf = sub.add_parser("bdf", help="print the BDF of an XRT device index")
    common(bdf)
    pro = sub.add_parser("probe", help="read each route a few times, choose one")
    common(pro)
    pro.add_argument("--routes", default="xrt-smi,pyxrt")
    pro.add_argument("--reads", type=int, default=5)
    pro.add_argument("--out", required=True)
    smp = sub.add_parser("sample", help="the sidecar sampler loop")
    common(smp)
    smp.add_argument("--source", default="xrt-smi")
    smp.add_argument("--interval-s", type=float, default=0.25)
    smp.add_argument("--duration-s", type=float, default=None)
    smp.add_argument("--max-samples", type=int, default=None)
    smp.add_argument("--stop-file", default=None)
    smp.add_argument("--out", required=True)
    par = sub.add_parser("parse", help="parse saved reports and print the readings")
    par.add_argument("--bdf", default="")
    par.add_argument("paths", nargs="+")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = build_parser().parse_args(argv)
    try:
        if options.command == "bdf":
            print(resolve_bdf(device_index=options.device_index,
                              fake_pyxrt=options.fake_pyxrt))
            return 0
        if options.command == "parse":
            status = 0
            for path in options.paths:
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read()
                try:
                    reading = parse_report(text, bdf=options.bdf or None)
                    print(json.dumps({"path": path, **reading}, sort_keys=True))
                except OdinTelemetryError as exc:
                    print(json.dumps({"path": path, "refusal": type(exc).__name__,
                                      "reason": str(exc)}, sort_keys=True))
                    status = 3
            return status
        if options.command in ("probe", "sample"):
            bdf = normalize_bdf(options.bdf) if options.bdf else None
            plant = _plant(options.fake_plant)
            scratch = os.path.dirname(os.path.abspath(options.out))
            if options.command == "probe":
                sources = []
                failures = []
                for name in [r for r in options.routes.split(",") if r]:
                    try:
                        sources.append(open_source(
                            name, bdf=bdf, device_index=options.device_index,
                            scratch=options.out, fake_pyxrt=options.fake_pyxrt,
                            plant=plant, smi_binary=options.smi))
                    except OdinTelemetryError as exc:
                        failures.append({"source": name, "refusal": type(exc).__name__,
                                         "reason": str(exc)})
                report = probe(sources, bdf=bdf, reads=options.reads, out_dir=options.out)
                report["unopened"] = failures
                with open(os.path.join(options.out, "probe.json"), "w",
                          encoding="utf-8") as handle:
                    json.dump(report, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                for route in report["routes"]:
                    print(f"[probe] {route['describe']['source']}: {route['ok']}/"
                          f"{route['reads']} readings, median read "
                          f"{route['latency_s_median']:.3f} s, power "
                          f"{route['power_w'][:3]}")
                for failure in failures:
                    print(f"[probe] {failure['source']}: not opened, "
                          f"{failure['refusal']}: {failure['reason']}")
                print(f"[probe] chosen route: {report['chosen']}")
                return 0 if report["chosen"] else 3
            source = open_source(options.source, bdf=bdf,
                                 device_index=options.device_index, scratch=scratch,
                                 fake_pyxrt=options.fake_pyxrt, plant=plant,
                                 smi_binary=options.smi)
            count = sample(source, out_path=options.out, bdf=bdf or getattr(source, "bdf", None),
                           interval_s=options.interval_s, duration_s=options.duration_s,
                           max_samples=options.max_samples, stop_file=options.stop_file)
            print(f"[sampler] {count} reading(s) written to {options.out}")
            return 0
        build_parser().print_help()
        return 2
    except OdinTelemetryError as exc:
        print(f"REFUSING: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
