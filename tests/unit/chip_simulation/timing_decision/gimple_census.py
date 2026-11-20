"""The census of the timing decision's closure, read from the compiler's intermediate form.

Every translation unit the two backends build for a timing program is compiled as its backend compiles it and dumped
by the GNU compiler as GIMPLE in SSA form (``-O0 -fdump-tree-ssa-lineno``: every temporary is declared with its type,
every statement carries its source location), so spacing, ``auto``, macros, helper headers and calls are normalized
before the census reads them:

- SANA-FE: every source the plugins' own CMake definition compiles, read from the compile commands CMake writes for
  the configuration ``scripts/build_sanafe_plugins.py`` runs (``configure_command``). No file name is matched: a new
  plugin source enters the census by entering the build, and a file of the plugins' directory that no built unit
  compiles or includes is itself a finding (``outside_the_build``).
- nevresim: the ``main.cpp`` that ``NevresimDriver.emit_main`` generates for a one-core program of every timing
  program the backend builds (each spiking mode whose policy decodes timing and supports nevresim, in every
  connectivity mode, comparator and weight type), compiled by ``compile_simulator`` (the toolchain's compiler, flags
  and include tree). A new compute, generator or helper header enters the census by being compiled into such a program.

From each dump the census reads every function whose body lies in the toolchain's sources (the plugins' directory,
nevresim's include tree, the generated program) and lists its floating decisions, each an item:

- ``line <source line>``: a relational or equality operator (or an unordered comparison) with a floating operand, a
  floating MIN_EXPR or MAX_EXPR, a conversion of a floating value to an integer type, a call to a rounding,
  remainder, minimum or maximum function (``ceil``, ``floor``, ``round``, ``lround``, ``llround``, ``rint``,
  ``lrint``, ``llrint``, ``nearbyint``, ``trunc``, ``fmod``, ``remainder``, ``remquo``, ``modf``, ``fmin``, ``fmax``,
  and ``std::min``, ``std::max``, ``std::clamp``, ``std::minmax`` on a floating type), or a call to a library function
  (its body outside the toolchain's sources: the C++ library, SANA-FE's headers) that holds such a decision itself or
  through its own calls;
- ``calls <function>``: a call to another toolchain function, outside the decision's namespace, that holds a decision
  of its own (so a site cannot hand its decision to a ruled-out function).

A function is named by its name without template arguments and the file of its body (``Name @ file``). The one
decision's namespace, ``nevresim::timing``, is where the decision is taken and is not listed. Every other item must be
covered by the ruled-out list held beside the census (``test_closure.RULED_OUT``).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from spiking_points import point_of_mode

REPO_ROOT = Path(__file__).resolve().parents[4]
PLUGIN_SOURCE = REPO_ROOT / "src" / "mimarsinan" / "chip_simulation" / "sanafe" / "plugins"
DECISION_NAMESPACE = "nevresim::timing::"
#: The GNU compiler's dump of GIMPLE in SSA form, with statement locations, at -O0 (nothing inlined or folded away).
DUMP_FLAGS: Tuple[str, ...] = ("-O0", "-fdump-tree-ssa-lineno")
#: The window of the one-core timing program each generated main is rendered for.
PROGRAM_WINDOW = 7

_FLOATING = {"float", "double", "long double", "_Float16", "_Float32", "_Float64", "_Float128", "_Float32x",
             "_Float64x", "__float128", "__ibm128", "__bf16"}
_INTEGRAL = {"bool", "char", "signed char", "unsigned char", "short int", "short unsigned int", "int",
             "unsigned int", "long int", "long unsigned int", "long long int", "long long unsigned int", "__int128",
             "__int128 unsigned", "wchar_t", "char8_t", "char16_t", "char32_t", "sizetype", "bitsizetype",
             "ssizetype"}
#: The typedef names the dumps print unresolved, each with its class (a new one fails the census until classified).
TYPEDEFS: Dict[str, str] = {
    "size_t": "integral", "ptrdiff_t": "integral", "size_type": "integral", "difference_type": "integral",
    "_Distance": "integral", "char_type": "integral",
}
_COMPARE = r"<=|>=|<|>|==|!=|unord|ord|unlt|unle|ungt|unge|uneq|ltgt"
_IF_COMPARE = re.compile(rf"^if \((\S+) ({_COMPARE}) (\S+)\)$")
_SET_COMPARE = re.compile(rf"^\S+ = (\S+) ({_COMPARE}) (\S+);$")
_MINMAX_EXPR = re.compile(r"^\S+ = (MIN|MAX)_EXPR <(\S+), (\S+)>;$")
_CONVERSION = re.compile(r"^\S+ = \(([^()]+)\) (\S+);$")
_CALL = re.compile(r"^(?:\S+ = )?([^\s=(][^=]*?) \((.*)\);$")
_LOCATION = re.compile(r"\[([^\[\]\s]+):(\d+):(\d+)(?: discrim \d+)?\]\s*")
_REAL_CONSTANT = re.compile(r"^-?(?:\d+\.\d*(?:e[+-]\d+)?|Inf|Nan)$")
_ROUNDING = re.compile(r"^(?:std::)?(?:__builtin_)?(?:ceil|floor|round|lround|llround|rint|lrint|llrint|nearbyint|"
                       r"trunc|fmod|remainder|remquo|modf|fmin|fmax)(?:f|l|f\d+x?)?$")
_LIBRARY_MINMAX = re.compile(r"^std::(?:min|max|clamp|minmax)<([^,>]+)")
_SSA_VERSION = re.compile(r"^(.*)_\d+(?:\(D\))?$")
_OPERATOR = re.compile(r"operator(<=>|<<=|>>=|<<|>>|<=|>=|->\*|->|<|>)")


class CensusUnavailable(RuntimeError):
    """The census cannot read the compiled form on this machine (the GNU compiler, cmake or SANA-FE's headers)."""


@dataclass
class TranslationUnit:
    """One translation unit as its backend builds it: the argv (no output), its directory, the source it compiles."""

    label: str
    argv: List[str]
    directory: Path
    source: Path


@dataclass
class Function:
    name: str
    declared: Dict[str, str] = field(default_factory=dict)
    statements: List[Tuple[Optional[Tuple[str, int]], str]] = field(default_factory=list)


@dataclass(frozen=True)
class Census:
    """One unit's reading: every toolchain function's decision items, the callers of the decision's namespace (by
    file), the files the unit compiled, and the operand types the census could not classify."""

    unit: str
    items: Dict[str, Tuple[str, ...]]
    evidence: Dict[str, Tuple[str, ...]]
    header_callers: Tuple[str, ...]
    files: Tuple[str, ...]
    unknown_types: Tuple[str, ...]


def normalized_name(name: str) -> str:
    """A function's name without its template arguments, so one key names every instantiation of it."""
    marks: List[str] = []

    def _mark(m: re.Match) -> str:
        marks.append(m.group(0))
        return f"\x00{len(marks) - 1}\x00"

    out, depth = [], 0
    for ch in _OPERATOR.sub(_mark, name):
        if ch == "<":
            depth += 1
        elif ch == ">" and depth:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return re.sub(r"\x00(\d+)\x00", lambda m: marks[int(m.group(1))], "".join(out))


# ---------------------------------------------------------------------------------------------- the dump's reading


def _split_top(text: str) -> List[str]:
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def _parameters(signature: str) -> Dict[str, str]:
    """The parameters of a dumped signature ``RET NAME (TYPE a, TYPE b)``: its last balanced parenthesized group."""
    end = signature.rfind(")")
    depth, start = 0, -1
    for i in range(end, -1, -1):
        if signature[i] == ")":
            depth += 1
        elif signature[i] == "(":
            depth -= 1
            if depth == 0:
                start = i
                break
    out: Dict[str, str] = {}
    for part in _split_top(signature[start + 1:end]):
        bits = part.rsplit(" ", 1)
        if len(bits) == 2 and re.fullmatch(r"[A-Za-z_][\w.$]*", bits[1]):
            out[bits[1]] = bits[0]
    return out


def parse_dump(text: str) -> List[Function]:
    """Every function of one ``-fdump-tree-ssa-lineno`` dump, with its declared names and located statements."""
    functions: List[Function] = []
    for block in re.split(r"^;; Function ", text, flags=re.M)[1:]:
        header, _, rest = block.partition("\n")
        name = re.match(r"^(.*) \(\S+, funcdef_no=", header)
        lines = rest.splitlines()
        sig_at = next((i for i, ln in enumerate(lines) if ln.strip()), None)
        if name is None or sig_at is None:
            continue
        fn = Function(name.group(1).strip())
        fn.declared.update(_parameters(lines[sig_at]))
        in_decls, pending = True, ""
        for raw in lines[sig_at + 1:]:
            line = raw.strip()
            if not line or line in ("{", "}"):
                continue
            if in_decls and not line.startswith(("<bb", "<L", "[", "#")) and line.endswith(";") and " = " not in line:
                decl = re.match(r"^(.*\S)\s+([A-Za-z_.$][\w.$]*)(?:\[[^\]]*\])*;$", line)
                if decl:
                    fn.declared[decl.group(2)] = decl.group(1)
                    continue
            in_decls = False
            if line.startswith(("<bb", "<L", "#", "goto", "else", "resx", "//")):
                continue
            located = _LOCATION.match(line)
            where = (located.group(1), int(located.group(2))) if located else None
            clean = _LOCATION.sub("", line).strip()
            if clean.startswith("if (") and not clean.endswith(")"):
                pending = clean
                continue
            if pending:
                clean, pending = pending + " " + clean, ""
            fn.statements.append((where, clean))
        functions.append(fn)
    return functions


def _base_type(type_name: str) -> str:
    t = re.sub(r"\b(?:const|volatile|restrict|__restrict)\b", "", type_name).strip()
    return re.sub(r"\s+", " ", t)


def type_class(type_name: str) -> str:
    """``floating``, ``integral``, ``other`` (pointers, references, aggregates, vectors) or ``unknown`` (a typedef
    the census has not classified)."""
    t = _base_type(type_name)
    if t.endswith(("*", "&")) or t.startswith(("struct ", "union ", "class ", "vector(", "enum ")):
        return "other"
    if t in _FLOATING:
        return "floating"
    if t in _INTEGRAL or t.startswith(("unsigned ", "signed ")) or re.fullmatch(r"<unnamed-[su]nsigned:\d+>", t):
        return "integral"
    if t == "void" or "(" in t:
        return "other"
    return TYPEDEFS.get(t, "unknown")


class _Reader:
    """The decision items of every function of one dump."""

    def __init__(self, functions: Sequence[Function], roots: Dict[str, Path]):
        self.roots = {label: str(Path(root).resolve()) for label, root in roots.items()}
        self.by_name: Dict[str, List[Function]] = {}
        for fn in functions:
            self.by_name.setdefault(fn.name, []).append(fn)
        self.unknown: Set[str] = set()
        self._library: Dict[str, bool] = {}
        self._lines: Dict[str, List[str]] = {}

    def label(self, path: str) -> Optional[str]:
        resolved = str(Path(path).resolve())
        for label, root in self.roots.items():
            if resolved == root or resolved.startswith(root + os.sep):
                return f"{label}/{Path(resolved).relative_to(root)}"
        return None

    def home(self, fn: Function) -> Optional[str]:
        """The toolchain file of a function's body, or None for a library function (or one with no statement)."""
        for where, _stmt in fn.statements:
            if where and (label := self.label(where[0])):
                return label
        return None

    def key(self, fn: Function) -> str:
        return f"{normalized_name(fn.name)} @ {self.home(fn)}"

    def source_line(self, where: Optional[Tuple[str, int]], stmt: str) -> str:
        if where is None:
            return f"<no location> {stmt}"
        path, line = where
        if path not in self._lines:
            try:
                self._lines[path] = Path(path).read_text(errors="replace").splitlines()
            except OSError:
                self._lines[path] = []
        lines = self._lines[path]
        text = lines[line - 1].strip() if 0 < line <= len(lines) else f"<line {line}>"
        return re.sub(r"\s+", " ", text)

    def operand_class(self, fn: Function, operand: str) -> str:
        if _REAL_CONSTANT.match(operand):
            return "floating"
        if re.match(r"^-?\d", operand) or operand.startswith("&"):
            return "integral" if re.match(r"^-?\d+[A-Za-z]*$", operand) else "other"
        declared = fn.declared.get(operand)
        if declared is None and (version := _SSA_VERSION.match(operand)):
            declared = fn.declared.get(version.group(1))
        if declared is None:
            return "other"
        cls = type_class(declared)
        if cls == "unknown":
            self.unknown.add(_base_type(declared))
        return cls

    def direct(self, fn: Function) -> List[Tuple[Optional[Tuple[str, int]], str]]:
        """The statements of ``fn`` that decide in floating point by themselves (library calls aside)."""
        found = []
        for where, stmt in fn.statements:
            hit = False
            for pattern in (_IF_COMPARE, _SET_COMPARE):
                m = pattern.match(stmt)
                if m and "floating" in (self.operand_class(fn, m.group(1)), self.operand_class(fn, m.group(3))):
                    hit = True
            m = _MINMAX_EXPR.match(stmt)
            if m and "floating" in (self.operand_class(fn, m.group(2)), self.operand_class(fn, m.group(3))):
                hit = True
            m = _CONVERSION.match(stmt)
            if m and self.operand_class(fn, m.group(2)) == "floating":
                target = type_class(m.group(1))
                if target == "unknown":
                    self.unknown.add(_base_type(m.group(1)))
                hit = hit or target in ("integral", "unknown")
            m = _CALL.match(stmt)
            if m:
                callee = m.group(1).strip()
                minmax = _LIBRARY_MINMAX.match(callee)
                hit = hit or bool(_ROUNDING.match(callee)) or bool(minmax and type_class(minmax.group(1)) == "floating")
            if hit:
                found.append((where, stmt))
        return found

    @staticmethod
    def calls(fn: Function) -> List[Tuple[Optional[Tuple[str, int]], str, str]]:
        return [(where, stmt, m.group(1).strip()) for where, stmt in fn.statements if (m := _CALL.match(stmt))]

    def library_decides(self, name: str, seen: Optional[Set[str]] = None) -> bool:
        if name in self._library:
            return self._library[name]
        seen = set() if seen is None else seen
        if name in seen:
            return False
        seen.add(name)
        decides = any(self.home(fn) is None and (self.direct(fn) or any(
            self.library_decides(callee, seen) for _w, _s, callee in self.calls(fn)))
            for fn in self.by_name.get(name, []))
        self._library[name] = decides
        return decides

    def items(self, fn: Function) -> Tuple[Set[str], List[str]]:
        items: Set[str] = set()
        evidence: List[str] = []
        for where, stmt in self.direct(fn):
            items.add(f"line {self.source_line(where, stmt)}")
            evidence.append(stmt)
        for where, stmt, callee in self.calls(fn):
            if callee.startswith(DECISION_NAMESPACE):
                continue
            for target in self.by_name.get(callee, []):
                if self.home(target) is None:
                    if self.library_decides(callee):
                        items.add(f"line {self.source_line(where, stmt)}")
                        evidence.append(f"{stmt} (a library function deciding in floating point)")
                        break
                elif not target.name.startswith(DECISION_NAMESPACE) and self.direct(target):
                    items.add(f"calls {self.key(target)}")
                    evidence.append(stmt)
        return items, evidence


def census_of_dump(text: str, *, unit: str, roots: Dict[str, Path]) -> Census:
    """Every toolchain function of one dump outside the decision's namespace with its decision items."""
    functions = parse_dump(text)
    reader = _Reader(functions, roots)
    items: Dict[str, Set[str]] = {}
    evidence: Dict[str, List[str]] = {}
    callers: Set[str] = set()
    files: Set[str] = set()
    for fn in functions:
        home = reader.home(fn)
        if home is None:
            continue
        files.add(home)
        if fn.name.startswith(DECISION_NAMESPACE):
            continue
        if any(callee.startswith(DECISION_NAMESPACE) for _w, _s, callee in reader.calls(fn)):
            callers.add(home)
        found, why = reader.items(fn)
        if found:
            items.setdefault(reader.key(fn), set()).update(found)
            evidence.setdefault(reader.key(fn), []).extend(why)
    return Census(unit, {k: tuple(sorted(v)) for k, v in sorted(items.items())},
                  {k: tuple(v) for k, v in evidence.items()}, tuple(sorted(callers)), tuple(sorted(files)),
                  tuple(sorted(reader.unknown)))


# ------------------------------------------------------------------------------------------- the units and dumps


def gnu_compiler() -> str:
    """The toolchain's C++20 compiler, which must be the GNU compiler: the census reads its GIMPLE."""
    from mimarsinan.common.build_utils import find_cpp20_compiler

    compiler, family = find_cpp20_compiler()
    if compiler is None or family != "gcc":
        raise CensusUnavailable(
            "the timing decision's closure census reads the GNU compiler's GIMPLE (-fdump-tree-ssa-lineno); the "
            f"toolchain resolved {compiler!r} ({family!r}): install g++ 10 or later as the toolchain's C++20 compiler")
    return compiler


def _build_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_census_build_sanafe_plugins",
                                                  REPO_ROOT / "scripts" / "build_sanafe_plugins.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sanafe_headers() -> Path:
    override = os.environ.get("MIMARSINAN_SANAFE_SRC", "").strip()
    candidates = [Path(override).expanduser()] if override else []
    candidates.append(REPO_ROOT / "build" / "mimarsinan_sanafe_plugins" / "_deps" / "sanafe_src-src" / "src")
    for path in candidates:
        if (path / "pipeline.hpp").is_file():
            return path.resolve()
    raise CensusUnavailable("the census compiles the SANA-FE plugins: SANA-FE's header tree is needed "
                            "(MIMARSINAN_SANAFE_SRC, or the tree a plugin build fetched into build/)")


def plugin_units(work: Path, source_dir: Optional[Path] = None,
                 nevresim_include: Optional[Path] = None) -> List[TranslationUnit]:
    """Every translation unit the plugins' CMake definition compiles, with the command CMake writes for it: the
    configuration ``scripts/build_sanafe_plugins.py`` runs (``source_dir`` may be a copy of the plugins' directory,
    then configured against ``nevresim_include``)."""
    gnu_compiler()
    build = work / "plugins_configure"
    command = _build_script().configure_command(build, sanafe_src=str(sanafe_headers()),
                                                source_dir=source_dir or PLUGIN_SOURCE)
    if nevresim_include is not None:
        command.append(f"-DNEVRESIM_INCLUDE={nevresim_include}")
    done = subprocess.run([*command, "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], capture_output=True, text=True)
    if done.returncode != 0:
        raise CensusUnavailable(f"the plugins' configuration failed: {done.stderr[-2000:]}")
    units = []
    for entry in json.loads((build / "compile_commands.json").read_text()):
        argv = shlex.split(entry["command"]) if "command" in entry else list(entry["arguments"])
        if "-o" in argv:
            at = argv.index("-o")
            del argv[at:at + 2]
        source = Path(entry["file"]).resolve()
        units.append(TranslationUnit(f"plugin:{source.name}", argv, Path(entry["directory"]), source))
    return sorted(units, key=lambda u: u.label)


def _read_outputs(dumps: Path, base: str) -> Tuple[str, Set[Path]]:
    found = sorted(dumps.glob(f"{base}*.ssa"))
    if len(found) != 1:
        raise CensusUnavailable(f"expected one SSA dump for {base} in {dumps}, found {[p.name for p in found]}")
    deps_text = (dumps / f"{base}.d").read_text().replace("\\\n", " ")
    deps = {Path(d).resolve() for d in deps_text.split(":", 1)[1].split()}
    return found[0].read_text(errors="replace"), deps


def compile_unit(unit: TranslationUnit, work: Path, source: Optional[Path] = None) -> Tuple[str, Set[Path]]:
    """Compile one plugin unit with its own command (``source`` may stand in for its file): the dump and the files
    the unit compiled."""
    dumps = work / ("dump_" + unit.source.stem)
    dumps.mkdir(parents=True, exist_ok=True)
    stand_in = str(source or unit.source)
    argv = [stand_in if a == str(unit.source) else a for a in unit.argv]
    argv += [*DUMP_FLAGS, "-MD", "-MF", str(dumps / f"{unit.source.stem}.d"), "-dumpdir", f"{dumps}/",
             "-dumpbase", unit.source.stem, "-o", str(dumps / "unit.o")]
    done = subprocess.run(argv, cwd=unit.directory, capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        raise CensusUnavailable(f"{unit.label} did not compile: {done.stderr[-3000:]}")
    return _read_outputs(dumps, unit.source.stem)


def dependencies(unit: TranslationUnit) -> Set[Path]:
    """The files one plugin unit compiles, by the compiler's own dependency listing (``-M``) on its command."""
    argv = [a for a in unit.argv if a != "-c"]
    done = subprocess.run([*argv, "-M", "-MT", "unit"], cwd=unit.directory, capture_output=True, text=True,
                          timeout=600)
    if done.returncode != 0:
        raise CensusUnavailable(f"{unit.label}: dependency listing failed: {done.stderr[-2000:]}")
    return {Path(unit.directory, d).resolve() for d in done.stdout.replace("\\\n", " ").split(":", 1)[1].split()}


def outside_the_build(deps: Set[Path], source_dir: Path = PLUGIN_SOURCE) -> List[str]:
    """Files of the plugins' source directory that no built unit compiles or includes (the build definition aside).
    ``source_dir`` may be a copy of the directory: files are compared by their path inside it."""
    used = {p.relative_to(PLUGIN_SOURCE.resolve()) for p in deps if PLUGIN_SOURCE.resolve() in p.parents}
    return sorted(str(p.relative_to(source_dir)) for p in source_dir.rglob("*")
                  if p.is_file() and p.name != "CMakeLists.txt" and "__pycache__" not in p.parts
                  and p.relative_to(source_dir) not in used)


def timing_programs() -> List[Tuple[str, str, str, str]]:
    """``(spiking mode, connectivity mode, comparator, weight type)`` of every timing program nevresim builds."""
    from typing import get_args

    from mimarsinan.chip_simulation.nevresim.connectivity import ConnectivityMode
    from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
    from mimarsinan.chip_simulation.spiking_semantics import ALL_SPIKING_MODES, THRESHOLDING_MODES

    modes = [m for m in sorted(ALL_SPIKING_MODES)
             if policy_for_spiking_mode(point_of_mode(m)).decode_mode() == "timing"
             and policy_for_spiking_mode(point_of_mode(m)).supports_backend("nevresim")]
    return list(product(modes, sorted(get_args(ConnectivityMode)), THRESHOLDING_MODES, ("int", "float")))


def program_label(program: Sequence[str]) -> str:
    return "main:" + "/".join(program)


def _one_core_program(weight_type: type):
    import numpy as np

    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

    core = HardCore(4, 4, has_bias_capability=False)
    core.core_matrix = np.zeros((4, 4), dtype=np.float64)
    core.core_matrix[0, 0] = weight_type(3)
    core.axon_sources = [SpikeSource(-2, 0, is_input=True)]
    core.available_axons = 3
    core.available_neurons = 3
    core.threshold = 2.0
    core.latency = 0
    segment = HardCoreMapping([])
    segment.cores = [core]
    segment.output_sources = np.asarray([SpikeSource(0, 0)], dtype=object)
    return segment


def render_main(work: Path, program: Sequence[str]) -> Path:
    """The ``main.cpp`` the nevresim driver generates for a one-core timing program; returns the program's directory."""
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.chip_simulation.spiking_semantics import derived_firing_mode, derived_spike_generation_mode
    from mimarsinan.common.env import nevresim_root

    mode, connectivity, comparator, weight = program
    weight_type = int if weight == "int" else float
    gen = work / "main_{}_{}_{}_{}".format(mode, connectivity, "le" if comparator == "<=" else "lt", weight)
    gen.mkdir(parents=True, exist_ok=True)
    NevresimDriver.nevresim_path = NevresimDriver.nevresim_path or nevresim_root()
    driver = NevresimDriver(
        1, _one_core_program(weight_type), str(gen), weight_type,
        spike_generation_mode=derived_spike_generation_mode(mode), thresholding_mode=comparator,
        firing_mode=derived_firing_mode(mode), point=point_of_mode(mode), threshold_type=float, verbose=False,
        connectivity_mode=connectivity)
    driver.emit_main(1, PROGRAM_WINDOW, 0, verbose=False)
    return gen


def compile_main(gen: Path, nevresim: Optional[Path] = None) -> Tuple[str, Set[Path]]:
    """Compile a generated main by ``compile_simulator`` (the toolchain's compiler, flags and include tree) at -O0
    with the dump flag: the dump and the files the unit compiled."""
    from mimarsinan.chip_simulation.nevresim.compile_nevresim import compile_simulator
    from mimarsinan.common.env import nevresim_root

    gnu_compiler()
    dumps = gen / "dump"
    dumps.mkdir(exist_ok=True)
    result = compile_simulator(str(gen), str(nevresim or nevresim_root()), output_path=str(gen / "main.o"),
                               verbose=False, optimization=DUMP_FLAGS[0],
                               extra_flags=["-c", DUMP_FLAGS[1], "-MD", "-MF", str(dumps / "main.d"),
                                            "-dumpdir", f"{dumps}/", "-dumpbase", "main"],
                               timeout_s=600)
    if result is None:
        raise CensusUnavailable(f"compile_simulator could not compile {gen / 'main' / 'main.cpp'}")
    return _read_outputs(dumps, "main")


def roots(nevresim: Optional[Path] = None, generated: Optional[Path] = None,
          plugins: Optional[Path] = None) -> Dict[str, Path]:
    """The toolchain's sources a unit compiles, by label: the plugins, nevresim's include tree, the generated program."""
    from mimarsinan.common.env import nevresim_root

    out = {"plugins": Path(plugins or PLUGIN_SOURCE), "nevresim/include": Path(nevresim or nevresim_root()) / "include"}
    if generated is not None:
        out["generated"] = generated
    return out
