"""[U-LIVE] The closure of the one liveness predicate over src's syntax trees: which functions decide an axon's
liveness for a schedule, read off a normalized form of the code (syntax trees with imports resolved, names followed
through their bindings, tuples of functions and attributes), never off names a site chose or source text by patterns.

A function is a unit (a module-level function or a method, its nested functions and lambdas included). Per unit the
analysis reads four facts from its syntax tree:

- ``mark``: it reads an axon source's kind: the off, input and always-on marks of a hard-core source (``is_off_``,
  ``is_input_``, ``is_always_on_``, as attributes or as ``getattr`` strings), an IR source's (``is_off()``,
  ``is_input()``, ``is_always_on()``), a source's core or node index compared with a constant (``node_id >= 0``,
  ``core_ < 0``), also where the index is first bound to a name of the same function and that name is compared
  (``ids = [s.node_id ...]``, then ``ids >= 0``), a call of a function on the finite list ``MARK_READS`` (the
  functions that hand a source's marks over), or a call of any function of the predicate's own module that is not
  the predicate (its arrays carry the marks the predicate has not read);
- ``weight``: it reads a core's weights: a load, or a ``getattr`` string, of a name on the finite list
  ``WEIGHT_READS`` (every function, method or attribute through which src hands over a core's matrix);
- ``sink``: it sets a schedule: it calls a core-graph order of latencies (``core_graph.alap_latencies``,
  ``core_graph.longest_paths``), writes a core's ``latency`` or a pass's pinned latencies;
- ``predicate``: it calls the one predicate (``mapping/latency/upstream.py``).

Calls are resolved through the module's own definitions, its imports (a relative import against the module's
package, a re-export followed to its definition), ``self``/``cls`` methods, an imported class's methods (``X.m``,
``X(...).m``) and the functions a name of the unit is bound to, also through a tuple or a list of functions
(``for hook in (f, g): hook(...)``); a call on any other object is not followed, which is why the two finite lists
are by name. The schedule's units are the latency engines' package (``mapping/latency``), every sink, and every
direct caller of a function that calls a core-graph order (it hands the order its sources); closed over their
callees. A site is a unit of that closure that reads a mark or a weight: the decision split over two functions (a
helper reading the marks, a sibling reading the order) is a site through the caller that joins them. A decision can
also be handed over through an attribute: a unit outside the closure that reads a mark or a weight and stores an
attribute that a unit of the closure loads is a site of its own (``handovers``). Each site is either a reader of the
predicate on the finite list of sites (``SITES`` in the test) or a reviewed reader outside the class (``OUTSIDE`` and
``HANDOVER``, each with its read).

The two finite lists are held beside searches (``mark_returners``, ``matrix_returners``) that find, by the same
normalized form, every function of src that returns a source's marks and every function or attribute through which
a core's matrix is handed over; the test fails on one not listed."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

PREDICATE_MODULE = "mimarsinan.mapping.latency.upstream"
PREDICATE = {"live_axons", "charged_rows", "core_reads", "ir_live_reads", "ir_reads", "direct_signals"}
MARK_ATTRS = {"is_off_", "is_input_", "is_always_on_"}
MARK_METHODS = {"is_off", "is_input", "is_always_on"}
INDEX_ATTRS = {"node_id", "core_"}
#: The functions that hand a source's marks over (by name: a method is called on an object of no known type).
MARK_READS = {"source_arrays", "_source_arrays", "source_pairs", "source_codes", "streamed_producer_nodes", "of_ir"}
#: Every name through which src hands over a core's matrix (by name: a method is called on an object of no known type).
WEIGHT_READS = {"core_matrix", "get_core_matrix", "resolve_core_matrix", "compact_source_matrix",
                "core_matrix_payloads", "weight_banks", "column_slice", "pre_pruning_heatmap", "pre_pruning_snapshot",
                "resolve_pre_pruning_heatmap"}
ORDERS = {("mimarsinan.mapping.latency.core_graph", "alap_latencies"),
          ("mimarsinan.mapping.latency.core_graph", "longest_paths")}
SCHEDULE_PACKAGE = "mimarsinan.mapping.latency."
LATENCY_ATTRS = {"latency"}
PIN_NAMES = {"PINNED_LATENCIES"}
#: Calls that hand their argument back as a value of the same kind (an array view, a copy, a cast).
_PASS_THROUGH = {"asarray", "array", "asanyarray", "ascontiguousarray", "copy", "astype"}
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class Unit:
    name: str
    module: str
    cls: Optional[str]
    mark: bool = False
    weight: bool = False
    sink: bool = False
    order: bool = False
    predicate: bool = False
    calls: Set[str] = field(default_factory=set)
    loads: Set[str] = field(default_factory=set)
    stores: Set[str] = field(default_factory=set)


def _const_number(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return True
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and _const_number(node.operand)


def _loads_index(node: ast.AST, names: Set[str]) -> bool:
    """The expression loads a source's core or node index, directly or through a name bound from one."""
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in INDEX_ATTRS and isinstance(n.ctx, ast.Load):
            return True
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names:
            return True
    return False


def _index_compare(node: ast.AST, names: Set[str]) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    parts = [node.left, *node.comparators]
    return any(_loads_index(p, names) for p in parts) and any(_const_number(p) for p in parts)


def _bindings(fn: ast.AST) -> List[Tuple[List[str], ast.AST]]:
    """Every (names bound, value) of a unit: assignments, augmented and annotated ones, ``for`` and comprehension
    targets (bound from their iterable) and assignment expressions."""
    out: List[Tuple[List[str], ast.AST]] = []

    def names(target: ast.AST) -> List[str]:
        # a name, or the names of a tuple or list target; an attribute or a subscript binds no name of the unit
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [n for e in target.elts for n in names(e)]
        if isinstance(target, ast.Starred):
            return names(target.value)
        return []

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            out.extend((names(t), node.value) for t in node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and node.value is not None:
            out.append((names(node.target), node.value))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            out.append((names(node.target), node.iter))
        elif isinstance(node, ast.NamedExpr):
            out.append((names(node.target), node.value))
    return out


def _bound_from(fn: ast.AST, accepts: Callable[[ast.AST, Set[str]], bool]) -> Set[str]:
    """The names a unit binds, directly or through its other names, from a value ``accepts`` takes."""
    bound = _bindings(fn)
    found: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for targets, value in bound:
            if any(t not in found for t in targets) and accepts(value, found):
                found.update(targets)
                changed = True
    return found


class _Module:
    def __init__(self, name: str, tree: ast.Module, package: bool = False):
        self.name = name
        self.tree = tree
        self.defs: Set[str] = set()
        self.classes: Dict[str, Set[str]] = {}
        self.imported: Dict[str, Tuple[str, str]] = {}  # local name -> (module, name)
        self.modules: Dict[str, str] = {}  # local alias -> module
        for node in tree.body:
            if isinstance(node, _FUNCTIONS):
                self.defs.add(node.name)
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = {b.name for b in node.body if isinstance(b, _FUNCTIONS)}
        # a relative import resolves against the module's package (the module itself for a package's __init__)
        package_parts = name.split(".") if package else name.split(".")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    keep = len(package_parts) - (node.level - 1)
                    base = package_parts[:keep] if keep >= 0 else []
                    source = ".".join(base + ([node.module] if node.module else []))
                else:
                    source = node.module or ""
                for alias in node.names:
                    self.imported[alias.asname or alias.name] = (source, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.modules[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else \
                        alias.name.split(".")[0]


class Analysis:
    """The units of a source tree and their facts (``root``: the ``src`` folder holding ``mimarsinan``)."""

    def __init__(self, root: Path, sources: Optional[Dict[str, str]] = None):
        self.units: Dict[str, Unit] = {}
        self.modules: Dict[str, _Module] = {}
        self.functions: Dict[str, ast.AST] = {}
        texts = dict(sources) if sources is not None else {
            ".".join(p.relative_to(root).with_suffix("").parts): p.read_text()
            for p in sorted(Path(root).rglob("*.py"))}
        for name, text in texts.items():
            package = name.endswith(".__init__")
            name = name[: -len(".__init__")] if package else name
            self.modules[name] = _Module(name, ast.parse(text), package)
        for module in self.modules.values():
            for node in module.tree.body:
                if isinstance(node, _FUNCTIONS):
                    self._unit(module, None, node)
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, _FUNCTIONS):
                            self._unit(module, node.name, item)

    # facts ---------------------------------------------------------------------------------------------------------
    def _unit(self, module: _Module, cls: Optional[str], fn: ast.AST) -> None:
        name = f"{module.name}:{cls + '.' if cls else ''}{fn.name}"  # type: ignore[attr-defined]
        unit = Unit(name, module.name, cls)
        index_names = _bound_from(fn, _loads_index)
        aliases = self._aliases(module, cls, fn)
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute):
                if node.attr in MARK_ATTRS:
                    unit.mark = True
                if isinstance(node.ctx, ast.Load):
                    unit.loads.add(node.attr)
                    if node.attr in WEIGHT_READS:
                        unit.weight = True
                elif isinstance(node.ctx, ast.Store):
                    unit.stores.add(node.attr)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in WEIGHT_READS and (
                    node.id in module.defs or node.id in module.imported):
                unit.weight = True
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in MARK_ATTRS:
                    unit.mark = True
                if node.value in WEIGHT_READS:
                    unit.weight = True
            elif isinstance(node, ast.Compare):
                if _index_compare(node, index_names):
                    unit.mark = True
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for leaf in ast.walk(t):
                        if isinstance(leaf, ast.Attribute) and leaf.attr in LATENCY_ATTRS and isinstance(
                                leaf.ctx, ast.Store):
                            unit.sink = True
            elif isinstance(node, ast.Call):
                self._call(module, cls, unit, node, aliases)
        self.units[name] = unit
        self.functions[name] = fn

    def _aliases(self, module: _Module, cls: Optional[str], fn: ast.AST) -> Dict[str, Set[str]]:
        """Per name of the unit, the functions it is bound to: a function itself, the elements of a tuple or a list of
        functions, or a ``for`` target over such a tuple, list or name (followed through the unit's bindings)."""
        bound = _bindings(fn)
        out: Dict[str, Set[str]] = {}

        def targets_of(value: ast.AST) -> Set[str]:
            if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
                return set().union(*(targets_of(e) for e in value.elts)) if value.elts else set()
            if isinstance(value, ast.Name) and value.id in out:
                return set(out[value.id])
            if isinstance(value, (ast.Name, ast.Attribute)):
                return set(self._resolve(module, cls, value))
            return set()

        changed = True
        while changed:
            changed = False
            for names, value in bound:
                found = targets_of(value)
                for n in names:
                    if found - out.get(n, set()):
                        out.setdefault(n, set()).update(found)
                        changed = True
        return out

    def _call(self, module: _Module, cls: Optional[str], unit: Unit, node: ast.Call,
              aliases: Dict[str, Set[str]]) -> None:
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in MARK_METHODS:
            unit.mark = True
        callee = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if callee in MARK_READS and module.name != PREDICATE_MODULE:
            unit.mark = True
        if isinstance(f, ast.Name) and f.id in ("setattr", "getattr", "hasattr") and len(node.args) >= 2:
            key = node.args[1]
            if f.id == "setattr" and ((isinstance(key, ast.Constant) and key.value in LATENCY_ATTRS) or (
                    isinstance(key, ast.Name) and key.id in PIN_NAMES)):
                unit.sink = True
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                (unit.stores if f.id == "setattr" else unit.loads).add(key.value)
        targets = list(self._resolve(module, cls, f))
        if isinstance(f, ast.Name) and f.id in aliases and not targets:
            targets.extend(aliases[f.id])
        for target in targets:
            mod, _, qual = target.partition(":")
            if (mod, qual) in ORDERS:
                unit.sink = unit.order = True
            if mod == PREDICATE_MODULE and module.name != PREDICATE_MODULE:
                if qual in PREDICATE:
                    unit.predicate = True
                elif target in self.units or qual in self.modules.get(mod, _EMPTY).defs:
                    unit.mark = True  # the predicate module's own arrays, read past the predicate
            unit.calls.add(target)

    def _class_of(self, module: _Module, local: str) -> Optional[Tuple[str, str]]:
        if local in module.classes:
            return module.name, local
        if local in module.imported:
            mod, name = self._reexport(*module.imported[local]).split(":")
            if mod in self.modules and name in self.modules[mod].classes:
                return mod, name
        return None

    def _resolve(self, module: _Module, cls: Optional[str], f: ast.AST) -> List[str]:
        if isinstance(f, ast.Name):
            if f.id in module.defs:
                return [f"{module.name}:{f.id}"]
            owner = self._class_of(module, f.id)
            if owner is not None:
                return [f"{owner[0]}:{owner[1]}.__init__"]
            if f.id in module.imported:
                return [self._reexport(*module.imported[f.id])]
            return []
        if not isinstance(f, ast.Attribute):
            return []
        base = f.value
        if isinstance(base, ast.Name) and base.id in ("self", "cls") and cls is not None:
            return [f"{module.name}:{cls}.{f.attr}"]
        if isinstance(base, ast.Name):
            owner = self._class_of(module, base.id)
            if owner is not None:
                return [f"{owner[0]}:{owner[1]}.{f.attr}"]
            if base.id in module.modules:
                return [self._reexport(module.modules[base.id], f.attr)]
            if base.id in module.imported:
                mod, name = module.imported[base.id]
                full = f"{mod}.{name}"
                if full in self.modules:
                    return [self._reexport(full, f.attr)]
        if isinstance(base, ast.Call) and isinstance(base.func, ast.Name):
            owner = self._class_of(module, base.func.id)
            if owner is not None:
                return [f"{owner[0]}:{owner[1]}.{f.attr}"]
        return []

    def _reexport(self, mod: str, name: str) -> str:
        """``mod:name``, followed through modules that only import ``name`` from another (a re-export)."""
        seen = set()
        while mod in self.modules and (mod, name) not in seen:
            m = self.modules[mod]
            if name in m.defs or name in m.classes or name not in m.imported:
                break
            seen.add((mod, name))
            mod, name = m.imported[name]
        return f"{mod}:{name}"

    # the closure ---------------------------------------------------------------------------------------------------
    def callees(self, seeds: Iterable[str]) -> Set[str]:
        seen: Set[str] = set()
        pending = [s for s in seeds if s in self.units]
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            pending.extend(c for c in self.units[name].calls if c in self.units and c not in seen)
        return seen

    def reaches_predicate(self, name: str) -> bool:
        return any(self.units[u].predicate or self.units[u].module == PREDICATE_MODULE for u in self.callees([name]))

    def schedule(self) -> Set[str]:
        base = {n for n, u in self.units.items() if u.sink or n.startswith(SCHEDULE_PACKAGE)}
        orders = {n for n, u in self.units.items() if u.order}
        callers = {n for n, u in self.units.items() if u.calls & orders}
        return self.callees(base | callers)

    def sites(self) -> Set[str]:
        """Every unit of the schedule's closure that reads a mark or a weight, the predicate's own module excepted."""
        return {n for n in self.schedule()
                if (self.units[n].mark or self.units[n].weight) and self.units[n].module != PREDICATE_MODULE}

    def handovers(self) -> Set[str]:
        """Every unit outside the schedule's closure that reads a mark or a weight and stores an attribute that a unit
        of the closure loads (a decision handed over through an attribute), the predicate's own module excepted."""
        schedule = self.schedule()
        loaded: Set[str] = set()
        for n in schedule:
            loaded |= self.units[n].loads
        return {n for n, u in self.units.items()
                if n not in schedule and (u.mark or u.weight) and u.module != PREDICATE_MODULE and u.stores & loaded}

    # the finite lists' searches ------------------------------------------------------------------------------------
    def _returners(self, carries: Callable[[ast.AST, Set[str]], bool]) -> Tuple[Set[str], Set[str]]:
        """The methods of src that return, and the attributes stored from, a value ``carries(value, the function's
        names bound from such a value)`` takes: (methods, attributes). A module-level function is reached through the
        imports and read as a unit of its own; a method can be called on an object of no known type, so its name is
        what a finite list holds."""
        methods: Set[str] = set()
        attributes: Set[str] = set()
        for name, fn in self.functions.items():
            for node in ast.walk(fn):
                if not isinstance(node, _FUNCTIONS):
                    continue
                names = _bound_from(node, carries)
                if node is fn and self.units[name].cls is not None and any(
                        carries(r, names) for r in _own_returns(node)):
                    methods.add(node.name)
                for a in ast.walk(node):
                    if isinstance(a, (ast.Assign, ast.AnnAssign)) and a.value is not None and carries(a.value, names):
                        for t in (a.targets if isinstance(a, ast.Assign) else [a.target]):
                            if isinstance(t, ast.Attribute):
                                attributes.add(t.attr)
        return methods, attributes

    def matrix_returners(self) -> Set[str]:
        """Every method or attribute of src through which a core's matrix is handed over: each method returning, and
        each attribute stored from, a value that is a matrix read through ``WEIGHT_READS`` (a load, a call or a
        ``getattr`` of a listed name, through an array view, a copy, a cast, a slice, a conditional or a container,
        or a name bound from one). Run against the list, it finds what the list must hold for a call on an object of
        no known type to be read: a name it finds and the list misses is red."""

        def matrix(value: ast.AST, names: Set[str]) -> bool:
            if isinstance(value, ast.Attribute):
                return isinstance(value.ctx, ast.Load) and value.attr in WEIGHT_READS
            if isinstance(value, ast.Name):
                return value.id in names
            if isinstance(value, ast.Subscript):
                return matrix(value.value, names)
            if isinstance(value, ast.IfExp):
                return matrix(value.body, names) or matrix(value.orelse, names)
            if isinstance(value, ast.BoolOp):
                return any(matrix(v, names) for v in value.values)
            if isinstance(value, (ast.Tuple, ast.List)):
                return any(matrix(e, names) for e in value.elts)
            if isinstance(value, ast.Call):
                f = value.func
                called = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
                if called in WEIGHT_READS:
                    return True
                if called == "getattr" and len(value.args) >= 2 and isinstance(value.args[1], ast.Constant):
                    return value.args[1].value in WEIGHT_READS
                if called in _PASS_THROUGH:
                    inner = list(value.args[:1]) + ([f.value] if isinstance(f, ast.Attribute) else [])
                    return any(matrix(a, names) for a in inner)
            return False

        methods, attributes = self._returners(matrix)
        return methods | attributes

    def mark_returners(self) -> Set[str]:
        """Every method of src whose returned value is read from a source's marks (a mark, a mark method, a function of
        ``MARK_READS``, a source's index compared with a constant, or a name bound from one): a call of it on an
        object of no known type hands the marks over unseen."""
        listed = MARK_READS | MARK_METHODS

        def marks(value: ast.AST, names: Set[str]) -> bool:
            index = names | (_bound_from(value, _loads_index) if isinstance(
                value, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)) else set())
            for n in ast.walk(value):
                if isinstance(n, ast.Attribute) and n.attr in MARK_ATTRS:
                    return True
                if isinstance(n, ast.Constant) and n.value in MARK_ATTRS:
                    return True
                if isinstance(n, ast.Call):
                    f = n.func
                    called = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
                    if called in listed:
                        return True
                if _index_compare(n, index):
                    return True
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names:
                    return True
            return False

        methods, _attributes = self._returners(marks)
        return methods


def _own_returns(fn: ast.AST) -> List[ast.AST]:
    """The values a function itself returns or yields (a nested function's or a lambda's are its own)."""
    out: List[ast.AST] = []
    pending = list(ast.iter_child_nodes(fn))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)) and node.value is not None:
            out.append(node.value)
        pending.extend(ast.iter_child_nodes(node))
    return out


_EMPTY = _Module("", ast.parse(""))
