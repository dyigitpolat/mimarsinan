"""[U-LIVE r2] The closure of the per-neuron column rule over src's syntax trees: which functions take a value a core
holds per neuron at some of its columns, read off a normalized form of the code (syntax trees, names followed through
their bindings and through the calls that hand a value on unchanged), never off source text by patterns.

The per-neuron names are the declarations themselves: every name a class of src declares ``PER_NEURON`` in its
``VALUE_LAYOUTS`` (``layouts(PER_NEURON=(...))``). A value expression is per neuron where it loads such an attribute
(``core.threshold``), reads it by ``getattr`` with the name as a string, or is a name of the function bound from one,
also through a call that hands its argument on in kind (an array view, a copy, a cast, a reshape, a tensor's host
copy: ``PASS_THROUGH``, and the functions of src that do so to their first parameter, found by search). A function
takes per-neuron columns where it subscripts such a value (reads or writes some of its entries), calls a readout
law's ``take_columns`` on one, or calls a function of the finite list ``TAKERS`` (the functions that take a value at
columns given beside it) with one as its first argument. ``takers`` finds, by the same form, every function of src that
subscripts one of its parameters by another parameter and is called somewhere with a per-neuron value there: the test
fails on one not listed. Each site is either the one function (``compaction.take_neuron_columns``) or a reviewed reader
or writer that drops, selects and reorders no core's neurons (``READERS`` in the test, each with its read). The other
half: a function that cuts a core's neurons (it stores a compaction's kept columns, or a core's matrix cut along its
neuron axis) calls the one function, or is reviewed (``CUTS_REVIEWED``)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

ONE_FUNCTION = "mimarsinan.mapping.packing.softcore.compaction:take_neuron_columns"
#: The attributes a core holds its weights in, and the one that holds a compaction's kept columns.
MATRIX_ATTRS = {"core_matrix", "compact_source_matrix"}
KEPT_COLUMNS_ATTRS = {"compact_keep_cols"}
#: Calls that hand their (first) argument or receiver on unchanged in kind.
PASS_THROUGH = {"asarray", "array", "asanyarray", "ascontiguousarray", "copy", "astype", "reshape", "ravel", "flatten",
                "detach", "cpu", "numpy", "clone", "double", "float", "contiguous"}
#: The functions that take a value at columns given beside it (first argument, or the receiver of a method).
TAKERS = {"take_columns", "threshold_columns", "placement_value", "neuron_entries", "row_entries",
          "threshold_of_neuron", "neuron_thresholds", "threshold_key", "_neuron_value_at", "generate_core_weights"}
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _call_name(node: ast.Call) -> Optional[str]:
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)


def _bound_names(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for e in target.elts for n in _bound_names(e)]
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return []


def _bindings(fn: ast.AST) -> List[Tuple[List[str], ast.AST]]:
    out: List[Tuple[List[str], ast.AST]] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            out.extend((_bound_names(t), node.value) for t in node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            out.append((_bound_names(node.target), node.value))
        elif isinstance(node, ast.NamedExpr):
            out.append((_bound_names(node.target), node.value))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            out.append((_bound_names(node.target), node.iter))
    return out


class Analysis:
    """The per-neuron names, the per-function facts and the two searches of a source tree (``root``: ``src``)."""

    def __init__(self, root: Path, sources: Optional[Dict[str, str]] = None):
        texts = dict(sources) if sources is not None else {
            ".".join(p.relative_to(root).with_suffix("").parts): p.read_text()
            for p in sorted(Path(root).rglob("*.py"))}
        self.trees: Dict[str, ast.Module] = {}
        for name, text in texts.items():
            name = name[: -len(".__init__")] if name.endswith(".__init__") else name
            self.trees[name] = ast.parse(text)
        self.names = self._declared_per_neuron()
        self.constants = self._name_constants()
        self.functions: Dict[str, ast.AST] = {}
        for module, tree in self.trees.items():
            for node in tree.body:
                if isinstance(node, _FUNCTIONS):
                    self.functions[f"{module}:{node.name}"] = node
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, _FUNCTIONS):
                            self.functions[f"{module}:{node.name}.{item.name}"] = item
        self.converters = self._converters()

    # the declarations -----------------------------------------------------------------------------------------------
    def _declared_per_neuron(self) -> Set[str]:
        """Every name a class of src declares per neuron (``layouts(PER_NEURON=(...))``)."""
        names: Set[str] = set()
        for tree in self.trees.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _call_name(node) == "layouts":
                    for kw in node.keywords:
                        if kw.arg == "PER_NEURON" and isinstance(kw.value, (ast.Tuple, ast.List)):
                            names |= {e.value for e in kw.value.elts if isinstance(e, ast.Constant)}
        return names

    def _name_constants(self) -> Dict[str, Set[str]]:
        """Per module, its names bound to a per-neuron name as a string: a module-level constant (``LANE_OUTPUTS =
        "lane_outputs"``) or one imported from the module that defines it."""
        own: Dict[str, Dict[str, str]] = {}
        for module, tree in self.trees.items():
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and \
                        node.value.value in self.names:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            own.setdefault(module, {})[target.id] = node.value.value
        out: Dict[str, Set[str]] = {m: set(c) for m, c in own.items()}
        for module, tree in self.trees.items():
            parts = module.split(".")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    source = ".".join(parts[: len(parts) - node.level] + ([node.module] if node.module else [])) \
                        if node.level else (node.module or "")
                    for alias in node.names:
                        if alias.name in own.get(source, {}):
                            out.setdefault(module, set()).add(alias.asname or alias.name)
        return out

    # the normalized form of a value ---------------------------------------------------------------------------------
    def _converters(self) -> Set[str]:
        """The module-level functions of src that return their first parameter handed on in kind (``_array``-like
        helpers; a method's first parameter is its object)."""
        found: Set[str] = set()
        for name, fn in self.functions.items():
            if "." in name.partition(":")[2]:
                continue
            params = [a.arg for a in fn.args.args]  # type: ignore[attr-defined]
            if not params:
                continue
            first = {params[0]}
            returns = [n.value for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
            if returns and all(self._carries(r, first, set()) for r in returns):
                found.add(fn.name)  # type: ignore[attr-defined]
        return found

    def _carries(self, value: ast.AST, names: Set[str], attributes: Set[str], keys: Set[str] = frozenset()) -> bool:
        """``value`` is a per-neuron value: a load of a per-neuron attribute (``attributes``), a ``getattr`` of one by
        its name or by a name of ``keys`` (bound from the per-neuron names), a name of ``names``, or one of these
        handed on by a pass-through call or a conditional."""
        if isinstance(value, ast.Name):
            return value.id in names
        if isinstance(value, ast.Attribute):
            return isinstance(value.ctx, ast.Load) and value.attr in attributes
        if isinstance(value, ast.IfExp):
            return self._carries(value.body, names, attributes) or self._carries(value.orelse, names, attributes)
        if isinstance(value, ast.Call):
            called = _call_name(value)
            if called == "getattr" and len(value.args) >= 2:
                key = value.args[1]
                return (isinstance(key, ast.Constant) and key.value in attributes) or (
                    isinstance(key, ast.Name) and key.id in keys)
            converters = PASS_THROUGH | getattr(self, "converters", set())
            if called in converters:
                inner = list(value.args[:1])
                if isinstance(value.func, ast.Attribute):
                    inner.append(value.func.value)
                return any(self._carries(a, names, attributes, keys) for a in inner)
        return False

    def _name_keys(self, fn: ast.AST, module: str = "") -> Set[str]:
        """The names of a function bound from the per-neuron names themselves: its module's constants holding one
        (``_name_constants``), a loop over a container holding one of them as a string, or over a holder's
        declaration (``declared_names``)."""
        keys: Set[str] = set(self.constants.get(module, set()))
        for targets, value in _bindings(fn):
            consts = {n.value for n in ast.walk(value) if isinstance(n, ast.Constant)}
            declared = any(isinstance(n, ast.Call) and _call_name(n) == "declared_names" for n in ast.walk(value))
            if consts & self.names or declared:
                keys.update(targets)
        return keys

    def _per_neuron_names(self, fn: ast.AST, module: str = "") -> Set[str]:
        bound = _bindings(fn)
        keys = self._name_keys(fn, module)
        found: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for targets, value in bound:
                if any(t not in found for t in targets) and self._carries(value, found, self.names, keys):
                    found.update(targets)
                    changed = True
        return found

    # the facts ------------------------------------------------------------------------------------------------------
    def takes(self, fn: ast.AST, module: str = "") -> bool:
        """The function takes a per-neuron value at some of its columns: it subscripts one, calls a law's
        ``take_columns`` on one, or hands one to a function of ``TAKERS`` as an argument."""
        names, keys = self._per_neuron_names(fn, module), self._name_keys(fn, module)
        for node in ast.walk(fn):
            if isinstance(node, ast.Subscript) and self._carries(node.value, names, self.names, keys):
                return True
            if isinstance(node, ast.Call) and _call_name(node) in TAKERS:
                handed = list(node.args) + [kw.value for kw in node.keywords]
                if isinstance(node.func, ast.Attribute):
                    handed.append(node.func.value)
                if any(self._carries(a, names, self.names, keys) for a in handed):
                    return True
        return False

    def sites(self) -> Set[str]:
        return {name for name, fn in self.functions.items() if self.takes(fn, name.partition(":")[0])}

    # the finite list's search ---------------------------------------------------------------------------------------
    def takers(self) -> Set[str]:
        """Every function of src that subscripts one of its parameters (or a name bound from it) by an expression of
        another parameter, and is called somewhere in src with a per-neuron value at that parameter (its position, its
        keyword, or the object of a method)."""
        slicing: Dict[str, Set[int]] = {}
        keywords: Dict[str, Set[str]] = {}
        for fn in self.functions.values():
            params = [a.arg for a in fn.args.args]  # type: ignore[attr-defined]
            if len(params) < 2:
                continue
            for k, p in enumerate(params):
                others = set(params) - {p}
                derived = {p}
                changed = True
                while changed:
                    changed = False
                    for targets, value in _bindings(fn):
                        if any(t not in derived for t in targets) and self._carries(value, derived, set()):
                            derived.update(targets)
                            changed = True
                from_others = set(others)
                for targets, value in _bindings(fn):
                    if any(isinstance(n, ast.Name) and n.id in others for n in ast.walk(value)):
                        from_others.update(targets)
                if any(isinstance(node, ast.Subscript) and self._carries(node.value, derived, set()) and any(
                        isinstance(n, ast.Name) and n.id in from_others for n in ast.walk(node.slice))
                       for node in ast.walk(fn)):
                    slicing.setdefault(fn.name, set()).add(k)  # type: ignore[attr-defined]
                    keywords.setdefault(fn.name, set()).add(p)  # type: ignore[attr-defined]
        called: Set[str] = set()
        for qual, fn in self.functions.items():
            module = qual.partition(":")[0]
            names, keys = self._per_neuron_names(fn, module), self._name_keys(fn, module)
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node)
                if name not in slicing:
                    continue
                method = isinstance(node.func, ast.Attribute)
                handed = [a for k, a in enumerate(node.args)
                          if k in slicing[name] or (method and k + 1 in slicing[name])]
                handed += [kw.value for kw in node.keywords if kw.arg in keywords[name]]
                if method and 0 in slicing[name]:
                    handed.append(node.func.value)
                if any(self._carries(a, names, self.names, keys) for a in handed):
                    called.add(name)  # type: ignore[arg-type]
        return called

    # the other half: a site that cuts a core's neurons carries its per-neuron values -------------------------------
    @staticmethod
    def _column_cut(value: ast.AST, names: Set[str]) -> bool:
        """``value`` holds a matrix cut along its neuron axis: a subscript ``m[:, keep]`` or ``m[np.ix_(rows, cols)]``
        of a core's matrix or of a name bound from one or from such a cut."""
        for node in ast.walk(value):
            if not isinstance(node, ast.Subscript):
                continue
            base = node.value
            matrix = (isinstance(base, ast.Attribute) and base.attr in MATRIX_ATTRS) or (
                isinstance(base, ast.Name) and base.id in names) or (
                isinstance(base, ast.Call) and _call_name(base) in {"get_core_matrix", "asarray", "array"})
            if not matrix:
                continue
            index = node.slice
            second = index.elts[1] if isinstance(index, ast.Tuple) and len(index.elts) >= 2 else None
            whole = isinstance(second, ast.Slice) and second.lower is None and second.upper is None
            if second is not None and not whole:
                return True
            if isinstance(index, ast.Call) and _call_name(index) == "ix_" and len(index.args) >= 2:
                return True
        return False

    def cuts(self, fn: ast.AST) -> bool:
        """The function cuts a core's neurons: it stores a compaction's kept columns, or a core's matrix cut along its
        neuron axis (directly or through a name bound from the cut)."""
        matrices: Set[str] = set()
        for targets, value in _bindings(fn):
            if any(isinstance(n, ast.Attribute) and n.attr in MATRIX_ATTRS for n in ast.walk(value)):
                matrices.update(targets)
        cut_names: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for targets, value in _bindings(fn):
                if any(t not in cut_names for t in targets) and (
                        self._column_cut(value, matrices | cut_names) or any(
                            isinstance(n, ast.Name) and n.id in cut_names for n in ast.walk(value))):
                    cut_names.update(targets)
                    changed = True
        for node in ast.walk(fn):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                stored = {n.attr for t in targets for n in ast.walk(t)
                          if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
                if stored & KEPT_COLUMNS_ATTRS:
                    return True
                if stored & {"core_matrix"} and (self._column_cut(node.value, matrices | cut_names) or any(
                        isinstance(n, ast.Name) and n.id in cut_names for n in ast.walk(node.value))):
                    return True
        return False

    def carries(self, fn: ast.AST) -> bool:
        return any(isinstance(n, ast.Call) and _call_name(n) == ONE_FUNCTION.rpartition(":")[2] for n in ast.walk(fn))

    def uncarried_cuts(self) -> Set[str]:
        """Every function of src that cuts a core's neurons and does not call the one function."""
        return {name for name, fn in self.functions.items() if self.cuts(fn) and not self.carries(fn)}

    def one_function_reads_the_declarations(self) -> bool:
        """``take_neuron_columns`` takes the names its holder declares per neuron (``declared_names`` with
        ``ValueLayout.PER_NEURON``), never a list of its own."""
        fn = self.functions.get(ONE_FUNCTION)
        if fn is None:
            return False
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and _call_name(node) == "declared_names" and any(
                    isinstance(a, ast.Attribute) and a.attr == "PER_NEURON" for a in node.args):
                return True
        return False
