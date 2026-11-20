"""[U-RSHIFT] The bias writers of the source tree, read from its syntax trees in a normalized form and closed over its
references.

The form (the ruled form of a closure test). Every function of ``src`` is read from its syntax tree with its module's
imports resolved and its names followed through their bindings:

* A bias chain is an expression whose chain of attributes, subscripts and calls holds a bias name (``BIAS``), where
  ``getattr(x, n)`` reads as the attribute ``n`` (``n`` a string, a module constant, or a name the function binds to
  string constants only); or a name the function binds to a bias chain (an assignment, an annotated or walrus
  assignment, a loop, ``with`` or comprehension target; through conditional and boolean expressions, tuples, lists
  and comprehensions, and the first argument of a library call such as ``torch.as_tensor``), followed to a fixpoint;
  or a call of a function of the tree that returns a bias chain.
* A function writes a bias directly where it writes on a bias chain: an in-place method (``x.add_``), an attribute,
  subscript or augmented assignment (on a bound name itself too), a deletion, ``setattr`` on it or of a bias name,
  ``register_buffer`` or ``register_parameter`` of a bias name, an ``out=`` argument, or passing it to a function that
  writes that parameter in place (followed through the callee's own bindings).
* A name or an attribute resolves to the functions it names: through the module's import table first (a name or a
  module imported under another name, a relative import, a package's re-export), then the enclosing function scopes,
  the module, ``self``'s class, a class's ``__init__``, a method of a class built in place, or a name only one
  function of the tree has (the ratchet holds every generic writer's name to one function of the tree, so a call of a
  generic writer on a receiver of unknown class always resolves). A name the function binds itself resolves at its
  binding.
* A reference to a writer, not only a call of one, makes the referring function a writer: a callee chosen by a
  conditional or boolean expression, a writer passed as an argument (``functools.partial``, a callback) or bound to a
  name. The writer set is closed over these references, except through a writer that decides its value itself
  (``contained``) and through the received shift's one writer (every function of ``mapping/support/received_shift.py``).

Stated limits (constructs the class's own modules use nowhere): a name built at run time (``getattr`` or ``setattr``
with a computed name), reflection over a module's tensors (``parameters()``, ``state_dict()``, ``__dict__``), a
reinterpretation of a tensor's storage, and a callee held in a container and chosen at run time.
"""

from __future__ import annotations

import ast
import fnmatch
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

#: The names under which a perceptron's or a host carrier's forward adds a bias, held beside the search that fails on
#: an unlisted one (``test_received_shift_ratchet.test_every_tensor_src_adds_is_a_listed_bias_or_a_reviewed_name``).
BIAS = frozenset({"bias", "in_proj_bias", "hardware_bias", "position_bias", "running_mean"})
ONE_WRITER = "mapping/support/received_shift.py"
PACKAGE = "mimarsinan"
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def is_bias(names: Iterable[str]) -> Set[str]:
    """The names among ``names`` that are bias names, a ``*`` in a name a wildcard."""
    return {n for n in names if n in BIAS or ("*" in n and any(fnmatch.fnmatchcase(b, n) for b in BIAS))}


def own_nodes(fn: ast.AST) -> list:
    """The nodes of ``fn``'s own body (nested functions and classes are their own functions; a lambda is part of it)."""
    out, stack = [], list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPES):
            if isinstance(node, ast.Lambda):
                stack.extend(ast.iter_child_nodes(node))
            continue
        out.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return out


def call_name(call: ast.Call) -> Optional[str]:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None


def values(node: ast.AST) -> Iterator[ast.AST]:
    """The sub-expressions whose value an expression may take (its branches, elements and comprehension items)."""
    if isinstance(node, ast.IfExp):
        yield from values(node.body)
        yield from values(node.orelse)
    elif isinstance(node, ast.BoolOp):
        for v in node.values:
            yield from values(v)
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for v in node.elts:
            yield from values(v)
    elif isinstance(node, (ast.Starred, ast.NamedExpr, ast.Await)):
        yield from values(node.value)
    elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        yield from values(node.elt)
    elif isinstance(node, ast.DictComp):
        yield from values(node.value)
    elif isinstance(node, ast.Dict):
        for v in node.values:
            yield from values(v)
    else:
        yield node


def _bindings(nodes: list) -> List[Tuple[List[str], ast.AST]]:
    """``(names, value)`` for every binding of a name among ``nodes`` (a loop, comprehension or ``with`` target is
    bound to its iterable or its context expression; a tuple bound to a tuple pairs its elements)."""
    out: List[Tuple[ast.AST, ast.AST]] = []
    for n in nodes:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, (ast.Tuple, ast.List)) and isinstance(n.value, (ast.Tuple, ast.List)) and len(
                        t.elts) == len(n.value.elts):
                    out.extend(zip(t.elts, n.value.elts))
                else:
                    out.append((t, n.value))
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            out.append((n.target, n.value))
        elif isinstance(n, ast.NamedExpr):
            out.append((n.target, n.value))
        elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
            out.append((n.target, n.iter))
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            out.append((n.optional_vars, n.context_expr))
    return [([x.id for x in ast.walk(t) if isinstance(x, ast.Name)], v) for t, v in out
            if not isinstance(t, (ast.Attribute, ast.Subscript))]


class Source:
    """The functions of a source tree (``{module path}::{Class.}function``), their direct bias writes and references."""

    def __init__(self, files: Iterable[Tuple[str, object]]):
        self.trees: Dict[str, ast.Module] = {
            rel: text if isinstance(text, ast.Module) else ast.parse(str(text)) for rel, text in files}
        self.consts: Dict[str, str] = {}
        for tree in self.trees.values():
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, str):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.consts[target.id] = node.value.value
        self.funcs: Dict[str, ast.AST] = {}
        self.classes: Dict[str, ast.ClassDef] = {}
        self.nodes: Dict[str, list] = {}
        self._import_nodes: Dict[str, list] = {}
        for rel, tree in self.trees.items():
            self._import_nodes[rel] = []
            self._collect(rel, tree, "", None)
        self._tables: Dict[str, Dict[str, Tuple[str, str]]] = {}
        self.imports = {rel: self._table(rel, set()) for rel in self.trees}
        self.by_name: Dict[str, Set[str]] = {}
        for q in self.funcs:
            self.by_name.setdefault(q.split("::")[1].split(".")[-1], set()).add(q)
        self.locals = {q: self._locals(q) for q in self.funcs}
        self.refs = {q: self._refs(q) for q in self.funcs}
        self.sites = {q: self._sites(q) for q in self.funcs}
        self.mentions = {q: self._mentions(q) for q in self.funcs}
        self.strings: Dict[str, Dict[str, Set[str]]] = {}
        self.bindings: Dict[str, list] = {}
        self.getters: Set[str] = set()
        self.aliases: Dict[str, Set[str]] = {}
        self._close_getters()
        self.param_writes = {q: self._param_writes(q) for q in self.funcs}
        self.direct = {q: self._direct(q) for q in self.funcs if q in self.aliases}
        self.direct = {q: lines for q, lines in self.direct.items() if lines}

    @classmethod
    def of_tree(cls, src: Path) -> "Source":
        return cls((p.relative_to(src).as_posix(), p.read_text(encoding="utf-8")) for p in sorted(src.rglob("*.py")))

    def _collect(self, rel: str, node: ast.AST, prefix: str, owner: Optional[str]) -> None:
        """One traversal: every function and class by its qualified name, each node in its function's own nodes (a
        lambda's nodes in the function that holds it), and every import of the module."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                self._import_nodes[rel].append(child)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{rel}::{prefix}{child.name}"
                self.funcs[q], self.nodes[q] = child, []
                if owner is not None:
                    self.nodes[owner].extend(child.decorator_list)
                self._collect(rel, child, f"{prefix}{child.name}.", q)
            elif isinstance(child, ast.ClassDef):
                self.classes[f"{rel}::{prefix}{child.name}"] = child
                self._collect(rel, child, f"{prefix}{child.name}.", None)
            else:
                if owner is not None:
                    self.nodes[owner].append(child)
                self._collect(rel, child, prefix, owner)

    # ---- imports --------------------------------------------------------------------------------------------------
    def _module(self, dotted: str) -> Optional[str]:
        """The tree's file of a dotted module path inside the package, or None."""
        parts = dotted.split(".")
        if not parts or parts[0] != PACKAGE:
            return None
        base = "/".join(parts[1:])
        for rel in (f"{base}.py", f"{base}/__init__.py" if base else "__init__.py"):
            if rel in self.trees:
                return rel
        return None

    @staticmethod
    def _from_base(rel: str, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        package = [PACKAGE] + rel.split("/")[:-1]
        package = package[:len(package) - (node.level - 1)]
        return ".".join(package + ([node.module] if node.module else []))

    def _member(self, module: str, name: str, seen: Set[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
        """What ``name`` of the module file ``module`` is: ("func", q), ("class", q) or ("module", file)."""
        if (module, name) in seen:
            return None
        seen.add((module, name))
        if f"{module}::{name}" in self.funcs:
            return "func", f"{module}::{name}"
        if f"{module}::{name}" in self.classes:
            return "class", f"{module}::{name}"
        if module.endswith("__init__.py"):
            sub = self._module(".".join([PACKAGE] + module.split("/")[:-1] + [name]))
            if sub is not None:
                return "module", sub
        return self._table(module, seen).get(name)

    def _table(self, rel: str, seen: Set[Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
        """The module's import table: each local name to the function, class or module of the tree it imports."""
        if rel in self._tables:
            return self._tables[rel]
        table: Dict[str, Tuple[str, str]] = {}
        for node in self._import_nodes.get(rel, []):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = self._module(alias.name)
                    if module is not None and alias.asname:
                        table[alias.asname] = ("module", module)
                    elif alias.name.split(".")[0] == PACKAGE and not alias.asname:
                        table[PACKAGE] = ("package", PACKAGE)
                continue
            base = self._from_base(rel, node)
            module = self._module(base) if base else None
            for alias in node.names:
                local = alias.asname or alias.name
                sub = self._module(f"{base}.{alias.name}") if base else None
                if sub is not None:
                    table[local] = ("module", sub)
                elif module is not None:
                    found = self._member(module, alias.name, set(seen))
                    if found is not None:
                        table[local] = found
        if not seen:
            self._tables[rel] = table
        return table

    # ---- resolution -----------------------------------------------------------------------------------------------
    def _locals(self, q: str) -> Set[str]:
        args = self.funcs[q].args
        names = {a.arg for a in args.args + args.kwonlyargs + args.posonlyargs}
        names |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
        names |= {n.id for n in self.nodes[q] if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
        return names

    def _strings(self, q: str) -> Dict[str, Set[str]]:
        """The names ``q`` binds to string constants only (a loop over a tuple of them, an assignment of one)."""
        seen: Dict[str, Set[str]] = {}
        other: Set[str] = set()
        for n in self.nodes[q]:
            pairs = []
            if isinstance(n, ast.Assign):
                pairs = [(t, n.value) for t in n.targets]
            elif isinstance(n, (ast.For, ast.comprehension)):
                pairs = [(n.target, n.iter)]
            for t, v in pairs:
                if not isinstance(t, ast.Name):
                    continue
                items = v.elts if isinstance(n, (ast.For, ast.comprehension)) and isinstance(
                    v, (ast.Tuple, ast.List, ast.Set)) else [v] if isinstance(n, ast.Assign) else None
                consts = [self._const_value(i) for i in items] if items is not None else [None]
                if all(c is not None for c in consts):
                    seen.setdefault(t.id, set()).update(consts)
                else:
                    other.add(t.id)
        return {k: v for k, v in seen.items() if k not in other}

    def _const_value(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return self.consts.get(node.id) if isinstance(node, ast.Name) else None

    def _init(self, cls_q: str) -> Set[str]:
        return {f"{cls_q}.__init__"} if f"{cls_q}.__init__" in self.funcs else set()

    def _of(self, found: Optional[Tuple[str, str]], attr: Optional[str] = None) -> Set[str]:
        if found is None:
            return set()
        kind, q = found
        if attr is None:
            return {q} if kind == "func" else self._init(q) if kind == "class" else set()
        if kind == "module":
            return self._of(self._member(q, attr, set()))
        if kind == "class":
            return {f"{q}.{attr}"} if f"{q}.{attr}" in self.funcs else set()
        return set()

    def _value_of(self, q: str, node: ast.AST) -> Optional[Tuple[str, str]]:
        """What a name or a dotted attribute chain names as a module, class or function, through the import table."""
        module = q.split("::")[0]
        if isinstance(node, ast.Name):
            if node.id in self.locals.get(q, ()):
                return None
            if f"{module}::{node.id}" in self.classes:
                return "class", f"{module}::{node.id}"
            return self.imports.get(module, {}).get(node.id)
        if isinstance(node, ast.Attribute):
            inner = self._value_of(q, node.value)
            if inner is None:
                return None
            if inner[0] == "package":
                sub = self._module(f"{PACKAGE}.{node.attr}")
                return ("module", sub) if sub is not None else None
            if inner[0] == "module":
                return self._member(inner[1], node.attr, set())
        return None

    def _unique(self, name: str) -> Set[str]:
        defs = self.by_name.get(name, ())
        return set(defs) if len(defs) == 1 else set()

    def resolve_ref(self, q: str, node: ast.AST) -> Set[str]:
        """The functions a name or an attribute (read in ``q``) names, where the source says which; else the one
        function of the tree with that name, or none."""
        module, local = q.split("::")
        if isinstance(node, ast.Name):
            if node.id in self.locals.get(q, ()):
                return set()
            parts = local.split(".")
            for k in range(len(parts), 0, -1):
                scope = f"{module}::{'.'.join(parts[:k])}"
                if scope in self.funcs and f"{scope}.{node.id}" in self.funcs:
                    return {f"{scope}.{node.id}"}
            if f"{module}::{node.id}" in self.funcs:
                return {f"{module}::{node.id}"}
            found = self._value_of(q, node)
            return self._of(found) if found is not None else self._unique(node.id)
        if isinstance(node, ast.Attribute):
            v = node.value
            if isinstance(v, ast.Name) and v.id in ("self", "cls"):
                owner = local.rsplit(".", 1)[0] + "."
                same = {d for d in self.by_name.get(node.attr, ())
                        if d.split("::")[0] == module and d.split("::")[1].startswith(owner)}
                if same:
                    return same
            inner = self._value_of(q, v)
            if inner is None and isinstance(v, ast.Call):
                built = self._value_of(q, v.func)
                inner = built if built is not None and built[0] == "class" else None
            got = self._of(inner, node.attr) if inner is not None else set()
            return got or self._unique(node.attr)
        return set()

    def resolve(self, q: str, call: ast.Call) -> Set[str]:
        """The functions ``call`` (made in ``q``) names: its callee read as a reference."""
        return self.resolve_ref(q, call.func)

    def _refs(self, q: str) -> List[Tuple[int, Set[str]]]:
        module, out = q.split("::")[0], []
        table = self.imports.get(module, {})
        for n in self.nodes[q]:
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id not in self.by_name and n.id not in table and f"{module}::{n.id}" not in self.classes:
                    continue
            elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
                if n.attr not in self.by_name:
                    continue
            else:
                continue
            targets = self.resolve_ref(q, n)
            if targets:
                out.append((n.lineno, targets))
        return out

    # ---- bindings -------------------------------------------------------------------------------------------------
    def chain(self, node: ast.AST, q: Optional[str] = None) -> list:
        """The names along an expression's chain of attributes, subscripts and calls, the root last; ``getattr(x, n)``
        reads as the attribute ``n`` (each string ``n`` may hold, where ``q`` binds it to string constants only)."""
        names: list = []
        while True:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
                    and len(node.args) >= 2:
                names.extend(self.names_of(node.args[1], q))
                node = node.args[0]
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
                node = node.value
            elif isinstance(node, (ast.Subscript, ast.Starred)):
                node = node.value
            elif isinstance(node, ast.Call):
                node = node.func
            else:
                return names + [node.id] if isinstance(node, ast.Name) else names

    def names_of(self, node: ast.AST, q: Optional[str] = None) -> List[str]:
        """The names a string expression may hold: a string, a module constant, a name ``q`` binds to string constants
        only, or an f-string (each part bound to constants expanded, any other part a ``*``; a name of ``*`` alone is
        a name built at run time and yields none)."""
        strings = self.strings.get(q, {}) if q else {}
        if isinstance(node, ast.JoinedStr):
            parts = [""]
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts = [p + str(v.value) for p in parts]
                elif isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name) and v.value.id in strings:
                    parts = [p + x for p in parts for x in sorted(strings[v.value.id])]
                else:
                    parts = [p + "*" for p in parts]
            return [p for p in parts if p.strip("*")]
        const = self._const_value(node)
        if const is not None:
            return [const]
        return sorted(strings.get(node.id, ())) if isinstance(node, ast.Name) else []

    def bias_name(self, node: ast.AST, q: Optional[str] = None) -> bool:
        return bool(is_bias(self.names_of(node, q)))

    def _mentions(self, q: str) -> Set[str]:
        """The attribute names, names and string constants ``q`` reads (a module constant as its value too)."""
        out: Set[str] = set()
        for n in self.nodes[q]:
            if isinstance(n, ast.Attribute):
                out.add(n.attr)
            elif isinstance(n, ast.Name):
                out.add(n.id)
                if n.id in self.consts:
                    out.add(self.consts[n.id])
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.add(n.value)
        return out

    def _sites(self, q: str) -> List[Tuple[int, ast.AST]]:
        """``(line, expression)`` for every in-place write of ``q``: an in-place method's receiver, ``setattr``'s
        object, an ``out=`` argument, an attribute or subscript target of an assignment, an augmented assignment or a
        deletion, and a name an augmented assignment updates."""
        sites = []
        for n in self.nodes[q]:
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Attribute) and f.attr.endswith("_") and not f.attr.startswith("_"):
                    sites.append((n.lineno, f.value))
                if call_name(n) == "setattr" and n.args:
                    sites.append((n.lineno, n.args[0]))
                sites.extend((n.lineno, k.value) for k in n.keywords if k.arg == "out")
                continue
            targets: list = []
            if isinstance(n, (ast.Assign, ast.Delete)):
                targets = n.targets
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
                targets = [n.target]
            for t in targets:
                if isinstance(t, (ast.Attribute, ast.Subscript)) or (
                        isinstance(t, ast.Name) and isinstance(n, ast.AugAssign)):
                    sites.append((n.lineno, t))
        return sites

    def _roots(self, q: str, node: ast.AST) -> Set[str]:
        """The root names of an expression's values (its chain's root, and the first argument of a library call)."""
        roots: Set[str] = set()
        for v in values(node):
            chain = self.chain(v, q)
            if chain and isinstance(v, (ast.Name, ast.Attribute, ast.Subscript, ast.Call, ast.Starred)):
                roots.add(chain[-1])
            at = v
            while isinstance(at, (ast.Call, ast.Attribute, ast.Subscript)):
                if isinstance(at, ast.Call):
                    if at.args and not self.resolve(q, at) and call_name(at) != "getattr":
                        roots |= self._roots(q, at.args[0])
                    at = at.func
                else:
                    at = at.value
        return roots

    def _holds(self, q: str, node: ast.AST, bound: Set[str]) -> bool:
        """``node`` is a bias chain: its chain holds a bias name or roots at a ``bound`` name, or it calls a getter,
        or a library call takes one as its first argument."""
        for v in values(node):
            chain = self.chain(v, q)
            if chain and (chain[-1] in bound or is_bias(chain)):
                return True
            at = v
            while isinstance(at, (ast.Call, ast.Attribute, ast.Subscript)):
                if isinstance(at, ast.Call):
                    callees = self.resolve(q, at)
                    if callees & self.getters:
                        return True
                    if not callees and at.args and call_name(at) != "getattr" and self._holds(q, at.args[0], bound):
                        return True
                    at = at.func
                else:
                    at = at.value
        return False

    def _relevant(self, q: str) -> bool:
        """``q`` reads a name or a string that holds a bias name (a part of an f-string too): only then, or when it
        refers to a getter (``_close_getters`` adds those), can it hold a bias chain."""
        return any(b in m for m in self.mentions[q] for b in BIAS)

    def _bound(self, q: str) -> Set[str]:
        """The names of ``q`` bound, directly or through each other, to a bias chain; a fixpoint."""
        if q not in self.strings:
            self.strings[q] = self._strings(q)
            self.bindings[q] = _bindings(self.nodes[q])
        bound: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for names, value in self.bindings[q]:
                new = [n for n in names if n not in bound]
                if new and self._holds(q, value, bound):
                    bound.update(new)
                    changed = True
        return bound

    def _close_getters(self) -> None:
        """Every relevant function's names bound to a bias chain, and the functions that return a bias chain (a call
        of one is a bias chain), closed over each other."""
        returns = {q: [n.value for n in self.nodes[q] if isinstance(n, ast.Return) and n.value is not None]
                   for q in self.funcs}
        pending = {q for q in self.funcs if self._relevant(q)}
        while pending:
            for q in pending:
                self.aliases[q] = self._bound(q)
            new = {q for q in pending if q not in self.getters and any(
                self._holds(q, r, self.aliases[q]) for r in returns[q])}
            if not new:
                break
            self.getters |= new
            pending = {c for c, refs in self.refs.items() if any(t & new for _, t in refs)}

    def _param_writes(self, q: str) -> Set[int]:
        """The positions of ``q``'s parameters it writes in place, on the parameter or on a name bound to it."""
        if not self.sites[q]:
            return set()
        args = self.funcs[q].args
        origin: Dict[str, Set[int]] = {a.arg: {i} for i, a in enumerate(args.posonlyargs + args.args)}
        if q not in self.strings:
            self.strings[q] = self._strings(q)
            self.bindings[q] = _bindings(self.nodes[q])
        changed = True
        while changed:
            changed = False
            for names, value in self.bindings[q]:
                got = set().union(*(origin.get(r, set()) for r in self._roots(q, value)))
                for name in names:
                    if got - origin.get(name, set()):
                        origin.setdefault(name, set()).update(got)
                        changed = True
        written: Set[int] = set()
        for _line, expr in self.sites[q]:
            for r in self._roots(q, expr):
                written |= origin.get(r, set())
        return written

    def _direct(self, q: str) -> list:
        bound = self.aliases[q]

        def bias_chain(node: ast.AST) -> bool:
            return self._holds(q, node, bound) or (isinstance(node, ast.Subscript) and self.bias_name(node.slice, q))
        lines = [line for line, expr in self.sites[q] if bias_chain(expr)]
        for n in self.nodes[q]:
            if not isinstance(n, ast.Call):
                continue
            name = call_name(n)
            if name in ("setattr", "register_buffer", "register_parameter"):
                at = 1 if name == "setattr" else 0
                if len(n.args) > at and self.bias_name(n.args[at], q):
                    lines.append(n.lineno)
            for target in self.resolve(q, n):
                if not self.param_writes[target]:
                    continue
                callee = self.funcs[target].args.posonlyargs + self.funcs[target].args.args
                offset = 1 if isinstance(n.func, ast.Attribute) and callee and callee[0].arg in ("self", "cls") else 0
                if any(0 <= i - offset < len(n.args) and bias_chain(n.args[i - offset])
                       for i in self.param_writes[target]):
                    lines.append(n.lineno)
                    break
        return sorted(set(lines))

    # ---- closure --------------------------------------------------------------------------------------------------
    def writers(self, contained: Iterable[str]) -> Dict[str, list]:
        """``{function: why}``: the direct writers and every function that refers to a writer (a call, a callee chosen
        by an expression, a writer passed or bound), closed; a reference to a ``contained`` writer or to the one
        writer makes no writer."""
        stops = set(contained) | {q for q in self.funcs if q.startswith(ONE_WRITER + "::")}
        found: Dict[str, list] = {q: [("writes", line) for line in lines] for q, lines in self.direct.items()}
        changed = True
        while changed:
            changed = False
            through = {q for q in found if q not in stops}
            for q, refs in self.refs.items():
                if q in found:
                    continue
                hits = [(line, sorted(targets & through)) for line, targets in refs if targets & through]
                if hits:
                    found[q] = [("refers", line, callees) for line, callees in hits]
                    changed = True
        return found


# ---- the bias names: the search the list ``BIAS`` is held beside ----------------------------------------------------

#: A library call's summands (its positions and keywords): the bias of a linear map or a convolution, the input of
#: ``addmm`` and its kin, the running mean and the bias of a batch or instance norm, the bias of a layer or group norm,
#: the projection biases of multi-head attention, and either operand of an addition or subtraction.
ADDITIVE = {
    "linear": (2, "bias"), "bilinear": (3, "bias"), "conv1d": (2, "bias"), "conv2d": (2, "bias"), "conv3d": (2, "bias"),
    "conv_transpose1d": (2, "bias"), "conv_transpose2d": (2, "bias"), "conv_transpose3d": (2, "bias"),
    "addmm": (0, "input"), "addmv": (0, "input"), "addbmm": (0, "input"), "baddbmm": (0, "input"), "addr": (0, "input"),
    "addcmul": (0, "input"), "addcdiv": (0, "input"),
    "batch_norm": (1, "running_mean", 4, "bias"), "instance_norm": (1, "running_mean", 4, "bias"),
    "layer_norm": (3, "bias"), "group_norm": (3, "bias"),
    "multi_head_attention_forward": (6, "in_proj_bias", 12, "out_proj_bias"),
    "add": (0, "input", 1, "other"), "sub": (0, "input", 1, "other"), "add_": (0, "other"), "sub_": (0, "other"),
}


def registered(source: Source) -> Dict[str, str]:
    """Every name a module of ``source`` registers as a parameter or a buffer (``register_buffer``,
    ``register_parameter``, a ``Parameter`` assigned to an attribute, ``_parameters[...]`` or ``_buffers[...]``
    assigned): its name a string, a module constant, a name bound to string constants only, or an f-string (each part
    bound to constants expanded, any other part a ``*``), with the first function that registers it."""
    out: Dict[str, str] = {}
    for q, nodes in source.nodes.items():
        source.strings.setdefault(q, source._strings(q))

        def names(node: ast.AST, q: str = q) -> List[str]:
            return source.names_of(node, q)
        for n in nodes:
            if isinstance(n, ast.Call) and call_name(n) in ("register_buffer", "register_parameter") and n.args:
                for name in names(n.args[0]):
                    out.setdefault(name, q)
            if not isinstance(n, ast.Assign):
                continue
            for t in n.targets:
                if isinstance(t, ast.Attribute) and isinstance(n.value, ast.Call) and call_name(n.value) == "Parameter":
                    out.setdefault(t.attr, q)
                if isinstance(t, ast.Subscript) and source.chain(t.value)[:1] in (["_parameters"], ["_buffers"]):
                    for name in names(t.slice):
                        out.setdefault(name, q)
    return out


def summands(source: Source, tensors: Iterable[str]) -> Dict[str, Set[str]]:
    """``{tensor name: sites}``: each registered tensor name (``tensors``, a ``*`` a wildcard suffix) that a forward
    of ``source``, or a function a forward reaches through its references, adds as a summand: an operand of an
    addition or a subtraction, or a summand of a library call (``ADDITIVE``), carrying the tensor through its chain
    (``getattr`` read as the attribute), a name bound to it, a product or quotient of it, a library call's first
    argument, or a function of the tree that returns it."""
    names = set(tensors)
    exact = {n for n in names if "*" not in n}
    prefixes = tuple(sorted({n.split("*")[0] for n in names if "*" in n and n.split("*")[0]}))

    def tensor(name: str) -> bool:
        if "*" in name:
            return name in names or any(fnmatch.fnmatchcase(e, name) for e in exact)
        return name in exact or (bool(prefixes) and name.startswith(prefixes))
    returns = {q: [n.value for n in nodes if isinstance(n, ast.Return) and n.value is not None]
               for q, nodes in source.nodes.items()}
    gives: Dict[str, Set[str]] = {q: set() for q in source.funcs}

    def carried(q: str, node: ast.AST, bound: Dict[str, Set[str]]) -> Set[str]:
        out: Set[str] = set()
        for v in values(node):
            if isinstance(v, ast.BinOp) and isinstance(v.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                out |= carried(q, v.left, bound) | carried(q, v.right, bound)
                continue
            if isinstance(v, ast.UnaryOp):
                out |= carried(q, v.operand, bound)
                continue
            chain, at = source.chain(v, q), v
            while isinstance(at, (ast.Call, ast.Attribute, ast.Subscript, ast.Starred)):
                if isinstance(at, ast.Call):
                    if call_name(at) == "getattr" and isinstance(at.func, ast.Name) and at.args:
                        at = at.args[0]
                        continue
                    callees = source.resolve(q, at)
                    for c in callees:
                        out |= gives.get(c, set())
                    if not callees and at.args:
                        out |= carried(q, at.args[0], bound)
                    at = at.func
                else:
                    at = at.value
            out |= {n for n in (chain[:-1] if isinstance(at, ast.Name) else chain) if tensor(n)}
            if isinstance(at, ast.Name):
                out |= bound.get(at.id, set())
            elif at is not v and not isinstance(at, ast.Constant):
                out |= carried(q, at, bound)
        return out

    def bind(q: str) -> Dict[str, Set[str]]:
        if q not in source.bindings:
            source.strings.setdefault(q, source._strings(q))
            source.bindings[q] = _bindings(source.nodes[q])
        bound: Dict[str, Set[str]] = {}
        changed = True
        while changed:
            changed = False
            for targets, value in source.bindings[q]:
                got = carried(q, value, bound)
                for name in targets:
                    if got - bound.get(name, set()):
                        bound.setdefault(name, set()).update(got)
                        changed = True
        return bound
    changed = True
    while changed:
        changed = False
        for q in (q for q in source.funcs if returns[q]):
            bound = bind(q)
            got = set().union(*(carried(q, r, bound) for r in returns[q]))
            if got - gives[q]:
                gives[q] |= got
                changed = True
    reach = {q for q in source.funcs if q.split("::")[1].split(".")[-1] == "forward"}
    stack = list(reach)
    while stack:
        for _line, targets in source.refs.get(stack.pop(), []):
            for c in targets - reach:
                reach.add(c)
                stack.append(c)
    found: Dict[str, Set[str]] = {}
    for q in reach:
        bound = bind(q)
        for n in source.nodes[q]:
            operands: list = []
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
                operands = [n.left, n.right]
            elif isinstance(n, ast.AugAssign) and isinstance(n.op, (ast.Add, ast.Sub)):
                operands = [n.target, n.value]
            elif isinstance(n, ast.Call) and call_name(n) in ADDITIVE:
                spec = ADDITIVE[call_name(n)]
                if isinstance(n.func, ast.Attribute) and call_name(n) in ("add", "sub", "add_", "sub_"):
                    operands.append(n.func.value)
                operands += [n.args[i] for i in spec[0::2] if i < len(n.args)]
                operands += [k.value for k in n.keywords if k.arg in spec[1::2]]
            for operand in operands:
                for name in carried(q, operand, bound):
                    found.setdefault(name, set()).add(f"{q}:{n.lineno}")
    return found
