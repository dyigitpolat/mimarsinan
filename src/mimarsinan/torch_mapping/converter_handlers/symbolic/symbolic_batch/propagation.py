"""One run of the symbolic-batch pass (``symbolic_batch/__init__.py`` states the pass and its lemma; review round
14): every node's domain value in traced order, a scalar operator through the domain, a shape query from its tensor's
domain extents, a tensor kernel through ``kernel_step.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Set, Tuple, cast

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode, disable_fake_tensor_cache
from torch.fx.experimental.symbolic_shapes import DimDynamic, StatelessSymbolicContext
from torch.fx.node import map_arg
from torch.nn.utils.stateless import _reparametrize_module
from torch.utils import _pytree as pytree

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.converter_handlers.refusals import node_operator
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import (
    BATCH,
    Opaque,
    evaluate,
    meaning,
    opaque_in,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.shapes import Shape, query
from mimarsinan.torch_mapping.converter_handlers.symbolic.decision_values import (
    DecisionValues,
    SymbolEnv,
    fp64,
    fp64_like,
    hint_of,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import is_conversion_time_operator
from mimarsinan.torch_mapping.converter_handlers.symbolic.pass_result import (
    DATA_DEPENDENT,
    SYMBOLIC_HINT,
    Decision,
    SymbolicPass,
    SymbolicShape,
    closure,
    first_decisions,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.shape_decisions import ShapeDecisions
from mimarsinan.torch_mapping.converter_handlers.symbolic.symbolic_batch.kernel_step import (
    MISSING as _MISSING,
    KernelStep,
    leaves as _leaves,
    origins as _origins,
)
from mimarsinan.torch_mapping.converter_handlers.symbolic.trace_values import BATCH as TRACED_BATCH
from mimarsinan.torch_mapping.converter_handlers.symbolic.trace_values import (
    SampledTensor,
    Trace,
    graph_device,
    placeholder_metas,
    rng_neutral,
    sample,
)

__all__ = ["symbolic_shapes"]

_QUERIES = frozenset({"size", "dim", "ndimension", "numel", "nelement", "shape", "ndim", "dtype", "is_floating_point",
                      "is_complex"})
_NATIVE_ATTRIBUTES = frozenset({"device", "is_cuda", "requires_grad", "layout", "is_sparse", "is_quantized"})


def _detail(reason: str) -> str:
    return f"a value outside the affine batch domain ({reason}), computed on the host at run time"


def _plain(value: Any) -> Any:
    """FX's immutable lists and dicts (and a namedtuple literal) of a node's operands as plain ones, holding the same
    values."""
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple) and not isinstance(value, torch.Size):  # a namedtuple literal holds the same values
        return tuple(_plain(item) for item in value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if type(value) is slice:
        return slice(_plain(value.start), _plain(value.stop), _plain(value.step))
    return value


def _with_decisions(domain: Any, attach) -> Any:
    """``domain`` with every opaque leaf given its decision (``attach``), through containers."""
    if isinstance(domain, Opaque):
        return attach(domain)
    if type(domain) in (tuple, list):
        return type(domain)(_with_decisions(item, attach) for item in domain)
    return domain


class _Propagation(KernelStep):
    """One run of the pass over ``gm`` in traced order (under the fake mode of ``symbolic_shapes``)."""

    def __init__(self, gm: fx.GraphModule, env: SymbolEnv, batch: torch.SymInt, traces: Tuple[Trace, ...]):
        self.gm, self.env, self.batch, self.traces = gm, env, batch, traces
        self.interpreter, self.modules = fx.Interpreter(gm), dict(gm.named_modules())
        self.values = DecisionValues(env, batch, graph_device(gm))
        self.domain: Dict[fx.Node, Any] = {}
        self.own: Dict[fx.Node, Decision] = {}
        self.shapes: Dict[fx.Node, SymbolicShape] = {}
        self.data_dependent: Set[fx.Node] = set()
        self.rebuilt: Set[fx.Node] = set()
        self.order = {n: i for i, n in enumerate(gm.graph.nodes)}
        self.decisions: ShapeDecisions[Decision] = ShapeDecisions(self.modules, self.order)

    def run(self, inputs: list) -> SymbolicPass:
        pending = iter(inputs)
        for node in self.gm.graph.nodes:
            if node.op == "output":
                break
            if node.op == "placeholder":
                value = next(pending)
                self._record(node, value, Shape((BATCH, *(int(d) for d in value.shape[1:]))))
            else:
                self._record(node, *self._evaluate(node))
        first = min(self.own.values(), key=lambda decision: self.order[decision.node]) if self.own else None
        return SymbolicPass(
            self.batch, self.shapes, first_decisions(self.gm, self.own), dict(self.decisions.decided),
            dict(self.decisions.batch_axis), closure(self.gm, self.rebuilt | set(self.own)), first,
        )

    # -- decisions ---------------------------------------------------------------------------------------------------

    def _decision(self, node: fx.Node, operator_name: str, kind: str, detail: str) -> Decision:
        """The node's own decision (its first one: one per node)."""
        if node not in self.own:
            self.own[node] = Decision(node, operator_name, kind, detail)
        return self.own[node]

    def _opaque(self, node: fx.Node, operator_name: str, found: Opaque) -> Opaque:
        """``found`` with its decision: an operand's (propagated), or the node's own."""
        if isinstance(found.origin, Decision) or found.origin is DATA_DEPENDENT:
            return found
        return Opaque(found.reason, self._decision(node, operator_name, "opaque", _detail(found.reason)))

    def _failed(self, node: fx.Node, operator_name: str, detail: str) -> Tuple[Any, Any]:
        """A node the pass cannot compute although it reads a batch-dependent value: a ``failed`` decision, the
        traces' structure with every extent and scalar opaque."""
        decision = self._decision(node, operator_name, "failed", detail)
        sampled = sample(node, self.traces)
        self.rebuilt.add(node)
        value = self.values.from_traces(sampled, decision)
        if sampled is None:
            return value, Opaque(detail, decision)
        leaves = [Shape(tuple(Opaque(detail, decision) for _ in leaf.axes)) if isinstance(leaf, SampledTensor)
                  else Opaque(detail, decision) for leaf in sampled[1]]
        return value, pytree.tree_unflatten(leaves, sampled[0])

    # -- evaluation --------------------------------------------------------------------------------------------------

    def _evaluate(self, node: fx.Node) -> Tuple[Any, Any]:
        operator_name = node_operator(node, self.modules)
        operands = _plain(map_arg((node.args, node.kwargs), lambda arg: self.domain.get(arg, _MISSING)))
        args, kwargs = tuple(operands[0]), dict(operands[1])
        if any(leaf is _MISSING for leaf in _leaves((args, kwargs))):
            return self._failed(node, operator_name, "an operand the pass could not compute")
        if node.op == "get_attr":
            value, exc = self.env.call(lambda: self.interpreter.run_node(node))
            return (None, _MISSING) if exc is not None else (value, self._batch_free(node, operator_name, value))
        if node.op == "call_function" and callable(node.target) and is_conversion_time_operator(node.target) \
                and not any(isinstance(arg, Shape) for arg in (*args, *kwargs.values())):
            return self._scalar(node, operator_name, args, kwargs)
        found = self._query(node, args, kwargs)
        return found if found is not None else self._kernel(node, operator_name, args, kwargs)

    def _scalar(self, node: fx.Node, operator_name: str, args: tuple, kwargs: dict) -> Tuple[Any, Any]:
        """A scalar operator, computed by the domain; its torch-side value is built from the domain value."""
        env_args, env_kwargs = self.interpreter.fetch_args_kwargs_from_env(node)
        hints = hint_of((tuple(env_args), dict(env_kwargs)))
        target = cast(Callable[..., Any], node.target)
        hint, hint_exc = guarded(lambda: target(*hints[0], **hints[1]))
        domain, exc = guarded(lambda: evaluate(target, args, kwargs))
        if exc is not None:
            return None, _MISSING  # batch-free and raising, natively too
        if any(isinstance(leaf, Shape) for leaf in _leaves(domain)):  # an element of a kernel's tuple of tensors
            return self.env.call(lambda: self.interpreter.run_node(node))[0], domain
        if isinstance(domain, Opaque):
            domain = self._opaque(node, operator_name, domain)
            if domain.origin is DATA_DEPENDENT:
                return self.env.call(lambda: self.interpreter.run_node(node))[0], domain
            domain = pytree.tree_map(lambda _: domain, hint) if type(hint) in (tuple, list) else domain
            return (None if hint_exc is not None else self.values.scalar(domain, hint)), domain
        if opaque_in(domain) is not None:  # a sequence holding opaque leaves (a slice of a shape)
            domain = _with_decisions(domain, lambda leaf: self._opaque(node, operator_name, leaf))
            if any(isinstance(leaf, Opaque) and leaf.origin is DATA_DEPENDENT for leaf in _leaves(domain)):
                return self.env.call(lambda: self.interpreter.run_node(node))[0], domain
            return (None if hint_exc is not None else self.values.scalar(domain, hint)), domain
        if hint_exc is not None or not _same_value(meaning(domain, SYMBOLIC_HINT), hint):
            return self._scalar_disagreement(node, operator_name, hint, hint_exc)
        return self.values.scalar(domain, hint), domain

    def _scalar_disagreement(self, node: fx.Node, operator_name: str, hint: Any, exc) -> Tuple[Any, Any]:
        found = self._opaque(node, operator_name, Opaque("the domain's value and the native one at the hint differ"))
        return (None if exc is not None else self.values.scalar(found, hint)), found

    def _query(self, node: fx.Node, args: tuple, kwargs: dict) -> Optional[Tuple[Any, Any]]:
        """A shape, rank, element-count or dtype query of a tensor, read from its domain extents."""
        name = node.target if node.op == "call_method" else node.args[1] if (
            node.op == "call_function" and node.target is getattr and len(node.args) == 2) else None
        if node.op == "call_function" and node.target is torch.numel:
            name = "numel"
        if not args or not isinstance(args[0], Shape) or not isinstance(name, str) or \
                name not in _QUERIES | _NATIVE_ATTRIBUTES:
            return None
        attribute = node.op == "call_function" and node.target is getattr
        dims, rest = args[0].dims, [] if attribute else [*args[1:], *kwargs.values()]
        meta = node.args[0].meta.get("tensor_meta") if isinstance(node.args[0], fx.Node) else None
        domain = query(name, dims, rest)
        native = getattr(meta, "dtype", None)
        if name in ("dtype", "is_floating_point", "is_complex") and not rest and native is not None:
            domain = native if name == "dtype" else getattr(native, name)
        elif name in _NATIVE_ATTRIBUTES and not rest:
            domain = None
        elif domain is NotImplemented:
            return None
        value, exc = self.env.call(lambda: self.interpreter.run_node(node))
        return (None if exc is not None else value), (value if name in _NATIVE_ATTRIBUTES else domain)

    # -- record ------------------------------------------------------------------------------------------------------

    def _keeps_batch_axis(self, domain: Any) -> bool:
        return isinstance(domain, Shape) and bool(domain.dims) and domain.dims[0] == BATCH

    def _record(self, node: fx.Node, value: Any, domain: Any) -> None:
        restored = self.decisions.record(node, self.own.get(node), _origins(domain), node in self.rebuilt,
                                         lambda: self._keeps_batch_axis(domain))
        if restored:  # a pure reshape restoring a per-sample shape: decision-free (``shape_decisions.py``)
            sampled = sample(node, self.traces)
            clean = self.values.from_traces(sampled, None)
            leaf = sampled[1][0] if sampled is not None and len(sampled[1]) == 1 else None
            if clean is not None and isinstance(leaf, SampledTensor) and leaf.axes[:1] == (TRACED_BATCH,) and all(
                    type(axis) is int for axis in leaf.axes[1:]):
                value, domain = clean, Shape((BATCH, *leaf.axes[1:]))
                self.rebuilt.add(node)
        self.interpreter.env[node] = value
        self.domain[node] = domain
        dependent = any(isinstance(d, Opaque) and d.origin is DATA_DEPENDENT for d in _leaves(domain))
        if dependent or self.data_dependent.intersection(node.all_input_nodes):
            self.data_dependent.add(node)
        if isinstance(domain, Shape):
            self.shapes[node] = SymbolicShape(domain.dims, node in self.data_dependent)


def _same_value(found: Any, want: Any) -> bool:
    if type(found) in (tuple, list, torch.Size) and type(want) in (tuple, list, torch.Size):
        return len(found) == len(want) and all(map(_same_value, found, want))
    return type(found) is type(want) and (found == want or found != found and want != want)


def symbolic_shapes(gm: fx.GraphModule, traces: Tuple[Trace, ...]) -> SymbolicPass:
    """Every traced value of the PURE program ``gm`` (``purity.program_purity``; the caller checks it) in the affine
    batch domain, and its opaque batch values; ``traces``: the seeded traces at batches 1, 2 and 3."""
    metas = placeholder_metas(gm)
    if metas is None:
        return SymbolicPass(None)
    env, device = SymbolEnv(), graph_device(gm)
    mode = FakeTensorMode(shape_env=env, allow_non_fake_inputs=True)
    example = torch.empty((SYMBOLIC_HINT, *tuple(metas[0].shape)[1:]), dtype=fp64(metas[0].dtype), device=device)
    dynamic = [DimDynamic.DYNAMIC] + [DimDynamic.STATIC] * (example.dim() - 1)
    state = [*gm.named_parameters(remove_duplicate=False), *gm.named_buffers(remove_duplicate=False)]
    with rng_neutral(gm), torch.no_grad(), disable_fake_tensor_cache(mode):
        first = mode.from_tensor(example, symbolic_context=StatelessSymbolicContext(dynamic_sizes=dynamic))
        batch = first.shape[0]
        assert isinstance(batch, torch.SymInt)  # DimDynamic.DYNAMIC at a hint above 1
        with mode:
            rest = [torch.empty((batch, *tuple(m.shape)[1:]), dtype=fp64(m.dtype), device=device) for m in metas[1:]]
            fp64_state = {name: fp64_like(t) for name, t in state}
            with _reparametrize_module(gm, fp64_state):
                return _Propagation(gm, env, batch, traces).run([first, *rest])
