"""A module's native call as ``nn.Module`` runs it (review round 11).

FX traces ``getattr(type(root), 'forward')`` and never runs the root's call. Round 10 refused a root whose call runs
code FX does not trace by reading the root's instance dict and a few class attributes. ``nn.Module``'s call reads its
names through Python's attribute lookup, which a class can intercept, and it tests the global hook tables for truth and
iterates them with ``items()``, where round 10 counted them with ``len()``. Five forms passed round 10's check: each
was pure with ``torch_rebinding()`` None, certified, and wrong by rel 0.5 at batches 9 and 13 under both contracts with
stage 2 passing (the round-10 reviewer's ``r10_mods.py``; w0-base refused each at its probe, ConversionProbeError at
'linear'):

- a class ``__getattribute__`` that answers ``forward`` (``RootGetattributeFwd``);
- a class property ``_compiled_call_impl`` (``RootCompiledProp``);
- a non-function descriptor ``forward`` whose class access is the traced function (``RootFwdDescriptor``);
- a class property ``_forward_hooks`` (``RootHooksProp``);
- a global forward-hook table rebound to a ``dict`` subclass whose ``len()`` is 0 and whose truth is ``True``
  (``LyingGlobalHookFunctional``); ``LyingGlobalHookLinear`` hooks every ``nn.Linear`` and fooled the called module's
  ``_has_hooks`` too (pure, its value program wrong at batches 9 and 13 with stage 2 passing).

Now the root check reads what the native call reads (``symbolic/module_call.py``). The names are taken from torch's
call code itself; each resolves statically (``inspect.getattr_static``) to ``nn.Module``'s own binding as imported or to
the instance state ``nn.Module`` keeps; attribute lookup is Python's own; the global tables are the objects imported,
of exact type ``OrderedDict``, counted with ``dict.__len__``; and the call's own code is the one imported.
"""

from __future__ import annotations

import abc
import contextlib
import copy
import types
import warnings
from collections import OrderedDict

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.module as module_globals

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import foreign_module
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import program_purity
from mimarsinan.torch_mapping.torch_graph_tracer import RootCallError, trace_model

from .harness import CONTRACTS, assert_converted_exact, convert, native

BATCHES_1_TO_13 = tuple(range(1, 14))
_ROOT = "(the root module, no FX node: its call is outside the traced program)"


class _Functional(nn.Module):
    """``F.linear`` on parameters: no ``call_module`` node."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(6, 8) * 0.3)
        self.b = nn.Parameter(torch.randn(6) * 0.1)

    def forward(self, x):
        return torch.relu(F.linear(x, self.w, self.b))


def _alternate_forward(self, x):
    y = torch.relu(F.linear(x, self.w, self.b))
    return y * 2 if x.shape[0] > 8 else y


class RootGetattributeFwd(_Functional):
    """The class's ``__getattribute__`` answers 'forward' with ``_alternate_forward``."""

    def __getattribute__(self, name):
        if name == "forward":
            return types.MethodType(_alternate_forward, self)
        return super().__getattribute__(name)


class RootCompiledProp(_Functional):
    """A class property ``_compiled_call_impl``: ``nn.Module._wrapped_call_impl`` calls it when it is not None."""

    @property
    def _compiled_call_impl(self):
        return types.MethodType(_alternate_forward, self)

    @_compiled_call_impl.setter
    def _compiled_call_impl(self, value):
        pass


class _TwoForward:
    """A non-function descriptor: the class gets the traced forward, an instance ``_alternate_forward``."""

    def __get__(self, obj, cls):
        return cls._traced if obj is None else types.MethodType(_alternate_forward, obj)


class RootFwdDescriptor(_Functional):
    def _traced(self, x):
        return torch.relu(F.linear(x, self.w, self.b))

    forward = _TwoForward()


def _double_above_8(module, inputs, output):
    return output * 2 if inputs[0].shape[0] > 8 else output


class RootHooksProp(_Functional):
    """``_forward_hooks`` is a class property: the instance dict holds the empty table ``nn.Module.__init__`` stored,
    every read (``nn.Module._call_impl``'s) returns a table holding ``_double_above_8``."""

    @property
    def _forward_hooks(self):
        return OrderedDict({987654: _double_above_8})

    @_forward_hooks.setter
    def _forward_hooks(self, value):
        self.__dict__["_forward_hooks"] = value


class LyingGlobalHookFunctional(_Functional):
    """The test installs a global forward-hook table whose ``len()`` is 0 and whose truth is True, holding a hook that
    doubles this class's output above batch 8."""


class LyingGlobalHookLinear(nn.Module):
    """A ``Linear`` call_module; the lying global table doubles every ``nn.Linear``'s output above batch 8."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return torch.relu(self.fc(x))


class _LyingHooks(dict):
    """``len()`` 0, truth True: ``nn.Module._call_impl`` tests truth and iterates ``items()``."""

    def __len__(self):
        return 0

    def __bool__(self):
        return True


@contextlib.contextmanager
def _lying_global_hooks(target):
    """``torch.nn.modules.module._global_forward_hooks`` rebound to a lying table whose hook doubles the output of a
    module of type ``target`` above batch 8 (nothing installed for ``target`` None)."""
    if target is None:
        yield
        return
    original = module_globals._global_forward_hooks
    lying = _LyingHooks()
    dict.__setitem__(lying, 424242, lambda m, i, o: o * 2 if type(m) is target and i[0].shape[0] > 8 else o)
    module_globals._global_forward_hooks = lying
    try:
        yield
    finally:
        module_globals._global_forward_hooks = original


_REBOUND_TABLE = "torch.nn.modules.module._global_forward_hooks is rebound since import"

ROOT_FORMS = {  # form -> (factory, the lying table's hooked type or None, the reason)
    "getattribute_forward": (RootGetattributeFwd, None, "its class RootGetattributeFwd defines __getattribute__"),
    "compiled_call_property": (RootCompiledProp, None, "its class overrides nn.Module._compiled_call_impl"),
    "forward_descriptor": (RootFwdDescriptor, None,
                           "its class's forward is a _TwoForward, not the plain function FX traces"),
    "hooks_property": (RootHooksProp, None, "its class binds _forward_hooks, which nn.Module keeps on the instance"),
    "lying_global_table": (LyingGlobalHookFunctional, LyingGlobalHookFunctional, _REBOUND_TABLE),
    "lying_global_table_linear": (LyingGlobalHookLinear, nn.Linear, _REBOUND_TABLE),
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(ROOT_FORMS))
def test_a_root_call_read_through_the_class_or_a_rebound_global_is_refused_by_name(how, contract):
    """Pure and certified on the round-10 tree, wrong by rel 0.5 at batches 9 and 13 with stage 2 passing."""
    factory, hooked, reason = ROOT_FORMS[how]
    model = native(factory)
    x = torch.randn(9, 8, dtype=torch.float64)
    with torch.no_grad():  # the witness: at batch 9 the native call is not what the class's forward computes
        traced = getattr(type(model), "_traced", type(model).forward)(model, x)
    with _lying_global_hooks(hooked):
        with torch.no_grad():
            assert not torch.equal(model(x), traced)
        with pytest.raises(RootCallError) as excinfo:
            convert(model, (8,), contract)
        with pytest.raises(RootCallError):
            trace_model(copy.deepcopy(model), (8,))
    error = excinfo.value
    assert isinstance(error, ToolchainRefusal) and error.operator == factory.__name__ and error.node_name is None
    assert str(error).startswith(f"{factory.__name__} {_ROOT}: {reason}"), str(error)


def test_a_called_module_reads_the_global_hook_tables_as_its_call_does():
    """The round-10 ``_has_hooks`` counted the global tables with ``len()``: LyingGlobalHookLinear's ``fc`` was a
    declared leaf, packaged into a perceptron. Its value program was wrong at batches 9 and 13 with stage 2 passing."""
    gm = trace_model(native(LyingGlobalHookLinear), (8,))
    assert program_purity(gm).pure and foreign_module(gm.fc) is None
    with _lying_global_hooks(nn.Linear):
        purity = program_purity(gm)
        assert foreign_module(gm.fc) == f"a module whose call is not torch's as imported ({_REBOUND_TABLE})"
    assert (purity.impurity.node, purity.impurity.operator) == ("fc", "Linear"), purity.impurity
    assert _REBOUND_TABLE in purity.impurity.reason and purity.foreign_modules == {"fc"}
    assert program_purity(gm).pure and foreign_module(gm.fc) is None


# ----------------------------------------------------------------------------- the names the call reads
_REVIEWED_NAMES = {
    "forward", "_compiled_call_impl", "_slow_forward", "_forward_hooks", "_forward_pre_hooks",
    "_forward_hooks_with_kwargs", "_forward_hooks_always_called", "_forward_pre_hooks_with_kwargs", "_backward_hooks",
    "_backward_pre_hooks",
}


def test_the_names_the_rule_reads_are_taken_from_torchs_call_code():
    from mimarsinan.torch_mapping.converter_handlers.symbolic.module_call import CALL_GLOBALS, CALL_READS

    assert _REVIEWED_NAMES | {"_call_impl"} <= CALL_READS, sorted(CALL_READS)
    tables = {"_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
              "_global_backward_pre_hooks"}
    assert {(name,) for name in tables} <= CALL_GLOBALS, sorted(CALL_GLOBALS)


class _Recorder(nn.Module):
    """Records every attribute read on it while ``_reads`` is a set."""

    def __getattribute__(self, name):
        reads = object.__getattribute__(self, "__dict__").get("_reads")
        if reads is not None:
            reads.add(name)
        return object.__getattribute__(self, name)

    def forward(self, x, fail=False):
        if fail:
            raise RuntimeError("forward failed")
        return x * 2


def _hooks(module):
    """Every kind of hook ``nn.Module``'s call runs, on ``module`` and globally (``_global_is_full_backward_hook``
    restored by the caller); returns the handles."""
    return [
        module.register_forward_pre_hook(lambda m, args: None),
        module.register_forward_pre_hook(lambda m, args, kwargs: None, with_kwargs=True),
        module.register_forward_hook(lambda m, args, out: None),
        module.register_forward_hook(lambda m, args, kwargs, out: None, with_kwargs=True, always_call=True),
        module.register_full_backward_pre_hook(lambda m, grad: None),
        module.register_full_backward_hook(lambda m, grad_in, grad_out: None),
        module_globals.register_module_forward_pre_hook(lambda m, args: None),
        module_globals.register_module_forward_hook(lambda m, args, out: None, always_call=True),
        module_globals.register_module_full_backward_pre_hook(lambda m, grad: None),
        module_globals.register_module_full_backward_hook(lambda m, grad_in, grad_out: None),
    ]


def test_every_name_torchs_call_reads_at_run_time_is_one_the_rule_reads(monkeypatch):
    """Fails when torch's call reads a name the rule does not cover: every attribute the native call reads on the
    module, with no hook, with every kind of hook (forward, pre, with keywords, always called, full backward and
    backward pre, the global ones), with a legacy backward hook, and through an exception in ``forward``, is in
    ``CALL_READS``."""
    from mimarsinan.torch_mapping.converter_handlers.symbolic.module_call import CALL_READS

    monkeypatch.setattr(module_globals, "_global_is_full_backward_hook", module_globals._global_is_full_backward_hook)
    reads: set = set()
    x = torch.randn(2, 3, requires_grad=True)

    def call(module, handles, **kwargs):
        state = object.__getattribute__(module, "__dict__")  # not a read through the module's lookup
        state["_reads"] = reads
        try:
            with contextlib.suppress(RuntimeError), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module(x, **kwargs)
        finally:
            state["_reads"] = None
            for handle in handles:
                handle.remove()

    call(_Recorder(), [])
    module = _Recorder()
    call(module, _hooks(module))
    module = _Recorder()
    call(module, _hooks(module), fail=True)
    module = _Recorder()
    call(module, [module.register_backward_hook(lambda m, grad_in, grad_out: None)])
    assert reads and reads - CALL_READS == set(), sorted(reads - CALL_READS)


def test_a_class_binding_of_each_name_the_call_reads_is_refused_by_name():
    """A class property for any name the call reads (an honest one included) is user code in the root's call."""
    from mimarsinan.torch_mapping.converter_handlers.symbolic.module_call import CALL_READS

    instance_state = set(vars(nn.Module()))
    for name in sorted(CALL_READS):
        accessor = property(lambda self, name=name: self.__dict__.get(name),
                            lambda self, value, name=name: self.__dict__.__setitem__(name, value))
        cls = type(f"Binds{name}", (_Functional,), {name: accessor})
        with pytest.raises(RootCallError) as excinfo:
            trace_model(native(cls), (8,))
        reason = ("its class's forward is a property, not the plain function FX traces" if name == "forward" else
                  f"its class binds {name}, which nn.Module keeps on the instance" if name in instance_state else
                  f"its class overrides nn.Module.{name}")
        assert str(excinfo.value).startswith(f"{cls.__name__} {_ROOT}: {reason}"), str(excinfo.value)


# ----------------------------------------------------------------------------- the call's code and globals as imported
@contextlib.contextmanager
def _replaced_call_impl_code():
    function = vars(nn.Module)["_call_impl"]
    original = function.__code__
    function.__code__ = original.replace()  # the same instructions, another code object
    try:
        yield
    finally:
        function.__code__ = original


@contextlib.contextmanager
def _rebound_tracing_state():
    original = torch._C._get_tracing_state
    torch._C._get_tracing_state = lambda: None
    try:
        yield
    finally:
        torch._C._get_tracing_state = original


@pytest.mark.parametrize("patch, reason", [
    (_replaced_call_impl_code, "torch.nn.modules.module.Module._call_impl.__code__ is replaced since import"),
    (_rebound_tracing_state, "torch._C._get_tracing_state is rebound since import"),
], ids=["call_impl_code", "tracing_state_global"])
def test_the_call_code_and_the_globals_it_reads_are_the_ones_imported(patch, reason):
    with patch():
        with pytest.raises(RootCallError) as excinfo:
            trace_model(native(_Functional), (8,))
    assert str(excinfo.value).startswith(f"_Functional {_ROOT}: {reason}"), str(excinfo.value)
    trace_model(native(_Functional), (8,))


# ----------------------------------------------------------------------------- controls: the call is torch's
class _AbstractRoot(_Functional, abc.ABC):
    """A root whose metaclass (``ABCMeta``) defines no attribute lookup and no ``forward``."""


class _BackwardHookRoot(_Functional):
    """A full backward hook on the root: the forward value is ``forward``'s (exact on w0-base)."""

    def __init__(self):
        super().__init__()
        self.register_full_backward_hook(lambda module, grad_in, grad_out: None)


class _WrappedCallImplRoot(_Functional):
    """``nn.Module.__call__`` is ``nn.Module``'s ``_wrapped_call_impl`` function itself: the call never reads the name
    ``_wrapped_call_impl``, so a class's own one never runs (exact on w0-base; refused by name on the round-10 tree)."""

    def _wrapped_call_impl(self, *args, **kwargs):
        return super()._wrapped_call_impl(*args, **kwargs) * 2


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", [_AbstractRoot, _BackwardHookRoot, _WrappedCallImplRoot],
                         ids=["abc_metaclass", "backward_hook", "unread_wrapped_call_impl"])
def test_a_root_whose_call_runs_its_forward_alone_keeps_its_conversion(factory, contract):
    assert_converted_exact(native(factory), (8,), contract, batches=BATCHES_1_TO_13, kinds=("normal",))


class _LookupMeta(type):
    def __getattr__(cls, name):
        raise AttributeError(name)


class _MetaRoot(_Functional, metaclass=_LookupMeta):
    pass


def test_a_metaclass_attribute_lookup_and_an_instance_graph_module_wrapper_are_refused_by_name():
    with pytest.raises(RootCallError) as excinfo:
        trace_model(native(_MetaRoot), (8,))
    assert str(excinfo.value).startswith(f"_MetaRoot {_ROOT}: its metaclass _LookupMeta defines __getattr__")
    graph_module = fx.symbolic_trace(native(_Functional))
    graph_module.__dict__["_wrapped_call"] = lambda module, *args: module.forward(*args) * 2
    with pytest.raises(RootCallError) as excinfo:
        trace_model(graph_module, (8,))
    assert "its GraphModule call is not FX's wrapper of nn.Module's" in str(excinfo.value)


# ----------------------------------------------------------------------------- an instance dict of a user type
class _HidingDict(dict):
    """An instance dict whose own methods hide ``forward``; Python's attribute lookup reads the dict itself."""

    def __contains__(self, key):
        return key != "forward" and dict.__contains__(self, key)

    def get(self, key, default=None):
        return default if key == "forward" else dict.get(self, key, default)

    def items(self):
        return [(key, value) for key, value in dict.items(self) if key != "forward"]

    def keys(self):
        return [key for key in dict.keys(self) if key != "forward"]


def _hide_forward(module, forward):
    module.__dict__ = _HidingDict(module.__dict__)
    dict.__setitem__(module.__dict__, "forward", types.MethodType(forward, module))
    return module


def _doubling_linear(self, x):
    out = F.linear(x, self.weight, self.bias)
    return out * 2 if x.shape[0] > 8 else out


def test_an_instance_dict_of_a_user_type_is_refused_by_name():
    """Found in review round 11: the round-10 checks read the instance dict through its own methods, so a hidden
    instance ``forward`` passed them (on the root: traced, the native call doubling above batch 8; on a called Linear:
    pure, packaged into a perceptron)."""
    root = _hide_forward(native(_Functional), _alternate_forward)
    x = torch.randn(9, 8, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(root(x), 2 * _Functional.forward(root, x))  # the native call runs the hidden forward
    with pytest.raises(RootCallError) as excinfo:
        trace_model(root, (8,))
    assert str(excinfo.value).startswith(f"_Functional {_ROOT}: its instance dict is a user container")
    model = native(LyingGlobalHookLinear)
    gm = trace_model(model, (8,))
    _hide_forward(gm.fc, _doubling_linear)
    purity = program_purity(gm)
    assert (purity.impurity.node, purity.impurity.reason) == ("fc", "a module whose instance dict is a user container")
    assert purity.foreign_modules == {"fc"}
