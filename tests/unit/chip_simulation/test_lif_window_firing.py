"""[U-LIF2 W1 to W5] The firing discipline of a synchronized LIF hop: one registry axis, ``lif_window_firing``, whose
declared set and default are the entry's data; the joint legal table, one function of the resolved point; one
resolution in the axes fold before the conversion recipe; one capability judgment every backend question reads.

Red on the set export: the axis does not exist (the run config carries no window firing, the recipe derives every
enable from the mode and the schedule, an execution extension is judged at assembly only, so a document declaring one
has SANA-FE and Loihi switched on and then refused). Green under U-LIF2's common part. Tests that need the second
value widen the entry's data for themselves (``declared``): the generic code at the set a later decision may declare.
The unit's new symbols are imported inside each test, so each test reds by itself on the set export."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimarsinan.config_schema.runtime import derive_run_config

FWI, ITE = "fire_while_integrating", "integrate_then_emit"
KEY = "lif_window_firing"
SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
_CORES = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 8}]}
_EXTERNAL = ("nevresim", "sanafe", "loihi", "odin_fpga", "odin_hacc")
_ENABLE = {"nevresim": "enable_nevresim_simulation", "sanafe": "enable_sanafe_simulation",
           "loihi": "enable_loihi_simulation", "odin_fpga": "enable_odin_fpga_simulation",
           "odin_hacc": "enable_odin_hacc_export"}
_ROWS = ("hcm", "unified", "hybrid", "training", "nevresim", "sanafe", "lava", "loihi", "odin_rtl", "odin_fpga",
         "odin_hacc", "scm")
_TORCH_EXECUTORS = ("hcm", "unified", "hybrid", "training")
#: [U-LIF2 W9] The rows the hybrid flow executes (the SCM twin is that flow on the identity program): the gated
#: cycle-level executor runs integrate then emit there.
_GATED_FLOWS = ("hcm", "unified", "hybrid", "scm")
#: [U-LIF2 W10] The rows that execute integrate then emit: the gated flows and the training walk's count branch.
_ITE_ROWS = _GATED_FLOWS + ("training",)
_EXTENSIONS = (("spike_phase_dither", True), ("lif_membrane_init", -0.25))


def _new(module: str, name: str):
    return getattr(importlib.import_module(module), name)


def _doc(**dp):
    return {"model_type": "simple_mlp", "spiking_family": "lif", "spiking_variant": "synchronized", **dp}


def _run(platform=None, **dp):
    return derive_run_config(_doc(**dp), {**_CORES, **(platform or {})})


@pytest.fixture
def declared(window_firing_declared):
    """Set the entry's data (the declared set and the default) for one test (the shared fixture)."""
    return window_firing_declared


# -- W1: the entry, its declared set and default, the joint legal table ----------------------------------------------

def test_t01_a_synchronized_lif_document_resolves_the_declared_default():
    assert _run()[KEY] == ITE  # [U-LIF2 option A] integrate then emit is the declared default


def test_t01_the_entry_holds_the_declared_set_and_its_default():
    from mimarsinan.config_schema.registry import REGISTRY, Category

    entry = REGISTRY[KEY]
    assert entry.resolved_options() == (FWI, ITE)  # the axis's values; the declared set gates the legal one
    assert entry.derived_default(_doc()) == ITE  # [U-LIF2 option A] both values declared, the default ITE
    assert entry.category is Category.ADVANCED and entry.declarable
    assert entry.legal_values(_doc()) == (FWI, ITE)


def test_t01_an_undeclared_value_is_refused_by_name(declared):
    declared((FWI,), FWI)  # a declared set that leaves a value out
    with pytest.raises(ValueError, match=r"lif_window_firing='integrate_then_emit' is not legal here: the current "
                                         r"config admits \{'fire_while_integrating'\}"):
        _run(lif_window_firing=ITE)


@pytest.mark.parametrize("axes", [{"spiking_variant": "streamed"}, {"spiking_family": "ttfs",
                                                                       "spiking_variant": "synchronized"}])
def test_t01_the_axis_does_not_apply_away_from_synchronized_lif(axes):
    assert KEY not in _run(**axes)
    with pytest.raises(ValueError, match="lif_window_firing='fire_while_integrating' is declared where the axis"):
        _run(lif_window_firing=FWI, **axes)


def test_t01_the_retired_key_names_the_new_axis_and_is_never_read_silently():
    from mimarsinan.config_schema.registry import retired_key_errors

    rows = retired_key_errors({"deployment_parameters": {"lif_execution_discipline": "synchronized"}})
    assert len(rows) == 1 and KEY in rows[0]["message"]
    with pytest.raises(ValueError, match="lif_execution_discipline='synchronized' is retired and read by no executor"):
        _run(lif_execution_discipline="synchronized")


def test_t19_the_joint_table_is_one_function_of_the_resolved_point():
    jointly_legal = _new("mimarsinan.chip_simulation.window_firing", "jointly_legal_window_firings")
    law = _new("mimarsinan.chip_simulation.soma_law", "SomaLaw")
    assert jointly_legal(law.resolve({})) == (FWI, ITE)
    assert jointly_legal(law.resolve({"firing_mode": "Novena"})) == (FWI,)
    assert jointly_legal(law.resolve({"membrane_bits": 8, "membrane_signed": True})) == (FWI, ITE)


_LEGAL_POINTS = {
    "re-timed one-level stages (every row of record)": ({}, {}),
    "fused stages": ({"lif_exact_qat": False}, {}),
    "relays off": ({"lif_depth_balancing_relays": False}, {}),
    "a signed register": ({}, {"membrane_bits": 8, "membrane_signed": True}),
}


@pytest.mark.parametrize("value", [FWI, ITE])
@pytest.mark.parametrize("point", sorted(_LEGAL_POINTS))
def test_t19_every_legal_point_admits_both_values(declared, point, value):
    declared((FWI, ITE), FWI)
    dp, platform = _LEGAL_POINTS[point]
    assert _run(platform, lif_window_firing=value, **dp)[KEY] == value


def test_t19_a_zero_reset_admits_firing_while_integrating_only(declared):
    declared((FWI, ITE), FWI)
    assert _run(firing_mode="Novena")[KEY] == FWI
    with pytest.raises(ValueError, match=r"lif_window_firing='integrate_then_emit' is not legal here: the current "
                                         r"config admits \{'fire_while_integrating'\}"):
        _run(firing_mode="Novena", lif_window_firing=ITE)


def test_t19_a_silent_document_whose_point_refuses_the_default_is_refused_by_name(declared):
    declared((FWI, ITE), ITE)
    with pytest.raises(ValueError, match="the declared default 'integrate_then_emit' is not legal at this point"):
        _run(firing_mode="Novena")


@pytest.mark.parametrize("value", [FWI, ITE])
def test_t19_per_event_and_an_unsigned_register_are_refused_under_both_values(declared, value):
    """Refused by their own keys under both values: [U-LIF2 r2, m-1; r3, n-2] a soma value its own key refuses never
    reaches the joint table, so a per-event law is refused by its own key at this point."""
    declared((FWI, ITE), FWI)
    with pytest.raises(ValueError, match="firing_granularity"):
        _run(firing_granularity="per_event", lif_window_firing=value)
    with pytest.raises(ValueError, match="saturating_unsigned"):
        _run({"membrane_bits": 8}, lif_window_firing=value)


# -- W2: one resolution, before the recipe; the removed key; the record every reader takes --------------------------

def test_t01a_the_value_is_resolved_before_the_recipe(monkeypatch):
    import mimarsinan.config_schema.deployment_derivation as derivation

    seen = []
    real = derivation.fold_conversion_recipe

    def spy(dp, *args, **kwargs):
        seen.append(dp.get(KEY))
        return real(dp, *args, **kwargs)

    monkeypatch.setattr(derivation, "fold_conversion_recipe", spy)
    assert _run()[KEY] == ITE and seen == [ITE]


@pytest.mark.parametrize("values,default", [((FWI,), FWI), ((FWI, ITE), FWI), ((FWI, ITE), ITE), ((ITE,), ITE)])
def test_t01b_a_silent_row_equals_the_row_declaring_the_default(declared, values, default):
    declared(values, default)
    silent, declaring = _run(), _run(lif_window_firing=default)
    assert silent == declaring and silent[KEY] == default


@pytest.mark.parametrize("default", [FWI, ITE])
def test_t01b_both_values_declared_with_default_v_give_the_one_value_option_of_v(declared, default):
    declared((FWI, ITE), default)
    both = _run()
    declared((default,), default)
    assert both == _run()


def test_t02_the_record_is_frozen_and_idempotent_over_raw_and_resolved_configs():
    resolve = _new("mimarsinan.chip_simulation.spiking_point", "resolve_spiking_point")
    cfg = _run()
    point = resolve(cfg)
    assert point == resolve(_doc()) == resolve(dict(cfg))
    with pytest.raises(Exception):
        point.window_firing = ITE  # type: ignore[misc]


def test_t02_the_contract_the_plan_and_the_behavior_carry_one_record():
    resolve = _new("mimarsinan.chip_simulation.spiking_point", "resolve_spiking_point")
    from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
    from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import firing_semantics_kwargs

    cfg = _run(spike_phase_dither=True)
    point = resolve(cfg)
    contract = SpikingDeploymentContract.from_pipeline_config(cfg)
    plan = DeploymentPlan.resolve(cfg)
    assert contract.spiking_point() == plan.require_spiking_point() == contract.behavior.spiking_point() == point
    assert point.execution_extensions == ("spike_phase_dither",) and point.window_firing == ITE
    kwargs = firing_semantics_kwargs(plan, cfg)
    assert (kwargs["spiking_mode"], kwargs["ttfs_cycle_schedule"]) == (point.spiking_mode, point.ttfs_cycle_schedule)
    assert kwargs["window_firing"] == point.window_firing  # [U-LIF2 W11] the search and its twin price one program
    assert contract.lif_execution_synchronized is True


#: [U-LIF2 r6, M-2] The finite list of the modules that read the key, each with its reason, held beside the search
#: (``_key_reads``) that fails on a read in any other module.
_RESOLVER_MODULES = {
    "mimarsinan.chip_simulation.window_firing": "the resolver: the declaration a document makes",
    "mimarsinan.chip_simulation.spiking_point": "the resolver: the run's point, a stored row's and a stored document's",
}
_READ_CALLS = frozenset({"get", "pop", "setdefault"})


def _key_reads(root: Path, package: str, key: str) -> list:
    """Every read of ``key`` in the package at ``root``, read from the syntax trees: a subscript load, a get, pop or
    setdefault call (the key among its first two arguments, so the unbound ``dict.get(d, key)`` too) and a membership
    test whose key is the literal or a name bound to it. A name is followed through imports (absolute, relative,
    aliased, starred), module attributes (``m.KEY``, ``a.b.m.KEY``), simple bindings (an assignment, an annotated or
    walrus binding, a for loop or comprehension over a literal tuple, list or set holding the key) and names bound to
    such a literal, to a fixpoint over the package. Returns sorted ``(module, line, kind)``."""
    trees = {}
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        name = ".".join([package, *(parts[:-1] if parts[-1] == "__init__" else parts)])
        trees[name] = (path.name == "__init__.py", ast.parse(path.read_text(encoding="utf-8")))
    keys, holders = set(), set()
    aliases = {m: {} for m in trees}
    imported = {m: [] for m in trees}
    for m, (is_package, tree) in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    aliases[m][alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else \
                        alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                base = m.split(".") if is_package else m.split(".")[:-1]
                source = (node.module or "") if not node.level else ".".join(
                    base[:len(base) - (node.level - 1)] + ([node.module] if node.module else []))
                for alias in node.names:
                    if f"{source}.{alias.name}" in trees:
                        aliases[m][alias.asname or alias.name] = f"{source}.{alias.name}"
                    imported[m].append((alias.asname or alias.name, source, alias.name))

    def module_of(m, node):
        if isinstance(node, ast.Name):
            return aliases[m].get(node.id)
        if isinstance(node, ast.Attribute):
            base = module_of(m, node.value)
            return f"{base}.{node.attr}" if base is not None and f"{base}.{node.attr}" in trees else None
        return None

    def bound(m, node, names):
        if isinstance(node, ast.Name):
            return (m, node.id) in names
        if isinstance(node, ast.Attribute):
            base = module_of(m, node.value)
            return base is not None and (base, node.attr) in names
        return False

    def is_key(m, node):
        return node.value == key if isinstance(node, ast.Constant) else bound(m, node, keys)

    def holds_key(m, node):
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return any(holds_key(m, e.value) if isinstance(e, ast.Starred) else is_key(m, e) for e in node.elts)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
            return node.func.id in ("tuple", "list", "set", "frozenset") and holds_key(m, node.args[0])
        return bound(m, node, holders)

    size = -1
    while size != len(keys) + len(holders):
        size = len(keys) + len(holders)
        for m, (_is_package, tree) in trees.items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.comprehension)):
                    if isinstance(node.target, ast.Name) and holds_key(m, node.iter):
                        keys.add((m, node.target.id))
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
                    node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None else []
                for target in (t for t in targets if isinstance(t, ast.Name)):
                    if is_key(m, node.value):
                        keys.add((m, target.id))
                    elif holds_key(m, node.value):
                        holders.add((m, target.id))
            for local, source, name in imported[m]:
                for names in (keys, holders):
                    if name == "*":
                        names.update({(m, n) for s, n in list(names) if s == source})
                    elif (source, name) in names:
                        names.add((m, local))
    reads = set()
    for m, (_is_package, tree) in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and is_key(m, node.slice):
                reads.add((m, node.lineno, "subscript"))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    node.func.attr in _READ_CALLS and any(is_key(m, arg) for arg in node.args[:2]):
                reads.add((m, node.lineno, node.func.attr))
            elif isinstance(node, ast.Compare) and is_key(m, node.left) and \
                    any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
                reads.add((m, node.lineno, "membership"))
    return sorted(reads)


def test_t02_only_the_resolver_reads_the_key_and_the_removed_reader_is_gone():
    """[U-LIF2 r6, M-2] The key is read only in the finite list's modules (the search over every src module's syntax
    tree fails on an unlisted read), and each listed module reads it (no stale entry)."""
    reads = _key_reads(SRC, "mimarsinan", KEY)
    assert [read for read in reads if read[0] not in _RESOLVER_MODULES] == []
    assert sorted({read[0] for read in reads}) == sorted(_RESOLVER_MODULES)
    semantics = (SRC / "chip_simulation" / "spiking_semantics.py").read_text()
    assert "def lif_execution_synchronized" not in semantics
    old_key = sorted(path.relative_to(SRC).as_posix() for path in SRC.rglob("*.py")
                     if "lif_execution_discipline" in path.read_text())
    assert old_key == ["chip_simulation/activation_axes.py", "chip_simulation/spiking_point.py",
                       "config_schema/registry/retired_keys.py"]


#: [U-LIF2 r6, M-2] The ratchet's self-check: shapes a read of the key takes (the verify's six, then shapes of this
#: delta's own), each a module of a package beside ``keys.py`` (``WINDOW_FIRING_KEY = <the key>``, as src binds it)
#: that the search must find. Each is red: a module in src holding it fails t02. A write and a mention that is no read
#: are found by none (the last two).
_K = "WINDOW_FIRING_KEY"
_SHAPES = {
    "a pop with a default": {"shape": f"from pkg.keys import {_K}\n\ndef f(doc):\n    return doc.pop({_K}, None)\n"},
    "a membership test": {"shape": f"from pkg.keys import {_K}\n\ndef f(doc):\n    return {_K} in doc\n"},
    "a setdefault": {"shape": f"from pkg.keys import {_K}\n\ndef f(doc, v):\n    return doc.setdefault({_K}, v)\n"},
    "an aliased import": {"shape": f"from pkg.keys import {_K} as FIRING\n\ndef f(doc):\n    return doc.get(FIRING)\n"},
    "a module attribute": {"shape": f"from pkg import keys\n\ndef f(doc):\n    return doc[keys.{_K}]\n"},
    "a bound alias": {"shape": f"from pkg.keys import {_K}\n\ndef f(doc):\n    name = {_K}\n    return doc[name]\n"},
    "a relative import with a negated membership test": {
        "shape": f"from .keys import {_K}\n\ndef f(doc):\n    return {_K} not in doc\n"},
    "a binding re-exported by another module, read by the unbound get": {
        "names": f"import pkg.keys as k\nFIRING = k.{_K}\n",
        "shape": "from pkg.names import FIRING\n\ndef f(doc):\n    return dict.get(doc, FIRING)\n"},
    "a loop over a tuple of keys bound in another module": {
        "names": f"from pkg.keys import *\nKEYS = ({_K}, 'other')\n",
        "shape": "from pkg import names\n\ndef f(doc):\n    for name in names.KEYS:\n        doc.pop(name, None)\n"},
    "a dotted module path": {"shape": f"import pkg.keys\n\ndef f(doc):\n    return doc[pkg.keys.{_K}]\n"},
    "the literal in a comprehension's subscript": {
        "shape": "def f(docs):\n    return [d['lif_window_firing'] for d in docs]\n"},
    "a write is no read": {"shape": f"from pkg.keys import {_K}\n\ndef f(doc):\n    doc[{_K}] = 1\n"},
    "a message is no read": {"shape": f"from pkg.keys import {_K}\n\ndef f():\n    return f'{{{_K}}} is declared'\n"},
}


@pytest.mark.parametrize("shape", tuple(_SHAPES))
def test_t02_self_check_the_search_finds_each_shape_of_a_read(tmp_path, shape):
    """[U-LIF2 r6, M-2] Each shape is found as a read of the key in its own module, and only there; a write and a
    message are found by none."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "keys.py").write_text(f"{_K} = {KEY!r}\n")
    for name, text in _SHAPES[shape].items():
        (package / f"{name}.py").write_text(text)
    found = {read[0] for read in _key_reads(package, "pkg", KEY)}
    assert found == (set() if shape.endswith("is no read") else {"pkg.shape"}), (shape, found)


_BUILDERS = {"policy_for_spiking_mode", "LifModePolicy", "TtfsSyncCycleModePolicy", "TtfsCascadeModePolicy",
             "TtfsAnalyticalModePolicy", "SpikingModePolicy", "_TtfsCycleModePolicy"}


def _builder_calls():
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name == "derive" and isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and \
                    func.value.id == "ConversionPolicy":
                yield path.relative_to(SRC).as_posix(), "ConversionPolicy.derive", node
            elif name in _BUILDERS:
                yield path.relative_to(SRC).as_posix(), name, node


def test_t02b_every_builder_and_recipe_call_takes_the_record_alone():
    """The record is the one required argument of the recipe, of the policy builder and of every policy constructor
    (the review's p-4: the constructors are scanned too); a constructor is called only by the builder."""
    bad = [f"{path}:{node.lineno} {name}" for path, name, node in _builder_calls()
           if len(node.args) != 1 or node.keywords or isinstance(node.args[0], ast.Constant)]
    assert bad == []
    constructors = sorted({path for path, name, _node in _builder_calls()
                           if name not in ("policy_for_spiking_mode", "ConversionPolicy.derive")})
    assert constructors == ["chip_simulation/spiking_mode_policy.py"]


# -- W3: the capability column and the extensions' sets, read in one judgment ----------------------------------------

@pytest.mark.parametrize("backend", _ROWS + ("an_undeclared_name",))
def test_t04_each_row_declares_the_window_firings_it_executes(backend):
    from mimarsinan.chip_simulation.spiking_semantics import backend_capabilities

    expected = frozenset() if backend == "an_undeclared_name" else frozenset(
        {FWI, ITE} if backend in _ITE_ROWS else {FWI})
    assert backend_capabilities(backend).window_firings == expected


def test_t04_every_declared_value_is_executed_by_the_toolchain_own_executors():
    from mimarsinan.chip_simulation.spiking_semantics import backend_capabilities

    declared_set = _new("mimarsinan.chip_simulation.window_firing", "declared_window_firings")()
    assert [(backend, value) for value in declared_set for backend in _TORCH_EXECUTORS
            if value not in backend_capabilities(backend).window_firings] == []


@pytest.mark.parametrize("backend", _ROWS + ("an_undeclared_name",))
def test_t04_the_one_judgment_names_the_window_firing(declared, backend):
    resolve = _new("mimarsinan.chip_simulation.spiking_point", "resolve_spiking_point")
    reason = _new("mimarsinan.chip_simulation.point_capability", "backend_refusal_reason")
    declared((FWI, ITE), FWI)
    assert reason(backend, resolve(_doc(lif_window_firing=ITE))) == (
        None if backend in _ITE_ROWS else "lif_window_firing='integrate_then_emit'")
    executes = reason(backend, resolve(_doc()))
    assert executes == (None if backend != "an_undeclared_name" else "lif_window_firing='fire_while_integrating'")


def test_t22_the_torch_executors_implement_both_extensions_and_no_other_row_does():
    implemented = _new("mimarsinan.chip_simulation.backend_extensions", "implemented_extensions")
    both = frozenset(key for key, _value in _EXTENSIONS)
    assert {backend: implemented(backend) for backend in _ROWS} == {
        backend: (both if backend in _TORCH_EXECUTORS else frozenset()) for backend in _ROWS}


@pytest.mark.parametrize("key,value", _EXTENSIONS)
def test_t22_a_declared_extension_derives_every_backend_lacking_it_off_with_its_reason(key, value):
    from mimarsinan.config_schema.registry import REGISTRY

    cfg = _run(**{key: value, KEY: FWI})  # [U-LIF2 W16] at firing while integrating the extension is the reason
    for backend in ("nevresim", "sanafe", "loihi"):
        enable = _ENABLE[backend]
        assert cfg[enable] is False, backend
        assert key in REGISTRY[enable].why(cfg), backend


@pytest.mark.parametrize("backend", ("nevresim", "sanafe", "loihi", "odin_fpga", "odin_hacc"))
@pytest.mark.parametrize("key,value", _EXTENSIONS)
def test_t22_an_explicit_on_for_a_backend_lacking_the_extension_is_refused_at_resolution(key, value, backend):
    with pytest.raises(ValueError, match=rf"{_ENABLE[backend]}=true contradicts the resolved point: the backend does "
                                         rf"not execute the declared execution extension\(s\) \['{key}'\]"):
        _run(**{key: value, _ENABLE[backend]: True, KEY: FWI})  # [U-LIF2 W16] the extension is the reason


# -- W5: the assembly, step and runner checks read the record, as guards --------------------------------------------

def _ite_config(declared):
    declared((FWI, ITE), FWI)
    return _run(lif_window_firing=ITE)


@pytest.mark.parametrize("backend", _EXTERNAL)
def test_t04b_the_assembly_check_refuses_the_window_firing_by_the_one_message(declared, backend):
    refusal = _new("mimarsinan.chip_simulation.point_capability", "BackendWindowFiringRefusal")
    from mimarsinan.chip_simulation.backend import BACKEND_REGISTRY
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

    plan = DeploymentPlan.resolve(_ite_config(declared))
    with pytest.raises(refusal, match=rf"ctx: backend '{backend}' does not execute "
                                      rf"lif_window_firing='integrate_then_emit'"):
        BACKEND_REGISTRY.get(backend).require_supported(plan, context="ctx")


@pytest.mark.parametrize("backend", _EXTERNAL)
def test_t04b_the_step_guard_refuses_by_the_one_message(declared, backend):
    refusal = _new("mimarsinan.chip_simulation.point_capability", "BackendWindowFiringRefusal")
    from mimarsinan.pipelining.core.engine.pipeline_helpers import require_spiking_mode_supported

    pipeline = SimpleNamespace(config=_ite_config(declared))
    with pytest.raises(refusal, match=rf"Step: backend '{backend}' does not execute lif_window_firing="):
        require_spiking_mode_supported(pipeline, "Step", backend=backend)


def test_t04b_every_runner_refuses_the_window_firing_directly(declared):
    refusal = _new("mimarsinan.chip_simulation.point_capability", "BackendWindowFiringRefusal")
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
    from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.chip_simulation.odin_fpga.runner import OdinFpgaRunner
    from mimarsinan.chip_simulation.odin_hacc.program_freeze import ProgramFreezer
    from mimarsinan.chip_simulation.sanafe.runner.core import SanafeRunner

    cfg = _ite_config(declared)
    contract = SpikingDeploymentContract.from_pipeline_config(cfg)
    message = "does not execute lif_window_firing='integrate_then_emit'"
    with pytest.raises(refusal, match=f"SanafeRunner: backend 'sanafe' {message}"):
        SanafeRunner(None, 4, contract=contract)
    with pytest.raises(refusal, match=f"OdinFpgaRunner: backend 'odin_fpga' {message}"):
        OdinFpgaRunner(None, 4, contract=contract, transport=None, weight_bits=8, effective_max_axons=256,
                       max_neurons=256, pass_transfer="collapse")
    with pytest.raises(refusal, match=f"ProgramFreezer: backend 'odin_fpga' {message}"):
        ProgramFreezer(None, contract=contract, timesteps=4, weight_bits=8, effective_max_axons=256,
                       max_neurons=256, membrane_init=0, weight_sign_granularity="per_axon")
    with pytest.raises(refusal, match=f"NeuralBehaviorConfig.for_lava: backend 'lava' {message}"):
        NeuralBehaviorConfig.for_lava(cfg)
    # [U-LIF2 r3, n-5] nevresim's driver judges the point it holds, as every other runner does, before any work.
    with pytest.raises(refusal, match=f"NevresimDriver: backend 'nevresim' {message}"):
        NevresimDriver(1, None, None, None, spike_generation_mode="Uniform", thresholding_mode="<=",
                       point=contract.spiking_point())


_POINTS = {
    "silent": {}, "fused": {"lif_exact_qat": False}, "zero reset": {"firing_mode": "Novena"},
    "relays off": {"lif_depth_balancing_relays": False},
    **{f"extension {key}": {key: value} for key, value in _EXTENSIONS},
}


@pytest.mark.parametrize("value", [FWI, ITE])
@pytest.mark.parametrize("point", sorted(_POINTS))
def test_t04c_no_derived_config_turns_a_backend_on_and_then_refuses_it(declared, point, value):
    from mimarsinan.chip_simulation.backend import BACKEND_REGISTRY
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

    declared((FWI, ITE), FWI)
    if point == "zero reset" and value == ITE:
        pytest.skip("refused at resolution by the joint table (t19)")
    cfg = _run(lif_window_firing=value, **_POINTS[point])
    plan = DeploymentPlan.resolve(cfg)
    BACKEND_REGISTRY.selected_step_specs(plan)
    for backend in _EXTERNAL:
        if cfg[_ENABLE[backend]]:
            BACKEND_REGISTRY.get(backend).require_supported(plan, context="ctx")
    if value == ITE:
        assert not any(cfg[_ENABLE[backend]] for backend in _EXTERNAL)
