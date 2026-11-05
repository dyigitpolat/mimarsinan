"""[U-ADC] Readout-aware training: Weight Quantization's endpoint recovery trains and reads the deployed readout law.

``readout_aware_training: true`` (absent or false: every run byte for byte) puts the deployed readout law in the forward
of Weight Quantization's endpoint recovery (P1''), the one tuning step that trains after the law is known on a value
row. The law is the one Soft Core Mapping deploys: per forward of the live model, when its weights or buffers moved
since the last solve, a copy of the model is handed over as Normalization Fusion hands it on (its normalization fused)
and Soft Core Mapping derives its IR with its own preparation and passes (``SoftCoreMappingStep.derive_law_ir``), so
every core carries the law ``install_readout_laws`` solves at the declared ``readout_law.adc_bits`` (its lane laws
where lanes execute) and the copy carries the laws stamped. The live model then reads the copy's stamps themselves
(``ReadoutQuantizer.adopt_law``, the pool lanes' stamps): its stamped forwards and lane ends evaluate the deployed law
on the live inputs, with the stamped quantizer's straight-through estimator (``straight_through``: the law's values
forward, the identity onto the live float output backward).

The law exists on integer weights only, so it enters where Weight Quantization's projection reaches rate 1.0: the
recovery's entry read, its training steps, its checks, its guards' deployed reads and its exit read run the
law-in-loop model; the projection ladder before it (no training step) and the step's own metric read the law-free
composition as before. The deployed program is the one Soft Core Mapping exports from the trained weights: no second
law. The first eval-mode forward after the first solve is compared with the law-carrying copy's forward on the same
inputs (``torch.equal``), and a difference is refused by name.

[U-RAW] The recovery's rule under the law (``recovery_lr``, the endpoint's ``RECOVERY_RULE``): the rate the LR explorer
finds non-destructive on the law-in-loop read, its probes at the rates they name, capped at the pipeline's ``lr`` (no
step where every rate is destructive), over the declared budget with keep-best (the funded convergence geometry).

Scope, refused by name at Weight Quantization's entry before any budget is spent: a value-domain (mvm) row with a
declared readout law, the fast optimization driver (whose fixed ladder runs the endpoint recovery), fixed hardware (no
architecture search), no structured pre-mapping pruning (it would change the mapped network's channels after the
loop) and no operand core type (its banks take their laws on the operand path, which this loop does not read)."""

from __future__ import annotations

import copy
import gc
import itertools
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.support.operand_lowering import operand_declaration
from mimarsinan.model_training.basic_trainer_steps import CONSTANT_RATE_EXACT
from mimarsinan.models.nn.activations.lane_quantization import LanePoolModule
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import SoftCoreMappingStep
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
from mimarsinan.tuning.orchestration.frontier.endpoint_recovery import RECOVERY_RULE

READOUT_AWARE_TRAINING = "readout_aware_training"
#: The tuner attribute naming the loop open on its endpoint recovery (absent outside one).
LOOP_ATTRIBUTE = "_readout_law_loop"


class ReadoutAwareTrainingError(ToolchainRefusal, ValueError):
    """``readout_aware_training`` declared where the deployed readout law cannot enter Weight Quantization's forward."""


class ReadoutAwareParityError(ToolchainRefusal, RuntimeError):
    """The law-in-loop forward differs from the law-carrying copy's: the law in the loop is not the deployed law."""


def readout_aware_declared(config: Mapping[str, Any]) -> bool:
    return bool(config.get(READOUT_AWARE_TRAINING) or False)


def _refuse(reason: str) -> None:
    raise ReadoutAwareTrainingError(f"{READOUT_AWARE_TRAINING}: {reason}")


def readout_aware_endpoint(pipeline) -> Optional[Callable[[Any], "ReadoutLawInLoop"]]:
    """``None`` when undeclared or false; else, the declaration judged in scope (refused by name otherwise), the factory
    Weight Quantization's tuner opens around its endpoint recovery."""
    config = pipeline.config
    if not readout_aware_declared(config):
        return None
    plan = DeploymentPlan.resolve(config)
    if not is_mvm_core_semantics(plan.core_semantics):
        _refuse(f"a spiking row (core_semantics {plan.core_semantics!r}) has no readout law; the deployed readout law "
                "is the value-domain (mvm) law")
    if config.get("readout_law") is None:
        _refuse("no readout_law is declared, so no deployed readout law exists to train through")
    if not plan.is_fast_driver:
        _refuse(f"the optimization driver {plan.optimization_driver!r} runs no Weight Quantization endpoint recovery "
                "(the fast driver's fixed ladder runs it), so nothing would train through the law")
    if plan.search_mode != "fixed":
        _refuse(f"under an architecture search (search_mode {plan.search_mode!r}) the mapped platform is the search's "
                "winner, known only at Soft Core Mapping")
    if float(plan.prune_sparsity) > 0.0:
        _refuse(f"structured pre-mapping pruning (prune_sparsity {plan.prune_sparsity}) changes the mapped network's "
                "channels at Soft Core Mapping, after the loop")
    if operand_declaration(config) is not None:
        _refuse("an operand core type's banks take their laws on the operand path (operand stamps), which this loop "
                "does not read")
    platform = build_platform_constraints_resolved(config)   # the one function Model Configuration resolves it by
    return lambda tuner: ReadoutLawInLoop(pipeline, platform, tuner)


def current_loop(tuner) -> Optional["ReadoutLawInLoop"]:
    """The loop open on ``tuner``'s endpoint recovery, else ``None``."""
    return getattr(tuner, LOOP_ATTRIBUTE, None)


class ReadoutLawInLoop:
    """While open, every forward of the tuner's live model applies the deployed readout law (module docstring)."""

    def __init__(self, pipeline, platform: Mapping[str, Any], tuner) -> None:
        self._pipeline = pipeline
        self._platform = dict(platform)
        self._tuner = tuner
        self._model: Any = tuner.model
        self._mapping = SoftCoreMappingStep(pipeline)
        self._model_hooks: List[Any] = []
        self._wires: List[Any] = []
        self._adopted: List[ReadoutQuantizer] = []
        self._pools: List[LanePoolModule] = []
        self._snapshot: Optional[List[torch.Tensor]] = None
        self._replica: Optional[nn.Module] = None
        self._parity_due = False
        self.last_ir = None
        self.solves = 0
        self.solve_wall_s = 0.0

    # -- context --------------------------------------------------------------------------------------------------

    def __enter__(self) -> "ReadoutLawInLoop":
        if getattr(self._tuner.trainer, "model", self._model) is not self._model:
            raise RuntimeError("[U-ADC] the tuner's trainer trains another model than the tuner's")
        self._hook_model()
        setattr(self._tuner, LOOP_ATTRIBUTE, self)
        setattr(self._tuner, RECOVERY_RULE, self.recovery_lr)   # [U-RAW] the recovery's rule under the law
        return self

    def __exit__(self, *_exc) -> bool:
        self._unhook_model()
        self._unwire()
        if getattr(self._tuner, LOOP_ATTRIBUTE, None) is self:
            delattr(self._tuner, LOOP_ATTRIBUTE)
            delattr(self._tuner, RECOVERY_RULE)
        self._replica, self._snapshot = None, None
        gc.collect()
        law = self._platform.get("readout_law") or {}
        print(f"[ReadoutAwareTraining] Weight Quantization's endpoint recovery read and trained the deployed readout "
              f"law (adc_bits {law.get('adc_bits')}): {self.solves} solve(s) of Soft Core Mapping's laws, "
              f"{self.solve_wall_s:.1f} s", flush=True)
        return False

    def recovery_lr(self, lr: float) -> Optional[float]:
        """The LR explorer's rate (``_find_lr``) on the law-in-loop read, its probes at the rates they name (the mark on
        the trainer for this call only), capped at ``lr``; None (no step) when every probed rate is destructive."""
        setattr(self._tuner.trainer, CONSTANT_RATE_EXACT, True)
        try:
            found = self._tuner._find_lr()
        finally:
            delattr(self._tuner.trainer, CONSTANT_RATE_EXACT)
        rate = None if found is None else min(float(found), float(lr))
        print(f"[ReadoutAwareTraining] the recovery trains at {rate} on the law-in-loop read (LR explorer {found}, "
              f"pipeline lr {lr})", flush=True)
        return rate

    def _hook_model(self) -> None:
        self._model_hooks = [self._model.register_forward_pre_hook(self._before_forward),
                             self._model.register_forward_hook(self._after_forward)]

    def _unhook_model(self) -> None:
        for handle in self._model_hooks:
            handle.remove()
        self._model_hooks = []

    # -- the live model's forward -------------------------------------------------------------------------------

    def _state(self) -> List[torch.Tensor]:
        return [t for t in itertools.chain(self._model.parameters(), self._model.buffers()) if t is not None]

    def _stale(self) -> bool:
        state = self._state()
        if self._snapshot is None or len(state) != len(self._snapshot):
            return True
        return not all(torch.equal(a, b) for a, b in zip(state, self._snapshot))

    def _before_forward(self, module, _args):
        if module is self._model and self._stale():
            self._solve()
        return None

    def _after_forward(self, module, args, output):
        if module is not self._model or not self._parity_due or module.training:
            return None
        self._parity_due = False
        replica = self._replica
        assert replica is not None
        with torch.no_grad():
            want = replica(*args)
        if not (isinstance(output, torch.Tensor) and isinstance(want, torch.Tensor) and torch.equal(output, want)):
            delta = (float((output.double() - want.double()).abs().max())
                     if isinstance(output, torch.Tensor) and isinstance(want, torch.Tensor)
                     and output.shape == want.shape else float("nan"))
            raise ReadoutAwareParityError(
                f"{READOUT_AWARE_TRAINING}: the law-in-loop forward differs from the law-carrying copy's forward "
                f"(max|delta| {delta:.3e}): the law in the loop would not be the deployed law")
        return None

    # -- the deployed law ----------------------------------------------------------------------------------------

    def _solve(self) -> None:
        """The deployed law for the live state: a copy handed over as Normalization Fusion hands the model on, mapped
        by Soft Core Mapping's own derivation; then the live model reads the copy's laws (``_wire``)."""
        started = time.perf_counter()
        self._unhook_model()   # neither these hooks nor the wires may be copied into the replica
        self._unwire()
        self._replica = None
        try:
            with torch.no_grad():
                memo: Dict[int, Any] = {}
                replica = copy.deepcopy(self._model, memo)
                replica.eval()
                device = self._pipeline.config["device"]
                for perceptron in replica.get_perceptrons():   # Normalization Fusion's transform
                    fuse_into_perceptron(perceptron, device=device)
                self.last_ir = self._mapping.derive_law_ir(replica, self._platform)
            self._replica = replica
            self._wire(memo)
            del memo
            self._snapshot = [t.detach().clone() for t in self._state()]
            self._parity_due = self.solves == 0
            self.solves += 1
        finally:
            self._hook_model()
            gc.collect()
            self.solve_wall_s += time.perf_counter() - started

    def _wire(self, memo: Mapping[int, Any]) -> None:
        """The live model reads the copy's laws through the copy's own stamps: every live readout quantizer the copy
        stamped adopts that stamp and its hand-off state (``ReadoutQuantizer.adopt_law``), every pool lane takes its
        copy's pool stamp, and each perceptron's input reaches its quantizer's state as the copy's input tap hands it
        on; a perceptron whose quantizer the copy dropped (one the program computes on the host) keeps its float
        output. The stamped forwards and lane ends are then the copy's, read on the live floats, with their
        straight-through estimators."""
        for module in self._model.modules():
            copied = memo.get(id(module))
            if isinstance(module, ReadoutQuantizer) and isinstance(copied, ReadoutQuantizer) \
                    and copied.stamped is not None:
                module.adopt_law(copied)
                self._adopted.append(module)
            elif isinstance(module, LanePoolModule) and isinstance(copied, LanePoolModule) and copied.stamp is not None:
                module.stamp = copied.stamp
                self._pools.append(module)
        wired = set()
        for live in self._model.get_perceptrons():
            quantizer, copied = live.scaler, memo[id(live)]
            if not isinstance(quantizer, ReadoutQuantizer):
                continue
            if copied.scaler is not memo.get(id(quantizer)):   # the copy computes this perceptron on the host
                if not isinstance(copied.scaler, nn.Identity):
                    raise ReadoutAwareParityError(
                        f"{READOUT_AWARE_TRAINING}: perceptron {getattr(live, 'name', '?')!r} reads its output "
                        f"through {type(copied.scaler).__name__} on the mapped copy")
                self._wires.append(quantizer.register_forward_hook(lambda _m, args, _out: args[0]))
                continue
            if quantizer.stamped is None:
                raise ReadoutAwareParityError(
                    f"{READOUT_AWARE_TRAINING}: perceptron {getattr(live, 'name', '?')!r} has a readout quantizer but "
                    "Soft Core Mapping stamped no law on it")
            wire = live.input_activation
            if id(wire) in wired:
                raise ReadoutAwareTrainingError(
                    f"{READOUT_AWARE_TRAINING}: perceptrons share one input wire module ({type(wire).__name__}): "
                    "the input each readout reads cannot be told apart")
            wired.add(id(wire))
            tap = quantizer.input_tap()   # on the adopted state: the copy's own hand-off
            self._wires += [live.register_forward_pre_hook(_entry_tap(tap)), wire.register_forward_hook(_wire_tap(tap))]

    def _unwire(self) -> None:
        for handle in self._wires:
            handle.remove()
        for quantizer in self._adopted:
            quantizer.release_law()
        for pool in self._pools:
            pool.stamp = None
        self._wires, self._adopted, self._pools = [], [], []


def _entry_tap(tap: nn.Module):
    """The perceptron's input at its forward's entry: the input the readout reads where the input wire is the identity
    its forward skips (a later wire hook replaces it where the wire runs)."""
    def hook(_module, args):
        tap(args[0])
        return None
    return hook


def _wire_tap(tap: nn.Module):
    """The perceptron's input after its input wire, where the copy's input tap reads it."""
    def hook(_module, _args, output):
        tap(output)
        return None
    return hook
