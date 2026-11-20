"""[U-POST2] Every restore returns what a reader reads: state held outside the state dictionary.

The toolchain's restores are state-dictionary restores: the keep-best of a training leg and the learning-rate finder's
trials (``restore_state_for_trainer``), the cycle's, the endpoint's and the run's keep-best through the checkpoint
guard (every scope and location), the fast ladder's rung keep-best, a fold step's rollback. A forward, an arming, a
projection or a gate that reads state held outside the state dictionary reads, after such a restore, whatever the
window last wrote there (the verify's M-1: the projection's membrane-lattice stamp outlived the keep-best restore of an
engaged Weight Quantization endpoint, so the published and deployed reads armed a lattice the chip does not run).

One rule closes the class: what describes the persisted parameters is derived from them at every read and never
cached (the membrane lattice, from the perceptron's recorded lattice kind and its persisted ``parameter_scale``; the
LIF node's armed lattice is written by the arming alone), and what records the parameters' history (a perceptron's
bake, install, bound and lattice-kind records) is carried with them by the toolchain's one snapshot. State the tuning
axes own (a blend rate, a window length, a rate buffer) returns through the cycle's extra state, and state a reader
re-derives at its own start (the wire gauge a twin walks, the membrane a forward resets) needs neither.

Each site is read here by a clone, a move and a restore with the toolchain's own instruments. The forms are converted
torch blocks of the toolchain's own classes (``test_wq_integer_program``'s residual block with a host BatchNorm).
"""

from __future__ import annotations

import ast
import copy
import pathlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.support.bias_compensation import (
    LIF_HALF_STEP_FLAG,
    SYNC_ENTRY_HALF_STEP_FLAG,
    TTFS_COMP_BAKED_FLAG,
    apply_lif_half_step_bias_compensation,
)
from mimarsinan.models.nn.lif_kernels import in_measurement_plane
from mimarsinan.models.perceptron_mixer.membrane_lattice import integer_program_read
from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice, unwrap_lif_activation
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.tuning.learning_rate_explorer import clone_state_for_trainer, restore_state_for_trainer
from mimarsinan.tuning.orchestration.checkpoint_guard import CheckpointGuard

from unit.pipelining.test_wq_integer_program import T, WQ_BITS, _form, _scm_program, _step_projection, _tuner
from unit.spiking.host_gather_forms import compare

SEMANTICS = ("streamed_lif", "windowed_lif")


def _cores(perceptrons):
    return [p for p in perceptrons if not getattr(p, "is_encoding_layer", False)]


def _scales(perceptrons):
    return [float(p.parameter_scale) for p in perceptrons]


def _armed(perceptrons):
    return [unwrap_lif_activation(p.activation).if_node.lattice_scale for p in perceptrons]


def _lattices_of_the_persisted_scales(perceptrons):
    """The lattice each perceptron's persisted parameters put its membrane on: 2 s on a projected core, none on the
    host-side encoder (a float input)."""
    return [None if getattr(p, "is_encoding_layer", False) else 2.0 * float(p.parameter_scale) for p in perceptrons]


def _move(perceptrons, factor: float = 0.6) -> None:
    """An endpoint training's parameter move that shifts the integer grid: every core's normalization affine scaled."""
    with torch.no_grad():
        for p in _cores(perceptrons):
            p.normalization.weight.mul_(factor)
            p.normalization.bias.mul_(factor)


# ---------------------------------------------------------------------------------------------------------------------
# The membrane lattice: derived from the persisted parameters at every arming, never cached
# ---------------------------------------------------------------------------------------------------------------------


class TestTheMembraneLatticeIsDerived:
    def test_v06_the_keep_best_rollback_arms_the_restored_scale_and_parity_holds(self):
        """The verify's v06 form: a projection, a snapshot, a parameter move that changes the integer scale, a
        projection again, the keep-best restore; then Soft Core Mapping's sequence arms the lattice of the restored
        scale and NF-SCM parity reads 0 flips on both LIF semantics (the stale stamp read 2 and 26 mismatches)."""
        for seed in (0, 1):
            flow, perceptrons = _form(seed)
            _step_projection(flow)
            trainer = SimpleNamespace(model=flow)
            snapshot = clone_state_for_trainer(trainer)
            restored_scales = _scales(perceptrons)
            _move(perceptrons)
            _step_projection(flow)
            assert _scales(perceptrons) != restored_scales  # the move changed the integer grid
            restore_state_for_trainer(trainer, snapshot)
            assert _scales(perceptrons) == restored_scales
            program = _scm_program(flow, perceptrons)
            assert _armed(perceptrons) == _lattices_of_the_persisted_scales(perceptrons)
            for semantics in SEMANTICS:
                mismatches, compared, out_diff = compare(program, semantics, n=24, seed=seed)
                assert compared > 0 and sum(mismatches.values()) == 0, (seed, semantics, mismatches)
                assert out_diff < 1e-3, out_diff

    def test_the_transform_trainers_keep_best_restore_arms_the_restored_scale(self):
        """The Weight Quantization endpoint's path: the transform trainer projects a copy of the float auxiliary model
        and writes the parameters back each step; its keep-best restores both models. The arming then reads the
        restored scale on every core, and the published read arms exactly that."""
        from mimarsinan.model_training.perceptron_transform_trainer import PerceptronTransformTrainer
        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

        flow, perceptrons = _form(0)
        trainer = PerceptronTransformTrainer.__new__(PerceptronTransformTrainer)
        trainer.device = "cpu"
        trainer.model = flow
        trainer.aux_model = copy.deepcopy(flow)
        trainer.perceptron_transformation = _tuner()._mixed_transform(1.0)
        trainer._perceptron_slots = []
        for name, module in flow.named_modules():
            if isinstance(module, Perceptron):
                aux = trainer._get_module_by_name(trainer.aux_model, name)
                trainer._perceptron_slots.append((module, aux, copy.deepcopy(aux)))
        with torch.no_grad():
            trainer._update_and_transform_model()
        snapshot = clone_state_for_trainer(trainer)
        restored_scales = _scales(perceptrons)
        _move(list(trainer.aux_model.get_perceptrons()), 0.55)
        with torch.no_grad():
            trainer._update_and_transform_model()
        assert _scales(perceptrons) != restored_scales
        restore_state_for_trainer(trainer, snapshot)
        assert arm_integer_membrane_lattice(flow) == len(_cores(perceptrons))
        assert _armed(perceptrons) == _lattices_of_the_persisted_scales(perceptrons)
        for p in perceptrons:
            unwrap_lif_activation(p.activation).set_membrane_lattice(None)  # a tuning stage recreated the activations
        with integer_program_read(flow) as entered:
            assert entered and in_measurement_plane()
            assert _armed(perceptrons) == _lattices_of_the_persisted_scales(perceptrons)

    @pytest.mark.parametrize("scope,location", [("full", "device"), ("full", "cpu_pinned")])
    def test_the_checkpoint_guards_restore_arms_the_restored_scale(self, scope, location):
        """The cycle's, the endpoint's and the run's keep-best restore through the checkpoint guard, at every location:
        the arming after the restore reads the restored scale. (The tunable scope keeps frozen parameters out of its
        snapshot; a projecting trainer carries an auxiliary model, and the guard takes its full snapshot: the next
        test.)"""
        flow, perceptrons = _form(1)
        _step_projection(flow)
        guard = CheckpointGuard(SimpleNamespace(model=flow, device="cpu"), scope=scope, location=location)
        handle = guard.snapshot()
        restored_scales = _scales(perceptrons)
        _move(perceptrons)
        _step_projection(flow)
        guard.restore(handle)
        assert _scales(perceptrons) == restored_scales
        arm_integer_membrane_lattice(flow)
        assert _armed(perceptrons) == _lattices_of_the_persisted_scales(perceptrons)

    def test_a_projecting_trainers_guard_restores_the_frozen_scale_in_every_scope(self):
        """A trainer that projects carries an auxiliary model; the guard then snapshots both models whole whatever its
        scope, so the frozen scale the projection writes returns with the restore."""
        flow, perceptrons = _form(1)
        _step_projection(flow)
        trainer = SimpleNamespace(model=flow, aux_model=copy.deepcopy(flow), device="cpu")
        guard = CheckpointGuard(trainer, scope="tunable", location="cpu_pinned")
        handle = guard.snapshot()
        restored_scales = _scales(perceptrons)
        _move(perceptrons)
        _step_projection(flow)
        guard.restore(handle)
        assert _scales(perceptrons) == restored_scales

    def test_the_projection_records_the_lattice_and_arms_no_node(self):
        """Never cached: a projection writes the parameters and the perceptron's lattice record, and leaves every LIF
        node as it found it; only the arming writes a node's lattice, from the persisted scale."""
        flow, perceptrons = _form(0)
        _step_projection(flow)
        assert _armed(perceptrons) == [None] * len(perceptrons)
        arm_integer_membrane_lattice(flow)
        assert _armed(perceptrons) == _lattices_of_the_persisted_scales(perceptrons)

    def test_a_restore_to_a_state_no_projection_reached_leaves_every_perceptron_unarmed(self):
        """The ladder's rollback to the step's entry: a snapshot before any projection, a projection, the restore. The
        perceptrons are unprojected again, so no lattice is armed and no read enters the measurement plane."""
        flow, perceptrons = _form(0)
        trainer = SimpleNamespace(model=flow)
        snapshot = clone_state_for_trainer(trainer)
        _step_projection(flow)
        restore_state_for_trainer(trainer, snapshot)
        assert _scales(perceptrons) == [1.0] * len(perceptrons)
        assert arm_integer_membrane_lattice(flow) == 0
        assert _armed(perceptrons) == [None] * len(perceptrons)
        with integer_program_read(flow) as entered:
            assert not entered and not in_measurement_plane()


# ---------------------------------------------------------------------------------------------------------------------
# A perceptron's records: carried with the parameters by the toolchain's one snapshot
# ---------------------------------------------------------------------------------------------------------------------


def _bake_lif_half_step(flow):
    apply_lif_half_step_bias_compensation(flow, T)


def _bake_sync_half_step(flow):
    from mimarsinan.mapping.support.bias_compensation import apply_sync_exact_entry_half_step

    apply_sync_exact_entry_half_step(flow, T, encoding_layer_placement="subsume")


def _bake_ttfs_shift(flow):
    from mimarsinan.mapping.support.bias_compensation import apply_ttfs_quantization_bias_compensation

    apply_ttfs_quantization_bias_compensation(flow, T)


def _bake_negative_shift(flow):
    from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias

    for p in _cores(flow.get_perceptrons()):
        apply_negative_shift_bias(p, torch.full((p.input_features,), 0.25))


def _record_charge_bounds(flow):
    from mimarsinan.tuning.orchestration.lif_exact_qat import record_lif_charge_bounds

    record_lif_charge_bounds(flow)


def _mark_lif_exact_qat(flow):
    from mimarsinan.tuning.orchestration.lif_exact_qat import mark_lif_exact_qat

    for p in flow.get_perceptrons():
        mark_lif_exact_qat(p)


def _mark_sync_exact_qat(flow):
    from mimarsinan.tuning.orchestration.adaptation_manager import mark_sync_exact_qat

    for p in flow.get_perceptrons():
        mark_sync_exact_qat(p)


def _stamp_hop_depths(flow):
    from mimarsinan.tuning.orchestration.adaptation_manager import HOP_DEPTH_ATTR

    for depth, p in enumerate(flow.get_perceptrons(), start=1):
        setattr(p, HOP_DEPTH_ATTR, depth)


def _declare_weight_bits(flow):
    from mimarsinan.transformations.quantization_bounds import LAYER_WEIGHT_BITS

    for p in flow.get_perceptrons():
        setattr(p, LAYER_WEIGHT_BITS, 6)


def _set_bias_row_drive(flow):
    from mimarsinan.transformations.perceptron.bias_drive import set_bias_row_drive

    for p in _cores(flow.get_perceptrons()):
        set_bias_row_drive(p, 3.0)


def _declare_value_domain(flow):
    from mimarsinan.models.perceptron_mixer.perceptron import declare_value_domain_input

    for p in flow.get_perceptrons():
        declare_value_domain_input(p)


def _fold_affine(flow):
    from mimarsinan.tuning.lif_affine_fold import AFFINE_FOLD_FLAG

    for p in _cores(flow.get_perceptrons()):
        with torch.no_grad():
            p.layer.weight.mul_(1.1)
        setattr(p, AFFINE_FOLD_FLAG, True)


def _fold_first_moment(flow):
    from mimarsinan.spiking.sync_first_moment import SYNC_FIRST_MOMENT_FLAG

    for p in _cores(flow.get_perceptrons()):
        with torch.no_grad():
            p.layer.bias.add_(0.01)
        setattr(p, SYNC_FIRST_MOMENT_FLAG, True)


def _project(flow):
    _step_projection(flow)


def _rederive_the_entry_fold(flow):
    """A theta move and the gauge seam's re-derivation (the fold each core applies to its input spikes, a
    non-persistent buffer)."""
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales

    with torch.no_grad():
        for p in flow.get_perceptrons():
            p.activation_scale.mul_(1.7)
    compute_per_source_scales(flow.get_mapper_repr(), arm_wire_value_ops=True)


#: Every record a toolchain writer puts on a perceptron (the search of src for writes outside the state dictionary),
#: each with the writer that moves it inside a window.
RECORD_WRITERS = {
    "the lattice record (the weight projection)": _project,
    "the LIF half-step bake": _bake_lif_half_step,
    "the synchronized half-step bake": _bake_sync_half_step,
    "the TTFS shift bake": _bake_ttfs_shift,
    "the negative-boundary input shift": _bake_negative_shift,
    "the per-cycle charge bound": _record_charge_bounds,
    "the LIF exact-QAT marker": _mark_lif_exact_qat,
    "the synchronized exact-QAT marker": _mark_sync_exact_qat,
    "the hop depth": _stamp_hop_depths,
    "the declared weight width": _declare_weight_bits,
    "the bias-row drive": _set_bias_row_drive,
    "the value-domain reader stamp": _declare_value_domain,
    "the affine fold record": _fold_affine,
    "the first-moment fold record": _fold_first_moment,
    "the entry fold (a theta move re-derived)": _rederive_the_entry_fold,
}


def _records(perceptron) -> dict:
    """What a perceptron holds outside its state dictionary: its instance attributes (not a parameter, buffer,
    submodule or torch's own module field) and its non-persistent buffers (tensors as values)."""
    fields = set(vars(nn.Module()))
    out = {}
    for k, v in vars(perceptron).items():
        if k in fields:
            continue
        out[k] = v.detach().clone() if torch.is_tensor(v) else v
    for k in perceptron._non_persistent_buffers_set:
        v = perceptron._buffers.get(k)
        out["buffer:" + k] = None if v is None else v.detach().clone()
    return out


def _same(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if torch.is_tensor(x) or torch.is_tensor(y):
            if not (torch.is_tensor(x) and torch.is_tensor(y) and torch.equal(x, y)):
                return False
        elif x != y:
            return False
    return True


class TestAPerceptronsRecordsReturnWithTheRestore:
    @pytest.mark.parametrize("record", sorted(RECORD_WRITERS))
    def test_every_record_returns_with_the_keep_best_restore(self, record):
        """A snapshot, the record's own writer inside the window (with any parameter move it makes), the keep-best
        restore: every perceptron's records equal the snapshot's (a record the window added is gone)."""
        flow, perceptrons = _form(0)
        trainer = SimpleNamespace(model=flow)
        before = [_records(p) for p in perceptrons]
        snapshot = clone_state_for_trainer(trainer)
        RECORD_WRITERS[record](flow)
        assert not all(_same(b, _records(p)) for b, p in zip(before, perceptrons)), "the writer moved nothing"
        restore_state_for_trainer(trainer, snapshot)
        assert all(_same(b, _records(p)) for b, p in zip(before, perceptrons)), record

    @pytest.mark.parametrize("scope,location", [("full", "device"), ("tunable", "device"), ("full", "cpu_pinned")])
    def test_records_return_with_the_checkpoint_guards_restore(self, scope, location):
        flow, perceptrons = _form(0)
        before = [_records(p) for p in perceptrons]
        guard = CheckpointGuard(SimpleNamespace(model=flow, device="cpu"), scope=scope, location=location)
        handle = guard.snapshot()
        _bake_lif_half_step(flow)
        _project(flow)
        guard.restore(handle)
        assert all(_same(b, _records(p)) for b, p in zip(before, perceptrons))

    def test_a_bake_inside_a_window_folds_exactly_once_after_the_restore(self):
        """The bake's record and the bias it describes return together: after a window that baked the half-step and
        a restore, the step's own bake folds the half-step once (a stale record skipped it: the bias lost it)."""
        from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

        flow, perceptrons = _form(0)
        pt = PerceptronTransformer()
        entry = [pt.get_effective_bias(p).detach().clone() for p in _cores(perceptrons)]
        trainer = SimpleNamespace(model=flow)
        snapshot = clone_state_for_trainer(trainer)
        _bake_lif_half_step(flow)
        once = [pt.get_effective_bias(p).detach().clone() for p in _cores(perceptrons)]
        restore_state_for_trainer(trainer, snapshot)
        assert all(torch.allclose(pt.get_effective_bias(p), e) for p, e in zip(_cores(perceptrons), entry))
        _bake_lif_half_step(flow)
        assert all(torch.allclose(pt.get_effective_bias(p), o, atol=1e-6) for p, o in zip(_cores(perceptrons), once))
        assert all(getattr(p, LIF_HALF_STEP_FLAG, False) for p in _cores(perceptrons))

    def test_the_lif_affine_fold_steps_rollback_returns_its_bake_and_fold_records(self, monkeypatch):
        """The LIF Affine Fold step bakes the half-step and folds, then rolls a destructive fold back: the rollback
        returns the bias and the records together, so Weight Quantization's own bake still folds the half-step."""
        import mimarsinan.pipelining.pipeline_steps.adaptation.lif_affine_fold_step as fold_step
        from mimarsinan.tuning.lif_affine_fold import AFFINE_FOLD_FLAG

        flow, perceptrons = _form(0)
        before = [_records(p) for p in perceptrons]
        entry_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
        x = torch.randn(32, 8, generator=torch.Generator().manual_seed(5))
        y = torch.zeros(32, dtype=torch.long)

        class _Trainer:
            def iter_validation_batches(self, n):
                yield x, y

        def destructive_fold(model, cal_x, steps, *, soma_law):
            with torch.no_grad():
                for p in _cores(model.get_perceptrons()):
                    p.layer.weight.mul_(-1.0)
                    setattr(p, AFFINE_FOLD_FLAG, True)
            return {"folded": 1, "consumer_folds": 1, "readout_folds": 0, "skipped": {}}

        monkeypatch.setattr(fold_step, "make_basic_trainer", lambda pipeline, model: _Trainer())
        monkeypatch.setattr(fold_step, "evaluate_crater_premise",
                            lambda *a, **k: SimpleNamespace(holds=True, deployed_read=2.0, reference_read=1.0,
                                                            standard_error=0.0))
        monkeypatch.setattr(fold_step, "apply_lif_affine_fold", destructive_fold)
        reported = []
        pipeline = SimpleNamespace(
            config={"simulation_steps": T, "firing_mode": "Default", "device": "cpu", "lif_half_step_bias": True},
            reporter=SimpleNamespace(report=lambda *a, **k: reported.append(a)))
        step = fold_step.LIFAffineFoldStep.__new__(fold_step.LIFAffineFoldStep)
        step.pipeline = pipeline
        entries = {"model": flow, "aq_reference_read": 1.0}
        step.get_entry = lambda key: entries[key]
        step.update_entry = lambda key, value, strategy: entries.__setitem__(key, value)
        step.process()
        assert reported and reported[-1][1].get("rolled_back") is True
        assert all(torch.equal(entry_state[k], v) for k, v in flow.state_dict().items())
        assert all(_same(b, _records(p)) for b, p in zip(before, perceptrons))
        assert not any(getattr(p, LIF_HALF_STEP_FLAG, False) for p in perceptrons)


# ---------------------------------------------------------------------------------------------------------------------
# State the tuning axes own: returned by the cycle's extra state
# ---------------------------------------------------------------------------------------------------------------------


def _cycle(tuner_model, axis):
    """The cycle's own snapshot and restore (``SmoothAdaptationCycle._clone_state``/``_restore_state``) bound on a
    tuner double: the checkpoint guard over the model and the axis's extra state."""
    from mimarsinan.tuning.orchestration.smooth_adaptation_cycle import SmoothAdaptationCycleMixin as Cycle

    ns = SimpleNamespace(_checkpoint_guard=CheckpointGuard(SimpleNamespace(model=tuner_model, device="cpu")),
                         _get_extra_state=axis.get_extra_state, _set_extra_state=axis.set_extra_state)
    return (lambda: Cycle._clone_state(ns)), (lambda s: Cycle._restore_state(ns, s))


def _blend_form():
    from mimarsinan.models.nn.activations.autograd import LeakyGradReLU
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFBlendActivation

    flow, perceptrons = _form(0)
    for p in perceptrons:
        lif = unwrap_lif_activation(p.activation)
        blend = LIFBlendActivation(LeakyGradReLU(), lif, 0.25)
        p.base_activation = blend
        p.activation = blend
    return flow, perceptrons


class TestAxisStateReturnsWithTheCycle:
    def test_the_blend_rate(self):
        from mimarsinan.tuning.axes.blend_axis import LIFAxis

        flow, perceptrons = _blend_form()
        axis = LIFAxis()
        axis.attach(flow, None, {})
        clone, restore = _cycle(flow, axis)
        state = clone()
        axis.set_rate(0.9)
        restore(state)
        assert [p.base_activation.rate for p in perceptrons] == [0.25] * len(perceptrons)

    def test_the_in_place_rate_buffer(self):
        """The activation-quantization rate a decorator reads from the manager's shared rate buffer (held outside the
        model's state dictionary): the axis's extra state returns it with the cycle's restore."""
        from mimarsinan.tuning.axes.manager_rate_axis import ActQuantAxis
        from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager

        flow, perceptrons = _form(0)
        from mimarsinan.models.nn.activations.autograd import LeakyGradReLU

        for p in perceptrons:
            p.base_activation = LeakyGradReLU()
            p.activation = LeakyGradReLU()
        config = {"spiking_mode": "lif", "firing_mode": "Default", "simulation_steps": T, "thresholding_mode": "<=",
                  "encoding_layer_placement": "subsume"}
        manager = AdaptationManager()
        axis = ActQuantAxis()
        axis.attach(flow, manager, config)
        axis.set_rate(0.3)
        clone, restore = _cycle(flow, axis)
        state = clone()
        axis.set_rate(0.9)
        assert manager._rate_value("quantization_rate") == pytest.approx(0.9)
        restore(state)
        assert manager._rate_value("quantization_rate") == pytest.approx(0.3)

    def test_the_genuine_forward_rate(self):
        """The teacher-to-genuine output blend's rate lives on the installed forward (outside the state dictionary):
        the genuine-blend axis's extra state returns it with the cycle's restore."""
        from mimarsinan.models.spiking.training.blended_genuine_forward import BlendedGenuineForward
        from mimarsinan.tuning.axes.blend_axis import GenuineBlendAxis

        flow, _ = _form(0)
        flow.forward = BlendedGenuineForward(flow, copy.deepcopy(flow), T, rate=0.25)
        axis = GenuineBlendAxis()
        axis.attach(flow, None, {})
        clone, restore = _cycle(flow, axis)
        state = clone()
        axis.set_rate(0.9)
        assert flow.forward.rate == pytest.approx(0.9)
        restore(state)
        assert flow.forward.rate == pytest.approx(0.25)

    def test_the_window_length(self):
        from mimarsinan.tuning.orchestration.mbh_tanneal import LIFTAnnealAxis, TAnnealSchedule

        flow, perceptrons = _blend_form()
        axis = LIFTAnnealAxis(TAnnealSchedule(target_T=T, ladder_rates=(0.5, 1.0)))
        axis.attach(flow, None, {})
        clone, restore = _cycle(flow, axis)
        state = clone()
        axis.set_rate(0.0)
        assert unwrap_lif_activation(perceptrons[1].activation).T != T
        restore(state)
        assert all(unwrap_lif_activation(p.activation).T == T for p in perceptrons)


# ---------------------------------------------------------------------------------------------------------------------
# State its reader re-derives at its own start
# ---------------------------------------------------------------------------------------------------------------------


class TestStateItsReaderRederives:
    def test_the_wire_gauge_a_twin_walks_is_rederived_after_a_restore(self):
        """A theta move, a re-derivation of the gauge on the live model (the LIF finalize's), the restore: the twin's
        precondition re-derives every slot at the restored thetas, equal to a fresh derivation there."""
        from mimarsinan.spiking.wire_gauge_seam import (
            establish_deployed_wire_gauge,
            require_established_wire_gauge,
        )

        flow, perceptrons = _form(0)
        establish_deployed_wire_gauge(flow, input_data_scale=1.0)
        reference = copy.deepcopy(flow)
        trainer = SimpleNamespace(model=flow)
        snapshot = clone_state_for_trainer(trainer)
        with torch.no_grad():
            for p in perceptrons:
                p.activation_scale.mul_(1.7)
        require_established_wire_gauge(flow, input_data_scale=1.0, reader="the finalize twin")
        restore_state_for_trainer(trainer, snapshot)
        require_established_wire_gauge(flow, input_data_scale=1.0, reader="the twin after the restore")
        require_established_wire_gauge(reference, input_data_scale=1.0, reader="the reference")
        slots = [n.per_source_scales for n in flow.get_mapper_repr().execution_order()
                 if getattr(n, "per_source_scales", None) is not None]
        expected = [n.per_source_scales for n in reference.get_mapper_repr().execution_order()
                    if getattr(n, "per_source_scales", None) is not None]
        assert slots and len(slots) == len(expected)
        for a, b in zip(slots, expected):
            assert all(torch.allclose(torch.as_tensor(u), torch.as_tensor(v)) for u, v in zip(a, b))

    def test_the_installed_twin_forward_walks_the_gauge_of_the_restored_thetas(self):
        """The LIF finalize installs the chip-aligned walk as the model's own forward and re-derives the gauge on the
        live model; a restore across it (a theta move inside the window) must leave the installed forward walking the
        gauge of the restored thetas: equal, output for output, to the same forward on a copy taken at the snapshot."""
        from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
        from mimarsinan.spiking.wire_gauge_seam import (
            establish_deployed_wire_gauge,
            require_established_wire_gauge,
        )
        from mimarsinan.tuning.forward_install import ChipAlignedNFForward
        from unit.pipelining.test_wq_integer_program import _inputs

        flow, perceptrons = _form(0)
        establish_deployed_wire_gauge(flow, input_data_scale=1.0)
        flow.forward = ChipAlignedNFForward(flow, T, retime=True, soma_law=DEFAULT_SOMA_LAW)
        x = _inputs(64)
        with torch.no_grad():
            flow(x)  # a walk before the window: its derivation is the one the window's writers replace
        reference = copy.deepcopy(flow)
        trainer = SimpleNamespace(model=flow)
        snapshot = clone_state_for_trainer(trainer)
        with torch.no_grad():
            for p in perceptrons:
                p.activation_scale.mul_(1.7)
        require_established_wire_gauge(flow, input_data_scale=1.0, reader="the finalize twin")
        restore_state_for_trainer(trainer, snapshot)
        with torch.no_grad():
            assert torch.equal(flow(x), reference(x))
            # A theta move in place (no writer replaces a slot): the next walk follows the moved thetas, as a fresh
            # copy's first walk derives them.
            for p in perceptrons:
                p.activation_scale.mul_(1.3)
            moved = flow(x)
            fresh = copy.deepcopy(flow)
            fresh.get_mapper_repr().__dict__.pop("_wire_gauge_walk_derivation", None)
            assert torch.equal(moved, fresh(x))
        # The derivation is values only: a training forward with trainable (moved) thetas builds no gradient path
        # through the gauge.
        for p in perceptrons:
            p.activation_scale.requires_grad_(True)
            with torch.no_grad():
                p.activation_scale.mul_(1.01)
        flow(x)
        slots = [s for n in flow.get_mapper_repr().execution_order() for s in (getattr(n, "per_source_scales", None)
                                                                             or ())]
        assert slots and not any(torch.as_tensor(s).requires_grad for s in slots)

    def test_an_installed_forwards_executor_holds_the_graph_and_reads_the_restored_parameters(self):
        """A lazily built executor (the genuine cascade an installed forward walks) holds references to the graph,
        never parameter values: built inside a window, it reads the restored parameters after the restore."""
        from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
        from mimarsinan.models.spiking.training.blended_genuine_forward import BlendedGenuineForward
        from mimarsinan.torch_mapping.converter import convert_torch_model

        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 4), nn.ReLU())
        flow = convert_torch_model(net, (6,), 4, device="cpu")
        for p in flow.get_perceptrons():
            p.set_activation(TTFSActivation(T=8, activation_scale=p.activation_scale,
                                            input_scale=p.input_activation_scale, bias=p.layer.bias,
                                            thresholding_mode="<=", encoding=getattr(p, "is_encoding_layer", False)))
        flow = flow.double()
        fresh = copy.deepcopy(flow)
        forward = BlendedGenuineForward(flow, copy.deepcopy(flow), 8, rate=1.0)
        trainer = SimpleNamespace(model=flow)
        snapshot = clone_state_for_trainer(trainer)
        x = torch.rand(16, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            forward(x)  # the executor is built inside the window
            for p in flow.get_perceptrons():
                p.layer.weight.mul_(0.5)
        restore_state_for_trainer(trainer, snapshot)
        with torch.no_grad():
            assert torch.equal(forward(x), BlendedGenuineForward(fresh, copy.deepcopy(fresh), 8, rate=1.0)(x))

    def test_the_readout_law_in_the_loop_is_resolved_whenever_the_state_it_reads_differs(self):
        """The readout-aware loop's laws are derived from the state of their last solve: a move makes the loop stale,
        and a restore to the solved state makes it current again (it serves the law of the state it reads)."""
        from mimarsinan.pipelining.pipeline_steps.quantization.readout_aware_training import ReadoutLawInLoop

        flow, _ = _form(0)
        loop = ReadoutLawInLoop.__new__(ReadoutLawInLoop)
        loop._model = flow
        loop._snapshot = [t.detach().clone() for t in loop._state()]  # the state of the last solve
        trainer = SimpleNamespace(model=flow)
        snapshot = clone_state_for_trainer(trainer)
        assert not loop._stale()
        _move(list(flow.get_perceptrons()))
        assert loop._stale()
        restore_state_for_trainer(trainer, snapshot)
        assert not loop._stale()

    def test_a_lif_forward_reads_no_membrane_a_previous_window_left(self):
        """Forwards in a window leave membranes in the LIF nodes' memories; after a restore, the forward equals a fresh
        copy's (every forward starts its window from the reset value)."""
        from unit.pipelining.test_wq_integer_program import _inputs, _walk

        flow, _ = _form(0)
        _step_projection(flow)
        fresh = copy.deepcopy(flow)
        trainer = SimpleNamespace(model=flow)
        snapshot = clone_state_for_trainer(trainer)
        x = _inputs(64)
        with torch.no_grad():
            _walk(flow, x)
            flow(x)
        restore_state_for_trainer(trainer, snapshot)
        with torch.no_grad():
            assert torch.equal(_walk(flow, x), _walk(fresh, x))
            assert torch.equal(flow(x), fresh(x))


# ---------------------------------------------------------------------------------------------------------------------
# The ratchet: every model restore goes through the one snapshot
# ---------------------------------------------------------------------------------------------------------------------

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "mimarsinan"

#: State-dictionary loads that are not a model restore inside a tuning window, each with its reason.
NOT_A_WINDOW_RESTORE = {
    "model_training/weight_loading.py": "pretrained weights from a file, before any step",
    "models/vit_leaf.py": "pretrained weights converted at build",
    "models/builders/torch/torch_vit_builder.py": "pretrained weights at build",
    "transformations/perceptron/bias_canonicalization.py": "one normalization's tensors inside a verified apply "
                                                           "that writes no record",
    "pipelining/pipeline_steps/quantization/host_normalization_recalibration.py": "the host normalizations' "
                                                                                  "statistics; their momentum and "
                                                                                  "mode are restored with them",
    "tuning/model_snapshot.py": "the one snapshot's own restore",
}


def _load_state_dict_calls(path: pathlib.Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict":
            owner = ast.unparse(node.func.value)
            if "optimizer" in owner or "schedule" in owner or "scaler" in owner:
                continue  # an optimizer's or a schedule's own state, not the model's
            calls.append((node.lineno, owner))
    return calls


def test_every_model_restore_goes_through_the_one_snapshot():
    """No tuning window restores a model by a bare state-dictionary load: every restore is the toolchain's one
    snapshot (``tuning.model_snapshot``), which carries the perceptrons' records with the parameters."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in NOT_A_WINDOW_RESTORE:
            continue
        offenders.extend(f"{rel}:{line} {owner}.load_state_dict" for line, owner in _load_state_dict_calls(path))
    assert not offenders, offenders
