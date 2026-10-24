"""The constant-OFF floor the bias clip reads is declared by the activation node (D19, fix round 1).

LIF Adaptation runs before Weight Quantization on every LIF plan and installs ``LIFBlendActivation`` (the
value-domain ramp) or ``LIFActivation`` (the genuine ramps) as the perceptron's base. Both have the floor: an IF node
with threshold 1 whose membrane starts below threshold never fires under per-cycle inputs at or below 0. So the clip
must fire on them exactly as the 75998141 clip did (the t01_19/t0_03 crater repair), while an affine package
(``nn.Identity``) is never clipped. The floor is a declaration of the node class (``models.nn.off_floor``), read by
the clip; the ReLU adaptation keeps its own ReLU-compatibility predicate.

New API names are imported inside the tests, so on a tree without them a test fails on its own assertion.
"""

from __future__ import annotations

import contextlib
import copy
import io
from typing import Callable, Dict

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer, LeakyGradReLU
from mimarsinan.models.nn.activations.ttfs_cycle import TTFSCycleActivation
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.nn.activations.value_quantizer import ValueGridQuantizer
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron, as_affine_package
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.perceptron.bias_saturation import clip_off_saturated_effective_bias
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.tuning.orchestration.blend_ramp import BlendActivation
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFBlendActivation

TR = PerceptronTransformer()
CRATER_BIAS = -12.6
"""The t01_19 dead channel's effective bias (test_bias_saturation.py), far below its [0, 1] bound."""


def _crater(seed: int = 8, wire: str = "none") -> Perceptron:
    """The crater anatomy of test_bias_saturation.py: channel 2 at b_eff -12.6 against max|w| about 0.6."""
    torch.manual_seed(seed)
    p = Perceptron(4, 8, normalization=nn.Identity())
    with torch.no_grad():
        p.layer.weight.data.mul_(0.3)
        p.layer.bias.data.mul_(0.3)
    if wire == "chip_input":
        p.append_input_wire_op(ChipInputQuantizer(8, 1.0))

    def setter(b):
        out = b.clone()
        out[2] = CRATER_BIAS
        return out

    TR.apply_effective_bias_transform(p, setter)
    return p


def _node(kind: str, p: Perceptron) -> nn.Module:
    if kind == "lif":
        return LIFActivation(T=16, activation_scale=p.activation_scale)
    if kind == "lif_blend":
        return LIFBlendActivation(LeakyGradReLU(), LIFActivation(T=16, activation_scale=p.activation_scale), 1.0)
    return LeakyGradReLU()


def _with_base(p: Perceptron, kind: str) -> Perceptron:
    node = _node(kind, p)
    p.base_activation = node
    p.activation = node
    return p.eval()


def _legacy_unit_clip_layer_bias(p: Perceptron) -> torch.Tensor:
    """75998141's clip on the event wire's [0, 1] (no normalization): b := max(b, -sum_i max(w_i, 0)), as layer bias."""
    w = TR.get_effective_weight(p).detach()
    b = TR.get_effective_bias(p).detach()
    bound = -w.flatten(1).clamp(min=0).sum(dim=1)
    return torch.where(b < bound, bound, b) * p.activation_scale.data


def _rates_and_spikes(n: int = 4096):
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, 8, generator=g), (torch.rand(n, 8, generator=g) < 0.5).float()


LIF_BASES = ["lif", "lif_blend"]


class TestLifCrater:
    """C1 and C2 of the exactness review: the crater channel on a LIF base is provably OFF and is clipped."""

    @pytest.mark.parametrize("kind", LIF_BASES)
    def test_the_dead_channel_never_fires(self, kind):
        p = _with_base(_crater(), kind)
        for x in _rates_and_spikes():
            with torch.no_grad():
                assert float(p(x)[:, 2].abs().max()) == 0.0

    @pytest.mark.parametrize("wire", ["none", "chip_input"])
    @pytest.mark.parametrize("kind", LIF_BASES)
    def test_clip_equals_the_legacy_unit_clip_byte_for_byte(self, kind, wire):
        p = _with_base(_crater(wire=wire), kind)
        expected = _legacy_unit_clip_layer_bias(p)
        with contextlib.redirect_stdout(io.StringIO()):
            assert clip_off_saturated_effective_bias(p) == 1
        assert torch.equal(p.layer.bias.data, expected)

    @pytest.mark.parametrize("kind", LIF_BASES)
    def test_clip_preserves_the_lif_forward(self, kind):
        p = _with_base(_crater(), kind)
        inputs = _rates_and_spikes()
        with torch.no_grad():
            before = [p(x).clone() for x in inputs]
        with contextlib.redirect_stdout(io.StringIO()):
            clip_off_saturated_effective_bias(p)
        with torch.no_grad():
            for x, ref in zip(inputs, before):
                assert torch.equal(p(x), ref)

    @pytest.mark.parametrize("kind", LIF_BASES)
    def test_projection_keeps_the_weight_grid(self, kind):
        p = _with_base(_crater(), kind)
        with contextlib.redirect_stdout(io.StringIO()):
            NormalizationAwarePerceptronQuantization(4, "cpu", rate=1.0).transform(p)
        w = TR.get_effective_weight(p).detach()
        assert float(p.parameter_scale) > 1.0
        assert float((w != 0).float().mean()) > 0.5


class TestLifAdaptationInstall:
    """The base LIF Adaptation installs through its own hooks keeps the [0, 1] verdict at WQ entry."""

    @staticmethod
    def _adapted_model(tmp_path, membrane_init: float = 0.0):
        from conftest import MockPipeline, default_config, make_tiny_supermodel
        from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

        cfg = default_config()
        cfg.update({"spiking_mode": "lif", "firing_mode": "Default", "thresholding_mode": "<",
                    "simulation_steps": 4, "lif_membrane_init": membrane_init})
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.5
        model = make_tiny_supermodel()
        tuner = LIFAdaptationTuner(pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
                                   adaptation_manager=AdaptationManager())
        tuner._install_blend()
        tuner._set_rate(1.0)
        tuner._finalize()
        return model.eval()

    def test_the_installed_base_keeps_the_unit_verdict(self, tmp_path):
        from mimarsinan.transformations.perceptron.bias_saturation import (
            UNIT_INPUT_RANGE,
            constant_off_input_range,
        )

        model = self._adapted_model(tmp_path)
        bases = {type(p.base_activation).__name__ for p in model.get_perceptrons()}
        assert bases == {"LIFBlendActivation"}
        for p in model.get_perceptrons():
            verdict = constant_off_input_range(p)
            if p.is_encoding_layer:
                assert verdict is None
            else:
                assert verdict is UNIT_INPUT_RANGE, p.name

    def test_a_planted_crater_channel_is_clipped_as_the_legacy_clip(self, tmp_path):
        model = self._adapted_model(tmp_path)
        p = [q for q in model.get_perceptrons() if not q.is_encoding_layer][0]

        def setter(b):
            out = b.clone()
            out[0] = CRATER_BIAS
            return out

        TR.apply_effective_bias_transform(p, setter)
        w = TR.get_effective_weight(p).detach()
        b = TR.get_effective_bias(p).detach()
        bound = -w.flatten(1).clamp(min=0).sum(dim=1)
        legacy = torch.where(b < bound, bound, b)
        with contextlib.redirect_stdout(io.StringIO()):
            assert clip_off_saturated_effective_bias(p) == 1
        assert torch.equal(TR.get_effective_bias(p).detach(), legacy)

    def test_a_membrane_init_at_threshold_declares_no_floor(self, tmp_path):
        from mimarsinan.transformations.perceptron.bias_saturation import constant_off_input_range

        model = self._adapted_model(tmp_path, membrane_init=1.0)
        assert all(constant_off_input_range(p) is None for p in model.get_perceptrons())


def _factories() -> Dict[type, Callable[[], nn.Module]]:
    """One instance per declared node class, in a state whose predicate is True."""
    return {
        nn.ReLU: nn.ReLU,
        LeakyGradReLU: LeakyGradReLU,
        LIFActivation: lambda: LIFActivation(T=8, activation_scale=1.0),
        TTFSActivation: lambda: TTFSActivation(T=8, activation_scale=1.0),
        TTFSCycleActivation: lambda: TTFSCycleActivation(T=8, activation_scale=1.0),
        BlendActivation: lambda: BlendActivation(LeakyGradReLU(), nn.ReLU(), 0.5, target_type="ReLU"),
        LIFBlendActivation: lambda: LIFBlendActivation(LeakyGradReLU(), LIFActivation(T=8, activation_scale=1.0), 0.5),
    }


def _loaded_subclasses(cls: type) -> set:
    found, frontier = set(), [cls]
    while frontier:
        for sub in frontier.pop().__subclasses__():
            if sub not in found:
                found.add(sub)
                frontier.append(sub)
    return found


class TestOffFloorDeclarations:
    """Contract over the declaration registry: every declared node outputs 0 for every pre-activation at or below 0."""

    def test_every_declared_class_and_loaded_subclass_has_a_contract_case(self):
        from mimarsinan.models.nn.off_floor import off_floor_declarations

        declared = set(off_floor_declarations())
        covered = set(_factories())
        assert declared <= covered, sorted(c.__name__ for c in declared - covered)
        for cls in declared:
            missing = {s for s in _loaded_subclasses(cls) if s.__module__.startswith("mimarsinan")} - covered
            assert not missing, (cls.__name__, sorted(s.__name__ for s in missing))

    @pytest.mark.parametrize("cls", list(_factories()), ids=lambda c: c.__name__)
    def test_declared_node_outputs_zero_at_or_below_zero(self, cls):
        from mimarsinan.models.nn.off_floor import has_off_floor

        node = _factories()[cls]()
        assert type(node) is cls
        assert has_off_floor(node)
        x = -torch.rand(64, 6, generator=torch.Generator().manual_seed(1)) * 10.0
        x[0] = 0.0
        with torch.no_grad():
            assert float(node(x).abs().max()) == 0.0

    @pytest.mark.parametrize("node", [nn.Identity(), nn.GELU(), nn.LeakyReLU(), nn.Tanh(), nn.Hardtanh(), None],
                             ids=lambda n: type(n).__name__)
    def test_undeclared_nodes_have_no_floor(self, node):
        from mimarsinan.models.nn.off_floor import has_off_floor

        assert not has_off_floor(node)

    def test_affine_package_has_no_floor(self):
        from mimarsinan.models.nn.off_floor import has_off_floor

        assert not has_off_floor(as_affine_package(Perceptron(2, 3)).base_activation)

    def test_lif_floor_follows_the_window_start_membrane(self):
        from mimarsinan.models.nn.off_floor import has_off_floor

        assert has_off_floor(LIFActivation(T=8, activation_scale=1.0, membrane_init=-0.5))
        assert has_off_floor(LIFActivation(T=8, activation_scale=1.0, membrane_init=0.9))
        assert not has_off_floor(LIFActivation(T=8, activation_scale=1.0, membrane_init=1.0))
        node = LIFActivation(T=8, activation_scale=1.0, membrane_init=0.9)
        node.set_membrane_lattice(4.0)  # quantum 1/8: 0.9 snaps to 0.875, below the threshold 1
        assert has_off_floor(node)
        node = LIFActivation(T=8, activation_scale=1.0, membrane_init=0.95)
        node.set_membrane_lattice(4.0)  # 0.95 snaps to 1.0, the threshold
        assert not has_off_floor(node)

    def test_a_lif_node_pickled_before_the_membrane_guard_starts_at_zero(self):
        from mimarsinan.models.nn.off_floor import has_off_floor

        node = LIFActivation(T=8, activation_scale=1.0)
        del node.__dict__["membrane_init"]  # the state of a cache that predates the window-start guard
        assert has_off_floor(node)
        with torch.no_grad():
            assert float(node(-torch.rand(16, 4)).abs().max()) == 0.0

    def test_blend_floor_reads_the_sides_its_rate_reads(self):
        from mimarsinan.models.nn.off_floor import has_off_floor

        def blend(old, target, rate):
            return BlendActivation(old, target, rate, target_type="T")

        lif = LIFActivation(T=8, activation_scale=1.0)
        assert has_off_floor(blend(nn.GELU(), lif, 1.0))
        assert not has_off_floor(blend(nn.GELU(), lif, 0.5))
        assert not has_off_floor(blend(nn.GELU(), lif, 0.0))
        assert has_off_floor(blend(LeakyGradReLU(), nn.GELU(), 0.0))
        assert has_off_floor(LIFBlendActivation(LeakyGradReLU(), lif, 0.5))
        assert not has_off_floor(LIFBlendActivation(nn.Identity(), lif, 0.5))

    def test_an_undeclared_subclass_of_an_undeclared_node_has_none(self):
        from mimarsinan.models.nn.off_floor import has_off_floor

        class Shifted(nn.Identity):
            pass

        assert not has_off_floor(Shifted())


class TestReluAdaptationKeepsItsOwnPredicate:
    """A LIF base has the floor but is not ReLU-compatible: the ReLU adaptation reads the base's own name list."""

    @pytest.mark.parametrize("make_base,needs", [
        (LeakyGradReLU, False),
        (nn.ReLU, False),
        (nn.GELU, True),
        (nn.LeakyReLU, True),
        (nn.Identity, True),
        (lambda: LIFActivation(T=4, activation_scale=1.0), True),
        (lambda: LIFBlendActivation(LeakyGradReLU(), LIFActivation(T=4, activation_scale=1.0), 1.0), True),
    ])
    def test_needs_relu_adaptation_is_the_name_list(self, make_base, needs):
        from mimarsinan.pipelining.pipeline_steps.activation_utils import (
            RELU_COMPATIBLE_TYPES,
            needs_relu_adaptation,
        )

        p = Perceptron(2, 3)
        p.base_activation = make_base()
        assert RELU_COMPATIBLE_TYPES == ("LeakyGradReLU", "ReLU")
        assert needs_relu_adaptation(p) is needs


class TestValueDomainDeclaration:
    """[0, 1] is the event wire's premise; a perceptron declared to read the value domain needs a grid for a range."""

    @staticmethod
    def _floored_signed_reader(declare: bool) -> Perceptron:
        from mimarsinan.models.perceptron_mixer.perceptron import declare_value_domain_input

        p = Perceptron(2, 6, normalization=nn.Identity())
        with torch.no_grad():
            p.layer.weight.copy_(torch.tensor([[0.5, -0.5, 0.2, 0.0, 0.1, -0.3], [0.3] * 6]))
            p.layer.bias.copy_(torch.tensor([-1.0, 0.1]))
        if declare:
            declare_value_domain_input(p)
        return p.eval()

    def test_a_declared_value_domain_reader_without_a_grid_has_no_range(self):
        from mimarsinan.transformations.perceptron.bias_saturation import declared_input_range

        p = self._floored_signed_reader(declare=True)
        x = (torch.rand(4096, 6, generator=torch.Generator().manual_seed(1)) * 2 - 1) * 3.0
        with torch.no_grad():
            before = p(x).clone()
        assert declared_input_range(p) is None
        assert clip_off_saturated_effective_bias(p) == 0
        with torch.no_grad():
            assert torch.equal(p(x), before)

    def test_a_declared_value_domain_reader_of_a_grid_reads_the_grid(self):
        from mimarsinan.transformations.perceptron.bias_saturation import declared_input_range

        p = self._floored_signed_reader(declare=True)
        q = ValueGridQuantizer(5)
        q.calibrate(2.0)
        p.append_input_wire_op(q)
        rng = declared_input_range(p)
        assert rng is not None and (float(rng.low), float(rng.high)) == (-2.0, 2.0)

    def test_an_undeclared_reader_keeps_the_event_wire_premise(self):
        from mimarsinan.transformations.perceptron.bias_saturation import UNIT_INPUT_RANGE, declared_input_range

        assert declared_input_range(self._floored_signed_reader(declare=False)) is UNIT_INPUT_RANGE

    def test_flow_birth_stamps_every_value_domain_perceptron(self):
        from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
        from mimarsinan.models.perceptron_mixer.perceptron import reads_value_domain
        from mimarsinan.torch_mapping.converter import convert_torch_model

        net = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
        value = convert_torch_model(copy.deepcopy(net).eval(), (6,), 3, device="cpu", packaging=MVM_PACKAGING)
        event = convert_torch_model(copy.deepcopy(net).eval(), (6,), 3, device="cpu", packaging=SPIKING_PACKAGING)
        assert [reads_value_domain(p) for p in value.get_perceptrons()] == [True, True]
        stamps = [reads_value_domain(p) for p in event.get_perceptrons()]
        assert stamps and not any(stamps)


class TestDeclaredRangeClipMargin:
    """On a declared grid the maximizing corner is a grid point: the clipped channel stays OFF in float32 there."""

    def test_the_maximizing_corner_stays_off_through_the_layer_forward(self):
        worst = -float("inf")
        for seed in range(200):
            g = torch.Generator().manual_seed(seed)
            p = Perceptron(3, 27, normalization=nn.Identity())
            scale = float(torch.rand(1, generator=g) * 3 + 0.5)
            with torch.no_grad():
                p.layer.weight.copy_(torch.randn(3, 27, generator=g))
                p.layer.bias.copy_(torch.tensor([-1000.0, -1000.0, -1000.0]))
            q = ValueGridQuantizer(5)
            q.calibrate(scale)
            p.append_input_wire_op(q)
            p.eval()
            with contextlib.redirect_stdout(io.StringIO()):
                assert clip_off_saturated_effective_bias(p) == 3
            corner = torch.sign(p.layer.weight.data) * scale  # +-levels * step: a grid point per input
            with torch.no_grad():
                pre = p.layer(q(corner)).diagonal()
                out = p(corner).diagonal()
            worst = max(worst, float(pre.max()))
            assert float(out.max()) == 0.0, seed
        assert worst <= 0.0

    def test_the_margin_is_small_against_the_bound(self):
        from mimarsinan.transformations.perceptron.bias_saturation import InputRange, off_saturation_bias_bound

        g = torch.Generator().manual_seed(7)
        w = torch.randn(5, 27, generator=g)
        exact = -torch.maximum(w.double() * -2.0, w.double() * 2.0).sum(dim=1)
        bound = off_saturation_bias_bound(w, InputRange(-2.0, 2.0)).double()
        assert bool((bound <= exact).all())
        assert bool(((exact - bound) <= 1e-4 * exact.abs()).all())


class TestEmpiricalShiftNeedsTheFloor:
    """The guarded empirical shift assumes OFF at ``v_max <= 0``: an affine package has no such state."""

    @staticmethod
    def _chain(affine: bool):
        torch.manual_seed(8)
        p1 = Perceptron(4, 8, normalization=nn.Identity())
        p2 = Perceptron(3, 4, normalization=nn.Identity())
        if affine:
            as_affine_package(p1)
        else:
            p1.activation = nn.Hardtanh(min_val=0.0, max_val=1.0)
        p2.activation = nn.Identity()
        with torch.no_grad():
            p1.layer.weight.data.mul_(0.3)
            p1.layer.bias.data.mul_(0.3)

        def setter(b):
            out = b.clone()
            out[2] = 21.0
            return out

        TR.apply_effective_bias_transform(p1, setter)

        class _Chain(nn.Module):
            def __init__(self):
                super().__init__()
                self.p1, self.p2 = p1, p2

            def forward(self, x):
                return self.p2(self.p1(x))

            def get_perceptrons(self):
                return [self.p1, self.p2]

        return _Chain().eval(), p1

    @pytest.mark.parametrize("affine", [False, True])
    def test_the_shift_rung_runs_only_on_a_floored_perceptron(self, affine, monkeypatch):
        from mimarsinan.transformations.perceptron import bias_canonicalization

        calls = []
        real = bias_canonicalization.empirical_bias_shift

        def spy(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(bias_canonicalization, "empirical_bias_shift", spy)
        model, _ = self._chain(affine)
        with contextlib.redirect_stdout(io.StringIO()):
            bias_canonicalization.canonicalize_starved_bias_outliers(model, [torch.rand(64, 8)], bits=4)
        assert bool(calls) is (not affine)
