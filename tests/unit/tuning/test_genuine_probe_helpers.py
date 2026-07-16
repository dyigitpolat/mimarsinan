"""Unit tests for the genuine-probe helpers (pure forward eval + clone probe)."""

import copy

import torch
import torch.nn as nn

from mimarsinan.tuning.orchestration.genuine_probe import (
    eval_forward_over_val,
    genuine_acc_on_clone,
    iter_val_batches,
)


class _TinyNet(nn.Module):
    """Deterministic 2-class classifier with a named submodule + an attr."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        self.foo = "original"

    def forward(self, x):
        return self.lin(x)


class _FakeTrainer:
    """Minimal trainer exposing iter_validation_batches over fixed CPU batches."""

    def __init__(self, batches):
        self._batches = batches

    def iter_validation_batches(self, n_batches):
        for i in range(int(n_batches)):
            yield self._batches[i % len(self._batches)]


def _hand_accuracy(model, batches, n_batches, device):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i in range(n_batches):
            x, y = batches[i % len(batches)]
            x, y = x.to(device), y.to(device)
            _, predicted = model(x).max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
    return correct / total if total else 0.0


def _make_batches():
    # Two batches; argmax of _TinyNet picks the larger of the first two features.
    x0 = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 0.0]])
    y0 = torch.tensor([0, 1])  # both correct
    x1 = torch.tensor([[1.0, 5.0, 0.0], [4.0, 0.0, 0.0]])
    y1 = torch.tensor([0, 0])  # first wrong (pred 1), second correct
    return [(x0, y0), (x1, y1)]


def test_iter_val_batches_is_a_pure_view_of_the_trainer_iterator():
    batches = _make_batches()
    trainer = _FakeTrainer(batches)

    got = list(iter_val_batches(trainer, 3))
    expected = list(trainer.iter_validation_batches(3))
    assert len(got) == len(expected) == 3
    for (gx, gy), (ex, ey) in zip(got, expected):
        assert torch.equal(gx, ex)
        assert torch.equal(gy, ey)


def test_eval_forward_over_val_matches_hand_computed():
    device = torch.device("cpu")
    model = _TinyNet()
    batches = _make_batches()
    trainer = _FakeTrainer(batches)
    n_batches = 3  # wraps: b0, b1, b0 -> 6 correct / 6 = ... compute by hand

    expected = _hand_accuracy(model, batches, n_batches, device)
    got = eval_forward_over_val(trainer, model.forward, model, n_batches, device)
    assert got == expected
    # b0: 2/2, b1: 1/2, b0: 2/2 -> 5/6
    assert abs(got - (5.0 / 6.0)) < 1e-9


def test_eval_forward_over_val_empty_returns_zero():
    device = torch.device("cpu")
    model = _TinyNet()
    trainer = _FakeTrainer(_make_batches())
    assert eval_forward_over_val(trainer, model.forward, model, 0, device) == 0.0


def test_eval_forward_over_val_does_not_install_forward():
    device = torch.device("cpu")
    model = _TinyNet()
    trainer = _FakeTrainer(_make_batches())
    original_forward = model.__dict__.get("forward", None)

    other = nn.Linear(3, 2, bias=False)
    eval_forward_over_val(trainer, other, model, 2, device)

    # forward_obj must NOT have been bound onto the model instance.
    assert model.__dict__.get("forward", None) is original_forward
    assert model.forward.__self__ is model


def test_eval_forward_over_val_uses_eval_mode_and_no_grad():
    device = torch.device("cpu")
    model = _TinyNet()
    model.train()
    trainer = _FakeTrainer(_make_batches())

    seen = {}

    def probe(x):
        seen["training"] = model.training
        seen["grad_enabled"] = torch.is_grad_enabled()
        return model.lin(x)

    eval_forward_over_val(trainer, probe, model, 1, device)
    assert seen["training"] is False
    assert seen["grad_enabled"] is False


def test_genuine_acc_on_clone_is_non_destructive():
    device = torch.device("cpu")
    model = _TinyNet()
    batches = _make_batches()
    trainer = _FakeTrainer(batches)

    before_state = copy.deepcopy(model.state_dict())
    original_lin = model.lin
    original_forward_entry = model.__dict__.get("forward", None)

    def prepare(clone):
        # Mutate the clone aggressively: zero params, swap submodule, set attr.
        with torch.no_grad():
            for p in clone.parameters():
                p.zero_()
        clone.lin = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            clone.lin.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )
        clone.foo = "mutated"

    def build_forward(clone):
        return clone.forward

    def evaluate(forward_obj, clone):
        assert clone.foo == "mutated"
        return eval_forward_over_val(trainer, forward_obj, clone, 3, device)

    acc = genuine_acc_on_clone(
        model, device, prepare=prepare, build_forward=build_forward, evaluate=evaluate
    )

    # Returned value reflects the (re-restored) mutated clone: same 5/6 surface.
    assert abs(acc - (5.0 / 6.0)) < 1e-9

    # Live model is untouched: state_dict, submodule identity, attr, forward binding.
    after_state = model.state_dict()
    assert set(after_state.keys()) == set(before_state.keys())
    for k in before_state:
        assert torch.equal(after_state[k], before_state[k])
    assert model.lin is original_lin
    assert model.foo == "original"
    assert model.__dict__.get("forward", None) is original_forward_entry


def test_genuine_acc_on_clone_passes_clone_not_live_model():
    device = torch.device("cpu")
    model = _TinyNet()
    trainer = _FakeTrainer(_make_batches())
    captured = {}

    def prepare(clone):
        captured["prepared"] = clone

    def build_forward(clone):
        captured["built"] = clone
        return clone.forward

    def evaluate(forward_obj, clone):
        captured["evaluated"] = clone
        return eval_forward_over_val(trainer, forward_obj, clone, 1, device)

    genuine_acc_on_clone(
        model, device, prepare=prepare, build_forward=build_forward, evaluate=evaluate
    )

    assert captured["prepared"] is not model
    assert captured["built"] is captured["prepared"]
    assert captured["evaluated"] is captured["prepared"]


class _OOMBelowChunk(nn.Module):
    """Forward that raises CUDA-style OOM for batches above ``max_batch`` —
    the genuine spike-train eval signature (S x batch x features exceeds VRAM
    at deploy-eval time on val batches sized for fp32 training)."""

    def __init__(self, inner, max_batch):
        super().__init__()
        self.inner = inner
        self.max_batch = int(max_batch)
        self.oom_raises = 0

    def forward(self, x):
        if x.size(0) > self.max_batch:
            self.oom_raises += 1
            raise torch.OutOfMemoryError("synthetic CUDA out of memory")
        return self.inner(x)


class TestChunkedGenuineEval:
    """[C4'] eval_forward_over_val halves the forward batch on OOM instead of
    dying: measured 36.94 GiB alloc = 32 cycles x 512 val batch on a ViT."""

    def _batches(self, n=2, batch=8):
        torch.manual_seed(0)
        return [
            (torch.randn(batch, 3), torch.randint(0, 2, (batch,)))
            for _ in range(n)
        ]

    def test_oom_falls_back_to_chunks_with_identical_accuracy(self):
        batches = self._batches()
        model = _TinyNet()
        plain = eval_forward_over_val(
            _FakeTrainer(batches), model, model, 2, "cpu",
        )
        guarded = _OOMBelowChunk(model, max_batch=2)
        chunked = eval_forward_over_val(
            _FakeTrainer(batches), guarded, model, 2, "cpu",
        )
        assert chunked == plain
        assert guarded.oom_raises >= 1

    def test_oom_at_chunk_one_reraises(self):
        batches = self._batches()
        model = _TinyNet()
        dead = _OOMBelowChunk(model, max_batch=0)
        try:
            eval_forward_over_val(_FakeTrainer(batches), dead, model, 2, "cpu")
        except torch.OutOfMemoryError:
            return
        raise AssertionError("an unchunkable OOM must fail loud")

    def test_wrapped_oom_is_unwrapped_through_the_cause_chain(self):
        # ModelRepresentation wraps forward failures in RuntimeError (its
        # fail-loud node context); the chunker must unwrap __cause__ to see
        # the OOM underneath (measured: the wrap masked the OOM and killed
        # the LIF probe despite the chunker being engaged).
        batches = self._batches()
        model = _TinyNet()

        class _WrappingOOM(nn.Module):
            def __init__(self, inner, max_batch):
                super().__init__()
                self.inner = inner
                self.max_batch = int(max_batch)

            def forward(self, x):
                if x.size(0) > self.max_batch:
                    try:
                        raise torch.OutOfMemoryError("synthetic OOM")
                    except torch.OutOfMemoryError as exc:
                        raise RuntimeError(
                            "[ModelRepresentation] forward failed at node X"
                        ) from exc
                return self.inner(x)

        plain = eval_forward_over_val(
            _FakeTrainer(batches), model, model, 2, "cpu",
        )
        wrapped = _WrappingOOM(model, max_batch=2)
        chunked = eval_forward_over_val(
            _FakeTrainer(batches), wrapped, model, 2, "cpu",
        )
        assert chunked == plain

    def test_non_oom_runtime_error_stays_loud(self):
        batches = self._batches()

        class _Broken(nn.Module):
            def forward(self, x):
                raise RuntimeError("genuine bug, not an OOM")

        try:
            eval_forward_over_val(
                _FakeTrainer(batches), _Broken(), _TinyNet(), 2, "cpu",
            )
        except RuntimeError as exc:
            assert "genuine bug" in str(exc)
            return
        raise AssertionError("a non-OOM RuntimeError must propagate")

    def test_no_oom_keeps_the_single_forward_path(self):
        batches = self._batches()
        model = _TinyNet()
        counting = _OOMBelowChunk(model, max_batch=1000)
        eval_forward_over_val(_FakeTrainer(batches), counting, model, 2, "cpu")
        assert counting.oom_raises == 0
