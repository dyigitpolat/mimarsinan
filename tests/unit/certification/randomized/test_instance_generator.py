"""The X3a generator: a pure function of its seed over the declared space, with named motifs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mimarsinan.certification.randomized.generator import DEFAULT_SPACE, MOTIFS, generate, with_semantics
from mimarsinan.certification.randomized.instance import (
    HOST,
    NEURAL,
    SEMANTICS,
    VALUE,
    Instance,
    InstanceError,
)

ROOT = Path(__file__).resolve().parents[4]


class TestDeterminism:
    @pytest.mark.parametrize("seed", [0, 7, 123])
    def test_two_runs_are_byte_identical(self, seed):
        assert generate(seed).to_json() == generate(seed).to_json()

    def test_another_interpreter_draws_the_same_bytes(self):
        """The generator reads no hash seed or global RNG: a fresh process with another PYTHONHASHSEED agrees."""
        code = ("import sys; from mimarsinan.certification.randomized.generator import generate; "
                "sys.stdout.write(generate(41).to_json())")
        env = {"PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": "12345", "OMP_NUM_THREADS": "1",
               "CUDA_VISIBLE_DEVICES": ""}
        other = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
        assert other.stdout == generate(41).to_json()

    def test_different_seeds_differ(self):
        assert len({generate(seed).to_json() for seed in range(20)}) == 20


class TestDeclaredSpace:
    @pytest.mark.parametrize("seed", range(40))
    def test_every_instance_lies_in_the_space(self, seed):
        instance = generate(seed)
        space, net = DEFAULT_SPACE, instance.network
        neural = net.neural_layers()
        assert space.timesteps[0] <= instance.timesteps <= space.timesteps[1]
        assert instance.semantics in SEMANTICS
        assert space.input_width[0] <= net.input_width <= space.input_width[1]
        for layer in neural:
            assert 1 <= layer.width <= space.width[1]
            assert net.fan_in(layer) <= space.max_fan_in
            assert all(space.bias[0] <= b <= space.bias[1] for b in layer.bias)
            assert all(space.weight[0] <= w <= space.weight[1] for row in layer.weights for w in row)
            assert space.theta[0] <= layer.theta <= space.theta[1]
        assert len(instance.inputs) == space.batch

    @pytest.mark.parametrize("seed", range(20))
    def test_every_neural_layer_is_observed(self, seed):
        """A node read only inside its segment is never published; observing every layer keeps none unseen."""
        net = generate(seed).network
        assert set(layer.name for layer in net.neural_layers()) <= set(net.outputs)

    @pytest.mark.parametrize("seed", range(20))
    def test_a_value_instance_keeps_every_partial_sum_exact(self, seed):
        """The value semantics sparsifies rows so the worst-case partial sum stays below the float64 run's
        exact bound (2^53)."""
        assert DEFAULT_SPACE.exact_bound == 2 ** 53
        instance = generate(seed, semantics=VALUE)
        bound = {"input": DEFAULT_SPACE.value_input[1]}
        for layer in instance.network.layers:
            src = max(bound[s] for s in layer.sources)
            if layer.kind == NEURAL:
                assert layer.theta == 1
                peak = max(sum(abs(w) for w in row) for row in layer.weights) * max(1, src)
                bound[layer.name] = peak + max(abs(b) for b in layer.bias)
                assert bound[layer.name] < DEFAULT_SPACE.exact_bound
            else:
                bound[layer.name] = src * (2 if layer.op == "sum" else 1)

    def test_deep_value_layers_keep_their_weights(self):
        """The exact bound sparsifies rows instead of narrowing the range to zero: over 200 value instances,
        at most 2% of the neural layers are all-zero (a constant of the bias, reached by no input), and
        every layer index draws the full declared range somewhere."""
        layers = [layer for seed in range(200) for layer in generate(seed, semantics=VALUE).network.neural_layers()]
        zero = sum(1 for layer in layers if all(w == 0 for row in layer.weights for w in row))
        assert zero <= 0.02 * len(layers), (zero, len(layers))
        peak = max(abs(w) for layer in layers for row in layer.weights for w in row)
        assert peak == DEFAULT_SPACE.weight[1]


class TestMotifs:
    @pytest.mark.parametrize("motif", MOTIFS)
    def test_a_forced_motif_appears(self, motif):
        for seed in range(5):
            assert motif in generate(seed, motifs=(motif,)).motifs

    def test_an_unknown_motif_refuses(self):
        with pytest.raises(ValueError, match="unknown motifs"):
            generate(0, motifs=("spiral",))

    def test_host_ops_follow_the_semantics(self):
        for seed in range(10):
            ops = {layer.op for layer in generate(seed, semantics=VALUE, motifs=("host",)).network.layers
                   if layer.kind == HOST and layer.op != "sum"}
            assert ops <= {"relu"}


class TestDocument:
    def test_round_trip(self):
        instance = generate(5, motifs=("residual", "skip"))
        assert Instance.from_json(instance.to_json()) == instance

    def test_with_semantics_keeps_the_network(self):
        instance = generate(9)
        other = with_semantics(instance, VALUE)
        assert other.network == instance.network and other.semantics == VALUE

    @pytest.mark.parametrize("mutate, message", [
        (lambda d: d.update(schema="x"), "schema"),
        (lambda d: d.update(semantics="quantum"), "semantics"),
        (lambda d: d["network"]["layers"][0].update(sources=["later"]), "before it is defined"),
        (lambda d: d["network"]["layers"][0].update(theta=0), "bias or theta"),
        (lambda d: d["network"].update(outputs=[]), "outputs"),
    ])
    def test_a_malformed_document_refuses_by_name(self, mutate, message):
        doc = json.loads(generate(3, semantics="streamed_lif").to_json())
        mutate(doc)
        with pytest.raises(InstanceError, match=message):
            Instance.from_dict(doc)
