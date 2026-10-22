"""The X3a generator: a seeded random integer network over a declared space, with named motifs."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Sequence, Tuple

from mimarsinan.certification.randomized.instance import (
    HOST,
    INPUT,
    NEURAL,
    SEMANTICS,
    VALUE,
    Instance,
    Layer,
    Network,
    validate,
)

#: Structural motifs an instance may carry; the coverage report counts each one.
MOTIFS = ("chain", "skip", "concat", "host", "residual", "host_residual", "wide_fan_in", "siblings",
          "narrow_chain", "uneven_branches")


@dataclass(frozen=True)
class GeneratorSpace:
    """The declared X3a generator space (portfolio X3a, Variables); every range is inclusive.

    ``exact_bound`` caps every partial sum a value-domain program forms (the exact-accumulation hypothesis
    H8): the value executor runs in float64, exact below 2^53 (G2's int64 golden holds it too). Weights are
    drawn from ``weight`` everywhere; where a row's worst case would pass the bound, the generator zeroes
    randomly chosen entries of that row until its L1 norm fits (a sparser row, never a narrower range), so
    a deep value layer keeps input-dependent rows. Spiking wires carry counts, so the bound never binds there.
    """

    depth: Tuple[int, int] = (2, 8)
    width: Tuple[int, int] = (1, 32)
    max_fan_in: int = 64
    input_width: Tuple[int, int] = (4, 16)
    weight: Tuple[int, int] = (-8, 8)
    theta: Tuple[int, int] = (1, 16)
    bias: Tuple[int, int] = (-4, 4)
    timesteps: Tuple[int, int] = (1, 8)
    value_input: Tuple[int, int] = (0, 8)
    batch: int = 4
    p_skip: float = 0.3
    p_host: float = 0.2
    p_residual: float = 0.15
    p_wide: float = 0.1
    exact_bound: int = 2 ** 53


DEFAULT_SPACE = GeneratorSpace()


def generate(seed: int, space: GeneratorSpace = DEFAULT_SPACE, *, semantics: Optional[str] = None,
             motifs: Iterable[str] = ()) -> Instance:
    """The instance of ``seed``; ``motifs`` forces each named motif at least once (a targeted generator)."""
    rng = random.Random(int(seed))
    sem = semantics if semantics is not None else rng.choice(SEMANTICS)
    forced = tuple(motifs)
    unknown = [m for m in forced if m not in MOTIFS]
    if unknown:
        raise ValueError(f"unknown motifs {unknown}; known: {MOTIFS}")
    timesteps = rng.randint(*space.timesteps)
    builder = _Builder(rng, space, sem, forced)
    network = builder.build()
    lo, hi = (space.value_input if sem == VALUE else (0, timesteps))
    inputs = tuple(tuple(rng.randint(lo, hi) for _ in range(network.input_width)) for _ in range(space.batch))
    instance = Instance(seed=int(seed), semantics=sem, timesteps=timesteps, hardware_bias=rng.random() < 0.5,
                        network=network, inputs=inputs, motifs=tuple(sorted(builder.used)))
    validate(instance)
    return instance


class _Builder:
    def __init__(self, rng: random.Random, space: GeneratorSpace, semantics: str, forced: Sequence[str]):
        self.rng, self.space, self.semantics = rng, space, semantics
        self.forced = set(forced)
        self.used: set = set()
        self.layers: List[Layer] = []
        self.widths = {INPUT: 0}
        self.bounds = {INPUT: 0}
        self.consumed: set = set()

    def build(self) -> Network:
        rng, space = self.rng, self.space
        input_width = rng.randint(*space.input_width)
        self.widths[INPUT] = input_width
        self.bounds[INPUT] = space.value_input[1] if self.semantics == VALUE else 1
        last = INPUT
        depth = rng.randint(*space.depth)
        for index in range(depth):
            last = self._neural_step(index, last, depth)
            if self._want("host", space.p_host) and index < depth - 1:
                last = self._host(last)
        self.used.add("chain")
        for motif in sorted(self.forced - self.used):
            last = self._force(motif, last)
        # Every neural layer is an output, so every neuron is observed: a node read only inside its own
        # segment is never published, and a comparison of published nodes alone would skip it.
        observed = tuple(layer.name for layer in self.layers
                         if layer.kind == NEURAL or layer.name not in self.consumed)
        return Network(input_width=input_width, layers=tuple(self.layers), outputs=observed)

    def _want(self, motif: str, p: float) -> bool:
        return motif in self.forced and motif not in self.used or self.rng.random() < p

    def _neural_step(self, index: int, last: str, depth: int) -> str:
        rng, space = self.rng, self.space
        earlier = [layer.name for layer in self.layers if layer.kind == NEURAL and layer.name != last]
        if earlier and self._want("residual", space.p_residual):
            twin = [n for n in earlier if self.widths[n] == self.widths[last] and self._sum_fits(last, n)]
            if twin and last != INPUT:
                return self._residual(last, rng.choice(twin))
        sources = [last]
        if earlier and self._want("skip", space.p_skip):
            other = rng.choice(earlier)
            if self.widths[last] + self.widths[other] <= space.max_fan_in:
                sources.append(other)
                self.used.update(("skip", "concat"))
        width = rng.randint(*space.width)
        if self._want("wide_fan_in", space.p_wide) and len(sources) == 1:
            pool = [n for n in earlier + [last] if n != INPUT]
            while pool and sum(self.widths[s] for s in sources) < space.max_fan_in // 2:
                extra = rng.choice(pool)
                pool.remove(extra)
                if sum(self.widths[s] for s in sources) + self.widths[extra] <= space.max_fan_in:
                    sources.append(extra)
            if len(sources) > 1:
                self.used.update(("wide_fan_in", "concat"))
        return self._neural(f"n{index}", tuple(sources), width)

    def _neural(self, name: str, sources: Tuple[str, ...], width: int,
                weights: Optional[Tuple[Tuple[int, ...], ...]] = None, theta: Optional[int] = None,
                bias: Optional[Tuple[int, ...]] = None) -> str:
        rng, space = self.rng, self.space
        fan_in = sum(self.widths[s] for s in sources)
        bound_in = max(self.bounds[s] for s in sources)
        if weights is None:
            weights = tuple(self._row(fan_in, bound_in) for _ in range(width))
        if bias is None:
            bias = tuple(rng.randint(*space.bias) for _ in range(width))
        if theta is None:
            theta = 1 if self.semantics == VALUE else rng.randint(*space.theta)
        self.layers.append(Layer(name=name, kind=NEURAL, sources=sources, width=width, weights=weights,
                                 bias=bias, theta=theta))
        self.widths[name] = width
        peak = max(sum(abs(w) for w in row) for row in weights) * max(1, bound_in) + max(abs(b) for b in bias)
        self.bounds[name] = peak if self.semantics == VALUE else 1
        self.consumed.update(sources)
        return name

    def _row(self, fan_in: int, bound_in: int) -> Tuple[int, ...]:
        """One weight row drawn from the declared range; under the value semantics its L1 norm is cut (by
        zeroing randomly chosen entries) to the budget that keeps the worst-case charge below the bound."""
        rng, space = self.rng, self.space
        row = [rng.randint(*space.weight) for _ in range(fan_in)]
        if self.semantics != VALUE:
            return tuple(row)
        budget = max(0, (space.exact_bound // 4 - space.bias[1]) // max(1, bound_in))
        order = list(range(fan_in))
        rng.shuffle(order)
        for index in order:
            if sum(abs(w) for w in row) <= budget:
                break
            row[index] = 0
        return tuple(row)

    def _host(self, source: str) -> str:
        name = f"h{len(self.layers)}"
        op = "relu" if self.semantics == VALUE else "identity"
        self.layers.append(Layer(name=name, kind=HOST, sources=(source,), width=self.widths[source], op=op))
        self.widths[name], self.bounds[name] = self.widths[source], self.bounds[source]
        self.consumed.add(source)
        self.used.add("host")
        return name

    def _sum_fits(self, a: str, b: str) -> bool:
        """Whether adding two branches keeps the value semantics' worst case inside the exact bound."""
        return self.semantics != VALUE or self.bounds[a] + self.bounds[b] < self.space.exact_bound // 4

    def _residual(self, a: str, b: str) -> str:
        """The residual add of two equal-width branches: an on-chip merge core or the host add."""
        if self.rng.random() < 0.5:
            return self._host_sum(a, b)
        return self._merge_core(a, b)

    def _merge_core(self, a: str, b: str) -> str:
        """An on-chip merge core ``[I | I]`` over ``concat(a, b)`` at threshold 1."""
        width = self.widths[a]
        eye = tuple(tuple(1 if c == r or c == r + width else 0 for c in range(2 * width)) for r in range(width))
        self.used.update(("residual", "concat"))
        return self._neural(f"r{len(self.layers)}", (a, b), width, weights=eye, theta=1,
                            bias=tuple(0 for _ in range(width)))

    def _host_sum(self, a: str, b: str) -> str:
        name = f"h{len(self.layers)}"
        self.layers.append(Layer(name=name, kind=HOST, sources=(a, b), width=self.widths[a], op="sum"))
        self.widths[name], self.bounds[name] = self.widths[a], self.bounds[a] + self.bounds[b]
        self.consumed.update((a, b))
        self.used.add("host_residual")
        return name

    def _force(self, motif: str, last: str) -> str:
        """Append the structure of one forced motif the random walk did not produce."""
        width = self.rng.randint(*self.space.width)
        anchor = last if last != INPUT else self._neural(f"f{len(self.layers)}", (INPUT,), width)
        if motif == "host":
            return self._neural(f"f{len(self.layers)}", (self._host(anchor),), width)
        if motif == "siblings":
            twins = [self._neural(f"f{len(self.layers)}", (anchor,), self.rng.randint(1, 4)) for _ in range(3)]
            self.used.add("siblings")
            return self._neural(f"f{len(self.layers)}", tuple(twins), width)
        if motif == "uneven_branches":
            short = self._neural(f"f{len(self.layers)}", (anchor,), self.rng.randint(1, 4))
            long = self._neural(f"f{len(self.layers)}", (anchor,), self.rng.randint(1, 4))
            long = self._neural(f"f{len(self.layers)}", (long,), self.rng.randint(1, 4))
            self.used.add("uneven_branches")
            return self._neural(f"f{len(self.layers)}", (short, long), width)
        if motif == "narrow_chain":
            node = anchor
            for _ in range(3):
                node = self._neural(f"f{len(self.layers)}", (node,), self.rng.randint(1, 4))
            self.used.add("narrow_chain")
            return node
        mid = self._neural(f"f{len(self.layers)}", (anchor,), self.widths[anchor])
        if motif in ("residual", "host_residual") and not self._sum_fits(mid, anchor):
            mid = self._neural(f"f{len(self.layers)}", (anchor,), self.widths[anchor],
                               weights=tuple(tuple(0 for _ in range(self.widths[anchor]))
                                             for _ in range(self.widths[anchor])))
        if motif == "residual":
            return self._merge_core(mid, anchor)
        if motif == "host_residual":
            return self._host_sum(mid, anchor)
        sources = [mid, anchor]
        if motif == "wide_fan_in":
            pool = [layer.name for layer in self.layers if layer.kind == NEURAL and layer.name not in sources]
            self.rng.shuffle(pool)
            for extra in pool:
                if sum(self.widths[s] for s in sources) + self.widths[extra] <= self.space.max_fan_in:
                    sources.append(extra)
            self.used.add("wide_fan_in")
        self.used.update(("skip", "concat"))
        return self._neural(f"f{len(self.layers)}", tuple(sources), width)


def with_semantics(instance: Instance, semantics: str) -> Instance:
    """The same network under another semantics (inputs re-clamped to its range)."""
    clipped = instance.inputs if semantics == VALUE else tuple(
        tuple(min(max(v, 0), instance.timesteps) for v in row) for row in instance.inputs)
    out = replace(instance, semantics=semantics, inputs=clipped)
    validate(out)
    return out
