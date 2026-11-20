"""[U-TGAUGE5] Generated per-cycle cascades for the window rule's property test: the placement rule's trees
(``layout_trees``) extended with in-segment fan-out at two depths and inner single-reader gaps, built through the
mapping (``cascade_windows``: Soft Core Mapping's own window calls), in both walk families.

A tree is drawn from a seed: one encoding perceptron over a token input, then chains of one to three perceptrons, each
starting at the encoding or, with probability one half, at a perceptron already drawn (so an in-segment producer is
read at two depths: by its first chain and by the later one's head), every chain's end a leaf and, with probability
one third per chain, an inner perceptron a leaf too (an inner core with one reader across a gap). The leaves are
joined by a random tree of the toolchain's layout mappers (concatenations along either axis, flattens, axis moves) and
read as the output, by a core, by a host operation, or by a host operation then a core, at random gauges (one value,
or one per channel). ``read`` compares the walk with the deployed executor element by element and each reading core's
fold with its sources' placed gauges (the fourth verify's m-5), or reports the toolchain's refusal by name.

[U-TGAUGE6] With ``dead`` the tree also draws dead in-segment reads before the leaves are joined: one or two readers,
each a perceptron over the concatenation of a drawn perceptron and a second part whose input columns are held at
zero. The second part is a drawn perceptron that other cores still read (input columns at zero), a new perceptron
read only through that dead read, or a chain of two or three such perceptrons; with probability one third some
columns of the live part are held at zero too. Each reader is a leaf. A tree drawn without ``dead`` is the tree the
generator drew before.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch

import unit.spiking.cascade_windows as cw
import unit.spiking.host_gather_forms as hg
import unit.spiking.layout_trees as lt

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.structural import ConcatMapper, InputMapper, PermuteMapper, ReshapeMapper
from mimarsinan.mapping.mapping_utils import ModelRepresentation, PerceptronMapper
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

TOKENS, FEATURES = 2, 4
SAMPLES = 8
TIMING_S = 8


def draw_cascade(seed: int, dead: bool = False) -> lt.Tree:
    """One generated cascade (no data: the weights and gauges are drawn from ``seed``); ``dead`` draws dead reads."""
    rng = np.random.default_rng(50_000 + seed)
    torch.manual_seed(50_000 + seed)
    perceptrons: List[Any] = []
    gauges: List[Any] = []
    text: List[str] = []

    def perceptron(src, out, inp, name):
        p = Perceptron(out, inp, name=name)
        with torch.no_grad():
            p.layer.weight.uniform_(-0.6, 0.9)
            p.layer.bias.uniform_(0.0, 0.15)
        perceptrons.append(p)
        if rng.random() < 0.25 and out > 1:
            gauges.append([float(v) for v in rng.uniform(0.4, 2.5, out)])
        else:
            gauges.append(float(rng.uniform(0.4, 2.5)))
        return PerceptronMapper(src, p)

    width = int(rng.integers(3, 6))
    inp = lt._tag(InputMapper((TOKENS, FEATURES)), (TOKENS, FEATURES))
    enc = lt._tag(perceptron(inp, width, FEATURES, "encoding"), (TOKENS, width))
    drawn: List[tuple] = []  # (node, width, depth) of every in-segment perceptron
    leaves: List[Any] = []
    for i in range(int(rng.integers(2, 5))):
        if drawn and rng.random() < 0.5:
            node, w, depth = drawn[int(rng.integers(len(drawn)))]
            text.append(f"fan{depth}")
        else:
            node, w, depth = enc, width, 0
        chain = []
        for j in range(int(rng.integers(1, 4))):
            out = int(rng.integers(2, 5))
            node = lt._tag(perceptron(node, out, w, f"p{i}_{j}"), (TOKENS, out))
            w, depth = out, depth + 1
            chain.append((node, w, depth))
        drawn.extend(chain)
        leaves.append(chain[-1][0])
        if len(chain) > 1 and rng.random() < 1 / 3:
            leaves.append(chain[int(rng.integers(len(chain) - 1))][0])
            text.append("inner")
    for i in range(int(rng.integers(1, 3)) if dead else 0):
        node, w, _depth = drawn[int(rng.integers(len(drawn)))]
        kind = str(rng.choice(["columns", "orphan", "chain"]))
        if kind == "columns":  # a drawn perceptron other cores still read
            part, pw, _ = drawn[int(rng.integers(len(drawn)))]
        else:  # a new perceptron (or a chain of them) read only through the dead read
            part, pw, _ = drawn[int(rng.integers(len(drawn)))] if rng.random() < 0.7 else (enc, width, 0)
            for j in range(1 if kind == "orphan" else int(rng.integers(2, 4))):
                out = int(rng.integers(2, 5))
                part = lt._tag(perceptron(part, out, pw, f"q{i}_{j}"), (TOKENS, out))
                pw = out
        cat = lt._tag(ConcatMapper([node, part], dim=2, rank=3), (TOKENS, w + pw))
        out = int(rng.integers(2, 5))
        reader = lt._tag(perceptron(cat, out, w + pw, f"dead{i}"), (TOKENS, out))
        zero = list(range(w, w + pw))
        if rng.random() < 1 / 3:
            zero += [int(c) for c in rng.choice(w, size=int(rng.integers(1, w + 1)), replace=False)]
        with torch.no_grad():
            perceptrons[-1].layer.weight[:, zero] = 0.0
        leaves.append(reader)
        text.append(f"dead_{kind}")

    def flatten(node):
        shape = lt._part_shape(node)
        if len(shape) == 1:
            return node
        return lt._tag(ReshapeMapper(node, (int(np.prod(shape)),)), (int(np.prod(shape)),))

    def move(node):
        shape = lt._part_shape(node)
        if len(shape) == 2 and rng.random() < 0.25:
            text.append("axis_move")
            return lt._tag(PermuteMapper(node, (0, 2, 1)), (shape[1], shape[0]))
        return node

    parts = [move(leaf) for leaf in leaves]
    while len(parts) > 1:
        a, b = parts.pop(int(rng.integers(len(parts)))), parts.pop(int(rng.integers(len(parts))))
        sa, sb = lt._part_shape(a), lt._part_shape(b)
        if len(sa) == 2 and len(sb) == 2 and sa[0] == sb[0] and rng.random() < 0.6:
            node = lt._tag(ConcatMapper([a, b], dim=2, rank=3), (sa[0], sa[1] + sb[1]))
            text.append("cat_last")
        elif len(sa) == 2 and len(sb) == 2 and sa[1] == sb[1]:
            node = lt._tag(ConcatMapper([a, b], dim=1, rank=3), (sa[0] + sb[0], sa[1]))
            text.append("cat_first")
        else:
            a, b = flatten(a), flatten(b)
            na, nb = lt._part_shape(a)[0], lt._part_shape(b)[0]
            node = lt._tag(ConcatMapper([a, b], dim=1, rank=2), (na + nb,))
            text.append("cat_flat")
        parts.append(node)
    root = parts[0]
    reader = str(rng.choice(["out", "core", "core", "host", "host_core"]))
    flat = flatten(root)
    n = lt._part_shape(flat)[0]
    if reader == "out":
        top = flat
    elif reader == "core":
        top = lt._tag(perceptron(flat, 3, n, "reader"), (3,))
    else:
        host = lt._tag(ComputeOpMapper(flat, lt.HostAffine(), name="host"), (n,))
        top = host if reader == "host" else lt._tag(perceptron(host, 3, n, "reader"), (3,))
    text.append(reader)
    depths = sorted({d for _n, _w, d in drawn})
    return lt.Tree(seed, " ".join(text), ModelRepresentation(top), perceptrons, gauges, reader=reader, depths=depths)


def read(seed: int, family: str, how: str = "mapping", *, semantics: Optional[str] = None,
         dead: bool = False) -> dict:
    """One generated cascade (``dead``: with dead reads) built ``how`` and read by both twins: ``{"seed", "tree",
    "max_abs", "fold_error", "relays", "window"}`` where compared, ``{"refused": name}`` where the toolchain declines
    the construction by name. Any other error propagates. The count family runs ``semantics``, else one LIF semantics
    by the seed's parity."""
    tree = draw_cascade(seed, dead)
    rec: dict = {"seed": seed, "tree": tree.text}
    flow = lt.GraphFlow(tree.repr_).eval()
    with torch.no_grad():
        flow(torch.rand(2, *tree.shape))
    gauges = [torch.as_tensor(g, dtype=torch.float32) for g in tree.gauges]
    try:
        if family == "timing":
            torch.manual_seed(7_000 + seed)
            x = torch.rand(SAMPLES, *tree.shape)
            got = cw.timing_program(flow, tree.shape, gauges, seed, how, S=TIMING_S, samples=x,
                                    perceptrons=tree.perceptrons)
        else:
            semantics = semantics or sorted(hg.SEMANTICS)[seed % len(hg.SEMANTICS)]
            rec["semantics"] = semantics
            torch.manual_seed(8_000 + seed)
            x = 2.0 * torch.rand(SAMPLES, *tree.shape)
            got = cw.count_program(flow, tree.shape, gauges, seed, how, semantics=semantics, samples=x,
                                   perceptrons=tree.perceptrons)
    except Exception as exc:  # noqa: BLE001 - a named refusal is a result; any other error is re-raised
        if lt.refused_by_name(exc):
            rec["refused"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            return rec
        raise
    rec["max_abs"] = float((got.walk - got.deployed).abs().max())
    rec["off"] = got.off()
    rec["relays"] = got.relays
    rec["window"] = got.window
    rec["folds_checked"], rec["fold_error"] = cw.fold_errors(got)
    return rec
