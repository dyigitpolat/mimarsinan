"""[U-PFOLD r3] The closure of the stamp's readers: outside the one reader, no site reads the stamp's form.

The one reader of a perceptron's input fold (``per_input_scales``) is ``transformations/perceptron/position_fold.py``.
Every other site of the source that names the stamp is listed below with its read, and the list is finite and exact:

- the one writer (``scale_propagation.assign_per_input_scales``) and the per-input reader
  (``PerceptronTransformer.input_scale``, behind ``refuse_per_input_read``);
- presence, save and restore sites, and reads of the stamp's values that are the same in either form (a mean, a check
  that every input holds one value): none reads its rows or its shape.

A reader of the rows of a fold per position or of their count takes them through ``position_fold.position_folds``
(each core position's fold row, ``rows[index]`` of ``position_rows``, on the weight's device). The scan reads the
syntax tree of every module under ``src/mimarsinan``: an attribute of the stamp's name (a load, a store or a delete)
and a string constant of its name (``getattr``, a buffer's key) are each one read of their site, the enclosing
function. A site that is not listed, or a listed site with another count of reads, is red; so is a listed site that is
gone. Red on revision 2 (three readers read the stamp directly: the twin's position count, the mappers' bank key and
assertion, the saturation bound's input divisor).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import mimarsinan

STAMP = "per_input_scales"
SRC = Path(mimarsinan.__file__).resolve().parent
THE_READER = "transformations/perceptron/position_fold.py"

REVIEWED: dict = {
    ("mapping/mappers/scale_propagation.py", "assign_per_input_scales"):
        (4, "the one writer: the stamp in either form, on the weight's device"),
    ("transformations/perceptron/perceptron_transformer.py", "PerceptronTransformer.input_scale"):
        (1, "the per-input reader: the one fold per input on the weight's input axis, behind refuse_per_input_read"),
    ("models/perceptron_mixer/perceptron.py", "Perceptron.__init__"):
        (1, "registration: the stamp declared a non-persistent buffer"),
    ("models/perceptron_mixer/perceptron.py", "Perceptron.__setstate__"):
        (3, "restore: an old cache's plain attribute moved into the buffer"),
    ("mapping/channel_axis_walk.py", "consumer_columns_unmediated"):
        (1, "presence: whether a stamp mediates the consumer's weight columns"),
    ("spiking/seam_audit.py", "_currency_certificate"):
        (2, "presence and the stamp's mean against the producer's currency, an audit certificate's value"),
    ("spiking/ttfs_cycle_reads.py", "input_fold"):
        (1, "presence and the one-value check over the stamp's values, the same in either form; each form's view is "
            "the one reader's (input_fold_view) or the per-input reader's"),
    ("spiking/wire_gauge_seam.py", "_gauge_state"):
        (1, "save: the stamp object held for the seam's restore"),
    ("spiking/wire_gauge_seam.py", "_restore"):
        (1, "restore: the saved stamp object written back"),
    ("spiking/wire_gauge_seam.py", "_walk_key"):  # [merge cut, U-POST2]
        (1, "identity: the stamp object in the walk's derivation key (a writer that replaced it re-derives), never its "
            "values or form"),
}


def _sites(tree: ast.AST, rel: str, scope: str, found: dict) -> None:
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _sites(child, rel, f"{scope}.{child.name}" if scope else child.name, found)
            continue
        if (isinstance(child, ast.Attribute) and child.attr == STAMP) or (
                isinstance(child, ast.Constant) and child.value == STAMP):
            found.setdefault((rel, scope or "<module>"), []).append(int(child.lineno))
        _sites(child, rel, scope, found)


def _reads_outside_the_reader() -> dict:
    found: dict = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel != THE_READER:
            _sites(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), rel, "", found)
    return found


def test_every_read_of_the_stamp_outside_the_one_reader_is_reviewed():
    found = _reads_outside_the_reader()
    unlisted = {f"{rel}:{scope} (lines {lines})" for (rel, scope), lines in found.items() if (rel, scope) not in REVIEWED}
    assert not unlisted, f"reads of the stamp outside the one reader and its reviewed list: {sorted(unlisted)}"
    recounted = {f"{rel}:{scope} reads {len(lines)} (lines {lines}), reviewed {REVIEWED[(rel, scope)][0]}"
                 for (rel, scope), lines in found.items() if len(lines) != REVIEWED[(rel, scope)][0]}
    assert not recounted, f"a reviewed site whose reads changed: {sorted(recounted)}"


def test_every_reviewed_site_still_reads_the_stamp():
    found = _reads_outside_the_reader()
    gone = sorted(f"{rel}:{scope}" for rel, scope in REVIEWED if (rel, scope) not in found)
    assert not gone, f"reviewed sites that no longer read the stamp (the list is exact): {gone}"
    assert all(read for _count, read in REVIEWED.values())


def test_the_one_reader_s_accessor_is_each_position_s_fold_row_on_the_weight_s_device():
    """``position_folds``: ``(positions, in_features)``, each position's own row (a linear's rows are its positions; a
    convolution's rows are its distinct patch folds, indexed per position), moved to the weight's device."""
    import torch.nn as nn

    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
    from mimarsinan.transformations.perceptron.position_fold import position_folds, position_rows

    p = Perceptron(3, 4, normalization=nn.Identity())
    stamp = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    p.per_input_scales = stamp.clone()
    folds = position_folds(p)
    assert torch.equal(folds, stamp) and folds.dtype == stamp.dtype
    rows, index = position_rows(p)
    assert torch.equal(folds, rows[index])
    p.output_channel_axis = 1  # a convolution's: its rows are the distinct folds, indexed per position
    p.per_input_scales = torch.tensor([[2.0, 1.0], [1.0, 3.0], [2.0, 1.0]])
    rows, index = position_rows(p)
    assert int(rows.shape[0]) == 2 and torch.equal(position_folds(p), p.per_input_scales)
    p.output_channel_axis = -1
    p.per_input_scales = stamp.clone()
    p.layer.to("meta")
    assert position_folds(p).device.type == "meta", "on the weight's device, as every reader reads the weight"


@pytest.mark.parametrize("form", ("token_mixing", "conv_seam"))
def test_the_accessor_equals_the_stamp_on_the_converter_s_forms(form):
    import unit.spiking.position_fold_forms as pf

    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.transformations.perceptron.position_fold import position_folds

    _model, flow, shape = pf.converted(form, 0)
    perceptrons = list(flow.get_perceptrons())
    for p, g in zip(perceptrons, pf.draw_gauges(flow, perceptrons, shape, 0, "one")):
        p.set_activation_scale(g)
    compute_per_source_scales(flow.get_mapper_repr())
    reader = perceptrons[-1]
    assert reader.per_input_scales.dim() == 2
    folds = position_folds(reader)
    assert torch.equal(folds, reader.per_input_scales)
    assert int(folds.shape[1]) == int(reader.layer.weight[0].numel())
    assert int(torch.unique(folds, dim=0).shape[0]) > 1, "the form holds more than one fold"
