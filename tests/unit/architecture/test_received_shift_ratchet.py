"""[U-RSHIFT] Ratchet: a reader's correction for a shift its inputs receive has one writer, and every other bias writer
of the tree is a reviewed writer outside the class.

The test reads a normalized form (``received_shift_writers``): every function of ``src`` from its syntax tree, its
module's imports resolved (a name or a module imported under another name, a relative import, a package's re-export),
its names followed through their bindings (a local name, a loop, ``with`` or comprehension target bound to a bias chain
or to ``getattr`` of a bias name, a function that returns a bias chain), and its references (a call, a callee chosen by
an expression, a writer passed as an argument or bound to a name). A site is a bias writer: every function of ``src``
that writes the bias of a perceptron or a host module, directly or through any function it refers to that does, closed
over the references. Each is either the received shift's one writer (``mapping/support/received_shift.py``:
``correct_for_received_shift`` and ``bake_received_shift`` with their parts), a function whose only bias writes go
through it (no writer then), or a function on the finite list below, each named with its read. A writer on none of these
is red, whatever value it writes; a listed writer that is gone is red too. ``GENERIC`` writers write a value their caller
passes, so their callers are writers in turn; every other listed writer decides its value itself (a copy, a
representation, a pruning, a quantization grid, a fold of the perceptron's own output), and the closure stops there.

The bias names (``BIAS``) are held beside a search: every parameter or buffer that a module of ``src`` or of
``torch.nn`` registers, which a forward (or a function a forward of ``src`` reaches) adds as a summand, is a bias name or
a reviewed name of ``NOT_A_BIAS`` with its read; a listed name the search no longer finds is red.

The detector's own test pins as red every shape written on paper: the review's six (a response computed by one helper
and written by another, a generic writer carrying a new field's response, an augmented assignment, a subscript
assignment, a convolution whose weight operand has another name, an ``out=`` write), the check's eight paper sites (its
own texts of the six, a bias held under another local name, a writer imported under another name), a callee chosen by a
conditional expression, and the unit's own: a writer passed as an argument, a writer bound to a name, a bias reached by
``getattr`` of a constant name in a loop, a bias returned by a function of the tree, the per-position bias through
the fold's accessor, a bias named by an f-string with a known suffix, a helper that writes its parameter through a name
it binds, and a library call that returns a view of the bias; and, red by the guard on the generic writers' names, a
generic writer's name that a second class defines, called on a receiver of unknown class. The search pins a new tensor a forward adds through a function that returns it.

Stated limits of the form (constructs the class's own modules use nowhere): a name built at run time (``getattr`` or
``setattr`` with a computed name), reflection over a module's tensors (``parameters()``, ``state_dict()``,
``__dict__``), a reinterpretation of a tensor's storage, and a callee held in a container and chosen at run time. A bias
written through one of them is not seen.
"""

from __future__ import annotations

import ast
import functools
import inspect
from pathlib import Path

import torch.nn.modules

from unit.architecture.received_shift_writers import BIAS, ONE_WRITER, Source, registered, summands

SRC = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"

#: Writers whose value comes from their caller (the closure passes through them).
GENERIC = {
    "transformations/perceptron/perceptron_transformer.py::PerceptronTransformer.write_shared_bias":
        "the shared effective bias written from a transform the caller passes",
    "transformations/perceptron/perceptron_transformer.py::PerceptronTransformer.apply_effective_bias_transform":
        "a caller's bias transform on the view the reader declares",
    "transformations/perceptron/perceptron_transformer.py::PerceptronTransformer.apply_effective_parameter_transform":
        "a caller's transform of weight and bias alike",
    "transformations/perceptron/perceptron_transformer.py::"
    "PerceptronTransformer._realize_effective_bias_through_normalization":
        "the caller's effective bias through the normalization",
    "transformations/perceptron/perceptron_transformer.py::PerceptronTransformer._apply_bias_free_effective_bias":
        "the caller's effective bias into a bias-free layer's normalization",
    "transformations/perceptron/position_bias.py::_set_offsets": "each position's offset as the caller computed it",
    "transformations/perceptron/position_bias.py::add_bias_corrections": "a caller's per-position corrections",
    "transformations/perceptron/position_bias.py::apply_bias_transform": "a caller's transform on every position's total",
    "mapping/support/bias_compensation.py::apply_additive_effective_bias_shift": "a caller's constant per output",
    "spiking/dfq_bias_correction.py::_effective_bias_shift": "a caller's measured shift per output",
    "tuning/lif_affine_fold.py::fold_affine_into_readout": "the caller's fitted affine on the readout's own rows",
}

#: Writers that decide their value themselves, each outside the class by its read.
CONTAINED = {
    # the perceptron's own output: folds, grids and fits of its own rows (no value an input receives)
    "mapping/support/bias_compensation.py::apply_half_step_entry_fold": "a half step of the entry's own output grid",
    "mapping/support/bias_compensation.py::apply_ttfs_quantization_bias_compensation":
        "a half step of each perceptron's own timing grid",
    "spiking/dfq_bias_correction.py::sequential_first_moment_fold": "a measured mean of the perceptron's own output",
    "spiking/dfq_bias_correction.py::dfq_correct_biases": "a measured output mean (no exactness claim)",
    "spiking/dfq_bias_correction.py::_restore_biases": "a saved bias restored",
    "transformations/normalization_aware_perceptron_quantization.py::NormalizationAwarePerceptronQuantization._transform":
        "the bias on its own grid (its weights' compensation through the one writer)",
    "transformations/normalization_aware_perceptron_quantization.py::"
    "NormalizationAwarePerceptronQuantization._transform_two_scale": "the bias on its own row grid",
    "transformations/normalization_fusion.py::fuse_into_perceptron": "the perceptron's own output normalization",
    "transformations/channel_scale_equalization.py::apply_scale_migration": "a per-channel gain of the producer's output",
    "transformations/perceptron/bias_canonicalization.py::_apply_decision_verified": "a verified decision restored",
    "transformations/perceptron/bias_canonicalization.py::canonicalize_starved_bias_outliers":
        "a shift of its own channels into their measured pre-activation range",
    "transformations/perceptron/bias_canonicalization.py::_remove_nuisance_grid_dominators.remove_channel":
        "a removed channel's bias at zero",
    "transformations/perceptron/bias_saturation.py::clip_off_saturated_effective_bias":
        "its own bias clipped to its saturation bound",
    "tuning/tuners/activation_shift_tuner.py::ActivationShiftTuner._apply_shift": "a constant of the own output",
    "tuning/lif_affine_fold.py::apply_lif_affine_fold":
        "the readout fold's own rows; consumer folds through the one writer",
    "models/nn/layers.py::FrozenStatsNormalization.__setstate__": "its own statistics moved into buffers on a load",
    "models/nn/layers.py::FrozenStatsNormalization.forward": "its own statistics moved to the input's device",
    # pruning: a removed output channel's bias, masks and their shrinks
    "transformations/pruning/apply.py::apply_pruning_masks": "a pruned row's bias scaled out",
    "transformations/pruning/committed_masks.py::commit_layer_pruning": "a pruned bias held at zero",
    "transformations/pruning/committed_masks.py::commit_norm_pruning": "a pruned channel's normalization bias at zero",
    "transformations/pruning/magnitude.py::_shrink_linear_inputs": "the bias carried to a shrunk layer",
    "transformations/pruning/magnitude.py::_shrink_linear_outputs": "the kept outputs' bias",
    "tuning/tuners/pruning/pruning_tuner_enforce.py::enforce_pruning_persistently": "a pruned row's bias at zero",
    "tuning/tuners/pruning/pruning_tuner_masks.py::register_recovery_hooks.make_hook.hook":
        "a pruned row's bias held at its recovery target",
    "search/problems/joint/candidate_pruning.py::_shrink_normalization": "the kept channels' normalization bias",
    "mapping/pruning/certificate/seed_reference.py::apply_admitted_seed_zeroing": "an admitted dead column at zero",
    "mapping/pruning/graph/constant_folding.py::apply_constant_folds":
        "an always-constant input line's charge folded into its core's bias, per axon by the IR's explicit sources "
        "(no received shift reaches the IR lattice: the review's m-5)",
    "mapping/pruning/certificate/dyadic_grid.py::snap_ir_graph_to_dyadic_grid": "a core's bias snapped to its grid",
    # the IR and the chip: the deployed bias copied, scaled, sliced or packed
    "mapping/export/chip_quantize.py::_scale_hardware_bias": "a core's bias on the chip's grid",
    "mapping/export/law_solver.py::solve_core_law": "a core's integer law, its bias on the law's own grid",
    "mapping/ir/legacy_convert.py::neural_core_to_soft_core": "a core's bias copied",
    "mapping/layout/layout_ir_mapping.py::LayoutIRMapping.__post_init__": "a layout's flag that cores hold a bias",
    "mapping/packing/softcore/hard_core.py::HardCore.__init__": "an empty hard core",
    "mapping/packing/softcore/hard_core.py::HardCore.add_softcore": "a soft core's bias placed in its hard core",
    "mapping/packing/softcore/hard_core_mapping.py::RuntimeMaterializer.fuse_hardcores._mk": "a fused core's copy",
    "mapping/packing/softcore/soft_core.py::SoftCore.__init__": "an empty soft core",
    "chip_simulation/sanafe/neuron_model.py::_ttfs_base_model_attributes": "a backend neuron's bias (a copy)",
    "chip_simulation/sanafe/neuron_model.py::lif_model_attributes": "a backend neuron's bias (a copy)",
    "code_generation/cpp_chip_model_types.py::Neuron.__init__": "a generated neuron's bias (a copy)",
    "gui/server/routes_layout.py::register_routes.api_hw_config_verify._run": "a view's copy of a core's bias",
    "gui/snapshot/model_snapshot.py::snapshot_model": "a snapshot's copy",
    "pipelining/pipeline_steps/mapping/soft_core_mapping_step.py::SoftCoreMappingStep.bring_back_bias":
        "the bias restored onto the mapped layer",
    # construction: the converter's and the builders' copies of a trained bias, and flags
    "mapping/mappers/conv1d_mapper.py::Conv1DPerceptronMapper.__init__": "the flag that the layer holds a bias",
    "mapping/mappers/conv2d_mapper.py::Conv2DPerceptronMapper.__init__": "the flag that the layer holds a bias",
    "models/nn/activations/lif_serial.py::SerialFoldSlot.__init__": "a twin's copy of the bias in its own units",
    "spiking/segment_policy_lif_fold.py::CycleFoldSlot.__init__": "a twin's copy of the bias",
    "models/nn/layers.py::FrozenStatsNormalization.__init__": "a frozen normalization's own bias",
    "pipelining/pipeline_steps/mapping/fused_linear.py::FusedLinear.__init__": "the fused layer's own bias",
    "models/builders/torch/torch_vit_builder.py::_adapt_patch_projection": "a builder's copy of a trained bias",
    "models/builders/torch/torchvision_builder_utils.py::adapt_conv_in_channels": "a builder's copy",
    "torch_mapping/converter_handlers/batchnorm_lowering.py::copy_batchnorm_state": "the converter's copy",
    "torch_mapping/converter_handlers/conv_mixin.py::ConvConvertMixin._convert_conv1d": "the converter's copy",
    "torch_mapping/converter_handlers/conv_mixin.py::ConvConvertMixin._convert_conv2d": "the converter's copy",
    "torch_mapping/converter_handlers/linear_mixin.py::LinearConvertMixin._convert_linear": "the converter's copy",
    "torch_mapping/graph_normalization.py::_fuse_linear_pair": "W2 b1 + b2 (every axon receives b1)",
    "torch_mapping/leaf_decomposition/attention.py::shared_linear": "a decomposed projection's own bias",
    "torch_mapping/leaf_decomposition/recurrence.py::input_projection": "a decomposed projection's own biases",
    "torch_mapping/value_parity/witness.py::randomize_batchnorm_statistics": "a witness's random statistics",
    "transformations/perceptron/position_fold.py::copy_position_state":
        "the per-position state copied from another copy of the perceptron",
    "certification/randomized/program.py::_perceptron": "a randomized program's own perceptron",
}

#: Tensors the search finds added by a forward that are not a reader's bias, each with its read.
NOT_A_BIAS = {
    "output_offset": "the lift itself: the shift's source inside its op, written by the seam and the install",
    "shift": "a lowered batch norm's own additive term (a host op the field passes through, its scale times the "
             "field), never a reader",
    "pos_embed": "a model's positional embedding added to its tokens by the model's own forward (a host op's operand)",
    "positional_emb": "a model's positional embedding added to its tokens by the model's own forward (a host op's "
                      "operand)",
    "cls_token": "a class token joined to the tokens before the positional embedding (a host op's operand)",
    "class_emb": "a class token joined to the tokens before the positional embedding (a host op's operand)",
    "alpha": "a surrogate gradient's or a local response norm's own constant (an activation's arithmetic)",
    "scale": "a local response norm's running sum of squares (an activation's arithmetic)",
    "max_value": "a scaler's running maximum (its own moving average, a divisor of its input)",
    "num_batches_tracked": "a batch norm's step counter (its momentum's bookkeeping)",
    "weight": "a weight, reached only through a block's output (a product with the input) in a residual sum",
    "in_proj_weight": "a weight, reached only through a block's output (a product with the input) in a residual sum",
    "bias_hh_*": "a decomposed recurrence's hidden-to-hidden bias: it adds to the hidden state's own product",
}

#: Bias names the search cannot find, being no module's tensor, each with where it lives.
NOT_A_TENSOR = {"hardware_bias": "the IR core's and the chip's bias (the deployed copy, written by the IR's own writers)"}


@functools.lru_cache(maxsize=1)
def _parsed() -> tuple:
    return tuple((p.relative_to(SRC).as_posix(), ast.parse(p.read_text(encoding="utf-8")))
                 for p in sorted(SRC.rglob("*.py")))


@functools.lru_cache(maxsize=1)
def _source() -> Source:
    return Source(_parsed())


def _unlisted(src: Source) -> tuple:
    found = src.writers(CONTAINED)
    listed = set(GENERIC) | set(CONTAINED)
    unlisted = {q: found[q] for q in found if q not in listed and not q.startswith(ONE_WRITER + "::")}
    return unlisted, sorted(listed - set(found))


def _shared_generic_names(src: Source) -> dict:
    """The generic writers whose name more than one function of the tree has: a call of one on a receiver of unknown
    class would resolve to none, and its caller would pass unseen."""
    names = {q.split("::")[1].split(".")[-1] for q in GENERIC}
    return {n: sorted(src.by_name.get(n, ())) for n in names if len(src.by_name.get(n, ())) != 1}


def test_every_generic_writers_name_is_one_function_of_the_tree():
    shared = _shared_generic_names(_source())
    assert not shared, f"generic writers' names that other functions of the tree share (rename one): {shared}"


def test_every_bias_writer_is_the_one_writer_or_a_reviewed_writer_outside_the_class():
    unlisted, gone = _unlisted(_source())
    assert not unlisted, (
        f"{len(unlisted)} bias writer(s) neither the received shift's one writer nor reviewed outside the class (route "
        f"a received shift's correction through correct_for_received_shift or bake_received_shift, or review the "
        f"writer onto the list with its read): " + "; ".join(f"{q} {why[:2]}" for q, why in sorted(unlisted.items())))
    assert not gone, f"listed writers that write no bias any more (remove them): {gone}"


def test_the_lifts_scalar_pass_through_is_gone_and_only_the_one_writer_reads_shift_rows():
    src, hits = _source(), []
    rows = "transformations/perceptron/position_fold.py::shift_rows"
    for q, refs in src.refs.items():
        if q.startswith((ONE_WRITER + "::", rows.split("::")[0] + "::")):
            continue
        hits += [f"{q}:{line} shift_rows" for line, targets in refs if rows in targets]
    for rel, tree in _parsed():
        for node in ast.walk(tree):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else \
                node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
            if name in ("_scalar_shift_equivariant", "_SHIFT_EQUIVARIANT_ADAPTER_FNS"):
                hits.append(f"{rel}:{node.lineno} {name}")
    assert not hits, hits


def _torch_modules() -> Source:
    folder = Path(inspect.getsourcefile(torch.nn.modules)).parent
    return Source((f"torch/nn/modules/{p.name}", p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.py")))


def test_every_tensor_a_forward_adds_is_a_bias_name_or_a_reviewed_name():
    src, lib = _source(), _torch_modules()
    tensors = set(registered(src)) | set(registered(lib))
    found = {}
    for source in (src, lib):
        for name, sites in summands(source, tensors).items():
            found.setdefault(name, set()).update(sites)
    unlisted = {name: sorted(sites)[:3] for name, sites in found.items() if name not in BIAS and name not in NOT_A_BIAS}
    assert not unlisted, (
        f"tensors a forward adds that are neither a bias name nor reviewed (add a reader's bias to BIAS in "
        f"received_shift_writers, or review the name onto NOT_A_BIAS with its read): {unlisted}")
    gone = sorted((set(BIAS) - set(NOT_A_TENSOR) | set(NOT_A_BIAS)) - set(found))
    assert not gone, f"listed names no forward adds any more (remove them): {gone}"


_REVIEW = '''
import torch
import torch.nn.functional as F

from mimarsinan.mapping.support.bias_compensation import apply_additive_effective_bias_shift
from mimarsinan.mapping.support.received_shift import correct_for_received_shift
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


def control_pinned(perceptron, op):
    s = op._negative_shift
    w = PerceptronTransformer().get_effective_weight(perceptron)
    PerceptronTransformer().apply_effective_bias_transform(perceptron, lambda b: b - (w * s).sum(dim=-1))


def _reader_response(perceptron, value):
    w = PerceptronTransformer().get_effective_weight(perceptron)
    return (w * value).sum(dim=-1)


def _install_correction(perceptron, correction):
    PerceptronTransformer().apply_effective_bias_transform(perceptron, lambda b: b - correction)


def e1_split(model, op):
    consumers = model.get_mapper_repr().consumer_map()
    for node in consumers.get(id(op), []):
        _install_correction(node.perceptron, _reader_response(node.perceptron, op.encode_offset))


def e2_generic_writer(model, op):
    consumers = model.get_mapper_repr().consumer_map()
    for node in consumers.get(id(op), []):
        w = PerceptronTransformer().get_effective_weight(node.perceptron)
        apply_additive_effective_bias_shift(node.perceptron, -(w * op.encode_offset).sum(dim=-1), baked_flag="_e2")


def e3_augmented(module, sigma):
    with torch.no_grad():
        module.bias -= sigma * module.weight.sum(dim=1)


def e4_subscript(perceptron, value):
    w = perceptron.layer.weight
    with torch.no_grad():
        perceptron.layer.bias.data[:] = perceptron.layer.bias.data - (w * value).sum(dim=-1)


def e5_own_forward(node, field):
    k = node.perceptron.layer.weight.view(node.out_channels, node.in_channels, *node.kernel_size)
    resp = F.conv2d(field, k, padding=node.padding)
    PerceptronTransformer().apply_effective_bias_transform(node.perceptron, lambda b: b - resp.flatten(2).mean(-1)[0])


def e6_out_write(perceptron, value):
    with torch.no_grad():
        torch.sub(perceptron.layer.bias, (perceptron.layer.weight * value).sum(-1), out=perceptron.layer.bias)


def routed_through_the_one_writer(perceptron, received):
    correct_for_received_shift(perceptron, received)
'''

#: The check's eight paper sites, its own text (check_evidence/probe/k3/k3_candidates.py 53fedda175b7) word for word.
_CHECK = r'''
"""Check k3: eight new sites written on paper by the checker (parsed only, never imported or run).

k3_split .. k3_out are my own texts of the review's six shapes; k3_alias and k3_import_alias are two shapes of my own.
Each writes a reader's bias from a value its inputs receive, decided by itself."""
import torch
import torch.nn.functional as F

from mimarsinan.mapping.support.bias_compensation import apply_additive_effective_bias_shift
from mimarsinan.mapping.support.bias_compensation import apply_additive_effective_bias_shift as _shift_bias
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


def _field_response(reader, field):
    w = PerceptronTransformer().get_effective_weight(reader)
    return torch.einsum("oi,i->o", w, field)


def _store(reader, delta):
    PerceptronTransformer().apply_effective_bias_transform(reader, lambda b: b - delta)


def k3_split(repr_, src):
    for node in repr_.consumer_map().get(id(src), []):
        _store(node.perceptron, _field_response(node.perceptron, src.received))


def k3_generic(reader, field):
    w = PerceptronTransformer().get_effective_weight(reader)
    apply_additive_effective_bias_shift(reader, -(w @ field), baked_flag="_k3_generic")


def k3_augmented(linear, offset):
    with torch.no_grad():
        linear.bias += -offset * linear.weight.sum(1)


def k3_subscript(reader, field):
    with torch.no_grad():
        reader.layer.bias[0:] = reader.layer.bias[0:] - reader.layer.weight.flatten(1) @ field


def k3_conv(node, field):
    kern = node.perceptron.layer.weight
    resp = F.conv2d(field.unsqueeze(0), kern, padding=node.padding)
    PerceptronTransformer().apply_effective_bias_transform(node.perceptron, lambda b: b - resp.mean((0, 2, 3)))


def k3_out(linear, offset):
    torch.addmv(linear.bias, linear.weight, -offset, out=linear.bias)


def k3_alias(reader, field):
    held = reader.layer.bias
    with torch.no_grad():
        held.sub_(reader.layer.weight.flatten(1) @ field)


def k3_import_alias(reader, field):
    w = PerceptronTransformer().get_effective_weight(reader)
    _shift_bias(reader, -(w @ field), baked_flag="_k3_import_alias")
'''

_OWN = '''
import functools

import torch

from mimarsinan.mapping.support import bias_compensation as compensation
from mimarsinan.models.perceptron_mixer.perceptron import layer_bias_or_zeros
from mimarsinan.transformations.perceptron import position_fold
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

BIAS_NAME = "bias"


def own_chosen_by_a_conditional_expression(reader, field, shared):
    transformer = PerceptronTransformer()
    w = transformer.get_effective_weight(reader)
    (transformer.write_shared_bias if shared else transformer.apply_effective_bias_transform)(
        reader, lambda b: b - w @ field)


def _later(fn, *args):
    return fn(*args)


def own_writer_passed_as_an_argument(reader, field):
    w = PerceptronTransformer().get_effective_weight(reader)
    _later(functools.partial(compensation.apply_additive_effective_bias_shift, baked_flag="_own"), reader, -(w @ field))


def own_writer_bound_to_a_name(reader, field):
    write = PerceptronTransformer().apply_effective_bias_transform
    w = PerceptronTransformer().get_effective_weight(reader)
    write(reader, lambda b: b - w @ field)


def own_bias_by_a_constant_name_in_a_loop(module, field):
    for name in (BIAS_NAME, "in_proj_bias"):
        held = getattr(module, name, None)
        if held is not None:
            with torch.no_grad():
                held -= field.sum()


def own_bias_returned_by_a_function_of_the_tree(reader, field):
    with torch.no_grad():
        layer_bias_or_zeros(reader).sub_(reader.layer.weight @ field)


def own_position_bias_through_the_folds_accessor(reader, field):
    offsets = position_fold.bias_offsets(reader)
    if offsets is not None:
        offsets.add_(field.mean())


def own_bias_by_a_name_built_with_a_known_suffix(module, field, prefix):
    with torch.no_grad():
        getattr(module, f"{prefix}_bias").sub_(field.sum())


def _write_through(tensor, value):
    held = tensor
    held.data.sub_(value)


def own_bias_through_a_helper_that_binds_its_parameter(reader, field):
    _write_through(reader.layer.bias, reader.layer.weight @ field)


def own_bias_through_a_library_view(reader, field):
    with torch.no_grad():
        torch.as_tensor(reader.layer.bias).sub_(reader.layer.weight @ field)


class _SecondWriterName:
    def write_shared_bias(self, reader, transform):
        return None


def own_generic_writer_called_by_a_name_a_second_class_defines(transformer, reader, field):
    transformer.write_shared_bias(reader, lambda b: b - field.sum())
'''

_PINNED = {
    "newsite/review.py": {"control_pinned", "_install_correction", "e1_split", "e2_generic_writer", "e3_augmented",
                          "e4_subscript", "e5_own_forward", "e6_out_write"},
    "newsite/check.py": {"_store", "k3_split", "k3_generic", "k3_augmented", "k3_subscript", "k3_conv", "k3_out",
                         "k3_alias", "k3_import_alias"},
    "newsite/own.py": {"own_chosen_by_a_conditional_expression", "own_writer_passed_as_an_argument",
                       "own_writer_bound_to_a_name", "own_bias_by_a_constant_name_in_a_loop",
                       "own_bias_returned_by_a_function_of_the_tree", "own_position_bias_through_the_folds_accessor",
                       "own_bias_by_a_name_built_with_a_known_suffix",
                       "own_bias_through_a_helper_that_binds_its_parameter", "own_bias_through_a_library_view"},
}


def test_the_detector_pins_every_shape_written_on_paper():
    texts = {"newsite/review.py": _REVIEW, "newsite/check.py": _CHECK, "newsite/own.py": _OWN}
    src = Source(_parsed() + tuple((rel, ast.parse(text)) for rel, text in texts.items()))
    unlisted, _gone = _unlisted(src)
    for rel, want in _PINNED.items():
        found = {q.split("::")[1] for q in unlisted if q.startswith(rel + "::")}
        assert found == want, (rel, sorted(want - found), sorted(found - want))
    assert set(_shared_generic_names(src)) == {"write_shared_bias"}, _shared_generic_names(src)


_NEW_TENSOR = '''
import torch
import torch.nn as nn

CARRY = "carry_offset"


class PaperReader(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lin = nn.Linear(width, width, bias=False)
        self.register_buffer(CARRY, torch.zeros(width))

    def forward(self, x):
        return self.lin(x) + _offset(self)


def _offset(module):
    return getattr(module, CARRY) * 2.0
'''


def test_the_search_pins_a_new_tensor_a_forward_adds_through_a_function_that_returns_it():
    paper = Source([("newsite/reader.py", _NEW_TENSOR)])
    found = summands(paper, set(registered(paper)) | {"weight", "bias"})
    assert set(found) == {"carry_offset"}, found
    assert "carry_offset" not in BIAS and "carry_offset" not in NOT_A_BIAS
