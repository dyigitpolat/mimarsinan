"""[U-POST2] The toolchain's one model snapshot: the state dictionary and what each perceptron records outside it.

Every restore of a model inside a tuning window (a training leg's keep-best, the learning-rate finder's trials, the
checkpoint guard of the cycle, the endpoint and the run, the fast ladder's rung keep-best, a fold step's rollback, the
deployed-risk stage's keep-best and resume) restores one of these snapshots. A state dictionary holds the parameters and
the persistent buffers; a perceptron also records its history outside it (a bake's flag and its input shift, an
install's marker, a per-cycle charge bound, its membrane-lattice kind, a declared width or drive) and holds its gauge
fold as a non-persistent buffer. A restore that returns the parameters without those records leaves a projection, an
arming, a fold or a gate reading the window's record of parameters the model no longer has.

The rule for the class (state outside the state dictionary that a reader reads): what describes the persisted
parameters is derived from them at its read and never cached (the membrane lattice from the kind and the restored
scale at every arming; the wire gauge, a function of the thetas, at every walk of the installed twin, as at every
seam and every twin's precondition), and what records their history travels with them here. Records are keyed by the
perceptron's module path and hold values only (tensors cloned), so a snapshot also pickles. No key of the state
dictionary changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

__all__ = ["ModelState", "perceptron_records", "restore_model_state", "snapshot_model_state"]

#: torch's own module fields (the mode, the hooks, the parameter, buffer and submodule tables): never a record.
_MODULE_FIELDS = frozenset(vars(nn.Module()))


def _value(value):
    return value.detach().clone() if torch.is_tensor(value) else value


class ModelState(dict):
    """A state dictionary (parameter and persistent-buffer values) with ``records``: ``{path: (attributes,
    non-persistent buffers)}`` of every perceptron of the model it was taken from."""

    def __init__(self, state=(), records=None):
        super().__init__(state)
        self.records = dict(records or {})


def perceptron_records(model) -> dict:
    """``{path: (attributes, buffers)}``: every perceptron's instance attributes outside torch's module fields and its
    non-persistent buffers, as values."""
    records = {}
    for path, module in model.named_modules():
        if not isinstance(module, Perceptron):
            continue
        attributes = {k: _value(v) for k, v in vars(module).items() if k not in _MODULE_FIELDS}
        buffers = {k: _value(module._buffers.get(k)) for k in module._non_persistent_buffers_set}
        records[path] = (attributes, buffers)
    return records


def snapshot_model_state(model, state=None) -> ModelState:
    """The snapshot: ``state`` (default: a copy of the state dictionary) with the perceptrons' records."""
    if state is None:
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return ModelState(state, perceptron_records(model))


def _placed(value, current, module):
    """A record's tensor on the device its holder lives on (a snapshot may sit on the host)."""
    if not torch.is_tensor(value):
        return value
    if torch.is_tensor(current):
        return value.detach().clone().to(current.device)
    anchor = next(iter(module.parameters()), None)
    return value.detach().clone() if anchor is None else value.detach().clone().to(anchor.device)


def _restore_records(module, attributes: dict, buffers: dict) -> None:
    for k in [k for k in vars(module) if k not in _MODULE_FIELDS and k not in attributes]:
        delattr(module, k)  # a record the window added
    for k, v in attributes.items():
        setattr(module, k, _placed(v, vars(module).get(k), module))
    for k in [k for k in module._non_persistent_buffers_set if k not in buffers]:
        module._buffers.pop(k, None)
        module._non_persistent_buffers_set.discard(k)
    for k, v in buffers.items():
        value = _placed(v, module._buffers.get(k), module)
        if k in module._buffers:
            module._buffers[k] = value
        else:
            module.register_buffer(k, value, persistent=False)


def restore_model_state(model, state, *, strict: bool = True) -> None:
    """Restore a snapshot: the state dictionary, then every perceptron's records (a plain state dictionary restores
    the state dictionary alone)."""
    model.load_state_dict(state, strict=strict)
    records = getattr(state, "records", None) or {}
    for path, module in model.named_modules():
        saved = records.get(path)
        if saved is not None and isinstance(module, Perceptron):
            _restore_records(module, *saved)
