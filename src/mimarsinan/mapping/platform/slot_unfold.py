"""How a hop's upstream cells fill its deployed cores' axon slot tables."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class SerialSlotUnfold(Protocol):
    """The mapper's unfold, in the currency the NF event-serial twin folds.

    A hop's deployed cores each see ONE slot table. ``unfold_events`` says which
    upstream cells fill it, ``group_major`` puts the hop's own fused
    pre-activation in the same (core, neuron) layout so the decomposition can be
    checked against it, and ``restore`` puts the folded counts back in the NF's
    feature layout. The three are one statement of the mapping's unfold — the
    twin never derives it. Shape refusals belong to the twin (the only place
    that knows the deployed weight), so these are pure transforms.
    """

    n_cores: int
    n_slots: int
    source_size: int

    def unfold_events(self, events: torch.Tensor) -> torch.Tensor:
        """``(B, cells)`` multiplicities -> ``(B, *cores, slots)``."""
        ...

    def group_major(self, activation: torch.Tensor) -> torch.Tensor:
        """``(B, *hop_output)`` -> ``(B, *cores, neurons)``."""
        ...

    def restore(self, grouped: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """``(B, *cores, neurons)`` -> ``like``'s shape."""
        ...


class WholeInputSlotUnfold:
    """ONE core per hop, consuming the whole input in feature order.

    The identity unfold: the base ``Mapper``'s answer, and the only one a
    fully-connected hop needs. It keeps no core axis at all, so the fold it
    feeds is byte-identically the one a whole-input hop has always run.
    """

    n_cores = 1

    def __init__(self, n_slots: int) -> None:
        self.n_slots = int(n_slots)
        self.source_size = int(n_slots)

    def unfold_events(self, events: torch.Tensor) -> torch.Tensor:
        return events.reshape(events.shape[0], -1)

    def group_major(self, activation: torch.Tensor) -> torch.Tensor:
        return activation.reshape(activation.shape[0], -1)

    def restore(self, grouped: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return grouped.reshape(like.shape)


class GatheredSlotUnfold:
    """MANY cores over one shared weight bank, each fed a gathered slot table.

    ``slot_source_index`` is the mapper's own ``(cores, slots)`` table of flat
    upstream cell indices, with ``-1`` marking a slot the mapping wired to the
    OFF source (a padded receptive-field tap): the gather reads it from an
    appended zero cell, which is exactly the charge an OFF axon delivers.
    """

    def __init__(
        self,
        *,
        slot_source_index: np.ndarray,
        source_size: int,
        output_grid: Sequence[int],
    ) -> None:
        index = np.asarray(slot_source_index, dtype=np.int64)
        if index.ndim != 2:
            raise ValueError(
                f"GatheredSlotUnfold: the slot table is (cores, slots); got "
                f"shape {index.shape}"
            )
        self._index = torch.as_tensor(index, dtype=torch.long)
        self._device_index: torch.Tensor | None = None
        self.n_cores = int(index.shape[0])
        self.n_slots = int(index.shape[1])
        self.source_size = int(source_size)
        self._output_grid = tuple(int(d) for d in output_grid)
        cells = 1
        for d in self._output_grid:
            cells *= d
        if cells != self.n_cores:
            raise ValueError(
                f"GatheredSlotUnfold: {self.n_cores} cores cannot lay out an "
                f"output grid of {self._output_grid}"
            )

    def _index_on(self, device) -> torch.Tensor:
        """The table lives where the events do; every cycle of every hop reads it."""
        if self._device_index is None or self._device_index.device != device:
            self._device_index = self._index.to(device)
        return self._device_index

    def unfold_events(self, events: torch.Tensor) -> torch.Tensor:
        flat = events.reshape(events.shape[0], -1)
        padded = torch.cat([flat, flat.new_zeros(flat.shape[0], 1)], dim=1)
        return padded[:, self._index_on(flat.device)]

    def group_major(self, activation: torch.Tensor) -> torch.Tensor:
        # Channels-first: the convolution mappers declare
        # ``output_channel_axis = 1``, so axis 1 IS the core's neurons and the
        # trailing axes are the positions its cores are laid out over.
        batch = activation.shape[0]
        neurons = int(activation.shape[1])
        return activation.reshape(batch, neurons, self.n_cores).transpose(1, 2)

    def restore(self, grouped: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return grouped.transpose(1, 2).reshape(like.shape)


__all__ = ["GatheredSlotUnfold", "SerialSlotUnfold", "WholeInputSlotUnfold"]
