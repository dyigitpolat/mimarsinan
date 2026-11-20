"""[U-LIF2 W1] The vocabulary of a synchronized LIF hop's firing discipline and the registry entry's data: the declared
set and the default. Constants only (no import), so every capability row and the entry read one datum.

``fire_while_integrating``: a neuron integrates and may fire on every cycle of its window (the paper's windowed LIF).
``integrate_then_emit``: it integrates the whole window with firing switched off, then emits the window's count on T
input-free cycles (the paper's synchronized LIF, calculus 16.3)."""

from __future__ import annotations

from typing import Tuple

WINDOW_FIRING_KEY = "lif_window_firing"
FIRE_WHILE_INTEGRATING = "fire_while_integrating"
INTEGRATE_THEN_EMIT = "integrate_then_emit"
#: The axis's values, the vocabulary the joint legal table judges.
WINDOW_FIRINGS: Tuple[str, ...] = (FIRE_WHILE_INTEGRATING, INTEGRATE_THEN_EMIT)

#: The entry's data: the values the toolchain offers and the value a silent document holds (the owner's decision D-1,
#: option A: both firing forms are declared values, integrate then emit is the default). The toolchain's own executors
#: run both; every external backend fires while integrating only (each capability row's column).
DECLARED_WINDOW_FIRINGS: Tuple[str, ...] = (FIRE_WHILE_INTEGRATING, INTEGRATE_THEN_EMIT)
DEFAULT_WINDOW_FIRING: str = INTEGRATE_THEN_EMIT
