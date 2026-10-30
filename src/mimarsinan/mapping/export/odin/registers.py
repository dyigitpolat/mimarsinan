"""The stock-ODIN global configuration registers, SYN_SIGN included."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from mimarsinan.mapping.export.odin.layout import OdinLayoutError

# CROSS-LANGUAGE CONTRACT — the address map is transcribed from
# ChFrenkel/ODIN @ 1781931, src/spi_slave.v:153-210 (one `spi_addr[15:0] == N`
# comparison per register), restated in doc/README.md Sec.4. Config registers have
# NO SPI readback and NO reset value: the runtime reprograms all of them after
# every power cycle, and the P5 testbench shadow-asserts them. SPI_SYN_SIGN is one
# 256-bit vector spread over 16 consecutive 16-bit registers
# (src/spi_slave.v:163-170), one bit per PRE-synaptic row, 1 = inhibitory.


@dataclass(frozen=True)
class ConfigRegister:
    """One configuration register: its base address, width, and word count."""

    name: str
    address: int
    width: int
    count: int = 1


@dataclass(frozen=True)
class RegisterWrite:
    """One 20-bit SPI configuration write."""

    address: int
    value: int


SYN_SIGN_BITS = 256
SYN_SIGN_BASE_ADDR = 2
SYN_SIGN_REGISTER_BITS = 16

CONFIG_REGISTERS: Tuple[ConfigRegister, ...] = (
    ConfigRegister("SPI_GATE_ACTIVITY", 0, 1),
    ConfigRegister("SPI_OPEN_LOOP", 1, 1),
    ConfigRegister("SPI_SYN_SIGN", SYN_SIGN_BASE_ADDR, SYN_SIGN_REGISTER_BITS,
                   count=SYN_SIGN_BITS // SYN_SIGN_REGISTER_BITS),
    ConfigRegister("SPI_BURST_TIMEREF", 18, 20),
    ConfigRegister("SPI_AER_SRC_CTRL_nNEUR", 19, 1),
    ConfigRegister("SPI_OUT_AER_MONITOR_EN", 20, 1),
    ConfigRegister("SPI_MONITOR_NEUR_ADDR", 21, 8),
    ConfigRegister("SPI_MONITOR_SYN_ADDR", 22, 8),
    ConfigRegister("SPI_UPDATE_UNMAPPED_SYN", 23, 1),
    ConfigRegister("SPI_PROPAGATE_UNMAPPED_SYN", 24, 1),
    ConfigRegister("SPI_SDSP_ON_SYN_STIM", 25, 1),
)

_REGISTERS_BY_NAME: Dict[str, ConfigRegister] = {r.name: r for r in CONFIG_REGISTERS}

# The inference programming (plan F16). SPI_OPEN_LOOP keeps locally generated
# spikes out of the scheduler so the HOST routes every event and the boundary
# transform stays host-side; the weight freeze is doubly guaranteed by writing
# every mapping bit 0 while SPI_UPDATE_UNMAPPED_SYN is 0 (SDSP can touch nothing)
# and SPI_PROPAGATE_UNMAPPED_SYN is 1 (the frozen magnitudes still reach the soma,
# src/neuron_core.v:89). SPI_GATE_ACTIVITY is NOT here: it is the caller's staging
# decision, asserted while the memories are programmed and cleared.
INFERENCE_REGISTER_VALUES: Dict[str, int] = {
    "SPI_OPEN_LOOP": 1,
    "SPI_BURST_TIMEREF": 0,
    "SPI_AER_SRC_CTRL_nNEUR": 0,
    "SPI_OUT_AER_MONITOR_EN": 0,
    "SPI_MONITOR_NEUR_ADDR": 0,
    "SPI_MONITOR_SYN_ADDR": 0,
    "SPI_UPDATE_UNMAPPED_SYN": 0,
    "SPI_PROPAGATE_UNMAPPED_SYN": 1,
    "SPI_SDSP_ON_SYN_STIM": 0,
}


def config_register(name: str) -> ConfigRegister:
    """The named register, refusing an unknown name rather than returning None."""
    register = _REGISTERS_BY_NAME.get(name)
    if register is None:
        raise OdinLayoutError(
            f"unknown ODIN configuration register {name!r}; known registers: "
            f"{', '.join(sorted(_REGISTERS_BY_NAME))}")
    return register


def pack_syn_sign(signs: Sequence[bool]) -> int:
    """The 256-bit SYN_SIGN vector: bit ``r`` set = physical row ``r`` is inhibitory."""
    if len(signs) != SYN_SIGN_BITS:
        raise OdinLayoutError(
            f"SYN_SIGN carries {SYN_SIGN_BITS} row signs, got {len(signs)}")
    vector = 0
    for row, inhibitory in enumerate(signs):
        if inhibitory:
            vector |= 1 << row
    return vector


def unpack_syn_sign(vector: int) -> Tuple[bool, ...]:
    """The per-row inhibitory flags of a packed SYN_SIGN vector."""
    _check_syn_sign(vector)
    return tuple(bool((vector >> row) & 1) for row in range(SYN_SIGN_BITS))


def syn_sign_register_writes(vector: int) -> Tuple[RegisterWrite, ...]:
    """The 16 consecutive config writes that install a SYN_SIGN vector."""
    _check_syn_sign(vector)
    register = config_register("SPI_SYN_SIGN")
    return tuple(
        RegisterWrite(
            address=register.address + index,
            value=(vector >> (SYN_SIGN_REGISTER_BITS * index))
            & ((1 << SYN_SIGN_REGISTER_BITS) - 1),
        )
        for index in range(register.count)
    )


def inference_register_writes(
    *, syn_sign: int, gate_activity: int
) -> Tuple[RegisterWrite, ...]:
    """Every configuration write one core needs, SYN_SIGN expanded to its 16 words."""
    writes = [_checked_write(config_register("SPI_GATE_ACTIVITY"), gate_activity)]
    writes.extend(syn_sign_register_writes(syn_sign))
    for name, value in INFERENCE_REGISTER_VALUES.items():
        writes.append(_checked_write(config_register(name), value))
    return tuple(sorted(writes, key=lambda write: write.address))


def _check_syn_sign(vector: int) -> None:
    if vector < 0 or vector >= (1 << SYN_SIGN_BITS):
        raise OdinLayoutError(f"{vector} is not a {SYN_SIGN_BITS}-bit SYN_SIGN vector")


def _checked_write(register: ConfigRegister, value: int) -> RegisterWrite:
    if value < 0 or value >= (1 << register.width):
        raise OdinLayoutError(
            f"{register.name} is {register.width} bits wide; {value} does not fit")
    return RegisterWrite(address=register.address, value=value)
