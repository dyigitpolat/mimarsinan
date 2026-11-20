"""[U-AREA fix round, MA-4] The optional blocks of a serialized dataclass (``deployment_record/optional_blocks.py``):
each block's field declares how it is read, so ``PlatformPhysics.to_dict`` and ``from_dict`` name no block and
``platform_physics/profile.py`` holds within the repository's 300-line module budget when composed with the host
pricing unit (the budget ratchet runs on the composed tree; never an allowlist entry).

Red on the build's tree: the module does not exist and profile.py names every block in its serialization. Green on
the fixed tree. The round trips and the key order are guards: a description serializes as it did, R5's blocks in R5's
order.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import textwrap
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import pytest

from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.registry import available_profiles

#: R5's optional blocks of a physics in R5's order, and the area rule's block.
R5_BLOCKS = ("external_components", "converter", "carry_buffer", "chip_components", "system_energy_rows",
             "backing_store")
HEAD = ("format_version", "name", "display_name", "description_file", "validity", "conversion_model", "constants")
TAIL = ("transfer_rows", "link_ends")


def _blocks():
    return importlib.import_module("mimarsinan.deployment_record.optional_blocks")


@dataclass(frozen=True)
class _Part:
    label: str

    def to_dict(self):
        return {"label": self.label}

    @classmethod
    def from_dict(cls, data):
        return cls(str(data["label"]))


def _toy():
    blocks = _blocks()

    @dataclass(frozen=True)
    class Toy:
        name: str
        part: Optional[_Part] = blocks.optional_block(_Part.from_dict)
        parts: Tuple[_Part, ...] = blocks.optional_block(_Part.from_dict, many=True)
        tags: Tuple[str, ...] = blocks.optional_block(str, many=True)
        named: Optional[str] = blocks.optional_block(lambda data, name: f"{data} of {name}", named=True)
    return blocks, Toy


class TestTheGenericBlocks:
    def test_a_block_is_written_only_when_present_in_field_order(self):
        blocks, Toy = _toy()
        assert blocks.blocks_to_dict(Toy("a")) == {}
        toy = Toy("a", part=_Part("p"), tags=("x", "y"))
        assert list(blocks.blocks_to_dict(toy)) == ["part", "tags"]
        assert blocks.blocks_to_dict(toy) == {"part": {"label": "p"}, "tags": ["x", "y"]}

    def test_every_block_is_read_by_its_fields_reader(self):
        blocks, Toy = _toy()
        kwargs = {"name": "a", "part": {"label": "p"}, "parts": [{"label": "q"}], "named": "n"}
        blocks.blocks_from_dict(Toy, kwargs, "the toy")
        assert Toy(**kwargs) == Toy("a", part=_Part("p"), parts=(_Part("q"),), tags=(), named="n of the toy")

    def test_an_absent_block_reads_as_absent(self):
        blocks, Toy = _toy()
        kwargs = {"name": "a"}
        blocks.blocks_from_dict(Toy, kwargs, "the toy")
        assert Toy(**kwargs) == Toy("a")


class TestThePhysicsBlocks:
    def test_the_physics_serialization_names_no_block(self):
        """No optional block is written as a key of ``to_dict`` or read into ``kwargs[...]`` by ``from_dict``: each is
        its field's (a use of a block's value beside the serialization, a host unit's judge, is not a serialization)."""
        marked = [f.name for f in fields(PlatformPhysics) if _blocks().OPTIONAL_BLOCK in f.metadata]
        assert set(R5_BLOCKS) | {"area_rule"} <= set(marked)
        written = {k.value for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(PlatformPhysics.to_dict))))
                   if isinstance(n, ast.Dict) for k in n.keys if isinstance(k, ast.Constant)}
        read = {t.slice.value for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(
            PlatformPhysics.from_dict)))) if isinstance(n, ast.Assign) for t in n.targets
            if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)}
        assert "constants" in written and "validity" in read  # the scan reads the two methods
        assert [b for b in marked if b in written | read] == []

    @pytest.mark.parametrize("name", available_profiles())
    def test_every_shipped_description_round_trips(self, name):
        """A guard: a description reads back equal and writes the same text."""
        physics = get_platform_physics(name)
        data = physics.to_dict()
        again = PlatformPhysics.from_dict(json.loads(json.dumps(data)))
        assert again == physics and json.dumps(again.to_dict()) == json.dumps(data)

    @pytest.mark.parametrize("name", available_profiles())
    def test_r5s_blocks_keep_r5s_key_order(self, name):
        """A guard: R5's head, R5's optional blocks in R5's order, the area rule's block, then R5's tail."""
        keys = list(get_platform_physics(name).to_dict())
        order = HEAD + R5_BLOCKS + ("area_rule",) + TAIL
        assert keys == [k for k in order if k in keys] and set(keys) <= set(order)
