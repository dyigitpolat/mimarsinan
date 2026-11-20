"""[U-AREA] The named unit constants of the area rule: the only numbers its modules hold (T-10, review m8).

Every published figure the rule prices with is data in ``data/`` (the registry and its evidence), read by id with
its quote, location and citation key; these names convert units, state an identity, or size a digest.
"""

#: Square micrometres per square millimetre.
UM2_PER_MM2 = 1e6
#: Nanometres per micrometre (a node name in nm over this is its feature size in um).
NM_PER_UM = 1e3
#: A percent figure over this is a fraction.
PERCENT = 100.0
#: The additive identity (no area, an outside part of 0).
NONE = 0.0
#: The multiplicative identity (a ratio at its anchor, one cell per weight, gamma of declared parts).
UNITY = 1.0
#: The base of a converter law's per-bit factor (a(b) = a_ref x BINARY ** (alpha (b - ref))).
BINARY = 2.0
#: The hexadecimal digits a block's digest keeps.
DIGEST_HEX_DIGITS = 16
#: The first element of a pair (an index, never a quantity).
FIRST = 0
#: The second element of a pair (an index, never a quantity).
SECOND = 1
#: The points a two-parameter law needs (a converter law fixed by two published bit points).
LAW_POINTS = 2
#: The relative tolerance of the registry check (T-9): a recomputed value against its stored one.
RECOMPUTE_TOLERANCE = 1e-9
#: The relative tolerance of a published value against its evidence entry (T-9).
ENTRY_TOLERANCE = 1e-12
#: The declared input domain's nesting depth (core/door.py): a container nested deeper is refused by name. A shipped
#: description nests its declared objects at most five deep. TOP is a door's argument itself, LEVEL one step deeper
#: (a nesting count, never a quantity).
NESTING = 32
TOP = 0
LEVEL = 1
#: An expression's declared bounds (evidence.parsed, the fourth delta round's MA-C5), refused by name beyond them before
#: it is parsed: its length in characters, and its depth (its bracket nesting plus its arithmetic operators, counted on
#: its text: a bound of the depth the restricted arithmetic recurses). The shipped expressions reach 341 and 18.
EXPRESSION_LENGTH = 2048
EXPRESSION_DEPTH = 64
