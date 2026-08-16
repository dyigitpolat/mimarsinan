"""A pipeline step never CAPTURES a config value at construction time.

The incident (found by the first search over deployment options): steps are
constructed when the pipeline is ASSEMBLED, but ``ArchitectureSearchStep``
rewrites ``pipeline.config`` mid-run with the winner's searched options. A
value captured in ``__init__`` therefore freezes the PRE-SEARCH declaration
while every process-time reader sees the winner's — a split brain that
surfaced as ``QuantizationVerificationStep`` asserting weights against a
5-bit grid (``q_max`` captured at assembly) after the search had stamped
``weight_bits=6`` and the quantizer had used it.

This is the pipeline-step form of the banned constructor-shaped
``get(key, default)``: contract-owned state read at construction instead of
at use. The ratchet is generic — it forbids the SHAPE, not the one key —
and only ever tightens.
"""

import ast
import pathlib

STEPS_ROOT = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src" / "mimarsinan" / "pipelining" / "pipeline_steps"
)

#: The ONE sanctioned shape: a read that decides the step's declared
#: ``requires`` must happen at construction, because the DAG is assembled
#: then. Its hazard (a searched value would freeze the contract) is closed
#: generically — every key here is REFUSED as a search axis, asserted below.
ALLOWLIST = frozenset({
    ("TTFSCycleAdaptationStep", "ttfs_scale_aware_boundaries"),
})


def _config_reads_in_init(path: pathlib.Path):
    """(class, key) for every ``…config[...]`` / ``…config.get(...)`` in an ``__init__``."""
    tree = ast.parse(path.read_text())
    found = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        inits = [
            n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        ]
        for fn in inits:
            for node in ast.walk(fn):
                if isinstance(node, ast.Subscript):
                    target = ast.unparse(node.value)
                    if target.endswith("config"):
                        key = (
                            node.slice.value
                            if isinstance(node.slice, ast.Constant) else "?"
                        )
                        found.append((cls.name, str(key)))
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and ast.unparse(node.func.value).endswith("config")
                ):
                    key = (
                        node.args[0].value
                        if node.args and isinstance(node.args[0], ast.Constant)
                        else "?"
                    )
                    found.append((cls.name, str(key)))
    return found


def test_no_step_captures_config_at_construction():
    offenders = []
    for path in sorted(STEPS_ROOT.rglob("*.py")):
        for cls, key in _config_reads_in_init(path):
            rel = str(path).split("pipeline_steps/")[-1]
            if (cls, key) in ALLOWLIST:
                continue
            offenders.append(f"{rel}: {cls}.__init__ reads config[{key!r}]")
    assert not offenders, (
        "a pipeline step captured a config value at construction; the "
        "architecture search rewrites config mid-run, so the captured value "
        "is the PRE-SEARCH declaration while every other reader sees the "
        "winner's. Read it in process() instead:\n  " + "\n  ".join(offenders)
    )


def test_every_contract_shaping_key_is_unsearchable():
    """The allowlist's hazard, closed generically: a key a step reads at
    ASSEMBLY to shape its contract must never become a decision variable, or
    the DAG would freeze at the pre-search value while the winner ran."""
    from mimarsinan.search.option_axes import REFUSED_OPTION_AXES

    for _cls, key in ALLOWLIST:
        assert key in REFUSED_OPTION_AXES, (
            f"{key!r} shapes a step's construction-time contract but is still "
            f"declarable as a search axis; add it to REFUSED_OPTION_AXES"
        )
