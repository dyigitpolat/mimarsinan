Always activate the virtual environment (env) before running and testing code.

Run the deployment pipeline from the **project root** using **`run.py`**.

# Commit messages

Never include AI-attribution lines in commit messages: no `Co-Authored-By:
Claude*` trailers, no `Claude-Session:` links, no "Generated with Claude Code"
lines. Plain, conventional commit messages only. This rule overrides any tool
default.

# Discipline
- Always implement contained unit tests covering the entire design hierarchy BEFORE adding, removing or editing any code. The tests dictate software design and implementation.
- Always run tests after changing the code.
- Do not remove assertions or silence the checks to match potentially incorrect code.
- Boy scout rule: While you are working on code, if you encounter duplicate, redundant or similar logic, directly or indirectly related to the task, create meaningful abstractions that enable reuse of the shared mechanisms. Write tests for these new abstractions.

## Testing & gates

- `python -m pytest tests` is the whole gate: parallel by default (`-n auto`),
  ≤2 minutes wall, always green. Unit tests are CPU-only and network-free by
  design (guards in `tests/conftest.py` and `tests/unit/conftest.py`); GPU
  coverage lives in the integration tiers.
- Ratchet tests under `tests/unit/architecture/` only ever tighten: lazy-import
  allowlist, broad-except counts, module size budgets, undefined-name baseline
  (empty), and ARCHITECTURE.md drift. Never grow an allowlist without a stated
  cycle/contract reason.
- `./scripts/typecheck.sh` (curated basedpyright, `pyrightconfig.json`) must
  report zero errors before committing. Inline ignores are a last resort: one
  per line, with a reason.
- End-to-end coverage lives in `test_configs/` (tiers 0-2). `generate.py` is
  the SSOT for the matrices — edit it and regenerate; never hand-edit the JSONs
  (a unit test enforces reproducibility). Run a tier with
  `python scripts/run_tier.py <tier>`.

## Error handling

Fail loud. The only sanctioned log-and-degrade seam is
`mimarsinan.common.best_effort.best_effort(...)` — use it exclusively for
telemetry/rendering side work whose failure must not kill a run. Never wrap
verification, mapping, or training logic in it.

## Where behavior lives (SSOTs)

- Config resolution: `pipelining/core/deployment_plan.py` (`DeploymentPlan`).
- Mode → proven recipe: `tuning/orchestration/conversion_policy.py`.
- Tuning-loop constants: `tuning/orchestration/tuning_policy.py`.
- Spiking-mode predicates: `chip_simulation/spiking_semantics.py` — never
  compare mode strings with literal ladders.
- Environment variables: `common/env.py` — never read `MIMARSINAN_*` directly.
- Config → running pipeline: `pipelining/session.py` (`PipelineSession`);
  `run.py`/`src/main.py` are thin frontends over it.

## Architecture Documentation

Read the root `ARCHITECTURE.md` (lean overview: flow, SSOTs, module map) and
the `ARCHITECTURE.md` of each **top-level module** you touch under
`src/mimarsinan/` before making changes. Docs exist ONLY at the root and at
top-level module roots — do not create per-subdirectory docs.

After editing code, update the touched module's `ARCHITECTURE.md` when you add,
remove, or rename a direct child file/subpackage, change cross-module
dependencies, or change `__init__.py` exports. The drift meta-test
(`tests/unit/architecture/test_architecture_docs.py`) enforces that every
direct child appears in the module's Key-files table and that every referenced
child exists.

### Package structure

Every directory under `src/mimarsinan/` that contains `.py` files **must** have
an `__init__.py`. Keep `__init__.py` exports conservative — only re-export
symbols that other modules actually import.

## Comment style

- Prefer self-documenting names and small extracted helpers over inline commentary.
- Comments are rare, single-sentence statements of non-obvious invariants,
  idempotency rules, numerical/hardware constants, or cross-language contracts
  that the code cannot express. Nothing else survives review.
- Module docstrings: one line. Class/method docstrings: at most 1–3 lines of intent (not a restatement of the body).
- Do not strip license headers, `# type:` / `# noqa` lines, or user-facing text in `raise` / log messages.
