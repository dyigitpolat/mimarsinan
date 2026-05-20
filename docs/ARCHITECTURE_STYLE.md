# Architecture documentation style

Use this guide when editing any `ARCHITECTURE.md` under mimarsinan.

## Rules

1. **Present tense, current state** — describe what the code does now.
2. **No sprint metadata** — do not use round numbers (`R4-3`), “post round N”, commit SHAs, or “Done/Partial” refactor tables.
3. **One home per topic** — the root doc summarizes and links; per-package files own module tables.
4. **Invariants over history** — prefer one sentence of required behavior over postmortem narratives (MNIST regressions, “we removed X”). Historical notes belong in `docs/ARCHITECTURE_NOTES.md` if needed.
5. **Verify claims** — before documenting wiring, check the source file.

## Consistency check

From the repo root:

```bash
rg -i 'round [0-9]|R4-|post round|f42ff58|still inline|deferred items|Input Activation Analysis' \
  mimarsinan/ARCHITECTURE.md mimarsinan/src/mimarsinan/**/ARCHITECTURE.md
```

Expect zero matches after a cleanup pass.
