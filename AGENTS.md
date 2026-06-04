Always activate the virtual environment (env) before running and testing code.

Run the deployment pipeline from the **project root** using **`run.py`**.

# Discipline 
- Always implement contained unit tests covering the entire design hierarchy BEFORE adding, removing or editing any code. The tests dictate software design and implementation. 
- Always run tests after changing the code. 
- Do not remove assertions or silence the checks to match potentially incorrect code. 
- Boy scout rule: While you are working on code, if you encounter duplicate, redundant or similar logic, directly or indirectly related to the task, create meaningful abstractions that enable reuse of the shared mechanisms. Write tests for these new abstractions. 

## Architecture Documentation

Before modifying any code, **read the `ARCHITECTURE.md` files** relevant to the
modules you are working on:

1. **Root** `ARCHITECTURE.md` — High-level overview of the entire framework,
   pipeline architecture, module dependency graph, and contributing guide.
2. **Per-directory** `ARCHITECTURE.md` — Each source directory under
   `src/mimarsinan/` has its own `ARCHITECTURE.md` describing that module's
   purpose, key components, internal/external dependencies, dependents, and
   exported API. Always read the `ARCHITECTURE.md` in the directory you are
   editing before making changes.

### When to update ARCHITECTURE.md

After editing code, update the relevant `ARCHITECTURE.md` file(s) whenever you:

- Add, remove, or rename a public class, function, or module.
- Change a module's dependencies (new imports from other mimarsinan submodules).
- Add or remove files from a directory.
- Change what is exported in `__init__.py`.

Keep updates concise — match the existing table/list format in each file.

### Package structure

Every directory under `src/mimarsinan/` that contains `.py` files **must** have
an `__init__.py`. When creating new subdirectories with Python files:

1. Add an `__init__.py` that exports the module's public API.
2. Add an `ARCHITECTURE.md` following the same format as sibling directories.
3. Keep `__init__.py` exports conservative — only re-export symbols that other
   modules actually import.

## Comment style

- Prefer self-documenting names and small extracted helpers over inline commentary.
- Do not use multiline `#` blocks unless they state a non-obvious invariant, idempotency rule, or cross-language contract that the code cannot express.
- Module docstrings: one line. Class/method docstrings: at most 1–3 lines of intent (not a restatement of the body).
- Do not strip license headers, `# type:` / `# noqa` lines, or user-facing text in `raise` / log messages.

