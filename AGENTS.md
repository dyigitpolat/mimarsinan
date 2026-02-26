Always activate the virtual environment (env) before running and testing code.

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
