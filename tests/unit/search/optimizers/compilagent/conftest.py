"""These tests require the external ``compilagent`` package (ICCD project)."""

import importlib.util

if importlib.util.find_spec("compilagent") is None:
    collect_ignore_glob = ["test_*.py"]
