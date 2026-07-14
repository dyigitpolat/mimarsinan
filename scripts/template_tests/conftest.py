"""Make ``mimarsinan`` importable when this suite is run standalone
(``pytest scripts/template_tests``), independent of the main tests/ conftest."""

import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_repo_root, "src"))
