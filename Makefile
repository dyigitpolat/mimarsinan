# One install entry point. Every third-party seam has exactly three places to
# look: the manifest (pyproject.toml + uv.lock), the patch directory
# (third_party/patches/), and the contract test
# (tests/unit/architecture/test_third_party_contract.py).
#
#   make install                 base + dev + loihi extras
#   make install SANAFE=1        ... and the GPL-3.0 SANA-FE extra + its plugins
#   make check-deps              the third-party contract test alone
#   make check-pins              submodule declarations and remote reachability
#
# SANA-FE is opt-in because it is GPL-3.0; mimarsinan itself stays MIT.

.PHONY: install check-deps check-pins plugins patches help

SANAFE_EXTRA := $(if $(SANAFE),--extra sanafe)
COMPILAGENT_DIR := $(abspath $(CURDIR)/../compilagent)

help:
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

install: ## venv + declared dependencies + SANA-FE plugins + third-party patches
	@test -d "$(COMPILAGENT_DIR)" || { \
	  echo "mimarsinan expects its sibling compilagent checkout at $(COMPILAGENT_DIR)."; \
	  echo "The outer research_stuff repository provides it; standalone clones need:"; \
	  echo "  git clone https://github.com/dyigitpolat/compilagent.git $(COMPILAGENT_DIR)"; \
	  exit 1; }
	uv sync --extra dev --extra loihi $(SANAFE_EXTRA)
	$(if $(SANAFE),uv run python scripts/build_sanafe_plugins.py)
	uv run python scripts/apply_patches.py

plugins: ## rebuild the mimarsinan SANA-FE plugins (headers fetched by CMake)
	uv run python scripts/build_sanafe_plugins.py

patches: ## (re-)apply third_party/patches/<dist>/*.patch to the installed dists
	uv run python scripts/apply_patches.py

check-deps: ## the third-party contract test alone
	uv run pytest tests/unit/architecture/test_third_party_contract.py -p no:cacheprovider -n 0

check-pins: ## every gitlink is declared and reachable on its remote (network)
	uv run python scripts/check_submodule_pins.py
