.DEFAULT_GOAL := help
.PHONY: help install fmt lint types boundaries test test-fast leakage e2e golden check clean demo fixtures ingest features recommend api web web-install

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Sync the uv workspace and install pre-commit hooks
	@# On Apple Silicon an x86_64 uv (Intel Homebrew) selects an x86_64 Python, which
	@# then runs under Rosetta. Polars' default wheel needs AVX2 and refuses to load
	@# there, so build the venv on a native arm64 interpreter instead.
	@if [ "$$(uname -m)" = "arm64" ]; then \
		if [ ! -x "$$HOME/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12" ]; then \
			echo "Installing native arm64 Python 3.12..."; \
			uv python install cpython-3.12-macos-aarch64-none; \
		fi; \
		uv venv --python "$$HOME/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12"; \
	fi
	uv sync --all-packages
	uv run pre-commit install
	@uv run python -c "import platform,sys; \
		m=platform.machine(); \
		sys.exit(0) if (m=='arm64' or platform.system()!='Darwin') else \
		sys.exit('ERROR: venv is %s under Rosetta; run make install again' % m)"
	@echo "Environment ready."

fmt: ## Format code
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint without fixing
	uv run ruff format --check .
	uv run ruff check .

types: ## Type-check
	uv run mypy .

boundaries: ## Enforce the package dependency direction
	uv run lint-imports --config .importlinter

test: ## Full test suite
	uv run pytest

test-fast: ## Test suite excluding network and e2e
	uv run pytest -m "not network and not e2e"

leakage: ## Point-in-time leakage tests only
	uv run pytest -m leakage -v

e2e: ## End-to-end slice tests only
	uv run pytest -m e2e -v

golden: ## Committed-fixture tests only
	uv run pytest -m golden -v

banned-strings: ## Fail if a corrected claim reappears in the docs
	@# `231 distinct` was the model-ready feature count in five documents and one
	@# docstring; the schema has returned 224 for some time. tests/docs/
	@# test_documented_counts.py checks the Markdown against the live schema — this
	@# line is here because that test does not read Python docstrings, which is
	@# where the stale figure was last found.
	@! grep -rniE '3000\+|thousands of candidate features|raw_understat_shots|231 distinct' \
		--include='*.md' --include='*.py' . \
		|| (echo "FAIL: banned string found above" && exit 1)
	@echo "OK: no banned strings"

check: lint types boundaries banned-strings test ## Everything CI runs

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------
# The offline demo. `make install && make demo` is the whole first run: it
# needs no network, no API key and no ingest, because `data/fixtures` is
# committed. See tools/build_demo_fixture.py for what those fixtures are.
# ---------------------------------------------------------------------------

demo: ## Run the whole pipeline offline on committed fixtures
	XG_ALONSO_OFFLINE=1 uv run xg demo

fixtures: ## Regenerate data/fixtures from a populated local .data
	uv run python tools/build_demo_fixture.py
	@uv run python tools/build_demo_fixture.py --check

# ---------------------------------------------------------------------------
# Slice-1 workflow
# ---------------------------------------------------------------------------

ingest: ## Fetch official FPL data into immutable bronze snapshots
	uv run xg ingest

features: ## Build point-in-time features
	uv run xg build-features

recommend: ## Recommend a transfer (usage: make recommend ENTRY=1234567)
	uv run xg recommend $(ENTRY)

# ---------------------------------------------------------------------------
# Local interfaces. Run `make api` and `make web` in two terminals; the web app
# proxies /api/* to port 8000, so the API must be up first.
# ---------------------------------------------------------------------------

api: ## Serve the decision API on 127.0.0.1:8000
	uv run uvicorn xg_alonso.api.main:app --host 127.0.0.1 --port 8000 --reload

web: ## Serve the web front end on 127.0.0.1:3000 (needs `make api`)
	cd apps/web && npm run dev

web-install: ## Install web dependencies
	cd apps/web && npm install
