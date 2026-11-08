# decoding_101 — see .claude/CLAUDE.md for the working agreements.
# `make ci` must pass before any commit.

.PHONY: help setup sync models lint format check typecheck test test-cov \
        test-offline notebooks run lab clean reset ci dev

.DEFAULT_GOAL := help

UV := uv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# --- setup -----------------------------------------------------------------

setup: ## First-time setup
	@command -v $(UV) >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	$(UV) sync --all-extras
	$(UV) run python -m ipykernel install --user --name decoding-101
	@echo
	@echo "Ready. Run 'make test'."

sync: ## Sync deps
	$(UV) sync --all-extras

models: ## (Optional) pull a tiny Ollama model for real-text generation
	@command -v ollama >/dev/null 2>&1 || (curl -fsSL https://ollama.com/install.sh | sh)
	ollama pull qwen2.5:0.5b
	@echo
	@echo "Optional local model ready (notebooks also run on the from-scratch char model)."

# --- quality ---------------------------------------------------------------

lint: format check typecheck ## All linters

format: ## ruff format
	$(UV) run ruff format src/ tests/ app/

check: ## ruff check
	$(UV) run ruff check --fix src/ tests/ app/

typecheck: ## ty check
	$(UV) run ty check src/

test: ## pytest
	$(UV) run pytest

test-cov: ## pytest with coverage
	$(UV) run pytest --cov=src --cov-report=term-missing

test-offline: ## pytest with no Hub access (fixtures only)
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(UV) run pytest -m "not network"

# --- artefacts -------------------------------------------------------------

notebooks: ## Execute all notebooks in place (outputs are committed)
	@for nb in notebooks/0*.ipynb; do \
		echo "Executing $$nb ..."; \
		$(UV) run jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=300 "$$nb" || exit 1; \
	done
	@echo
	@echo "All notebooks executed."

run: ## Streamlit app
	$(UV) run streamlit run app/streamlit_app.py --server.port 8501

lab: ## JupyterLab
	$(UV) run jupyter lab --no-browser --port 8888

# --- housekeeping ----------------------------------------------------------

clean: ## Remove caches and coverage artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage .coverage.* .pytest_cache .ruff_cache .ty_cache .mypy_cache
	@echo "Cleaned."

reset: clean ## Full reset (also drops the virtualenv)
	rm -rf .venv

ci: sync lint test ## CI
dev: lint test ## Fast loop
