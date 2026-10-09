# The sandboxed dev environment cannot write paths containing "venv"; override if needed:
#   make UV_ENV=.venv test
UV_ENV ?= .uvenv
export UV_PROJECT_ENVIRONMENT := $(abspath $(UV_ENV))
UV ?= uv

.PHONY: setup lint format test contracts contracts-check fixtures dev

setup:
	$(UV) sync --frozen

lint:
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts

format:
	$(UV) run ruff check --fix src tests scripts
	$(UV) run ruff format src tests scripts

test:
	$(UV) run pytest

contracts:
	$(UV) run python scripts/export_contracts.py

contracts-check:
	$(UV) run python scripts/export_contracts.py --check

fixtures:
	$(UV) run python scripts/generate_contract_fixtures.py

dev:
	@echo "make dev is wired after integration (API + worker)."; exit 1
