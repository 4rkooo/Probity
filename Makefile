# The sandboxed dev environment cannot write paths containing "venv"; override if needed:
#   make UV_ENV=.venv test
UV_ENV ?= .uvenv
export UV_PROJECT_ENVIRONMENT := $(abspath $(UV_ENV))
UV ?= uv

.PHONY: setup lint format test contracts contracts-check fixtures dev workshop-dev

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
	@echo "Probity dev: API and worker. Press Ctrl-C to stop both."
	@port=$$($(UV) run python -c 'from probity.config import get_settings; print(get_settings().api_port)'); \
	echo "API health: http://127.0.0.1:$$port/v1/health"; \
	echo "Worker logs JSON lines on stdout. Stop both with Ctrl-C."; \
	trap 'trap - INT TERM EXIT; echo "Stopping API and worker"; kill 0' INT TERM EXIT; \
	PROBITY_HANDLER_FACTORY=probity.wiring:handler_factory \
		$(UV) run uvicorn probity.api.main:app --host 127.0.0.1 --port $$port & \
	PROBITY_HANDLER_FACTORY=probity.wiring:handler_factory \
		$(UV) run python -m probity.worker & \
	wait

# Builders Challenge: source /config/<team>.config into PROBITY_* then run API+worker.
workshop-dev:
	@echo "Probity workshop-dev: live Cosmos/VAST adapters from /config."
	@set -a; . ./scripts/workshop_env.sh; set +a; \
	port=$$($(UV) run python -c 'from probity.config import get_settings; print(get_settings().api_port)'); \
	echo "API health: http://127.0.0.1:$$port/v1/health"; \
	trap 'trap - INT TERM EXIT; echo "Stopping API and worker"; kill 0' INT TERM EXIT; \
	PROBITY_HANDLER_FACTORY=probity.wiring:handler_factory \
		$(UV) run uvicorn probity.api.main:app --host 127.0.0.1 --port $$port & \
	PROBITY_HANDLER_FACTORY=probity.wiring:handler_factory \
		$(UV) run python -m probity.worker & \
	wait
