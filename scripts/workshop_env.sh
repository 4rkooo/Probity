#!/usr/bin/env bash
# Map Builders Challenge /config/<team>.config into PROBITY_* for live sponsor adapters.
# Usage:  set -a && source scripts/workshop_env.sh && set +a
# Never prints secret values.

set -euo pipefail

mapfile -t TEAM_CONFIGS < <(find /config -maxdepth 1 -type f -name '*.config' | sort)
if (( ${#TEAM_CONFIGS[@]} != 1 )); then
  echo "workshop_env: expected exactly one /config/*.config" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
set -a && source "${TEAM_CONFIGS[0]}" && set +a

# Optional overrides: export COSMOS3_REASON_URL / COSMOS_EMBED1_URL / YOLO_URL before sourcing.
: "${GPU_HOST:=166.19.38.112}"
: "${COSMOS3_REASON_URL:=http://${GPU_HOST}:8001}"
: "${COSMOS_EMBED1_URL:=http://${GPU_HOST}:8003}"
: "${YOLO_URL:=http://${GPU_HOST}:8002}"

export PROBITY_MODE="${PROBITY_MODE:-AUTO}"
export PROBITY_HANDLER_FACTORY="${PROBITY_HANDLER_FACTORY:-probity.wiring:handler_factory}"
export PROBITY_SYNTHETIC_DESCRIPTIONS="${PROBITY_SYNTHETIC_DESCRIPTIONS:-false}"

export PROBITY_COSMOS_ENABLED=true
export PROBITY_COSMOS_ENDPOINT="${COSMOS3_REASON_URL}"
export PROBITY_COSMOS_EMBED_ENDPOINT="${COSMOS_EMBED1_URL}"
export PROBITY_COSMOS_TOKEN="${GPU_BEARER_TOKEN:-}"
# Omit empty model ids so AdapterHealth accepts model_id=None until discovery.
if [[ -n "${COSMOS3_REASON_MODEL:-}" ]]; then
  export PROBITY_COSMOS_MODEL_ID="${COSMOS3_REASON_MODEL}"
else
  unset PROBITY_COSMOS_MODEL_ID || true
fi
if [[ -n "${COSMOS_EMBED1_MODEL:-}" ]]; then
  export PROBITY_COSMOS_EMBED_MODEL_ID="${COSMOS_EMBED1_MODEL}"
else
  unset PROBITY_COSMOS_EMBED_MODEL_ID || true
fi

export PROBITY_VAST_ENABLED=true
export PROBITY_VAST_ENDPOINT="${INGRESS_URL:-}"
# Optional: set PROBITY_VAST_TOKEN to a retrieval JWT; health works without it on /health.

export PROBITY_YOLO_ENABLED=true
export PROBITY_YOLO_ENDPOINT="${YOLO_URL}"

# W&B query planning is opt-in: set PROBITY_WANDB_ENABLED=true explicitly when the
# serverless inference endpoint is confirmed. Auto-enabling invents bad time windows.
export PROBITY_WANDB_ENABLED="${PROBITY_WANDB_ENABLED:-false}"

export PROBITY_API_BASE="${PROBITY_API_BASE:-http://127.0.0.1:8000}"

echo "workshop_env: cosmos=${PROBITY_COSMOS_ENDPOINT} embed=${PROBITY_COSMOS_EMBED_ENDPOINT} vast=${PROBITY_VAST_ENDPOINT} yolo=${PROBITY_YOLO_ENDPOINT}" >&2
