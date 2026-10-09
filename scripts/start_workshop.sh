#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data /tmp

pkill -f 'uvicorn probity.api.main:app' 2>/dev/null || true
pkill -f 'python -m probity.worker' 2>/dev/null || true
sleep 1

set -a
# shellcheck disable=SC1091
source ./scripts/workshop_env.sh
set +a

common=(
  PROBITY_MODE="$PROBITY_MODE"
  PROBITY_HANDLER_FACTORY=probity.wiring:handler_factory
  PROBITY_SYNTHETIC_DESCRIPTIONS=false
  PROBITY_COSMOS_ENABLED="$PROBITY_COSMOS_ENABLED"
  PROBITY_COSMOS_ENDPOINT="$PROBITY_COSMOS_ENDPOINT"
  PROBITY_COSMOS_EMBED_ENDPOINT="$PROBITY_COSMOS_EMBED_ENDPOINT"
  PROBITY_COSMOS_TOKEN="$PROBITY_COSMOS_TOKEN"
  PROBITY_COSMOS_MODEL_ID="${PROBITY_COSMOS_MODEL_ID:-}"
  PROBITY_COSMOS_EMBED_MODEL_ID="${PROBITY_COSMOS_EMBED_MODEL_ID:-}"
  PROBITY_VAST_ENABLED="$PROBITY_VAST_ENABLED"
  PROBITY_VAST_ENDPOINT="$PROBITY_VAST_ENDPOINT"
  PROBITY_YOLO_ENABLED="$PROBITY_YOLO_ENABLED"
  PROBITY_YOLO_ENDPOINT="$PROBITY_YOLO_ENDPOINT"
  PROBITY_WANDB_ENABLED="${PROBITY_WANDB_ENABLED:-false}"
)

env "${common[@]}" .uvenv/bin/uvicorn probity.api.main:app --host 127.0.0.1 --port 8000 \
  >/tmp/probity-api.log 2>&1 &
echo "api_pid=$!"

env "${common[@]}" .uvenv/bin/python -m probity.worker \
  >/tmp/probity-worker.log 2>&1 &
echo "worker_pid=$!"

sleep 2
curl -sf http://127.0.0.1:8000/v1/health >/tmp/probity-health.json
python3 - <<'PY'
import json
d=json.load(open("/tmp/probity-health.json"))
print("status", d.get("status"), "mode", d.get("mode"))
for a in d.get("adapters", []):
    print(f"  {a['adapter_name']}: {a['status']} ({a['mode']}) detail={a.get('detail')}")
PY
