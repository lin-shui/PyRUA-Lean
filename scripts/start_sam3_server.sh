#!/usr/bin/env bash
# Start RPent's SAM3 HTTP service for the twin (one process, 3.3 GB checkpoint).
#
#   bash scripts/start_sam3_server.sh          # uses the paths found on spark
#   PORT=8114 bash scripts/start_sam3_server.sh
#
# Then export SAM3_ENDPOINT=127.0.0.1:8114 (spark_env_setup.sh writes it into
# piper-twin.env). The policy interpreter never imports torch: it sends the
# image to this service and gets a mask back. Re-running is a no-op when the
# port already answers.
set -euo pipefail

CHECKPOINT="${SAM3_CHECKPOINT:-/home/liaoheng/robodojo/src/XPolicyLab/policy/PyRUA_Lean/checkpoints/sam3.pt}"
RPENT_PYTHON="${RPENT_PYTHON:-/home/liaoheng/miniconda3/envs/rpent/bin/python}"
RPENT_ROOT="${RPENT_ROOT:-/home/liaoheng/rpent_deps/rpent}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8114}"
LOG="${LOG:-/home/lilin/game/sam3/server.log}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "SAM3 checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [ ! -x "$RPENT_PYTHON" ]; then
  echo "RPent interpreter not found: $RPENT_PYTHON" >&2
  exit 1
fi

if (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; then
  echo "SAM3 already listening on http://$HOST:$PORT"
  echo "export SAM3_ENDPOINT=$HOST:$PORT"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
cd "$RPENT_ROOT"
SAM3_CHECKPOINT_PATH="$CHECKPOINT" nohup "$RPENT_PYTHON" \
  -m rpent.robots.components.sam3_server \
  --transport http --host "$HOST" --port "$PORT" >"$LOG" 2>&1 &
echo "starting SAM3 (pid $!) -> $LOG"

for _ in $(seq 1 120); do
  if (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; then
    echo "SAM3 ready: http://$HOST:$PORT"
    echo "export SAM3_ENDPOINT=$HOST:$PORT"
    exit 0
  fi
  sleep 2
done
echo "SAM3 did not open $HOST:$PORT; see $LOG" >&2
exit 1
