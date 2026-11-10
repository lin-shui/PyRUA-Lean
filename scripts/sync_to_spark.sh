#!/usr/bin/env bash
# Copy the three trees the twin needs from this Mac to spark.
#
#   SPARK_HOST=spark SPARK_ROOT=/home/lilin/game bash scripts/sync_to_spark.sh
#
# The twin needs only PyRUA-Lean (host + robot), steward's Python package and
# steward's data root (var/steward: scenes, scenarios, assets) — 4 MB + 224 KB,
# so this runs in seconds. .git, virtualenvs, runs and caches never travel.
set -euo pipefail

SPARK_HOST="${SPARK_HOST:-spark}"
SPARK_ROOT="${SPARK_ROOT:-/home/lilin/game}"
MAC_ROOT="${MAC_ROOT:-/Users/lilin/game}"

RSYNC=(rsync -az --human-readable
  --exclude '.git/' --exclude '.venv/' --exclude 'venv/' --exclude '__pycache__/'
  --exclude 'runs/' --exclude 'uv.lock' --exclude '*.pyc' --exclude '.pytest_cache/')

echo "== PyRUA-Lean -> $SPARK_HOST:$SPARK_ROOT/PyRUA-Lean"
ssh "$SPARK_HOST" "mkdir -p '$SPARK_ROOT/PyRUA-Lean'"
"${RSYNC[@]}" "$MAC_ROOT/PyRUA-Lean/" "$SPARK_HOST:$SPARK_ROOT/PyRUA-Lean/"

echo "== uni-robot-agent -> $SPARK_HOST:$SPARK_ROOT/uni-robot-agent"
ssh "$SPARK_HOST" "mkdir -p '$SPARK_ROOT/uni-robot-agent'"
"${RSYNC[@]}" "$MAC_ROOT/uni-robot-agent/" "$SPARK_HOST:$SPARK_ROOT/uni-robot-agent/"

echo "== steward package -> $SPARK_HOST:$SPARK_ROOT/steward/steward"
ssh "$SPARK_HOST" "mkdir -p '$SPARK_ROOT/steward/steward'"
"${RSYNC[@]}" "$MAC_ROOT/steward/steward/" "$SPARK_HOST:$SPARK_ROOT/steward/steward/"

echo "== steward data root -> $SPARK_HOST:$SPARK_ROOT/steward/var"
ssh "$SPARK_HOST" "mkdir -p '$SPARK_ROOT/steward/var'"
"${RSYNC[@]}" "$MAC_ROOT/steward/var/" "$SPARK_HOST:$SPARK_ROOT/steward/var/"

echo "== third_party (piper_description) -> $SPARK_HOST:$SPARK_ROOT/steward/third_party"
"${RSYNC[@]}" "$MAC_ROOT/steward/third_party/" "$SPARK_HOST:$SPARK_ROOT/steward/third_party/"

echo "OK: synced to $SPARK_HOST:$SPARK_ROOT"
