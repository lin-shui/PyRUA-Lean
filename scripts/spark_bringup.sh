#!/usr/bin/env bash
# One-shot: sync the repos, build the spark env, boot the twin and check it.
#
#   SPARK_HOST=spark bash scripts/spark_bringup.sh              # full chain
#   SKIP_SYNC=1 SPARK_HOST=spark bash scripts/spark_bringup.sh  # env + smoke
#
# The smoke needs EGL (MUJOCO_GL=egl) and the twin's data root; both come from
# the env file the setup writes. Nothing here needs a model endpoint or SAM3.
set -euo pipefail

SPARK_HOST="${SPARK_HOST:-spark}"
SPARK_ROOT="${SPARK_ROOT:-/home/lilin/game}"
ENV_DIR="${ENV_DIR:-/home/lilin/env/piper-twin}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${SKIP_SYNC:-0}" != "1" ]; then
  echo "===== 1/3 sync"
  SPARK_HOST="$SPARK_HOST" SPARK_ROOT="$SPARK_ROOT" bash "$HERE/sync_to_spark.sh"
fi

echo "===== 2/3 env on $SPARK_HOST"
ssh "$SPARK_HOST" "SPARK_ROOT='$SPARK_ROOT' ENV_DIR='$ENV_DIR' bash '$SPARK_ROOT/PyRUA-Lean/scripts/spark_env_setup.sh'"

echo "===== 3/3 twin smoke"
ssh "$SPARK_HOST" "source '$SPARK_ROOT/piper-twin.env' && cd '$SPARK_ROOT/PyRUA-Lean' && \
  MUJOCO_GL=egl STEWARD_TWIN_ROOT='$SPARK_ROOT/steward/var/steward' \
  '$ENV_DIR/bin/python' scripts/piper_twin_smoke.py --out /tmp/piper-twin-smoke ${SMOKE_ARGS:-}"

echo
echo "NEXT: a hand-written episode (no model endpoint needed):"
echo "  ssh $SPARK_HOST \"source $SPARK_ROOT/piper-twin.env && cd $SPARK_ROOT/PyRUA-Lean && \\"
echo "    $ENV_DIR/bin/python -m pyrualean run --robot piper_twin --task drawer-open \\"
echo "    --scenario piper-drawer --instruction 'Open the cabinet drawer.' \\"
echo "    --policy /tmp/policy.py --out /tmp/piper-run --max-episode-steps 20000\""
echo "  (build the drawer scene first: $ENV_DIR/bin/python scripts/make_piper_drawer_scenario.py)"
