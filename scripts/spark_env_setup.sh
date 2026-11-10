#!/usr/bin/env bash
# Build the episode interpreter for the Cobot Magic twin on spark. Runs ON spark:
#
#   ssh spark "SPARK_ROOT=/home/lilin/game bash /home/lilin/game/PyRUA-Lean/scripts/spark_env_setup.sh"
#
# One venv holds the twin server (uni-robot-agent + steward + MuJoCo), the
# PyRUA-Lean host and the smoke test. Idempotent: re-running upgrades nothing
# but reinstalls the editable packages.
set -euo pipefail

SPARK_ROOT="${SPARK_ROOT:-/home/lilin/game}"
ENV_DIR="${ENV_DIR:-/home/lilin/env/piper-twin}"
PYTHON="${PYTHON:-python3}"
UNI_DIR="$SPARK_ROOT/uni-robot-agent"
STEWARD_DIR="$SPARK_ROOT/steward/steward"
PYRUA_DIR="$SPARK_ROOT/PyRUA-Lean"
TWIN_ROOT="${STEWARD_TWIN_ROOT:-$SPARK_ROOT/steward/var/steward}"

for path in "$UNI_DIR/pyproject.toml" "$STEWARD_DIR/pyproject.toml" "$PYRUA_DIR/pyproject.toml" \
            "$TWIN_ROOT/scenarios/cobot-table.json"; do
  [ -e "$path" ] || { echo "missing $path — run sync_to_spark.sh first" >&2; exit 1; }
done

if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "== creating $ENV_DIR"
  "$PYTHON" -m venv "$ENV_DIR" || {
    echo "venv failed; on Debian/Ubuntu install python3-venv (and python3-pip)" >&2; exit 1; }
fi
PIP="$ENV_DIR/bin/pip"
"$PIP" install --quiet --upgrade pip setuptools wheel

echo "== runtime dependencies"
# The twin server imports fastapi/uvicorn/pydantic; steward needs mujoco + mink.
# uni-robot-agent is installed --no-deps below: its planner SDKs are not used
# by the environment service and are heavy on aarch64.
"$PIP" install --quiet \
  "numpy>=1.26" "pillow>=10" "fastapi>=0.115,<1" "uvicorn>=0.29,<1" "pydantic>=2,<3" \
  "jsonschema>=4.18" "httpx>=0.28,<1"
# play_episode serves the arm tools over MCP even when the policy runtime is the
# in-process OpenAI loop; the piper env needs that package too.
"$PIP" install --quiet "mcp>=1.8,<2"

echo "== steward (mujoco, mink, zstandard)"
"$PIP" install --quiet -e "$STEWARD_DIR"

echo "== uni-robot-agent (no planner deps)"
"$PIP" install --quiet -e "$UNI_DIR" --no-deps

echo "== pyrualean (host + piper_twin)"
"$PIP" install --quiet -e "$PYRUA_DIR"

echo "== import check"
env MUJOCO_GL=egl STEWARD_TWIN_ROOT="$TWIN_ROOT" "$ENV_DIR/bin/python" - <<'PY'
import mujoco, numpy, fastapi, pydantic
from steward.world.sim import WorldSim
from steward.ego.ik import CobotIK
from steward.assets.cobot import TCP_OFFSET
from steward.authoring.service import AuthoringService
import uni_robot_agent.integrations.cobot_magic.environment.adapter  # noqa: F401
from pyrualean.hosts.piper_twin import Cell, Client, TwinToolkit  # noqa: F401
print("imports OK: mujoco", mujoco.__version__, "| TCP_OFFSET", TCP_OFFSET)
PY

cat > "$SPARK_ROOT/piper-twin.env" <<EOF
# source this before running the twin on spark
export PIPER_TWIN_PYTHON="$ENV_DIR/bin/python"
export STEWARD_TWIN_ROOT="$TWIN_ROOT"
export MUJOCO_GL=egl
# RPent's SAM3 HTTP service (scripts/start_sam3_server.sh); text prompts score
# near zero on the twin's synthetic renders, point prompts work.
export SAM3_ENDPOINT="\${SAM3_ENDPOINT:-127.0.0.1:8114}"
EOF
echo "== wrote $SPARK_ROOT/piper-twin.env"
echo "OK: env at $ENV_DIR"
