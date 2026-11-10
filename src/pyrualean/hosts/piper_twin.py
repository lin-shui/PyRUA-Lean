# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The host of ``--robot piper_twin``: the Cobot Magic twin as an episode service.

The twin runs in its own interpreter (uni-robot-agent plus steward; MuJoCo) and
is started here as a subprocess, one process per episode. This process speaks
the environment service's HTTP protocol (world poses in, primitive steps and
calibrated RGB-D frames out), keeps SAM3 for perception, and records one step
per motion call.

Needs: ``--twin-python`` (or ``PIPER_TWIN_PYTHON``), an interpreter with
``uni_robot_agent`` and ``steward``; ``--twin-root`` (or ``STEWARD_TWIN_ROOT``),
the steward data root holding assets/, scenes/ and scenarios/; and
``SAM3_CHECKPOINT_PATH`` for ``segment``/``locate``.

The HTTP contract is the one in uni-robot-agent's
``services/environment``: ``GET /info``, ``POST /captures``, ``PSOT /steps``
with a versioned request envelope, ``GET /records?index=N``,
``GET /artifacts/<id>`` and ``POST /finish``. A step reply carries the
recorded snapshot (the service records one per step), so no extra capture is
needed after a motion.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .._backend import Ledger
from ..piper_twin import (
    HOME_TOL_RAD,
    MOVE_TOL_DEG,
    MOVE_TOL_M,
    PiperTwinRobot,
)
from ..prompt import TaskCard, render_prompt
from ..runner import PolicyLoadError, load_policy, policy_sha256, run_policy
from ..sandbox import DESCRIPTION as SANDBOX_DESCRIPTION
from ..sandbox import audit_source, make_builtins
from . import piper_sight as sight
from .common import RESULT_SCHEMA

ROBOT = "piper_twin"
TWIN_MODULE = "uni_robot_agent.integrations.cobot_magic.environment.server"
#: Policy camera name -> the twin's mounted camera name.
CAMERAS = {
    "cam_head": "head",
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
}
POLICY_CAMERA = {twin: policy for policy, twin in CAMERAS.items()}
BOOT_TIMEOUT_S = 600.0
#: One step's target may be at most this far (m) / turned this much (rad) from
#: the measured pose, so every step is an increment the twin's executor accepts.
STEP_MAX_M = 0.05
STEP_MAX_RAD = 0.30
#: ``move_to`` gives up after this many increments (5 cm each).
MOVE_MAX_STEPS = 60
#: ``set_gripper`` repeats the command until the reading stops changing.
GRIP_MAX_STEPS = 10
#: ``home`` swings the joints back within this many steps.
HOME_MAX_STEPS = 60


@dataclass(frozen=True)
class Cell:
    """One twin episode: a task category pair, a scenario and an instruction.

    ``task`` is the twin's own final check: ``"<upper>-on-<lower>"`` (for
    example ``bowl-on-plate``), matched by category across the scene graph.
    """

    task: str
    scenario: str = "cobot-table"
    instruction: str = ""
    seed: int = 0
    max_episode_steps: int = 2500
    cuda_device: int | None = None

    @property
    def tag(self) -> str:
        return f"{self.task}_{self.scenario}_s{self.seed}"

    @property
    def suite(self) -> str:
        """``play_episode``/``result.json`` call the scenario the cell's suite."""
        return self.scenario


def add_cell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, help="twin task pair, e.g. bowl-on-plate")
    parser.add_argument("--scenario", default="cobot-table", help="steward scenario name")
    parser.add_argument(
        "--instruction", default="", help="task instruction shown to the policy"
    )
    parser.add_argument("--seed", type=int, default=0, help="scene seed")


def add_boot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--twin-python",
        default=os.environ.get("PIPER_TWIN_PYTHON"),
        help="interpreter with uni_robot_agent + steward that runs the twin"
        " (default: $PIPER_TWIN_PYTHON)",
    )
    parser.add_argument(
        "--twin-root",
        default=os.environ.get("STEWARD_TWIN_ROOT"),
        help="steward data root with assets/, scenes/ and scenarios/"
        " (default: $STEWARD_TWIN_ROOT)",
    )


def boot_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "twin_python": getattr(args, "twin_python", None),
        "twin_root": getattr(args, "twin_root", None),
    }


def cell_from_args(args: argparse.Namespace) -> Cell:
    return Cell(
        task=str(args.task),
        scenario=str(getattr(args, "scenario", None) or "cobot-table"),
        instruction=str(getattr(args, "instruction", None) or ""),
        seed=int(getattr(args, "seed", 0) or 0),
        max_episode_steps=int(getattr(args, "max_episode_steps", 2500)),
        cuda_device=getattr(args, "cuda_device", None),
    )


# ---------------------------------------------------------------------------
# the twin process and its HTTP contract
# ---------------------------------------------------------------------------


class Twin:
    """The twin subprocess and the lock that serialises requests to it."""

    def __init__(self, proc: subprocess.Popen, log: Any, url: str) -> None:
        self.proc, self.log, self.url = proc, log, url
        self._lock = threading.Lock()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self.proc.terminate()
            self.proc.wait(timeout=20)
        with contextlib.suppress(Exception):
            self.proc.kill()
        with contextlib.suppress(Exception):
            self.log.close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _info(url: str) -> dict[str, Any] | None:
    """``GET /info`` when the service answers, else None."""

    try:
        with urlopen(url + "/info", timeout=2.0) as reply:  # noqa: S310 - loopback only
            return json.loads(reply.read().decode("utf-8"))
    except (URLError, HTTPError, OSError, ValueError):
        return None


def start_twin(
    cell: Cell,
    out_dir: Path,
    twin_python: str | None = None,
    twin_root: str | None = None,
    timeout_s: float = BOOT_TIMEOUT_S,
) -> Twin:
    python = twin_python or os.environ.get("PIPER_TWIN_PYTHON")
    if not python:
        raise FileNotFoundError(
            "the twin's interpreter: pass --twin-python or set PIPER_TWIN_PYTHON"
        )
    root = twin_root or os.environ.get("STEWARD_TWIN_ROOT")
    if not root:
        raise FileNotFoundError(
            "the twin's data root: pass --twin-root or set STEWARD_TWIN_ROOT"
        )
    port = _free_port()
    env = dict(os.environ)
    # EGL is the offscreen backend on Linux; macOS uses CGL. PIPER_TWIN_GL overrides.
    gl = os.environ.get("PIPER_TWIN_GL") or ("cgl" if sys.platform == "darwin" else "egl")
    env.update(MUJOCO_GL=gl, PYTHONDONTWRITEBYTECODE="1", STEWARD_TWIN_ROOT=str(root))
    if cell.cuda_device is not None and gl == "egl":
        env["MUJOCO_EGL_DEVICE_ID"] = str(cell.cuda_device)
    log = open(out_dir / "twin.log", "ab")  # noqa: SIM115 - lives with the process
    cmd = [
        python,
        "-m",
        TWIN_MODULE,
        "--root",
        str(root),
        "--scenario",
        cell.scenario,
        "--task",
        cell.task,
        "--instruction",
        cell.instruction or cell.task,
        "--max-steps",
        str(cell.max_episode_steps),
        # The solver may tilt the tool to keep the grasp point on target, but a
        # large allowance lets a commanded orientation drift by tens of degrees
        # while every position reading still looks right. A grasp needs the
        # finger axis, so the tilt is a servo tolerance, not a free variable.
        "--max-tilt",
        os.environ.get("PIPER_TWIN_MAX_TILT", "0.2"),
        "--output-dir",
        str(out_dir),
        "--run-id",
        cell.tag,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(  # noqa: S603 - our own module, fixed arguments
        cmd, stdout=subprocess.DEVNULL, stderr=log, env=env, start_new_session=True
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            log.close()
            tail = (out_dir / "twin.log").read_text(errors="replace")[-4000:]
            raise RuntimeError(f"the twin process exited while booting:\n{tail}")
        if _info(url) is not None:
            return Twin(proc, log, url)
        time.sleep(0.5)
    proc.kill()
    raise TimeoutError("the twin did not boot in time")


class Client:
    """The versioned episode protocol over HTTP, with no physical retries."""

    def __init__(self, twin: Twin, timeout_s: float = 900.0) -> None:
        self.twin, self.timeout_s = twin, timeout_s
        self.info = self._get("/info")
        self.episode_id = str(self.info.get("episode_id") or "episode")
        self._sequence = int(self.info.get("next_sequence") or 1)

    # -- reads --------------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        with urlopen(self.twin.url + path, timeout=self.timeout_s) as reply:  # noqa: S310
            return json.loads(reply.read().decode("utf-8"))

    def _get_bytes(self, path: str) -> bytes:
        with urlopen(self.twin.url + path, timeout=self.timeout_s) as reply:  # noqa: S310
            return reply.read()

    def record(self, index: int = -1) -> dict[str, Any]:
        return self._get(f"/records?index={int(index)}")

    def artifact_bytes(self, artifact_id: str) -> bytes:
        return self._get_bytes(f"/artifacts/{artifact_id}")

    def outcome(self) -> dict[str, Any] | None:
        try:
            return self._get("/outcome")
        except HTTPError as error:
            if error.code in (404, 409, 400, 422):
                return None
            raise

    # -- writes -------------------------------------------------------------

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.twin._lock:
            for attempt in (0, 1):
                body = json.dumps(
                    {
                        "identity": {
                            "episode_id": self.episode_id,
                            "request_id": uuid.uuid4().hex,
                            "sequence": self._sequence,
                            "execution_id": None,
                        },
                        "payload": payload,
                    }
                ).encode("utf-8")
                request = Request(  # noqa: S310 - loopback only
                    self.twin.url + path,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urlopen(request, timeout=self.timeout_s) as reply:
                        result = json.loads(reply.read().decode("utf-8"))
                except HTTPError as error:
                    detail = error.read().decode("utf-8", errors="replace")
                    # A request that the twin executed but whose reply we lost (a
                    # timeout or a truncated body) leaves our counter one behind
                    # the twin's; every later call is then refused as a sequence
                    # mismatch and the episode is dead. The refusal says where
                    # the twin is: adopt it and resend once.
                    if attempt == 0 and error.code == 409:
                        try:
                            info = json.loads(detail)
                        except json.JSONDecodeError:
                            info = {}
                        failure = info.get("error") if isinstance(info, dict) else None
                        next_sequence = ((failure or {}).get("details") or {}).get("next_sequence")
                        if (isinstance(failure, dict)
                                and failure.get("code") == "sequence_mismatch"
                                and isinstance(next_sequence, int)
                                and next_sequence != self._sequence):
                            self._sequence = next_sequence
                            continue
                    raise RuntimeError(f"{path} rejected ({error.code}): {detail[:800]}") from error
                self._sequence += 1
                return result
            raise RuntimeError(f"{path} rejected: sequence did not resynchronise")

    def capture(self) -> dict[str, Any]:
        return self.post("/captures", {"cameras": None, "high_resolution": True})

    def finish(self, reason: str, status: str | None = None) -> dict[str, Any]:
        return self.post("/finish", {"reason": reason, "status": status})


# ---------------------------------------------------------------------------
# recorded steps
# ---------------------------------------------------------------------------


class Steps:
    """The recorded steps of an episode: state and calibration in memory, images on disk."""

    def __init__(self, out_dir: Path, instruction: str) -> None:
        self.root = Path(out_dir) / "steps"
        self.root.mkdir(parents=True, exist_ok=True)
        self.instruction = instruction
        self.records: list[dict[str, Any]] = []

    def add(self, snapshot: dict[str, Any], client: Client) -> int:
        step = len(self.records)
        folder = self.root / f"{step:04d}"
        folder.mkdir(exist_ok=True)
        cameras: dict[str, Any] = {}
        for twin_name, frame in (snapshot.get("cameras") or {}).items():
            policy_name = POLICY_CAMERA.get(twin_name, twin_name)
            rgb = frame.get("rgb") or {}
            artifact_id = rgb.get("artifact_id")
            if artifact_id:
                (folder / f"{policy_name}.png").write_bytes(client.artifact_bytes(artifact_id))
            size = frame.get("image_size") or [0, 0]
            cameras[policy_name] = {
                "shape": [int(size[0]), int(size[1]), 3],
                "intrinsic_matrix": frame.get("intrinsic_K"),
                "extrinsic_matrix": frame.get("T_world_from_optical"),
                "has_depth": frame.get("depth_m") is not None,
            }
        self.records.append(
            {
                "state": snapshot.get("state") or {},
                "truncated": bool(snapshot.get("truncated")),
                "terminated": bool(snapshot.get("terminated")),
                "cameras": cameras,
            }
        )
        return step

    def index(self, step: int) -> int:
        if not self.records:
            raise IndexError("no step recorded")
        index = len(self.records) + step if step < 0 else step
        if not 0 <= index < len(self.records):
            raise IndexError(f"step {step} is not recorded (0 .. {len(self.records) - 1})")
        return index

    def artifact_path(self, name: str, step: int = -1) -> Path:
        return self.root / f"{self.index(step):04d}" / name

    def load_bytes(self, name: str, step: int = -1) -> bytes:
        return self.artifact_path(name, step).read_bytes()


# ---------------------------------------------------------------------------
# the toolkit
# ---------------------------------------------------------------------------


def _pose_of(state: dict[str, Any], arm: str) -> dict[str, Any]:
    return ((state.get("arms") or {}).get(arm) or {}).get("control_pose") or {}


def _joints_of(state: dict[str, Any], arm: str) -> list[float]:
    arm_state = (state.get("arms") or {}).get(arm) or {}
    return [float(v) for v in arm_state.get("joints") or []]


def _gripper_of(state: dict[str, Any], arm: str) -> float:
    arm_state = (state.get("arms") or {}).get(arm) or {}
    try:
        return float(arm_state.get("gripper", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _holding_of(state: dict[str, Any], arm: str) -> bool:
    """The twin's own grasp flag: something is between the jaws (not just "closed").

    A close on nothing leaves a few millimetres of jaw gap, so ``opening > 0``
    is not evidence of a grasp; the twin's contact flag is.
    """
    arm_state = (state.get("arms") or {}).get(arm) or {}
    metadata = arm_state.get("metadata") or {}
    return bool(metadata.get("gripper_holding"))


def _distance(a: Any, b: Any) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _slerp(a: Any, b: Any, fraction: float) -> list[float]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    left /= np.linalg.norm(left) or 1.0
    right /= np.linalg.norm(right) or 1.0
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right, dot = -right, -dot
    if dot > 0.9995:
        out = left + fraction * (right - left)
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        out = (
            math.sin((1.0 - fraction) * theta) * left + math.sin(fraction * theta) * right
        ) / math.sin(theta)
    return (out / (np.linalg.norm(out) or 1.0)).tolist()


def _angle_between(a: Any, b: Any) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    dot = abs(float(np.dot(left, right)))
    norm = (np.linalg.norm(left) * np.linalg.norm(right)) or 1.0
    return 2.0 * math.acos(max(-1.0, min(1.0, dot / norm)))


class TwinToolkit:
    """The robot's tools: motions in the twin, perception here, a recorded step per motion."""

    def __init__(
        self,
        twin: Twin,
        client: Client,
        cell: Cell,
        out_dir: Path,
        sam3: Any = None,
    ) -> None:
        self.twin, self.client, self.cell = twin, client, cell
        self.state = Steps(out_dir, cell.instruction or cell.task)
        self.sam3 = sam3
        self._initial: dict[str, list[float]] = {}
        self._initial_pose: dict[str, tuple[Any, Any]] = {}
        snapshot = client.capture()
        index = self.state.add(snapshot, client)
        self._latest_snapshot = snapshot
        for arm in ("left", "right"):
            initial_state = snapshot.get("state") or {}
            self._initial[arm] = _joints_of(initial_state, arm)
            pose = _pose_of(initial_state, arm)
            self._initial_pose[arm] = (
                pose.get("position") or (0.0, 0.0, 0.0),
                pose.get("quaternion") or (1.0, 0.0, 0.0, 0.0),
            )
        self._latest_index = index
        self._outcome: dict[str, Any] | None = None

    # -- host interface -----------------------------------------------------

    def solved(self) -> bool:
        return bool((self._outcome or {}).get("task_success"))

    def episode_over(self) -> bool:
        record = self.state.records[-1] if self.state.records else {}
        return bool(record.get("truncated") or record.get("terminated"))

    def verdict_record(self) -> dict[str, Any] | None:
        return self._outcome

    def conclude(self) -> None:
        """Seal the episode and keep the twin's own outcome; nothing else judges it."""
        if self._outcome is not None:
            return
        try:
            result = self.client.finish(reason="policy_finished", status="give_up")
        except RuntimeError:
            return
        outcome = result.get("outcome") if isinstance(result, dict) else None
        self._outcome = dict(outcome) if isinstance(outcome, dict) else None

    def cancel_active_and_wait(self) -> None:
        pass

    def cancel(self) -> None:
        """Nothing is queued: a primitive has already returned when the policy sees it."""

    def close(self) -> None:
        self.twin.stop()

    def env_steps(self) -> int | None:
        state = self._latest_snapshot.get("state") or {}
        value = state.get("native_steps")
        return int(value) if value is not None else None

    def artifact(self, name: str, step: int = -1) -> bytes:
        """The recorded image bytes of ``name`` (``cam_head.png``, ...) at ``step``."""
        return self.state.load_bytes(name, step=int(step))

    # -- tool dispatch ------------------------------------------------------

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(input_dict or {})
        if name == "view_env_state":
            try:
                index = self.state.index(int(kwargs.get("step", -1)))
            except IndexError as error:
                return {"error": str(error)}
            return self._envelope(index, name, {})
        handler = {
            "move_pose": self.move_pose,
            "move_joints": self.move_joints,
            "grip": self.grip,
            "home": self.home,
            "segment": self.segment,
            "locate": self.locate,
            "back_project": self.back_project,
        }.get(name)
        if handler is None:
            return {"error": f"this host has no tool {name!r}"}
        try:
            payload = handler(**kwargs)
        except TypeError as error:
            return {"error": f"{name}: {error}"}
        if name in ("segment", "locate", "back_project"):
            return payload
        return self._envelope(self._latest_index, name, payload)

    def _envelope(self, index: int, tool: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        record = self.state.records[self.state.index(index)]
        return {
            "step": self.state.index(index),
            "task_language": self.state.instruction,
            "terminated": bool(record.get("terminated")),
            "truncated": bool(record.get("truncated")),
            "state": {
                "state": record.get("state") or {},
                "instruction": self.state.instruction,
                "cameras": record.get("cameras") or {},
            },
            "log": {"tool": tool, "result": payload},
        }

    # -- motion -------------------------------------------------------------

    def _current(self) -> tuple[dict[str, Any], int]:
        """The latest state and its version (one capture, no motion)."""

        snapshot = self._latest_snapshot
        return snapshot.get("state") or {}, int(snapshot.get("state_version", 0))

    def _refresh(self, reply: dict[str, Any]) -> None:
        snapshot = reply.get("snapshot")
        if isinstance(snapshot, dict) and snapshot.get("cameras"):
            self._latest_index = self.state.add(snapshot, self.client)
            self._latest_snapshot = snapshot
            return
        snapshot = self.client.capture()
        self._latest_index = self.state.add(snapshot, self.client)
        self._latest_snapshot = snapshot

    def _arms_command(
        self,
        targets: dict[str, tuple[Any, Any]],
        openings: dict[str, float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """One pose command: ``targets`` move, every other arm is held at its measured pose.

        Holding through a pose (not a joint target) keeps a measured joint that
        sits a hair past a hardware stop from being rejected by the service's
        joint-limit check; the twin's executor holds an already-at-target arm
        with its measured joints.
        """
        state, version = self._current()
        arms: dict[str, Any] = {}
        for name in ("left", "right"):
            if name in targets:
                position, quaternion = targets[name]
            else:
                pose = _pose_of(state, name)
                position = pose.get("position") or (0.0, 0.0, 0.0)
                quaternion = pose.get("quaternion") or (1.0, 0.0, 0.0, 0.0)
            arms[name] = {
                "pose": {
                    "position": [float(v) for v in position],
                    "quaternion": [float(v) for v in quaternion],
                },
                "gripper": float(openings.get(name, _gripper_of(state, name))),
            }
        command = {"kind": "pose", "solver": "native", "arms": arms}
        reply = self.client.post(
            "/steps", {"command": command, "expected_state_version": version}
        )
        self._refresh(reply)
        return reply, reply.get("feedback") or {}

    def _send_pose(
        self, arm: str, position: Any, quaternion: Any, opening: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._arms_command({arm: (position, quaternion)}, {arm: opening})

    def _joints_command(
        self,
        targets: dict[str, Sequence[float]],
        openings: dict[str, float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """One joint command: ``targets`` move, every other arm holds its measured joints."""
        state, version = self._current()
        arms: dict[str, Any] = {}
        for name in ("left", "right"):
            joints = targets.get(name)
            if joints is None:
                joints = _joints_of(state, name)
            arms[name] = {
                "joints": [float(v) for v in joints],
                "gripper": float(openings.get(name, _gripper_of(state, name))),
            }
        command = {"kind": "joints", "arms": arms}
        reply = self.client.post(
            "/steps", {"command": command, "expected_state_version": version}
        )
        self._refresh(reply)
        return reply, reply.get("feedback") or {}

    def move_pose(
        self,
        xyz: Any,
        arm: str,
        gripper: int = 0,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
        path: str = "auto",
        preview: bool = False,
    ) -> dict[str, Any]:
        from ..piper_twin import angles_from_quat, quat_from_angles

        state, version = self._current()
        pose = _pose_of(state, arm)
        start = [float(v) for v in pose.get("position") or (0.0, 0.0, 0.0)]
        start_quat = [float(v) for v in pose.get("quaternion") or (1.0, 0.0, 0.0, 0.0)]
        current = angles_from_quat(start_quat)
        target_angles = (
            current[0] if pitch is None else float(pitch),
            current[1] if yaw is None else float(yaw),
            current[2] if roll is None else float(roll),
        )
        target_quat = quat_from_angles(*target_angles)
        target = [float(v) for v in xyz]
        if gripper > 0:
            opening = 0.0
        elif gripper < 0:
            opening = 1.0
        else:
            opening = _gripper_of(state, arm)

        if preview:
            if "goal_region_search" not in (self.client.info.get("capabilities") or []):
                # The service advertises what it can answer; this twin currently
                # advertises motion+capture only. Report it instead of guessing.
                return {
                    "arm": arm,
                    "target_xyz": target,
                    "start_xyz": start,
                    "final_xyz": start,
                    "dist_to_target_m": math.inf,
                    "orientation_error_deg": math.nan,
                    "reached": False,
                    "preview": True,
                    "ik_error": "goal-region search is not advertised by this twin",
                    "note": None,
                    "steps_used": 0,
                }
            reply = self.client.post(
                "/goal-regions",
                {
                    "arm": arm,
                    "goal": {"position": target, "quaternion": list(target_quat)},
                    "position_tolerance_m": MOVE_TOL_M,
                    "orientation_tolerance_rad": math.radians(MOVE_TOL_DEG),
                    "expected_state_version": version,
                },
            )
            found = bool(reply.get("found"))
            solved = ((reply.get("pose") or {}).get("position")) or target
            return {
                "arm": arm,
                "target_xyz": target,
                "start_xyz": start,
                "final_xyz": [float(v) for v in solved],
                "dist_to_target_m": _distance(solved, target),
                "orientation_error_deg": math.degrees(
                    _angle_between((reply.get("pose") or {}).get("quaternion") or target_quat, target_quat)
                ),
                "reached": found,
                "preview": True,
                "ik_error": None if found else str(reply.get("diagnostics") or "no solution"),
                "note": None,
                "steps_used": 0,
            }

        steps = 0
        ik_error: str | None = None
        note: str | None = None
        best = math.inf
        stalled = 0
        for _ in range(MOVE_MAX_STEPS):
            state, _ = self._current()
            pose = _pose_of(state, arm)
            position = [float(v) for v in pose.get("position") or start]
            quaternion = [float(v) for v in pose.get("quaternion") or start_quat]
            distance = _distance(position, target)
            rotation = _angle_between(quaternion, target_quat)
            if distance <= MOVE_TOL_M and math.degrees(rotation) <= MOVE_TOL_DEG:
                break
            # A target the arm cannot quite reach plateaus; stop paying settle
            # ticks for it instead of walking the whole MOVE_MAX_STEPS budget.
            if distance < best - 0.001:
                best, stalled = distance, 0
            else:
                stalled += 1
                if stalled >= 8:
                    note = f"no progress toward the target (remaining {distance * 1000:.1f} mm)"
                    break
            fraction = 1.0
            if distance > STEP_MAX_M:
                fraction = STEP_MAX_M / distance
            if rotation > STEP_MAX_RAD:
                fraction = min(fraction, STEP_MAX_RAD / rotation)
            next_position = [
                position[i] + (target[i] - position[i]) * fraction for i in range(3)
            ]
            # Hold the commanded orientation once it is within one step: slerping
            # from the measured orientation every step lets the solver's tilt
            # drift accumulate, and a descent then lands with the fingers turned.
            if rotation <= STEP_MAX_RAD:
                next_quat = list(target_quat)
            else:
                next_quat = _slerp(quaternion, target_quat, STEP_MAX_RAD / rotation)
            reply, feedback = self._send_pose(arm, next_position, next_quat, opening)
            steps += 1
            if not feedback.get("accepted", True):
                diagnostics = feedback.get("diagnostics") or {}
                ik_error = str(
                    diagnostics.get("message")
                    or diagnostics.get("reason")
                    or feedback.get("error")
                    or "step rejected"
                )
                break
            tilt = (feedback.get("diagnostics") or {}).get("tool_tilt_rad") or {}
            if tilt.get(arm):
                note = f"tool tilted {float(tilt[arm]):.3f} rad to reach the pose"

        state, _ = self._current()
        pose = _pose_of(state, arm)
        final = [float(v) for v in pose.get("position") or start]
        final_quat = [float(v) for v in pose.get("quaternion") or start_quat]
        return {
            "arm": arm,
            "target_xyz": target,
            "start_xyz": start,
            "final_xyz": final,
            "dist_to_target_m": _distance(final, target),
            "orientation_error_deg": math.degrees(_angle_between(final_quat, target_quat)),
            "reached": _distance(final, target) <= MOVE_TOL_M
            and math.degrees(_angle_between(final_quat, target_quat)) <= MOVE_TOL_DEG,
            "preview": False,
            "ik_error": ik_error,
            "note": note,
            "steps_used": steps,
        }

    def grip(self, arm: str, gripper: int, opening: float | None = None) -> dict[str, Any]:
        if gripper > 0:
            target, commanded = 0.0, "closed"
        else:
            target, commanded = (1.0 if opening is None else float(opening)), "open"
        steps = 0
        previous = _gripper_of(self._latest_snapshot.get("state") or {}, arm)
        stable = 0
        for _ in range(GRIP_MAX_STEPS):
            openings = {arm: target}
            _, feedback = self._arms_command({}, openings)
            steps += 1
            if not feedback.get("accepted", True):
                break
            now = _gripper_of(self._latest_snapshot.get("state") or {}, arm)
            if abs(now - previous) < 0.003:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            previous = now
        now = _gripper_of(self._latest_snapshot.get("state") or {}, arm)
        return {
            "arm": arm,
            "gripper": commanded,
            "opening": now,
            "holding": _holding_of(self._latest_snapshot.get("state") or {}, arm),
            "steps_used": steps,
        }

    def home(self, arm: str) -> dict[str, Any]:
        names = ("left", "right") if arm == "both" else (arm,)
        steps = 0
        error = math.inf
        pose_error = [math.inf, math.inf]
        best = math.inf
        stalled = 0
        for _ in range(HOME_MAX_STEPS):
            targets = {name: self._initial[name] for name in names}
            _, feedback = self._joints_command(targets, {})
            steps += 1
            if not feedback.get("accepted", True):
                break
            state = self._latest_snapshot.get("state") or {}
            error = 0.0
            pose_error = [0.0, 0.0]
            for name in names:
                now = np.asarray(_joints_of(state, name), dtype=float)
                start = np.asarray(self._initial.get(name) or now, dtype=float)
                if now.shape == start.shape and now.size:
                    error = max(error, float(np.max(np.abs(now - start))))
                pose = _pose_of(state, name)
                position = pose.get("position") or (0.0, 0.0, 0.0)
                quaternion = pose.get("quaternion") or (1.0, 0.0, 0.0, 0.0)
                pose_error[0] = max(
                    pose_error[0], _distance(position, self._initial_pose[name][0])
                )
                pose_error[1] = max(
                    pose_error[1], _angle_between(quaternion, self._initial_pose[name][1])
                )
            # Home is a joint-space return: the boot posture is fully extended, where
            # pose IK is singular, but the joint targets themselves are reachable.
            if error <= HOME_TOL_RAD:
                break
            if error < best - 1e-3:
                best, stalled = error, 0
            else:
                stalled += 1
                if stalled >= 8:
                    break
        return {
            "arm": arm,
            "reached": error <= HOME_TOL_RAD,
            "max_joint_error_rad": error,
            "pose_error_m": pose_error[0],
            "pose_error_deg": math.degrees(pose_error[1]),
            "steps_used": steps,
        }

    def move_joints(self, arm: str, joints: Sequence[float]) -> dict[str, Any]:
        """One arm to a joint posture; the other arm holds its measured joints.

        The joint targets go through repeated one-increment commands until the
        servos settle, because the twin's ``joints`` step is one control tick
        and a large posture change takes tens of them.
        """
        arm = str(arm)
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        target = [float(v) for v in joints]
        if len(target) != 6:
            raise ValueError("joints must have six angles")
        steps, error = 0, math.inf
        best = math.inf
        stalled = 0
        for _ in range(HOME_MAX_STEPS):
            _, feedback = self._joints_command({arm: target}, {})
            steps += 1
            if not feedback.get("accepted", True):
                break
            state = self._latest_snapshot.get("state") or {}
            now = np.asarray(_joints_of(state, arm), dtype=float)
            error = (
                float(np.max(np.abs(now - np.asarray(target, dtype=float))))
                if now.size == len(target)
                else math.inf
            )
            if error <= HOME_TOL_RAD:
                break
            if error < best - 1e-3:
                best, stalled = error, 0
            else:
                stalled += 1
                if stalled >= 8:
                    break
        pose = _pose_of(self._latest_snapshot.get("state") or {}, arm)
        return {
            "arm": arm,
            "reached": error <= HOME_TOL_RAD,
            "max_joint_error_rad": error,
            "tcp_pos": tuple(pose.get("position") or (0.0, 0.0, 0.0)),
            "steps_used": steps,
        }

    # -- perception ---------------------------------------------------------

    def _need_sam(self) -> Any:
        if self.sam3 is None:
            from . import sam3_local

            try:
                self.sam3 = sam3_local.shared(
                    cuda_device=self.cell.cuda_device
                )
            except Exception as error:  # noqa: BLE001 - report unavailability to the policy
                raise RuntimeError(
                    f"SAM3 is not available in this host: {type(error).__name__}: {error}"
                ) from error
        return self.sam3

    def segment(
        self,
        text_prompt: str = "",
        camera: str = "cam_head",
        min_score: float = 0.2,
        point: list[int] | None = None,
    ) -> dict[str, Any]:
        frame = sight.frame(self._latest_snapshot, camera, self.client.artifact_bytes)
        seg = sight.segment(
            self._need_sam(), frame["rgb"], text_prompt or None, min_score, point=point
        )
        return {
            "found": seg["found"],
            "centroid_rc": seg.get("centroid"),
            "area_px": seg.get("area_px", 0),
            "score": seg.get("score"),
            "box_px": seg.get("box"),
            "camera": camera,
        }

    def locate(
        self,
        text_prompt: str = "",
        camera: str = "cam_head",
        min_score: float = 0.2,
        point: list[int] | None = None,
    ) -> dict[str, Any]:
        return sight.locate(
            self._latest_snapshot,
            camera,
            text_prompt or None,
            self._need_sam(),
            min_score,
            self.client.artifact_bytes,
            point=point,
        )

    def back_project(self, row: int, col: int, camera: str) -> dict[str, Any]:
        return sight.back_project(
            self._latest_snapshot, camera, int(row), int(col), self.client.artifact_bytes
        )


# ---------------------------------------------------------------------------
# boot
# ---------------------------------------------------------------------------


def make_backend(toolkit: Any) -> Any:
    """The toolkit is already a Backend; no RPent adaptation is involved."""

    return toolkit


def boot(
    cell: Cell, out_dir: Path, twin_python: str | None = None, twin_root: str | None = None,
    **_: Any
) -> tuple[TwinToolkit, list[Any]]:
    """Start the twin for one episode and return its toolkit."""

    twin = start_twin(cell, out_dir, twin_python=twin_python, twin_root=twin_root)
    try:
        client = Client(twin)
        toolkit = TwinToolkit(twin, client, cell, out_dir)
    except BaseException:
        twin.stop()
        raise
    return toolkit, []


def dump_card(toolkit: TwinToolkit, cell: Cell, out_dir: Path) -> TaskCard:
    """What the policy author is told about this episode: the task, the scenario, the cameras."""

    state = toolkit.state.records[0]["state"] if toolkit.state.records else {}
    facts: dict[str, Any] = {
        "scenario": cell.scenario,
        "twin task check": cell.task,
        "cameras": list(CAMERAS),
    }
    for arm in ("left", "right"):
        pose = _pose_of(state, arm)
        facts[f"{arm}_grasp_point"] = [round(float(v), 4) for v in pose.get("position") or ()]
        facts[f"{arm}_quaternion_wxyz"] = [
            round(float(v), 4) for v in pose.get("quaternion") or ()
        ]
        facts[f"{arm}_gripper_opening"] = round(_gripper_of(state, arm), 3)
    images = {
        camera: str(toolkit.state.artifact_path(f"{camera}.png", 0))
        for camera in CAMERAS
        if toolkit.state.artifact_path(f"{camera}.png", 0).exists()
    }
    card = TaskCard(
        suite=ROBOT,
        task=cell.task,
        seed=cell.seed,
        task_language=cell.instruction or cell.task,
        object_names=[],
        max_env_steps=cell.max_episode_steps,
        images=images,
        robot=ROBOT,
        facts=facts,
    )
    (out_dir / "card.json").write_text(card.to_json() + "\n", encoding="utf-8")
    return card


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def card_only(cell: Cell, out_dir: Path, **options: Any) -> TaskCard:
    """Boot only the twin, dump the task card and shut it down."""

    out_dir.mkdir(parents=True, exist_ok=True)
    toolkit, daemons = boot(cell, out_dir, **options)
    try:
        return dump_card(toolkit, cell, out_dir)
    finally:
        with contextlib.suppress(Exception):
            toolkit.close()
        for daemon in reversed(daemons):
            with contextlib.suppress(Exception):
                daemon.stop()


def run_episode(
    policy_path: Path,
    cell: Cell,
    out_dir: Path,
    *,
    timeout_s: float = 1200.0,
    **options: Any,
) -> dict[str, Any]:
    """Run one policy file against one twin episode and write ``result.json``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "system": "pyrualean",
        "backend": ROBOT,
        "suite": cell.scenario,
        "task": cell.task,
        "seed": cell.seed,
        "task_language": None,
        "policy": {
            "path": str(policy_path),
            "sha256": policy_sha256(policy_path),
            "bytes": policy_path.stat().st_size,
        },
        "environment_success": False,
        "status": "boot_error",
        "reason": None,
        "error": None,
        "calls": 0,
        "stateful_calls": 0,
        "env_steps": 0,
        "env_steps_total": None,
        "policy_wall_s": None,
        "startup_s": None,
        "total_wall_s": None,
        "timeout_s": timeout_s,
        "max_episode_steps": cell.max_episode_steps,
        "prompt": None,
        "host": {
            "python": sys.executable,
            "cuda_device": cell.cuda_device,
            "scenario": cell.scenario,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "notes": [],
    }
    shutil.copyfile(policy_path, out_dir / "policy.py")

    toolkit = None
    daemons: list[Any] = []
    ledger = Ledger()
    try:
        toolkit, daemons = boot(cell, out_dir, **options)
        result["startup_s"] = time.perf_counter() - started
        backend = make_backend(toolkit)
        card = dump_card(toolkit, cell, out_dir)
        result["task_language"] = card.task_language
        bundle = render_prompt(card, budget_s=int(timeout_s))
        prompt_text = bundle.text()
        (out_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        result["prompt"] = {
            "chars": len(prompt_text),
            "sha256": _sha256_text(prompt_text),
            "images": list(bundle.images),
        }

        robo = PiperTwinRobot(backend, ledger=ledger)
        _install_sigterm()
        scratch = out_dir / "workspace"
        scratch.mkdir(exist_ok=True)
        result["host"]["sandbox"] = SANDBOX_DESCRIPTION
        result["policy"]["sandbox_flags"] = audit_source(
            policy_path.read_text(encoding="utf-8", errors="replace"), workspace=scratch
        )
        try:
            policy = load_policy(policy_path, builtins=make_builtins(scratch))
        except PolicyLoadError as exc:
            result["status"] = "error"
            result["reason"] = "load"
            result["error"] = {"type": "PolicyLoadError", "message": str(exc)}
        else:
            with open(out_dir / "policy_output.txt", "w", encoding="utf-8") as handle:
                with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                    outcome = run_policy(policy, robo, timeout_s=timeout_s)
            result["status"] = outcome.status
            result["reason"] = outcome.reason
            result["policy_wall_s"] = outcome.elapsed_s
            if outcome.error_type:
                result["error"] = {
                    "type": outcome.error_type,
                    "message": outcome.error_message,
                    "traceback": outcome.traceback,
                }
        toolkit.conclude()
        result["environment_success"] = bool(toolkit.solved())
        result["env_steps_total"] = backend.env_steps()
    except BaseException as exc:  # noqa: BLE001 - the result records boot failures
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if result["status"] == "boot_error":
            result["notes"].append("host failed before the policy started")
        else:
            result["status"] = "error"
        if isinstance(exc, KeyboardInterrupt):
            result["reason"] = "interrupted"
    finally:
        result["calls"] = len(ledger)
        result["stateful_calls"] = ledger.stateful_calls()
        result["env_steps"] = ledger.env_steps()
        ledger.write_jsonl(out_dir / "calls.jsonl")
        if toolkit is not None:
            with contextlib.suppress(Exception):
                toolkit.close()
        for daemon in reversed(daemons):
            with contextlib.suppress(Exception):
                daemon.stop()
        result["total_wall_s"] = time.perf_counter() - started
        (out_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return result


def _install_sigterm() -> None:
    def handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise KeyboardInterrupt(f"signal {signum}")

    with contextlib.suppress(ValueError):  # not in the main thread
        signal.signal(signal.SIGTERM, handler)


def _add_cell_args(parser: argparse.ArgumentParser) -> None:
    add_cell_args(parser)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=20000)
    add_boot_args(parser)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyrualean.hosts.piper_twin", description=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one policy file for one twin episode")
    run.add_argument("--policy", type=Path, required=True)
    run.add_argument("--timeout-s", type=float, default=1200.0)
    _add_cell_args(run)
    card = sub.add_parser("card", help="boot the twin and dump the task card")
    _add_cell_args(card)
    args = parser.parse_args(argv)
    cell = cell_from_args(args)
    options = boot_options(args)
    if args.command == "run":
        result = run_episode(
            args.policy.resolve(), cell, args.out.resolve(),
            timeout_s=args.timeout_s, **options,
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "status",
                        "reason",
                        "environment_success",
                        "calls",
                        "env_steps",
                        "policy_wall_s",
                    )
                },
                ensure_ascii=False,
            )
        )
        return 0 if result["status"] != "boot_error" else 2
    card_obj = card_only(cell, args.out.resolve(), **options)
    print(card_obj.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMERAS",
    "Cell",
    "Client",
    "Twin",
    "TwinToolkit",
    "add_boot_args",
    "add_cell_args",
    "boot",
    "boot_options",
    "card_only",
    "cell_from_args",
    "dump_card",
    "main",
    "make_backend",
    "run_episode",
    "start_twin",
]
