"""Behaviour of the Cobot Magic twin API over a native-shaped backend."""

from __future__ import annotations

import math

import pytest

from pyrualean.piper_twin import (
    MOVE_TOL_DEG,
    PiperTwinRobot,
    angles_from_quat,
    quat_from_angles,
)
from pyrualean._backend import EpisodeFinished


class _Result:
    def __init__(self, name: str, result: dict) -> None:
        self.tool, self.result = name, result


class FakeTwinBackend:
    """Two arms, one state version, and the payload shapes the host returns."""

    TASK = "put the bowl on the plate"

    def __init__(self) -> None:
        self.version = 1
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.calls: list[tuple[str, dict]] = []
        self.last_prompt: dict | None = None
        self.arms = {
            "left": {"position": [-0.3, -0.3, 0.9], "quaternion": list(quat_from_angles(0, 0, 0)),
                     "joints": [0.0] * 6, "gripper": 1.0},
            "right": {"position": [0.3, -0.3, 0.9], "quaternion": list(quat_from_angles(0, 0, 0)),
                      "joints": [0.0] * 6, "gripper": 1.0},
        }

    # -- Backend contract ---------------------------------------------------

    def execute_tool(self, name: str, input_dict: dict) -> _Result:
        self.calls.append((name, dict(input_dict)))
        if name == "view_env_state":
            return _Result(name, self._envelope(None))
        handler = getattr(self, f"_do_{name}", None)
        if handler is None:
            return _Result(name, {"error": f"unknown tool {name}"})
        payload = handler(**input_dict)
        if name in ("segment", "locate", "back_project"):
            # read-only tools answer with their payload, not an observation envelope
            return _Result(name, payload)
        envelope = self._envelope(payload)
        envelope.update({k: v for k, v in payload.items() if k in ("error", "interrupted")})
        return _Result(name, envelope)

    def solved(self) -> bool:
        return False

    def artifact(self, name: str, step: int = -1) -> bytes:
        return b""

    def env_steps(self) -> int:
        return self.steps

    def cancel(self) -> None:
        pass

    # -- payloads -----------------------------------------------------------

    def _state(self) -> dict:
        return {
            "arms": {
                name: {
                    "control_pose": {"position": list(arm["position"]),
                                     "quaternion": list(arm["quaternion"])},
                    "joints": list(arm["joints"]),
                    "gripper": arm["gripper"],
                }
                for name, arm in self.arms.items()
            },
            "state_version": self.version,
            "native_steps": self.steps,
        }

    def _envelope(self, payload: dict | None) -> dict:
        return {
            "step": 0,
            "task_language": self.TASK,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "state": {"state": self._state(), "instruction": self.TASK, "cameras": {}},
            "log": {"tool": None, "result": payload},
        }

    def _do_move_pose(self, xyz, arm, gripper=0, pitch=None, yaw=None, roll=None,
                      path="auto", preview=False) -> dict:
        self.steps += 1
        start = list(self.arms[arm]["position"])
        if preview:
            return {
                "arm": arm, "target_xyz": list(xyz), "start_xyz": start,
                "final_xyz": list(xyz), "dist_to_target_m": 0.0,
                "orientation_error_deg": 0.0, "reached": True, "preview": True,
                "ik_error": None, "note": None, "steps_used": 0,
            }
        self.arms[arm]["position"] = list(xyz)
        if pitch is not None or yaw is not None or roll is not None:
            current = angles_from_quat(self.arms[arm]["quaternion"])
            self.arms[arm]["quaternion"] = list(
                quat_from_angles(
                    current[0] if pitch is None else pitch,
                    current[1] if yaw is None else yaw,
                    current[2] if roll is None else roll,
                )
            )
        return {
            "arm": arm, "target_xyz": list(xyz), "start_xyz": start,
            "final_xyz": list(xyz), "dist_to_target_m": 0.0,
            "orientation_error_deg": 0.0, "reached": True, "preview": False,
            "ik_error": None, "note": None, "steps_used": 1,
        }

    def _do_grip(self, arm, gripper, opening=None) -> dict:
        self.steps += 1
        self.arms[arm]["gripper"] = 0.0 if gripper > 0 else 1.0
        return {
            "arm": arm, "gripper": "closed" if gripper > 0 else "open",
            "opening": self.arms[arm]["gripper"],
            "holding": gripper > 0, "steps_used": 1,
        }

    def _do_home(self, arm) -> dict:
        self.steps += 1
        return {"arm": arm, "reached": True, "max_joint_error_rad": 0.0, "steps_used": 1}

    def _do_locate(self, text_prompt="", camera="cam_head", min_score=0.2, point=None) -> dict:
        self.last_prompt = {"text": text_prompt, "point": point, "camera": camera}
        return {
            "found": True, "score": 0.9, "centroid_rc": [100, 200], "area_px": 500,
            "camera": camera, "center_xyz": [0.1, 0.2, 0.85], "support_z": 0.8,
            "top_z": 0.9, "height_m": 0.1, "length_m": 0.06, "width_m": 0.04,
            "long_axis_deg": 10.0, "grasp_yaw": 100.0, "geometry_error": None,
        }

    def _do_segment(self, text_prompt="", camera="cam_head", min_score=0.2, point=None) -> dict:
        self.last_prompt = {"text": text_prompt, "point": point, "camera": camera}
        return {"found": True, "centroid_rc": [100, 200], "area_px": 500,
                "score": 0.9, "box_px": [180, 80, 40, 40], "camera": camera}

    def _do_back_project(self, row, col, camera="cam_head") -> dict:
        return {"pixel": [row, col], "world_xyz": [0.1, 0.2, 0.85],
                "depth_m": 0.5, "camera": camera}


@pytest.fixture()
def backend():
    return FakeTwinBackend()


@pytest.fixture()
def robo(backend):
    return PiperTwinRobot(backend)


# -- orientation conventions -------------------------------------------------


@pytest.mark.parametrize(
    "angles",
    [(0.0, 0.0, 0.0), (90.0, 0.0, 0.0), (45.0, 30.0, 0.0), (-30.0, -120.0, 15.0)],
)
def test_angles_round_trip(angles):
    pitch, yaw, roll = angles
    recovered = angles_from_quat(quat_from_angles(pitch, yaw, roll))
    assert recovered[0] == pytest.approx(pitch, abs=1e-6)
    assert recovered[1] == pytest.approx(yaw, abs=1e-6)
    assert recovered[2] == pytest.approx(roll, abs=1e-6)


def test_straight_down_has_the_tool_axis_pointing_down():
    w, x, y, z = quat_from_angles(90.0, 0.0, 0.0)
    # third column of the rotation: where the twin's grasp-site z axis points
    r = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    assert r[0][2] == pytest.approx(0.0, abs=1e-6)
    assert r[1][2] == pytest.approx(0.0, abs=1e-6)
    assert r[2][2] == pytest.approx(-1.0, abs=1e-6)


# -- observation -------------------------------------------------------------


def test_state_and_task_come_from_the_envelope(robo, backend):
    state = robo.state()
    assert state.task == backend.TASK
    assert state.left.tcp_pos == pytest.approx((-0.3, -0.3, 0.9))
    assert state.left.pitch == pytest.approx(0.0, abs=1e-6)
    assert state.left.gripper == pytest.approx(1.0)
    assert state.step == 0
    assert not state.truncated


def test_move_to_dispatches_the_twin_payload_and_returns_reached(robo, backend):
    move = robo.move_to([0.1, 0.2, 0.85], arm="right", gripper=PiperTwinRobot.CLOSE,
                        pitch=90.0, yaw=10.0)
    assert move.reached and move.preview is False
    name, args = backend.calls[-1]
    assert name == "move_pose"
    assert args["xyz"] == [0.1, 0.2, 0.85]
    assert args["arm"] == "right"
    assert args["gripper"] == 1
    assert args["pitch"] == 90.0 and args["yaw"] == 10.0
    assert args["preview"] is False


def test_preview_solves_without_moving(robo, backend):
    before = list(backend.arms["left"]["position"])
    move = robo.move_to([0.2, 0.0, 0.95], arm="left", preview=True)
    assert move.preview is True and move.reached
    assert backend.arms["left"]["position"] == before


def test_gripper_and_home_are_typed(robo, backend):
    closed = robo.set_gripper("left", PiperTwinRobot.CLOSE)
    assert closed.commanded == "closed" and closed.holding
    assert backend.arms["left"]["gripper"] == pytest.approx(0.0)
    home = robo.home("both")
    assert home.reached and home.max_joint_error_rad == pytest.approx(0.0)


def test_locate_and_segment_parse_the_host_payload(robo):
    sight = robo.locate("the bowl")
    assert sight.found and sight.center_xyz == pytest.approx((0.1, 0.2, 0.85))
    assert sight.grasp_yaw == pytest.approx(100.0)
    assert sight.height_m == pytest.approx(0.1)
    segment = robo.segment("the bowl")
    assert segment.found and segment.centroid == (100, 200)
    point = robo.back_project(100, 200)
    assert point.xyz == pytest.approx((0.1, 0.2, 0.85))
    assert point.depth_m == pytest.approx(0.5)


def test_point_prompts_reach_the_host(robo, backend):
    sight = robo.locate(point=[120, 240], camera="cam_left_wrist")
    assert sight.found
    assert backend.last_prompt == {"text": "", "point": [120, 240],
                                   "camera": "cam_left_wrist"}
    segment = robo.segment(point=[5, 6])
    assert segment.found and backend.last_prompt["point"] == [5, 6]
    with pytest.raises(ValueError):
        robo.locate("the bowl", point=[1, 2])
    with pytest.raises(ValueError):
        robo.locate()


def test_truncated_envelope_freezes_motion(robo, backend):
    backend.truncated = True
    robo.state()
    with pytest.raises(EpisodeFinished):
        robo.move_to([0.1, 0.2, 0.85])
