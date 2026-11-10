# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""The Cobot Magic twin (two PiPER arms) as a code-policy robot.

The twin is steward's MuJoCo scene of an AgileX Cobot Magic station, served by
uni-robot-agent's environment service (``cobot_magic`` integration) on the
same machine: world poses in, primitive steps and calibrated RGB-D frames out.
The same class is meant to drive the physical station later through the same
service contract; nothing here talks to a simulator directly.

The host is :mod:`pyrualean.hosts.piper_twin`. There is no VLA: run the robot
with ``--primitives no-vla``.

Conventions
-----------
World frame: metres. Two 6-DoF PiPER arms named ``"left"`` and ``"right"``;
a motion call names one arm and holds the other where it is. The frames and
limits that the twin publishes are described in ``knowledge/piper_twin.md``.

``move_to`` drives an arm's GRASP POINT — the pinch point between the jaws —
to a world position, with an orientation given in degrees: ``pitch`` is how
far the gripper points below the horizontal (0 level, 90 straight down),
``yaw`` the heading of the pointing direction on the table, ``roll`` the turn
of the fingers about the pointing direction. An angle left ``None`` keeps its
measured value.

``move_to(..., preview=True)`` solves the target pose and returns the pose the
arm would reach WITHOUT moving the robot (the twin's own IK, same solver the
motion uses). Every other motion call blocks until the primitive finishes and
returns a typed result; nothing is queued.

Gripper: COMMANDS are signed — ``robo.CLOSE`` (1) closes, ``robo.OPEN`` (-1)
opens, ``robo.KEEP`` (0) leaves the gripper. MEASUREMENTS
(``state().left.gripper``) are openings: 1.0 = open, 0.0 = closed on nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ._robot import RobotBase
from .robots import RobotAdapter

#: A move has arrived when the grasp point is this close (metres) ...
MOVE_TOL_M = 0.01
#: ... and the gripper is turned less than this from the asked orientation (degrees).
MOVE_TOL_DEG = 5.0
#: ``home`` stops once every joint is this close to its starting angle (radians).
HOME_TOL_RAD = 0.03


# ---------------------------------------------------------------------------
# Result types (plain values, returned to the policy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmState:
    """One arm at a recorded step.

    tcp_pos: the grasp point (pinch point between the jaws) [x, y, z], metres.
        ``move_to`` drives this point.
    pitch / yaw / roll: the gripper's orientation, degrees (class docstring).
    gripper: measured opening, 1.0 = open, 0.0 = closed on nothing.
    joints: the six arm joint angles, radians.
    """

    tcp_pos: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    gripper: float
    joints: tuple[float, ...]


@dataclass(frozen=True)
class State:
    """Snapshot of both arms at one recorded step.

    step: index of the recorded step (0 = initial scene). A new step is
        recorded after every motion call.
    task: the task instruction the episode was started with (authoritative).
    left / right: the two arms (``state.arm("left")`` picks one by name).
    truncated: the episode is over (budget spent or the twin sealed it).
    """

    step: int
    task: str
    left: ArmState
    right: ArmState
    truncated: bool

    def arm(self, name: str) -> ArmState:
        if name == "left":
            return self.left
        if name == "right":
            return self.right
        raise ValueError("arm must be 'left' or 'right'")


@dataclass(frozen=True)
class Move:
    """Outcome of ``move_to``.

    target_xyz: the requested grasp-point position.
    start_xyz / final_xyz: where the grasp point was before and after.
    dist_to_target_m: distance from ``final_xyz`` to ``target_xyz``.
    orientation_error_deg: how far the final orientation is from the one asked.
    reached: within 0.01 m and 5 degrees of the target.
    preview: this was ``preview=True``: ``final_xyz`` is the pose the solver
        would reach; nothing moved.
    ik_error: why the pose was not reached (out of reach, no solution), or None.
    note: a measured remark from the twin (tool tilt used, rejected increment).
    steps_used: native actions consumed.
    truncated: the episode is over.
    """

    arm: str
    target_xyz: tuple[float, float, float]
    start_xyz: tuple[float, float, float]
    final_xyz: tuple[float, float, float]
    dist_to_target_m: float
    orientation_error_deg: float
    reached: bool
    preview: bool
    ik_error: str | None
    note: str | None
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Gripper:
    """Outcome of ``set_gripper``.

    commanded: ``"closed"`` or ``"open"``.
    opening: the opening once the fingers stopped (1.0 = open, 0.0 = closed).
    holding: after a close, the fingers stopped apart: something is between
        them. It does not say what — confirm the intended object moves with
        the arm.
    steps_used: native actions consumed.
    truncated: the episode is over.
    """

    arm: str
    commanded: str
    opening: float
    holding: bool
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Home:
    """Outcome of ``home``.

    arm: as requested (``"left"``, ``"right"`` or ``"both"``).
    reached: every named arm returned to the grasp pose it started in
        (within 1 cm and 5 degrees).
    max_joint_error_rad: the largest joint difference that remains (the pose
        is what is guaranteed; IK may return a different joint solution).
    steps_used: native actions consumed.
    truncated: the episode is over.
    """

    arm: str
    reached: bool
    max_joint_error_rad: float
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Joints:
    """Outcome of ``move_joints``.

    arm: ``"left"`` or ``"right"``.
    reached: the measured joints are within 0.01 rad of the target.
    max_joint_error_rad: the largest remaining joint difference.
    tcp_pos: where the grasp point ended [x, y, z], metres.
    steps_used: native actions consumed.
    truncated: the episode is over.
    """

    arm: str
    reached: bool
    max_joint_error_rad: float
    tcp_pos: tuple[float, float, float]
    steps_used: int
    truncated: bool


@dataclass(frozen=True)
class Segment:
    """Outcome of ``segment``: one mask from SAM3 in the current image.

    found: SAM3 produced a mask scoring at least ``min_score``.
    centroid: (row, col) of the mask centroid, or None.
    area_px: number of mask pixels.
    score: SAM3 confidence, None when not found.
    box: mask bounding box as SAM3 reports it, or None.
    camera: the camera segmented.
    """

    found: bool
    centroid: tuple[int, int] | None
    area_px: int
    score: float | None
    box: Any
    camera: str


@dataclass(frozen=True)
class WorldPoint:
    """Outcome of ``back_project``: the surface point seen at one pixel.

    xyz: world [x, y, z], metres.
    depth_m: distance along the camera's optical axis.
    camera: the camera the pixel was read from.
    """

    pixel: tuple[int, int]
    xyz: tuple[float, float, float]
    depth_m: float
    camera: str


@dataclass(frozen=True)
class ObjectSight:
    """Outcome of ``locate``: an object measured from its mask and the depth image.

    found: SAM3 produced a mask scoring at least ``min_score``.
    center_xyz: [x, y, z] of the middle of the footprint, at half its height
        above the support surface, metres. None when no 3D measurement was
        possible (see ``note``).
    support_z: the support surface height the footprint was measured against.
    top_z: height of the object's top surface.
    height_m: ``top_z`` above ``support_z``.
    length_m / width_m: the long and the short side of the footprint.
    long_axis_deg: direction of the long side on the table, degrees from +x
        (-90 .. 90).
    grasp_yaw: the ``yaw`` for ``move_to(..., pitch=90, yaw=grasp_yaw)``:
        the fingers then close across the short side.
    score: SAM3 confidence.
    centroid: (row, col) of the mask in the image.
    area_px: mask pixels.
    camera: the camera measured from.
    note: what limits the measurement (no depth, too few points), else None.
    """

    found: bool
    center_xyz: tuple[float, float, float] | None
    support_z: float | None
    top_z: float | None
    height_m: float | None
    length_m: float | None
    width_m: float | None
    long_axis_deg: float | None
    grasp_yaw: float | None
    score: float | None
    centroid: tuple[int, int] | None
    area_px: int
    camera: str
    note: str | None


# ---------------------------------------------------------------------------
# Orientation helpers: the library's pitch/yaw/roll <-> the twin's quaternion
# ---------------------------------------------------------------------------

#: Rotation taking the library's approach axis (x) onto the twin grasp site's
#: approach axis (z). Confirmed against the twin by the primitives smoke test
#: (``scripts/piper_twin_smoke.py``); change here only with that evidence.
_R_Z_FROM_X = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transpose(a: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _rot_from_quat(quat: Sequence[float]) -> list[list[float]]:
    w, x, y, z = (float(v) for v in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _quat_from_rot(r: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        w, x, y, z = (r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        w, x, y, z = (r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        w, x, y, z = (r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s
    return (w, x, y, z)


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _robodojo_from_angles(pitch: float, yaw: float, roll: float) -> list[list[float]]:
    """R with x = pointing direction, y = the line the fingers close along."""
    a, p, r = math.radians(90.0 + yaw), math.radians(pitch), math.radians(roll)
    ca, sa, cp, sp = math.cos(a), math.sin(a), math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    rz = [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]]
    ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    return _matmul(_matmul(rz, ry), rx)


def _angles_from_robodojo(r: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    point = (r[0][0], r[1][0], r[2][0])
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    if math.hypot(point[0], point[1]) < math.sin(math.radians(1.0)):
        jaw = (r[0][1], r[1][1])
        return pitch, _wrap(math.degrees(math.atan2(-jaw[1], -jaw[0]))), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0])) - 90.0
    a, p = math.radians(90.0 + yaw), math.radians(pitch)
    ca, sa, cp, sp = math.cos(a), math.sin(a), math.cos(p), math.sin(p)
    base = [[ca * cp, -sa, ca * sp], [sa * cp, ca, sa * sp], [-sp, 0.0, cp]]
    r21 = sum(base[k][2] * r[k][1] for k in range(3))
    r11 = sum(base[k][1] * r[k][1] for k in range(3))
    return pitch, _wrap(yaw), _wrap(math.degrees(math.atan2(r21, r11)))


def quat_from_angles(pitch: float, yaw: float, roll: float) -> tuple[float, float, float, float]:
    """The twin's grasp-site quaternion [w, x, y, z] for the library's angles."""
    return _quat_from_rot(_matmul(_robodojo_from_angles(pitch, yaw, roll), _R_Z_FROM_X))


def angles_from_quat(quat: Sequence[float]) -> tuple[float, float, float]:
    """The library's (pitch, yaw, roll) degrees of the twin's quaternion."""
    return _angles_from_robodojo(_matmul(_rot_from_quat(quat), _transpose(_R_Z_FROM_X)))


def angle_between_quat(a: Sequence[float], b: Sequence[float]) -> float:
    """Rotation between two quaternions, radians."""
    dot = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
    norm = math.sqrt(sum(float(x) ** 2 for x in a) * sum(float(y) ** 2 for y in b)) or 1.0
    return 2.0 * math.acos(max(-1.0, min(1.0, dot / norm)))


# ---------------------------------------------------------------------------
# The robot
# ---------------------------------------------------------------------------


def _arm(name: Any) -> str:
    if name not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {name!r}")
    return str(name)


def _camera(name: Any, cameras: tuple[str, ...]) -> str:
    if name not in cameras:
        raise ValueError(f"camera must be one of {cameras}, got {name!r}")
    return str(name)


def _floats(value: Any, n: int, name: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {n} numbers, got {value!r}") from exc
    if len(vals) != n or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} must be {n} finite numbers, got {value!r}")
    return vals


def _tuple3(value: Any) -> tuple[float, float, float]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        vals = ()
    if len(vals) != 3:
        return (math.nan, math.nan, math.nan)
    return vals  # type: ignore[return-value]


def _command(value: Any, *, allow_keep: bool) -> int:
    allowed = (-1, 0, 1) if allow_keep else (-1, 1)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if number not in allowed:
        words = "1 (close), -1 (open) or 0 (keep)" if allow_keep else "1 (close) or -1 (open)"
        raise ValueError(f"gripper must be {words}, got {value!r}")
    return int(number)


def _angle(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number of degrees or None, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


class PiperTwinRobot(RobotBase):
    """The Cobot Magic twin: two PiPER arms with parallel grippers at a table.

    ``move_to`` drives an arm's GRASP POINT (the pinch point between the jaws,
    ``state().left.tcp_pos``) to a world position with an orientation in
    degrees: ``pitch`` 0 level and 90 straight down, ``yaw`` the heading of the
    pointing direction on the table, ``roll`` the turn of the fingers about it.
    A pose the arm cannot reach does not move it; ``ik_error`` says why.

    Gripper commands are signed (``robo.CLOSE`` 1 / ``robo.OPEN`` -1 /
    ``robo.KEEP`` 0 for ``move_to``); measurements are openings
    (``state().left.gripper``: 1.0 open, 0.0 closed on nothing).

    Every motion call blocks until the primitive finishes and returns a typed
    result; nothing is queued. Each motion call also records a step: its
    images, calibration and state stay readable through ``image`` and
    ``state(step)``.

    Perception: a head camera (``cam_head``) and one wrist camera per arm
    (``cam_left_wrist``, ``cam_right_wrist``) with calibrated RGB-D.
    ``locate`` measures an object from its SAM3 mask and the depth image;
    ``segment`` / ``back_project`` give a mask centroid and the world point at
    one pixel.
    """

    ARMS: tuple[str, str] = ("left", "right")
    LEFT: str = "left"
    RIGHT: str = "right"
    CLOSE: int = 1
    OPEN: int = -1
    KEEP: int = 0

    NAME: ClassVar[str] = "piper_twin"
    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        State,
        ArmState,
        Move,
        Gripper,
        Home,
        ObjectSight,
        Segment,
        WorldPoint,
    )
    CAMERAS: ClassVar[tuple[str, ...]] = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("cam_head", "high"): "cam_head.png",
        ("cam_left_wrist", "high"): "cam_left_wrist.png",
        ("cam_right_wrist", "high"): "cam_right_wrist.png",
    }
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {}
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = (
        "cam_head.png",
        "cam_left_wrist.png",
        "cam_right_wrist.png",
    )
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = frozenset({"move_pose", "grip", "home"})
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (
        "reached",
        "dist_to_target_m",
        "orientation_error_deg",
        "final_xyz",
        "ik_error",
        "steps_used",
        "max_joint_error_rad",
        "holding",
        "opening",
        "found",
        "score",
        "centroid_rc",
        "center_xyz",
        "top_z",
        "grasp_yaw",
        "world_xyz",
        "truncated",
        "step",
    )

    # -- observation --------------------------------------------------------

    def state(self, step: int = -1) -> State:
        """Read the recorded state of both arms.

        Args:
            step: recorded step to read; ``-1`` is the latest, ``0`` the
                initial scene. A new step is recorded after every motion call.
        Returns:
            State with both arms' grasp points, orientations, measured gripper
            openings, joint angles, and the ``truncated`` flag.
        """
        return super().state(step)

    def image(self, camera: str = "cam_head", *, resolution: str = "high", step: int = -1) -> Any:
        """Load a recorded camera image as a ``numpy`` uint8 array (H, W, 3), RGB.

        Args:
            camera: ``"cam_head"`` (the whole table), ``"cam_left_wrist"`` or
                ``"cam_right_wrist"`` (on the gripper).
            resolution: one resolution per camera, ``"high"``. Pixel
                coordinates are (row, col), row 0 at the top.
            step: recorded step, ``-1`` = latest.
        """
        return super().image(camera, resolution=resolution, step=step)

    def show(self, camera: str = "cam_head") -> None:
        """Attach the current image of ``camera`` to the feedback of this turn.

        Images are not returned automatically; call ``show("cam_head")`` or a
        wrist camera when *you* need to look before writing the next turn.
        Costs tokens: look only when a decision depends on it.
        """
        super().show(camera)

    def camera_meta(self, camera: str = "cam_head", *, step: int = -1) -> dict[str, Any]:
        """Return the calibration of one camera at a recorded step as a dict.

        Keys: ``shape``, ``intrinsic_matrix`` (3x3), ``extrinsic_matrix``
        (4x4 optical-to-world; the camera looks down its own +z axis) and
        ``has_depth``. Wrist cameras move with their arm.
        """
        camera = _camera(camera, self.CAMERAS)
        payload = self._readonly(self.STATE_TOOL, {"step": int(step)})
        cameras = (payload.get("state") or {}).get("cameras") or {}
        return dict(cameras.get(camera) or {})

    def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Return the raw bytes of a recorded artifact (advanced).

        Names are ``cam_head.png``, ``cam_left_wrist.png`` and
        ``cam_right_wrist.png``.
        """
        return super().artifact(name, step=step)

    # -- perception ---------------------------------------------------------

    def segment(
        self,
        text_prompt: str = "",
        camera: str = "cam_head",
        min_score: float = 0.2,
        *,
        point: list[int] | None = None,
    ) -> Segment:
        """Segment an object in the current image of ``camera`` (SAM3).

        Exactly one prompt: a text prompt, or a positive ``point``. On the
        twin's synthetic renders the text head scores near zero; a point
        picked from the image works. Text prompts are for the real robot.

        Args:
            text_prompt: one object by its visible properties, e.g. ``"the
                red block on the left"`` (omit when ``point`` is given).
            camera: ``"cam_head"`` (default), ``"cam_left_wrist"`` or
                ``"cam_right_wrist"``.
            min_score: masks scoring below this are reported as not found.
            point: ``[row, col]`` inside the object (the twin's working prompt).
        Returns:
            Segment; when ``found``, pass ``centroid`` to ``back_project``
            with the same camera to get a world point. The mask is the best
            match for the prompt, not a guarantee of identity.
        Raises:
            ToolError: the image or the SAM3 service is unavailable.
        """
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt or "").strip()
        if bool(prompt) == (point is not None):
            raise ValueError("provide exactly one of text_prompt or point")
        request: dict[str, Any] = {
            "text_prompt": prompt,
            "camera": camera,
            "min_score": float(min_score),
        }
        if point is not None:
            request["point"] = [int(point[0]), int(point[1])]
        p = self._readonly("segment", request)
        centroid = p.get("centroid_rc") or ()
        return Segment(
            found=bool(p.get("found")),
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(p.get("area_px") or 0),
            score=float(p["score"]) if p.get("score") is not None else None,
            box=p.get("box_px"),
            camera=str(p.get("camera", camera)),
        )

    def back_project(self, row: int, col: int, camera: str = "cam_head") -> WorldPoint:
        """Return the world point seen at one pixel of the current image of ``camera``.

        Args:
            row / col: pixel, row 0 at the top, col 0 at the left.
            camera: the camera the pixel was read from.
        Returns:
            WorldPoint: where the pixel's ray meets the measured surface.
        Raises:
            ToolError: pixel out of range, or no valid depth there.
        """
        camera = _camera(camera, self.CAMERAS)
        p = self._readonly("back_project", {"row": int(row), "col": int(col), "camera": camera})
        pixel = p.get("pixel") or (int(row), int(col))
        return WorldPoint(
            pixel=(int(pixel[0]), int(pixel[1])),
            xyz=_tuple3(p.get("world_xyz")),
            depth_m=float(p.get("depth_m", math.nan)),
            camera=str(p.get("camera", camera)),
        )

    def locate(
        self,
        text_prompt: str = "",
        camera: str = "cam_head",
        min_score: float = 0.2,
        *,
        point: list[int] | None = None,
    ) -> ObjectSight:
        """Find an object by a text prompt or a point and measure it in 3D from RGB-D.

        SAM3 segments the current image of ``camera``; the mask's pixels are
        projected through the depth image, the support surface is estimated
        from the same view, and the points above it give the object's centre,
        top, footprint and heading. On the twin's synthetic renders the text
        head scores near zero: use a point picked from the image. Text prompts
        are for the real robot.

        Args:
            text_prompt: one object by its visible properties, e.g. ``"the
                blue block"`` (omit when ``point`` is given).
            camera: ``"cam_head"`` (default) or a wrist camera (close up:
                re-measure before a precise grasp).
            min_score: masks scoring below this are reported as not found.
            point: ``[row, col]`` inside the object (the twin's working prompt).
        Returns:
            ObjectSight. For a top grasp: ``move_to([x, y, top_z + 0.08],
            pitch=90, yaw=grasp_yaw)`` above it, then down to the height to
            hold it at. With one view and no depth the height is not
            measured: ``note`` says so.
        Raises:
            ToolError: the image, the depth image or the SAM3 service is
            unavailable.
        """
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt or "").strip()
        if bool(prompt) == (point is not None):
            raise ValueError("provide exactly one of text_prompt or point")
        request: dict[str, Any] = {
            "text_prompt": prompt,
            "camera": camera,
            "min_score": float(min_score),
        }
        if point is not None:
            request["point"] = [int(point[0]), int(point[1])]
        p = self._readonly("locate", request)
        centroid = p.get("centroid_rc") or ()

        def num(key: str) -> float | None:
            return float(p[key]) if p.get(key) is not None else None

        center = p.get("center_xyz")
        return ObjectSight(
            found=bool(p.get("found")),
            center_xyz=_tuple3(center) if center is not None else None,
            support_z=num("support_z"),
            top_z=num("top_z"),
            height_m=num("height_m"),
            length_m=num("length_m"),
            width_m=num("width_m"),
            long_axis_deg=num("long_axis_deg"),
            grasp_yaw=num("grasp_yaw"),
            score=num("score"),
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(p.get("area_px") or 0),
            camera=str(p.get("camera", camera)),
            note=str(p["geometry_error"]) if p.get("geometry_error") else None,
        )

    # -- motion -------------------------------------------------------------

    def move_to(
        self,
        xyz: Sequence[float],
        arm: str = "right",
        gripper: int = 0,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
        path: str = "auto",
        preview: bool = False,
    ) -> Move:
        """Move one arm's grasp point to a world position, with an orientation.

        Args:
            xyz: target grasp-point position [x, y, z], metres.
            arm: ``"left"`` or ``"right"``. The other arm holds its joints.
            gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1) to command the
                moving arm's gripper on every action of the motion,
                ``robo.KEEP`` (0) to keep the last command (a held object
                stays squeezed).
            pitch / yaw / roll: the orientation to arrive with, degrees. An
                angle left None keeps its measured value.
            path: ``"auto"`` (default) or ``"line"``. The twin walks either in
                short increments; a command that cannot be executed does not
                move the arm.
            preview: solve the pose and return it without moving the robot
                (``Move.preview``; ``final_xyz`` is where the arm would go).
        Returns:
            Move; test ``reached`` (within 0.01 m and 5 degrees).
        Raises:
            ValueError: a malformed argument (nothing is sent).
        """
        arm = _arm(arm)
        target = _floats(xyz, 3, "xyz")
        command = _command(gripper, allow_keep=True)
        if path not in ("auto", "line"):
            raise ValueError(f"path must be 'auto' or 'line', got {path!r}")
        kwargs: dict[str, Any] = {
            "xyz": list(target),
            "arm": arm,
            "gripper": command,
            "path": path,
            "preview": bool(preview),
        }
        for name, value in (("pitch", pitch), ("yaw", yaw), ("roll", roll)):
            angle = _angle(value, name)
            if angle is not None:
                kwargs[name] = angle
        return self._stateful("move_pose", kwargs, lambda p: self._move(arm, target, p))

    def set_gripper(self, arm: str, gripper: int, opening: float | None = None) -> Gripper:
        """Close or open one gripper, without moving the arms.

        After a close, ``holding`` tells whether something stopped the fingers
        apart.

        Args:
            arm: ``"left"`` or ``"right"``.
            gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1).
            opening: with ``robo.OPEN`` only, open to this fraction (0, 1]
                instead of all the way.
        Returns:
            Gripper. ``holding`` is not proof that the INTENDED object is
            held: confirm it moves with the arm.
        """
        arm = _arm(arm)
        command = _command(gripper, allow_keep=False)
        kwargs: dict[str, Any] = {"arm": arm, "gripper": command}
        if opening is not None:
            try:
                fraction = float(opening)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"opening must be a number in (0, 1], got {opening!r}") from exc
            if command != -1 or not 0.0 < fraction <= 1.0:
                raise ValueError("opening goes with robo.OPEN and must be in (0, 1]")
            kwargs["opening"] = fraction

        def build(p: dict[str, Any]) -> Gripper:
            return Gripper(
                arm=arm,
                commanded=str(p.get("gripper") or ("closed" if command > 0 else "open")),
                opening=float(p.get("opening", math.nan)),
                holding=bool(p.get("holding")),
                steps_used=int(p.get("steps_used", 0)),
                truncated=self._latest_state().truncated,
            )

        return self._stateful("grip", kwargs, build)

    def home(self, arm: str = "both") -> Home:
        """Return an arm to the joint posture it had when the episode started.

        The starting posture is replayed as joint targets, so it returns exactly
        (no IK and no path planning beyond that). An arm that is not named holds
        its measured pose; the gripper keeps its command.

        Args:
            arm: ``"left"``, ``"right"`` or ``"both"`` (default).
        Returns:
            Home; test ``reached`` and call again when it is False.
        """
        if arm not in (*self.ARMS, "both"):
            raise ValueError(f"arm must be 'left', 'right' or 'both', got {arm!r}")
        which = str(arm)

        def build(p: dict[str, Any]) -> Home:
            return Home(
                arm=str(p.get("arm", which)),
                reached=bool(p.get("reached")),
                max_joint_error_rad=float(p.get("max_joint_error_rad", math.inf)),
                steps_used=int(p.get("steps_used", 0)),
                truncated=self._latest_state().truncated,
            )

        return self._stateful("home", {"arm": which}, build)

    def move_joints(self, arm: str, joints: Sequence[float]) -> Joints:
        """Drive one arm's six joints to a posture, the other arm holding.

        Joint space is how a skill reaches a pre-grasp posture that the
        Cartesian walk cannot reorient into: ``move_to`` interpolates position
        and orientation together, so a large reorientation from the folded
        boot posture stalls at an unreachable wrist. A fixed, calibrated
        pre-grasp posture has no such problem -- record it once, replay it.

        Args:
            arm: ``"left"`` or ``"right"``.
            joints: the six joint angles in radians, ``[joint1..joint6]``.
        Returns:
            Joints; test ``reached`` and call again when it is False.
        """
        arm = _arm(arm)
        target = _floats(joints, 6, "joints")

        def build(p: dict[str, Any]) -> Joints:
            return Joints(
                arm=str(p.get("arm", arm)),
                reached=bool(p.get("reached")),
                max_joint_error_rad=float(p.get("max_joint_error_rad", math.inf)),
                tcp_pos=_tuple3(p.get("tcp_pos")),
                steps_used=int(p.get("steps_used", 0)),
                truncated=self._latest_state().truncated,
            )

        return self._stateful("move_joints", {"arm": arm, "joints": list(target)}, build)

    # -- internals ----------------------------------------------------------

    def _latest_state(self) -> State:
        if self._last_state is None:
            return self.state()
        return self._last_state

    def _move(self, arm: str, target: tuple[float, ...], p: dict[str, Any]) -> Move:
        return Move(
            arm=str(p.get("arm", arm)),
            target_xyz=_tuple3(p.get("target_xyz", target)),
            start_xyz=_tuple3(p.get("start_xyz")),
            final_xyz=_tuple3(p.get("final_xyz")),
            dist_to_target_m=float(p.get("dist_to_target_m", math.inf)),
            orientation_error_deg=float(p.get("orientation_error_deg", math.nan)),
            reached=bool(p.get("reached")),
            preview=bool(p.get("preview")),
            ik_error=str(p["ik_error"]) if p.get("ik_error") else None,
            note=str(p["note"]) if p.get("note") else None,
            steps_used=int(p.get("steps_used", 0)),
            truncated=self._latest_state().truncated,
        )

    def _state_from_envelope(self, envelope: dict[str, Any]) -> State:
        summary = envelope.get("state") or {}
        raw = summary.get("state") or {}
        arms = raw.get("arms") or {}
        return State(
            step=int(envelope.get("step", -1)),
            task=str(envelope.get("task_language") or summary.get("instruction") or ""),
            left=_arm_state(arms.get("left"), "left"),
            right=_arm_state(arms.get("right"), "right"),
            truncated=bool(envelope.get("truncated")),
        )

    def _summary_dict(self) -> dict[str, Any]:
        state = self.state()
        out: dict[str, Any] = {"step": state.step}
        for name in self.ARMS:
            arm = state.arm(name)
            out[f"{name}_tcp"] = [round(v, 4) for v in arm.tcp_pos]
            out[f"{name}_pitch_yaw_roll"] = [round(arm.pitch, 1), round(arm.yaw, 1), round(arm.roll, 1)]
            out[f"{name}_gripper"] = round(arm.gripper, 3)
        out["truncated"] = state.truncated
        return out


def _arm_state(raw: Any, name: str) -> ArmState:
    raw = raw if isinstance(raw, dict) else {}
    pose = raw.get("control_pose") or {}
    position = _tuple3(pose.get("position"))
    quat = pose.get("quaternion") or (1.0, 0.0, 0.0, 0.0)
    pitch, yaw, roll = angles_from_quat(quat)
    try:
        joints = tuple(float(v) for v in raw.get("joints") or ())
    except (TypeError, ValueError):
        joints = ()
    try:
        gripper = float(raw.get("gripper", math.nan))
    except (TypeError, ValueError):
        gripper = math.nan
    return ArmState(
        tcp_pos=position,
        pitch=round(pitch, 2),
        yaw=round(yaw, 2),
        roll=round(roll, 2),
        gripper=gripper,
        joints=joints,
    )


ADAPTER = RobotAdapter(
    name="piper_twin",
    robot_cls=PiperTwinRobot,
    host="pyrualean.hosts.piper_twin",
    knowledge="piper_twin",
    blurb="a dual-arm AgileX Cobot Magic (two PiPER arms) in its MuJoCo twin",
    guides_subdir="",
    guide_names=(),
    show_examples=(
        "`robo.show('cam_head')`, `robo.show('cam_left_wrist')` or `robo.show('cam_right_wrist')`"
    ),
    services="a SAM3 segmentation service",
    services_novla="a SAM3 segmentation service",
)


__all__ = [
    "ADAPTER",
    "ArmState",
    "Gripper",
    "Home",
    "Move",
    "ObjectSight",
    "PiperTwinRobot",
    "Segment",
    "State",
    "WorldPoint",
    "angles_from_quat",
    "quat_from_angles",
]
