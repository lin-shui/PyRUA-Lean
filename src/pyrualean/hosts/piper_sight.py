# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Perception for the Cobot Magic twin: SAM3 masks measured through RGB-D.

The twin publishes calibrated frames (intrinsics, optical-to-world, metric
depth). This module turns one into a numpy bundle, segments an object with
SAM3 and measures its 3D extent from the depth image. It holds no camera
constants: everything comes from the published calibration, so the same code
serves the physical station once its service publishes the same frames.
"""

from __future__ import annotations

import io
import math
from typing import Any, Callable

import numpy as np

#: Points are taken above this height over the estimated support surface (m).
MIN_ABOVE_SUPPORT_M = 0.005
#: At most this many mask pixels are projected (stride keeps it bounded).
MAX_POINTS = 20000
#: Sampling step over the full frame when estimating the support height.
SUPPORT_STRIDE = 8
#: Radius around the object's footprint used for its support-plane median (m).
SUPPORT_RADIUS_M = 0.25


def _to_rgb(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


def _to_depth(data: bytes) -> np.ndarray:
    return np.asarray(np.load(io.BytesIO(data), allow_pickle=False), dtype=np.float32)


def frame(
    snapshot: dict[str, Any], camera: str, fetch: Callable[[str], bytes]
) -> dict[str, Any]:
    """The current capture of ``camera``: RGB, depth (or None), K and optical-to-world."""

    twin_name = {"cam_head": "head", "cam_left_wrist": "left_wrist",
                 "cam_right_wrist": "right_wrist"}[camera]
    entry = (snapshot.get("cameras") or {}).get(twin_name) or {}
    rgb_ref = (entry.get("rgb") or {}).get("artifact_id")
    if not rgb_ref:
        raise RuntimeError(f"no image for {camera} in the latest capture")
    depth_ref = (entry.get("depth_m") or {}).get("artifact_id")
    intrinsic = entry.get("intrinsic_K")
    extrinsic = entry.get("T_world_from_optical")
    if intrinsic is None or extrinsic is None:
        raise RuntimeError(f"{camera} has no calibration in the latest capture")
    return {
        "rgb": _to_rgb(fetch(rgb_ref)),
        "depth": _to_depth(fetch(depth_ref)) if depth_ref else None,
        "K": np.asarray(intrinsic, dtype=float),
        "T": np.asarray(extrinsic, dtype=float),
    }


def segment(
    sam3: Any,
    rgb: np.ndarray,
    prompt: str | None,
    min_score: float,
    point: list[int] | None = None,
) -> dict[str, Any]:
    """The best SAM3 mask for ``prompt`` or a positive ``point`` (row, col).

    Text prompts are the real robot's path. SAM3's text head scores synthetic
    MuJoCo renders near zero, so on the twin a point picked from the image is
    the working prompt.
    """

    if point is not None:
        result = sam3.segment(
            rgb, point=[int(point[0]), int(point[1])], min_score=float(min_score)
        )
    else:
        result = sam3.segment(rgb, text_prompt=prompt, min_score=float(min_score))
    if not result.found or result.mask is None:
        return {"found": False, "score": result.score, "box": result.box,
                "centroid": None, "area_px": 0}
    mask = np.asarray(result.mask).astype(bool)
    rows, cols = np.nonzero(mask)
    centroid = (int(rows.mean()), int(cols.mean())) if rows.size else None
    return {
        "found": True,
        "score": result.score,
        "box": result.box,
        "mask": mask,
        "centroid": centroid,
        "area_px": int(mask.sum()),
    }


def _project(depth: np.ndarray, K: np.ndarray, T: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    """World points of the pixels with valid depth, and the depth values."""

    d = depth[rows, cols]
    valid = np.isfinite(d) & (d > 0.0)
    rows, cols, d = rows[valid], cols[valid], d[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (cols - cx) / fx * d
    y = (rows - cy) / fy * d
    optical = np.stack([x, y, d], axis=1)
    world = optical @ T[:3, :3].T + T[:3, 3]
    return world, d


def _support_z(
    frame_data: dict[str, Any], center_xy: tuple[float, float] | None = None
) -> float | None:
    """A robust support-surface height near the object (or the whole view).

    The head camera sees the floor as well as the table: a global low
    percentile would report the floor and make every object a metre tall.
    Around the mask, the table dominates the neighbourhood, so its median is
    the surface the object stands on.
    """

    depth = frame_data.get("depth")
    if depth is None:
        return None
    rows, cols = np.mgrid[0 : depth.shape[0] : SUPPORT_STRIDE, 0 : depth.shape[1] : SUPPORT_STRIDE]
    world, _ = _project(depth, frame_data["K"], frame_data["T"], rows.ravel(), cols.ravel())
    if world.shape[0] < 50:
        return None
    if center_xy is not None:
        dx = world[:, 0] - float(center_xy[0])
        dy = world[:, 1] - float(center_xy[1])
        local = world[(dx * dx + dy * dy) <= SUPPORT_RADIUS_M * SUPPORT_RADIUS_M, 2]
        if local.size >= 50:
            return float(np.median(local))
    return float(np.percentile(world[:, 2], 5.0))


def back_project(
    snapshot: dict[str, Any], camera: str, row: int, col: int, fetch: Callable[[str], bytes]
) -> dict[str, Any]:
    data = frame(snapshot, camera, fetch)
    depth = data.get("depth")
    if depth is None:
        raise RuntimeError(f"{camera} has no depth in this episode")
    if not (0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]):
        raise RuntimeError(f"pixel ({row}, {col}) is outside the {depth.shape} image")
    value = float(depth[row, col])
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"no valid depth at pixel ({row}, {col})")
    world, _ = _project(depth, data["K"], data["T"], np.array([row]), np.array([col]))
    return {
        "pixel": (row, col),
        "world_xyz": [float(v) for v in world[0]],
        "depth_m": value,
        "camera": camera,
    }


def locate(
    snapshot: dict[str, Any],
    camera: str,
    prompt: str | None,
    sam3: Any,
    min_score: float,
    fetch: Callable[[str], bytes],
    point: list[int] | None = None,
) -> dict[str, Any]:
    """Measure the object ``prompt`` names (or the segment under ``point``) from RGB-D."""

    data = frame(snapshot, camera, fetch)
    seg = segment(sam3, data["rgb"], prompt, min_score, point=point)
    out: dict[str, Any] = {
        "found": bool(seg["found"]),
        "score": seg.get("score"),
        "centroid_rc": seg.get("centroid"),
        "area_px": seg.get("area_px", 0),
        "camera": camera,
        "center_xyz": None,
        "support_z": None,
        "top_z": None,
        "height_m": None,
        "length_m": None,
        "width_m": None,
        "long_axis_deg": None,
        "grasp_yaw": None,
        "geometry_error": None,
    }
    if not seg["found"]:
        return out
    depth = data.get("depth")
    if depth is None:
        out["geometry_error"] = "no depth image for this episode"
        return out
    mask = seg["mask"]
    rows, cols = np.nonzero(mask)
    if rows.size > MAX_POINTS:
        stride = int(math.ceil(rows.size / MAX_POINTS))
        rows, cols = rows[::stride], cols[::stride]
    world, _ = _project(depth, data["K"], data["T"], rows, cols)
    measured = world.shape[0] >= 20
    mask_center_xy = (
        (float(world[:, 0].mean()), float(world[:, 1].mean())) if measured else None
    )
    support = _support_z(data, mask_center_xy)
    out["support_z"] = support
    if support is None or not measured:
        out["geometry_error"] = "too few measurable points"
        return out
    above = world[world[:, 2] > support + MIN_ABOVE_SUPPORT_M]
    if above.shape[0] < 20:
        out["geometry_error"] = "the mask's points do not stand above the support surface"
        return out
    top = float(above[:, 2].max())
    height = max(0.0, top - support)
    xy = above[:, :2]
    mean = xy.mean(axis=0)
    centered = xy - mean
    covariance = np.cov(centered.T) if centered.shape[0] > 2 else np.eye(2)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    minor = vectors[:, int(np.argmin(values))]
    along = centered @ major
    across = centered @ minor
    length = float(along.max() - along.min())
    width = float(across.max() - across.min())
    angle = math.degrees(math.atan2(major[1], major[0]))
    angle = (angle + 90.0) % 180.0 - 90.0
    out.update(
        center_xyz=[float(mean[0]), float(mean[1]), float(support + height / 2.0)],
        top_z=top,
        height_m=height,
        length_m=max(length, width),
        width_m=min(length, width),
        long_axis_deg=angle,
        # pointing straight down, the fingers close across the short side
        grasp_yaw=(angle + 90.0 + 180.0) % 360.0 - 180.0,
    )
    return out


__all__ = ["back_project", "frame", "locate", "segment"]
