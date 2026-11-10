#!/usr/bin/env python3
"""Build the twin's drawer scenarios in a steward data root.

    STEWARD_TWIN_ROOT=/home/lilin/game/steward/var/steward \\
        $PIPER_TWIN_PYTHON scripts/make_piper_drawer_scenario.py

Publishes a room, a cabinet-with-drawer asset and the twin's robot asset (the
current ``cobot_magic`` kinematics plus the three cameras the environment
service expects: head, left wrist, right wrist), then composes one scene per
layout. The cabinet stands in front of the twin, its drawer facing the robot
(yaw pi); the drawer's handle is 0.27 m in front of the cabinet centre at
z=0.676 m. The robot parks at the standoff the twin's own drawer skill uses
(``0.58 - ARM_MOUNT_X`` = 0.34 m) with the lift at 0.60 m, putting the handle
0.31-0.39 m from the arm mount -- inside the Piper's reach with a downward
tool.

Rebuilding matters: a scene carries a snapshot of the robot asset's
kinematics. A scene composed before the lift joint gained its 0.24 m forward
arm-mount offset stays 0.24 m off the IK model even though the model and the
adapter agree with each other, so every commanded pose lands behind its
target. Re-running this script republishes the current asset and recomposes
the scenes.

Idempotent: re-running republishes the same assets and overwrites the scenes
and scenarios.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

#: The robot's boot pose: the same standoff the twin's drawer skill navigates
#: to (``0.58 - ARM_MOUNT_X``).  The camera calibration moves with it.
ROBOT_X = 0.34
#: The lift height the drawer skill uses: the handle is reachable with a level
#: tool at 0.60 m.
LIFT = 0.60
#: Layout name -> cabinet centre (x, y).  Dev a/b, held-out test c/d.
LAYOUTS = {
    "a": (1.16, -0.10),
    "b": (1.24, 0.10),
    "c": (1.18, 0.12),
    "d": (1.22, -0.12),
}

#: The cameras the environment service expects, mounted on the current robot
#: template.  The stock ``cobot_magic`` generator makes no sensors; these
#: mounts come from the published twin asset and are fixed by the contract
#: (``uni_robot_agent.integrations.cobot_magic.environment.adapter.CAMERAS``).
CAMERAS = {
    "frame": [
        {
            "id": "top_camera",
            "kind": "rgbd",
            "intrinsics": {"fovy": 60.0, "height": 480, "width": 640},
            "mount": {
                "p": [0.42, 0.05, 0.7450000000000001],
                "q": [0.7071067811865476, 0.0, 0.7071067811865475, 0.0],
            },
            "rateHz": 10,
        }
    ],
    "L_link6": [
        {
            "id": "L_wrist_camera",
            "kind": "rgbd",
            "intrinsics": {"fovy": 60.0, "height": 480, "width": 640},
            "mount": {
                "p": [0.0, 0.075, 0.03],
                "q": [0.6994450614237633, 0.10562663504969944,
                      -0.10571078187807806, 0.6988882962338909],
            },
            "rateHz": 10,
        }
    ],
    "R_link6": [
        {
            "id": "R_wrist_camera",
            "kind": "rgbd",
            "intrinsics": {"fovy": 60.0, "height": 480, "width": 640},
            "mount": {
                "p": [0.0, 0.075, 0.03],
                "q": [0.6994450614237633, 0.10562663504969944,
                      -0.10571078187807806, 0.6988882962338909],
            },
            "rateHz": 10,
        }
    ],
}


def _parse_layout(text: str) -> tuple[str, float, float]:
    key, _, xy = text.partition("=")
    x, _, y = xy.partition(",")
    return key, float(x), float(y)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("STEWARD_TWIN_ROOT", "")),
        help="steward data root (default: $STEWARD_TWIN_ROOT)",
    )
    parser.add_argument("--prefix", default="piper-drawer", help="scene name prefix")
    parser.add_argument(
        "--layouts",
        nargs="+",
        type=_parse_layout,
        default=[(k, *v) for k, v in LAYOUTS.items()],
        metavar="KEY=X,Y",
        help="cabinet centre per layout (default: dev a/b, test c/d)",
    )
    parser.add_argument("--robot-x", type=float, default=ROBOT_X, help="chassis x (m)")
    parser.add_argument("--lift", type=float, default=LIFT, help="torso lift (m)")
    args = parser.parse_args()
    if not str(args.root) or not args.root.is_dir():
        parser.error("--root (or STEWARD_TWIN_ROOT) must name the steward data root")

    from steward.assets.cobot import cobot_magic
    from steward.assets.generators import GENERATORS
    from steward.authoring.service import AuthoringService
    from steward.world.sim import WorldSim

    GENERATORS.setdefault("cobot_magic", cobot_magic)
    service = AuthoringService(args.root)

    def publish(name: str, generator: str, params: dict) -> None:
        created = service.create_asset(generator=generator, params=params)
        service.publish_asset(
            name, created["template"], meta={"generator": generator, "params": params}
        )

    publish(f"{args.prefix}-room", "building", {"width": 5.0, "depth": 4.0, "height": 2.6})
    publish(
        f"{args.prefix}-cabinet",
        "cabinet_with_drawer",
        {"width": 0.6, "depth": 0.5, "height": 0.8, "drawer_height": 0.2},
    )

    # The current kinematics, with the contract's cameras mounted on it.  A
    # scene keeps its own copy of the robot, so this template must be the one
    # the IK model has (the lift joint at ARM_MOUNT_X, 0.24 m ahead).
    robot = cobot_magic().to_dict()
    for link in robot["body"]["links"]:
        sensors = CAMERAS.get(link["id"])
        if sensors:
            link["sensors"] = [dict(sensor) for sensor in sensors]
    service.publish_asset(
        "cobot-magic", robot, meta={"generator": "cobot_magic", "params": {},
                                    "cameras": "twincam"}
    )

    built = []
    for key, cabinet_x, cabinet_y in args.layouts:
        name = f"{args.prefix}-{key}"
        service.compose_scene(
            {
                "name": name,
                "entities": [
                    {"id": "w:room", "asset": f"{args.prefix}-room", "at": {"x": 0.0, "y": 0.0}},
                    {"id": "w:cobot-1", "asset": "cobot-magic",
                     "at": {"x": args.robot_x, "y": 0.0}},
                    {
                        "id": "w:cabinet-1",
                        "asset": f"{args.prefix}-cabinet",
                        "at": {"x": cabinet_x, "y": cabinet_y, "yaw": math.pi},
                    },
                ],
                "regions": [
                    {
                        "id": "flat",
                        "label": "flat",
                        "polygon": [[-2.5, -2], [2.5, -2], [2.5, 2], [-2.5, 2]],
                    }
                ],
            }
        )
        # Compose installs the asset's boot posture (lift at LIFT_HOME); this
        # scene's skill expects the drawer height instead.
        path = args.root / "scenes" / f"{name}.json"
        scene = json.loads(path.read_text(encoding="utf-8"))
        robot_entity = next(e for e in scene["entities"] if e["role"] == "robot")
        for joint in robot_entity["body"]["joints"]:
            if joint["id"] == "lift":
                joint["q"]["value"] = [args.lift]
        path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        service.save_scenario(
            {
                "name": name,
                "scene": name,
                "robots": ["w:cobot-1"],
                "task": "Open the cabinet drawer.",
                "mission": "drawer-open",
                "view": {"lookat": [0.7, 0.0, 0.8], "distance": 1.8,
                         "azimuth": 90.0, "elevation": -30.0},
            }
        )
        loaded = service.load_scenario(name)
        sim = WorldSim(loaded["sceneState"])
        cabinet = next(e for e in sim.world_state()["entities"] if e["id"] == "w:cabinet-1")
        slide = next(j for j in cabinet["body"]["joints"] if j["id"] == "drawer_slide")
        built.append({"scenario": name, "cabinet": [cabinet_x, cabinet_y],
                      "drawer_slide": slide["q"]["value"]})
        sim.stop()

    print(json.dumps({"built": built, "scenarios": service.list_scenarios()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
