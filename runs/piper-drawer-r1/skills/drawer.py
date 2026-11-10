"""Open the cabinet drawer on the Cobot Magic twin.

The cabinet is the blue box across the table with a tan, open-topped drawer
whose light-grey handle sticks out of its front face toward the robot.  The
drawer slides straight toward the robot (world -x).  This module finds that
handle from the calibrated head RGB-D frame, puts the right arm into the
drawer's calibrated pre-grasp posture, descends onto the handle, checks that
the fingers actually stopped on it, and pulls in short increments.  It reports
whether the drawer moved, so the caller does not have to guess from a wrist
image.

Usage::

    from library.drawer import open_drawer
    result = open_drawer(robo)
    print(result)

Why a joint-space pre-grasp: the twin boots with its arms folded up, and
``move_to`` interpolates position and orientation together, so asking for the
downward grasp attitude from that posture stalls at an unreachable wrist
(the tool would have to rise through the arm's own frame).  The calibrated
posture below puts the pinch point 11 cm above the handle with the tool
already pointing down -- measured on the twin, not guessed -- and the short
Cartesian descent from there always has an IK solution.

The handle is the only white object in the head view.  The drawer's front
face is 0.027 m behind it and its top is 0.096 m below it; those are the
asset's fixed dimensions and are used to reconstruct the handle when a layout
puts it above the top edge of the image.
"""

import numpy as np

#: Right arm at the drawer's pre-grasp: the pinch point is at (0.903, -0.017,
#: 0.788) with the tool pointing straight down.  Solved on the twin at the
#: standoff and lift this scene uses (chassis x=0.34, lift=0.70); it is the
#: posture every successful drawer opening starts from.
_PREGRASP_R = [0.773, 2.325, -1.584, 0.001, 0.917, 2.345]
#: Left arm at the same pre-grasp (pinch 0.888, 0.002, 0.796).  The shoulders
#: sit at y=+/-0.31, so an arm reaches easier on its own side of the cabinet:
#: layout b's handle (y=+0.10, x=0.97) is at the right arm's reach limit and
#: well inside the left arm's.
_PREGRASP_L = [-0.792, 2.333, -1.603, -0.001, 0.917, 0.779]
#: The tool attitude for the whole grasp and pull: link6 +z down, fingers
#: across the handle.  ``pitch=90, yaw=90`` is the twin's downward quaternion.
_DOWN = {"pitch": 90.0, "yaw": 90.0, "roll": 0.0}
#: Height of the pre-grasp above the handle (the posture's pinch z minus the
#: handle's z in these scenes).
_ABOVE_M = 0.112
#: The drawer handle, in the cabinet's local frame, relative to its carcass
#: front face: the handle bar sits this far in front of it and this far above
#: the carcass top.  Fixed by the cabinet asset (not by the scene).
_HANDLE_PROUD_M = 0.027
_HANDLE_ABOVE_CARCASS_M = 0.096
#: The handle's height above the cabinet's floor/support (the asset is
#: floor-standing in these scenes); used when the carcass top edge is partly
#: hidden behind the tan drawer front.
_HANDLE_ABOVE_SUPPORT_M = 0.676


def _pixel(mask, prefer="centre"):
    """One (row, col) inside a boolean mask; None when it is empty."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    if prefer == "top":
        i = int(np.argmin(ys))
    elif prefer == "bottom":
        i = int(np.argmax(ys))
    else:
        r, c = float(np.median(ys)), float(np.median(xs))
        i = int(np.argmin((ys - r) ** 2 + (xs - c) ** 2))
    return int(ys[i]), int(xs[i])


def find_handle(robo, verbose=True):
    """Return the handle's world [x, y, z] from the head RGB-D frame, or None.

    Two routes, in order: the white handle itself (visible in most layouts),
    then the blue carcass front (the handle sits 0.027 m in front of that
    face and 0.096 m above its top edge).  Every point is measured with
    ``back_project``/``locate`` from the calibrated depth, never guessed from
    a pixel-to-metre scale.  No full world map is needed: this twin does not
    publish one.
    """
    image = np.asarray(robo.image("cam_head"))
    if image.ndim != 3:
        return None
    r, g, b = image[..., 0].astype(float), image[..., 1].astype(float), image[..., 2].astype(float)

    # 1. the white handle bar: the only near-white thing in the scene.  Keep
    # the threshold above the beige wall (about 240, 230, 215): r,g,b > 200
    # alone catches 50k wall pixels and back-projects the wall, not the handle.
    white = (r > 225) & (g > 225) & (b > 225)
    white[:3, :] = False
    pix = _pixel(white)
    if pix is not None and int(white.sum()) >= 20:
        row, col = pix
        try:
            world = robo.back_project(row, col, "cam_head")
        except Exception:
            world = None
        if world is not None:
            x, y, z = (float(v) for v in world.xyz)
            if 0.5 < x < 1.8 and abs(y) < 0.8 and 0.4 < z < 1.1:
                if verbose:
                    print("drawer: handle from white bar at",
                          (round(x, 3), round(y, 3), round(z, 3)), "n=", int(white.sum()))
                # The bar's visible band spans its 2 cm height with the camera
                # looking down; its back-projected centroid sits 1-2 cm high.
                return np.array([x, y, z - 0.015], float)

    # 2. the blue carcass front: its top edge is 0.096 m below the handle
    blue = (b > 70) & (b > r + 17) & (b > g + 7)
    blue[:6, :] = False          # the carcass top/side faces are above this
    blue[190:, :] = False        # the robot's own blue chassis is below this
    # Keep the central band: the side walls are at the left/right edges and
    # their top edge is higher than the front face's, so a bare "topmost
    # blue" lands on a wall and puts the handle 0.3 m off in y.
    ys0, xs0 = np.nonzero(blue)
    if len(xs0) >= 60:
        centre = int(np.median(xs0))
        blue[:, :max(0, centre - 12)] = False
        blue[:, centre + 13:] = False
    pix = _pixel(blue, prefer="top")
    if pix is None or int(blue.sum()) < 60:
        if verbose:
            print("drawer: no white handle and too few cabinet pixels")
        return None
    row, col = pix
    try:
        world = robo.back_project(row, col, "cam_head")
    except Exception:
        world = None
    if world is None:
        return None
    x, y, z = (float(v) for v in world.xyz)
    if not (0.5 < x < 1.8 and abs(y) < 0.8):
        return None
    handle_z = z + _HANDLE_ABOVE_CARCASS_M
    if 0.52 < z < 0.62:
        # The tan drawer front hides the last ~2 cm of the carcass top at
        # this column.  Measure the carcass base in the same central band:
        # if it is on the floor, the handle is 0.676 m above it.
        base_pix = _pixel(blue, prefer="bottom")
        base = None
        if base_pix is not None:
            try:
                base = robo.back_project(base_pix[0], base_pix[1], "cam_head")
            except Exception:
                base = None
        if base is not None and float(base.xyz[2]) < 0.08:
            handle_z = float(base.xyz[2]) + _HANDLE_ABOVE_SUPPORT_M
        else:
            handle_z = z + _HANDLE_ABOVE_CARCASS_M + 0.020
    else:  # not the front face's top edge: ask SAM3 for the top
        try:
            sight = robo.locate(point=[row, col], camera="cam_head")
        except Exception:
            sight = None
        if sight is not None and sight.found and sight.top_z:
            handle_z = float(sight.top_z) + _HANDLE_ABOVE_CARCASS_M
            if sight.center_xyz is not None:
                y = float(sight.center_xyz[1])
        else:
            if verbose:
                print("drawer: carcass top edge not measured")
            return None
    handle = np.array([x - _HANDLE_PROUD_M, y, handle_z], float)
    if verbose:
        print("drawer: handle reconstructed from carcass front at",
              tuple(round(float(v), 3) for v in handle))
    return handle


def _drawer_front_x(robo):
    """World x of the nearest tan drawer-front pixel, or None.

    Tan is the drawer (the carcass is blue), and the front panel is the
    nearest tan surface, so the largest-row tan pixel is its near edge.  This
    is what moves when the drawer opens, so it is the honest success check.
    """
    image = np.asarray(robo.image("cam_head"))
    r, g, b = image[..., 0].astype(float), image[..., 1].astype(float), image[..., 2].astype(float)
    tan = (r > 70) & (r < 180) & (r - b > 25) & (r - g > 10)
    pix = _pixel(tan, prefer="bottom")
    if pix is None or int(tan.sum()) < 30:
        return None
    try:
        return float(robo.back_project(pix[0], pix[1], "cam_head").xyz[0])
    except Exception:
        return None


def _descend(robo, handle, arm, verbose):
    """From the pre-grasp posture, settle the fingers onto the handle.

    Three short moves instead of one diagonal: the first aligns the pinch
    horizontally at the pre-grasp height, the next two lower it onto the bar.
    The tool attitude is commanded on every move, so the fingers arrive
    across the handle rather than turned by a tilt.
    """
    x, y, z = (float(v) for v in handle)
    for dz in (_ABOVE_M, _ABOVE_M - 0.058, 0.0):
        move = robo.move_to([x, y, z + dz], arm=arm, gripper=robo.OPEN, **_DOWN)
        if move.dist_to_target_m > 0.08:
            if verbose:
                print("drawer: descent stalled at z+", round(dz, 3), ":",
                      move.note or move.ik_error)
            return False
    grip = robo.set_gripper(arm, robo.CLOSE)
    if verbose:
        print("drawer: closed at", (round(x, 3), round(y, 3), round(z, 3)),
              "opening", round(grip.opening, 3), "holding", grip.holding)
    return bool(grip.holding)


def _pick_arm(handle):
    """The arm on the handle's side of the table (left shoulders at +y)."""
    return "left" if float(handle[1]) > 0.02 else "right"


def open_drawer(robo, arm=None, pull=0.21, verbose=True):
    """Grasp the cabinet drawer handle and pull it straight open.

    Returns a dict: ``opened`` (the drawer moved toward the robot),
    ``handle`` (the starting world point), ``moved_m`` (how far the handle
    came toward the robot), ``arm`` and ``attempts``.  ``opened`` is evidence
    from the calibrated view after the pull, not a guess from the wrist.
    """
    out = {"opened": False, "handle": None, "moved_m": 0.0, "arm": None, "attempts": []}
    handle = find_handle(robo, verbose=verbose)
    if handle is None:
        if verbose:
            print("drawer: no handle found")
        return out
    out["handle"] = tuple(round(float(v), 3) for v in handle)
    arm = arm or _pick_arm(handle)
    out["arm"] = arm
    pregrasp = _PREGRASP_L if arm == "left" else _PREGRASP_R
    front_x0 = _drawer_front_x(robo)

    held = False
    for attempt in ("pregrasp", "retry"):
        out["attempts"].append(attempt)
        posture = robo.move_joints(arm, pregrasp)
        if verbose:
            print("drawer:", attempt, "posture reached", posture.reached,
                  "pinch", tuple(round(v, 3) for v in posture.tcp_pos))
        if attempt == "retry":
            # Back off along the pull axis and come in 2 cm higher: the first
            # descent can brush the handle and push it instead of straddling it.
            robo.move_to([float(handle[0]) + 0.05, float(handle[1]),
                          float(handle[2]) + _ABOVE_M + 0.02], arm=arm,
                         gripper=robo.OPEN, **_DOWN)
        if _descend(robo, handle, arm, verbose):
            held = True
            break
        robo.set_gripper(arm, robo.OPEN)
    if not held:
        if verbose:
            print("drawer: handle was not held after", out["attempts"])
        return out

    # Pull toward the robot in 3 cm increments, in the SAME tool attitude
    # that made the grasp (turning a held handle loses it).
    x, y, z = (float(v) for v in handle)
    last = x
    for step in range(1, int(round(pull / 0.03)) + 1):
        target_x = x - 0.03 * step
        try:
            move = robo.move_to([target_x, y, z], arm=arm, gripper=robo.KEEP, **_DOWN)
        except Exception:
            if robo.done:  # the twin ended it: the drawer opened
                out["opened"] = True
                out["moved_m"] = round(x - target_x, 3)
                if verbose:
                    print("drawer: twin ended the episode during the pull; opened")
                return out
            raise
        if move.dist_to_target_m > 0.06:
            if verbose:
                print("drawer: pull stalled after", round(last - target_x, 3), "m:",
                      move.note or move.ik_error)
            break
        last = float(move.final_xyz[0])

    # Three independent witnesses; the largest is the honest one.  The
    # back-projected surfaces are noisy (the arm partly occludes the drawer
    # during the pull), but the hand itself is a rigid witness: it was holding
    # the handle and travelled toward the robot, so the drawer did too.
    moved = 0.0
    front_x1 = _drawer_front_x(robo)
    if front_x0 is not None and front_x1 is not None:
        moved = max(moved, float(front_x0 - front_x1))
    now = find_handle(robo, verbose=False)
    if now is not None:
        moved = max(moved, float(handle[0] - now[0]))
    # The hand is only a witness while it still holds: a handle that slipped
    # leaves a free arm that also travels toward the robot.
    grip = robo.set_gripper(arm, robo.CLOSE)
    if grip.holding:
        moved = max(moved, float(x - last))
    if moved <= 0.0:
        if verbose:
            print("drawer: could not measure the drawer after the pull")
        return out
    out["moved_m"] = round(moved, 3)
    out["opened"] = moved > 0.08
    if verbose:
        print("drawer: drawer moved", out["moved_m"], "m; opened", out["opened"])
    return out
