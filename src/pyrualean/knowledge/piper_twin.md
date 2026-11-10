# Operating knowledge for the Cobot Magic twin (two PiPER arms)

No task recipes here: this is how the tools behave in this world.

## Frames, units, robot
- Two 6-DoF PiPER arms named `"left"` and `"right"`, each with a parallel
  gripper and a wrist camera. World frame: metres; orientations in the tools
  are degrees (`pitch`, `yaw`, `roll`); quaternions, where they appear, are
  `[w, x, y, z]`.
- The support surface height and the workspace limits belong to the scenario.
  Read them from `locate` (`support_z`) and from `state()`; never assume a
  table height and never assume a reach envelope — ask the arm.
- `robo.task` is the instruction the episode was started with; it is
  authoritative. Read it before planning.
- Every motion call blocks until the primitive has finished; nothing is
  queued. Each motion call also records a step: `state(step)` and
  `image(camera, step=...)` read it later.

## The grasp point and the gripper's orientation
- `move_to` drives an arm's GRASP POINT: the pinch point between the jaws,
  `state().left.tcp_pos`. `state()` reports the point the arm actually
  reached, not the one asked for.
- The orientation is three angles:
  - `pitch`: how far the gripper points below the horizontal; 0 = level,
    90 = straight down.
  - `yaw`: the heading of the pointing direction on the table, degrees.
  - `roll`: the turn of the fingers about the pointing direction.
  An angle you leave out keeps its measured value: a plain position move does
  not turn the gripper.
- The twin's wrist has a limited range: a strictly vertical tool cannot be
  held over most of the table, and the twin's solver keeps the grasp point and
  tilts the tool instead. `Move.note` says when that happened and by how much;
  a small tilt (up to 0.2 rad) is accepted, and a pose that needs more is
  refused with `Move.ik_error`. Do not keep re-asking: change the position,
  the pitch, or the arm -- or start from a posture that already has the tool
  pointing the right way (`move_joints`).
- If the pose is out of reach, nothing moves and `Move.ik_error` says why.
  Do not repeat the same pose; change the position, the pitch, or the arm.
- With an EMPTY open gripper the two fingers are interchangeable, so a pose
  turned half a turn about the pointing direction is the same grasp. Holding
  something, the pose is taken exactly as asked.

## Motion
- `move_to(..., path="line")` walks the grasp point in short straight
  increments (at most 5 cm and about 0.3 rad per step) and stops when it is
  within 1 cm and 5 degrees. Use `path="line"` for every approach, descent and
  lift near objects: the arm must not sweep through anything.
- `move_to(..., preview=True)` solves the pose and returns it **without
  moving**: `Move.final_xyz` is the pose the arm would reach. Use it before a
  risky descent when the route or reachability is uncertain.
- Test `Move.reached` (within 1 cm and 5 degrees). A move that ends short is
  not a completed grasp approach; correct the target from fresh `state()` and
  the wrist image.
- The other arm holds its joints during a motion. Move one arm at a time and
  keep the idle arm out of the working arm's way.
- To turn something held, `move_to` the same point with another `roll` (or,
  pointing straight down, another `yaw`).
- `move_joints(arm, joints)` drives the six joint angles directly (radians,
  `[joint1..joint6]`) and returns `Joints`: `reached`, `max_joint_error_rad`,
  `tcp_pos` of the grasp point it ends at, `steps_used`. The idle arm holds.
  Use it for a LARGE tool reorientation: `move_to` interpolates position and
  orientation together, so a big rotation from the folded posture can stall at
  an unreachable wrist; a recorded pre-grasp posture has no such problem, and
  the short `move_to` descent from it does. Find a posture once by moving the
  arm there (or by reading `state().arm(arm).joints` at a good spot) and replay
  it; do not guess joint angles.

## Gripper
- COMMANDS are signed: `robo.CLOSE` (1) closes, `robo.OPEN` (-1) opens,
  `robo.KEEP` (0) leaves the last command (`move_to` only).
- MEASUREMENTS are openings: `state().left.gripper` is 1.0 open and 0.0 closed
  on nothing.
- After a close, `Gripper.holding` says the fingers stopped apart: something is
  between them. It does not say WHAT. Confirm with the wrist camera, or lift a
  little and `locate` the object again.
- `set_gripper(arm, robo.OPEN, opening=0.5)` opens to a fraction when the full
  stroke would push neighbours apart.

## Perception
- Three calibrated cameras: `cam_head` (the whole table), `cam_left_wrist` and
  `cam_right_wrist` (on each gripper). Each has RGB and metric depth;
  `camera_meta` returns the intrinsics and the optical-to-world pose.
- **On this twin, prompt SAM3 with a POINT, not text.** Its synthetic renders
  score near zero on the text head (`score` < 0.05), while a point prompt
  scores ~0.97. Read `image(camera)`, pick a pixel inside the object, then call
  `locate(point=[row, col], camera=...)` or `segment(point=[row, col], ...)`.
  `row 0` is the top, `col 0` the left. Check `found`, `box_px` and the mask
  area; a point on the robot's own arm segments the arm.
- `locate(point=[row, col])` measures the object from its mask and the depth
  image: `center_xyz`, `top_z`, `height_m`, `length_m`, `width_m`,
  `long_axis_deg`, `grasp_yaw`, and the support height `support_z`.
  - For a top grasp: `move_to([x, y, top_z + 0.08], pitch=90,
    yaw=grasp_yaw)`, then `locate` again from the wrist camera (it then looks
    straight down and the reading is closer), then go down to the height to
    hold the object at.
  - `note` (or a missing `center_xyz`) says what limited the measurement: no
    depth, too few points, or a mask that does not stand above the support.
  - The mask is the segment under the point, not proof of identity. Check it
    against the image before committing.
- A text prompt (`locate("the bowl")`) is the real robot's path; on the twin it
  usually reports `found=False` with a low score. Do not repeat it.
- `back_project(row, col, camera)` returns the exact world point at one pixel
  (it uses the depth image; there is no guessing from a plane).
- `segment` gives a mask centroid and box without a 3D measurement; prefer
  `locate` when you need the object's extent.
- A held object is often hidden from the head camera; look from the wrist
  camera of the holding arm.

## Budget and ending
- Every native step (one increment of a motion, one gripper stroke) counts
  against the episode's step budget. `state().truncated` says the episode is
  over; further motion calls raise `EpisodeFinished`. Long approach routes
  cost steps: plan them, do not wander.
- `home(arm)` returns an arm to the joint posture it started in (replayed as
  joint targets; no path planning beyond that). Most task checks reward leaving
  the scene tidy, and the next episode starts from `home`; return the arms when
  you are done.
- Keep each cell short (a few dozen lines). A reply that reaches the output
  token limit is cut off mid tool call, so it cannot run: it costs the turn
  and the episode's clock.
