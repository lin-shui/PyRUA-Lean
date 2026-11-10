# piper-drawer round 1: frozen skill

- revision `7c6a703dd827ef0d` (parent `2845e466f77db2ab`, root v0 `e3b0c44298fc1c14`)
- task: open the cabinet drawer, Cobot Magic twin (steward WorldSim via the
  cobot_magic environment service), 2 dev + 2 held-out layouts
- skill: calibrated joint-space pre-grasp (right/left), three-step descent with
  the tool-down attitude, side-aware arm choice, 3 cm pull increments,
  three-witness verification (drawer front, handle, held hand)
- verdict (the twin's own check, drawer_slide >= 0.10 m):
  direct replay a/b/c/d 4/4; 4B policy dev 2/2 (2 cells each), test 2/2
  (5 + 4 cells), every episode claim agreed (no overclaim)
- worker: dsh + deepseek-flash round 1 wrote the skill; the substrate fixes
  (stale scene snapshot, wrist-limit reorientation, attitude drift) are in
  commits c19f2b1..485a83a (branch piper-rsi)
- full campaign evidence: /Users/lilin/game/piper-campaign
  (episodes.jsonl, library/, runs/, videos/)
