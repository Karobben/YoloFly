# Head Binding in `utils/Post_track.py`

This document explains exactly how head assignment works in the current tracking pipeline.

---

## Scope

Head assignment is implemented across:

- `utils/Post_track.py`
- `utils/Head_bind.py`

The tracker first assigns body IDs, then assigns one head per fly ID for each frame.

---

## Input assumptions for head assignment

The detection table is expected to contain:

- class `0`: body detections
- class `1`: head detections
- normalized `x_center, y_center, w, h` in columns `2:6`

For head assignment in frame `f`:

- body state comes from `FLY_matrix[f][fly]["body"]`
- head candidates come from `TB_head = TMP[TMP[1] == 1].iloc[:, 1:6]`

`TB_head` columns used by head bind logic:

- `TB_head.iloc[i, 1]` -> head `x_center`
- `TB_head.iloc[i, 2]` -> head `y_center`
- `TB_head.iloc[i, 3]` -> head `w`
- `TB_head.iloc[i, 4]` -> head `h`

---

## Stage 1: Primary head bind (overlap-based matching)

Primary matching is done by:

```python
head_bind.main(FLY_matrix, frame, TB_head)
```

Inside `head_match.main(...)` in `utils/Head_bind.py`:

1. Build body rectangle from each tracked fly body box.
2. Build head rectangle for each detected head candidate.
3. Compute intersection area between body and head rectangles.
4. Convert to a score:
   - `score = intersection_area / head_area`
5. Keep candidate if `score >= Thread` (default `Thread = 0.7`).
6. Resolve one-to-one assignment in three phases:
   - **Phase A (iterative singleton propagation):**
     - repeatedly assign any body that has exactly one remaining head candidate
     - after each round, remove assigned heads from all other bodies and recount
     - repeat until no single-candidate bodies remain
   - **Phase B1 (global vector-ranking resolver):**
     - collect all unresolved `(body, head)` candidate pairs
     - for each pair, compute cost from previous-frame body->head vector consistency
     - fallback to overlap score when previous vector is unavailable
     - globally sort candidates by cost (then by constrained-body priority)
     - assign best non-conflicting pairs
     - after each assignment, immediately remove that head from all other option lists
   - **Phase B2 (greedy for unresolved bodies):**
     - process remaining bodies with fewer candidates first
     - for each body, pick highest-score unassigned head
7. Save final result to `head_bind.MATCH_result` as:
   - `{fly_id: "head_index_as_string"}`

### Why this works better in crowded scenes

Singleton propagation removes easy ambiguity early, and the global vector-ranking resolver reduces swaps in general crowded cases (not only strict 2-body/2-head patterns) by preferring temporal direction consistency over raw local overlap alone.

---

## Stage 2: Per-fly validation and uniqueness guard

Back in `Post_track.py`, for each fly in the current frame:

1. Read assigned index from `head_bind.MATCH_result[fly]`.
2. Validate index and bounds with `_safe_head_row(tb_head, idx)`.
3. Enforce per-frame uniqueness with `used_head_idx`:
   - if this head index was already used by another fly, treat as bind failure for this fly.

If successful:

- assign detected head directly
- reset fallback streak for this fly
- increment `qa["head_bind_success"]`

If not successful:

- increment `qa["head_fallback"]`
- enter fallback flow

---

## Stage 3: Fallback head prediction (motion-consistent inheritance)

When direct bind fails, the tracker predicts head position from previous frame geometry:

1. From previous frame (`prev_frame`) get:
   - `last_body = FLY_matrix[prev_frame][fly]["body"]`
   - `last_head = FLY_matrix[prev_frame][fly]["head"]`
2. Compute prior relative offset:
   - `rel_pos = [last_head_x - last_body_x, last_head_y - last_body_y]`
3. Apply offset to current body center:
   - `new_head_xy = current_body_xy + rel_pos`
4. Reuse prior head size:
   - `new_head_wh = last_head_wh`
5. Save fallback head:
   - `fallback_head = [new_x, new_y, last_w, last_h]`

This preserves orientation/offset continuity when detector head outputs are noisy or missing.

---

## Stage 4: Reacquire detected head after repeated fallback

If a fly has too many consecutive fallback frames, tracker tries to re-lock to a detected head.

Controlled by CLI:

- `--head-reacquire-after` (default `8`)
- `--head-reacquire-max-dist` (default `0.03`, normalized distance)

Condition:

- `head_fallback_streak[fly] >= HEAD_REACQUIRE_AFTER`

Action:

1. Use `_nearest_head_idx(tb_head, target_xy, used_idx, max_dist)` where:
   - `target_xy` is fallback head center `(fallback_x, fallback_y)`
   - `used_idx` is current-frame assigned head index set
2. `_nearest_head_idx` scans unassigned detected heads and computes:
   - `d = sqrt((hx - tx)^2 + (hy - ty)^2)`
3. Choose nearest candidate if `d <= max_dist`.
4. If valid, replace fallback head with detected head and:
   - add index to `used_head_idx`
   - reset streak to `0`
   - increment `qa["head_reacquire_success"]`

Every reacquire attempt increments:

- `qa["head_reacquire_attempts"]`

---

## Startup frame behavior vs ongoing frames

There are two places where start-frame head assignment appears:

1. Legacy/simple start assignment block (direct try/except assignment)
2. Start-frame QA verification block (`used_head_idx` uniqueness counting)

During normal ongoing frames, the full robust logic runs (bind -> fallback -> optional reacquire).

Practical implication:

- Ongoing frame behavior is stricter and deterministic with uniqueness guard + fallback streak logic.
- Start-frame assignment and start-frame QA counting are not perfectly symmetrical in implementation.

---

## QA counters related to head bind

The tracker writes head-related QA metrics to `<output>.qa.json`:

- `head_bind_success`
- `head_fallback`
- `head_reacquire_attempts`
- `head_reacquire_success`
- `head_reacquire_fail` (computed at end)
- `head_fallback_streak_max` (computed at end)

Interpretation:

- high `head_bind_success`: detector and overlap logic are stable
- high `head_fallback`: frequent missing/ambiguous head detections
- high attempts but low success: reacquire thresholds may be too strict or detections too noisy

---

## Tuning guidance

### `--head-reacquire-after`

- Lower value: reacquires sooner, faster relock, more risk of wrong relock.
- Higher value: more conservative, longer fallback drifting before relock attempt.

### `--head-reacquire-max-dist`

- Lower value: safer relock, may miss valid nearby heads.
- Higher value: more relocks, but higher swap risk in crowded scenes.

Recommended tuning pattern:

1. Keep `after` moderate (`6-10`) first.
2. Sweep `max-dist` from `0.02` to `0.04`.
3. Compare QA: `head_reacquire_success`, `head_reacquire_fail`, and downstream identity stability.

---

## Failure modes to watch

1. **Crowded overlap**: one head candidate overlaps multiple bodies.
2. **Detector dropout**: no head detections for several frames, forcing fallback drift.
3. **Fast turns**: body center moves but true head offset changes quickly, making inherited offset stale.
4. **Start-frame asymmetry**: initial assignment path differs from regular per-frame path.

---

## Minimal pseudocode

```text
for each frame:
  detect bodies -> assign body IDs
  detect heads -> build candidates_by_body from overlap score >= Thread
  assigned_heads = {}
  matches = {}

  # phase A: iterative singleton propagation with recount
  while True:
    remaining = remove already-assigned heads from every body candidate list
    singleton_bodies = bodies with exactly one remaining candidate
    if singleton_bodies is empty:
      break
    assign those singleton pairs
    update assigned_heads

  # phase B1: global vector-ranking on all unresolved options
  options = all unresolved (body, head) candidate pairs
  rank options by vector-consistency cost (fallback: overlap score)
  for option in sorted(options):
    if body/head still unassigned:
      assign
      remove selected head from all other option lists

  # phase B2: greedy resolve leftovers
  process unresolved bodies by fewest remaining candidates first
  for each unresolved body:
    choose highest-score unassigned head
    assign if available

  for each fly:
    if valid unique bound head exists:
      use it; reset fallback streak
    else:
      fallback = previous(head-body offset) + current body center
      use fallback; streak += 1
      if streak >= head_reacquire_after:
        try nearest unassigned detected head within max_dist
        if found: replace fallback and reset streak
  emit CSV + JSON + QA counters
```

---

## Implementation notes

- The recount loop is the critical anti-switch mechanism in crowded scenes. It repeatedly simplifies the matching graph before greedy assignment.
- The vector resolver now runs on all unresolved candidates (global ranking), not only on exact 2-body/2-head pairs.
- Each head assignment is immediately removed from all other option lists to prevent duplicated head use in later ranking/greedy stages.
- Bodies with zero candidates are not assigned in `Head_bind.py`; they are handled downstream by fallback logic in `Post_track.py`.
- When multiple bodies have equal candidate counts in the greedy phase, body order is stabilized by fly ID string (`str(fly)`) for deterministic behavior.

---

## Related files

- `utils/Post_track.py`
- `utils/Head_bind.py`
- `utils/Post_track.md` (full tracker design notes)

---

## Real code (detection and tracking)

### Detection output rows (`detect_2.py`)

```python
# detect_2.py
for *xyxy, conf, cls in reversed(det):
    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    # Always keep confidence in label rows saved to csv outputs.
    cls_id = int(round(cls.item()))
    label_tmp = [str(cls_id)] + [str(i) for i in xywh] + [str(float(conf.item()))]
    Lable_Result.append(" ".join(label_tmp))
```

### Head matching core (`utils/Head_bind.py`)

```python
# utils/Head_bind.py
# ... build candidates_by_body from overlap score >= Thread ...
assigned_heads = set()
matches = {}
remaining_candidates = {
    fly: sorted(candidates_by_body[fly], key=operator.itemgetter(1), reverse=True)
    for fly in candidates_by_body
}

def _remove_assigned_from_options(head_ids):
    remove_set = set(int(h) for h in head_ids)
    for fly in list(remaining_candidates.keys()):
        if fly in matches:
            continue
        remaining_candidates[fly] = [
            (hid, sc) for hid, sc in remaining_candidates[fly]
            if int(hid) not in remove_set
        ]

# Phase A: singleton propagation
# ... assign unique bodies, then:
assigned_heads.add(head_id)
_remove_assigned_from_options([head_id])

# Phase B1: global vector ranking for unresolved candidates
ranked_options = []
for fly in unresolved:
    for head_id, score in remaining_candidates.get(fly, []):
        cost = self._vector_cost_for_pair(FLY_matrix, Num_frame, fly, TB_head, head_id, score)
        ranked_options.append((cost, len(remaining_candidates[fly]), -score, str(fly), int(head_id), fly))
ranked_options.sort()

for _cost, _cand_n, _neg_score, _fly_key, head_id, fly in ranked_options:
    if fly in matches or head_id in assigned_heads:
        continue
    matches[fly] = str(head_id)
    assigned_heads.add(head_id)
    _remove_assigned_from_options([head_id])
```

### Tracking-side head assignment and fallback (`utils/Post_track.py`)

```python
# utils/Post_track.py
TB_head = TMP[TMP[1] == 1].iloc[:, 1:6]
head_bind.main(FLY_matrix, frame, TB_head)
used_head_idx = set()
for fly in FLY_matrix[frame].keys():
    fly_key = str(fly)
    match_idx = None
    try:
        match_idx = int(head_bind.MATCH_result[fly])
    except Exception:
        match_idx = None
    head_row = _safe_head_row(TB_head, match_idx) if match_idx is not None else None
    if head_row is not None and match_idx not in used_head_idx:
        FLY_matrix[frame][fly].update({"head": head_row})
        used_head_idx.add(match_idx)
        head_fallback_streak[fly_key] = 0
        qa["head_bind_success"] += 1
        continue

    qa["head_fallback"] += 1
    head_fallback_streak[fly_key] = head_fallback_streak.get(fly_key, 0) + 1
    last_body = FLY_matrix[prev_frame][fly]['body']
    last_head = FLY_matrix[prev_frame][fly]['head']
    new_body = FLY_matrix[frame][fly]['body']
    rel_pos = [last_head[0] - last_body[0], last_head[1] - last_body[1]]
    rel_pos_new = [rel_pos[0] + new_body[0], rel_pos[1] + new_body[1]]
    fallback_head = rel_pos_new + last_head[2:4]
    FLY_matrix[frame][fly].update({"head": fallback_head})
```
