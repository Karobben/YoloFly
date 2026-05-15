# Main Fly Body Tracking in `utils/Post_track.py`

This document explains how the main fly body tracking works (ID continuity over frames), independent of head assignment.

---

## Scope

Main body tracking is implemented in:

- `utils/Post_track.py`

This document focuses on:

- body detection selection
- fly ID assignment across frames
- overlap correction
- missing-fly recovery
- state/cache updates and outputs

Head matching is covered separately in:

- `documentation/tracking/head_bind.md`

---

## Input assumptions

Expected detection table columns:

- `0`: frame index
- `1`: class (`0` = body, `1` = head)
- `2,3,4,5`: normalized `x_center, y_center, w, h`
- optional `6`: confidence (if missing, code inserts `1.0`)

Body tracking uses class `0` rows only for **candidate** bodies. Class `4` (wing) and class `5` (mount) rows on the **same frame** are still read from `TB` for protection and mount-pair logic.

---

## Initialization (start frame)

At tracking start:

1. Select rows at `Start` frame.
2. Keep class `0` body detections.
3. Keep top `--num-fly` by confidence.
4. Assign fixed IDs: `fly_0`, `fly_1`, ...
5. Save first frame to output CSV.
6. Initialize state:
   - `S_TMP_B`: previous tracked bodies
   - `Dots_from`: previous body centers (`x,y`)
   - `TB_cache`: sliding body-history table
   - `FLY_matrix`: per-frame dict used for JSON output

---

## Per-frame body tracking pipeline

For each next frame, `Post_track.py` runs this sequence:

1. **Select body candidates**
2. **ID matching (two-step nearest pairing)**
3. **Mount-aware ID correction** (optional swap for a class-5 mount pair; see below)
4. **Box-size overlap correction**
5. **Missing object recovery** (mount-relative path when applicable, else `Obj_los_test`)
6. **Write frame CSV rows**
7. **Update runtime state/cache**

The progress labels in code are:

- `select-body`
- `id-match` (immediately after this step, mount-aware ID swap runs before the next progress update)
- `box-adjust`
- `missing-recover`
- `save-csv`
- (then head stage)
- `flush-cache`

---

## 1) Select body candidates

Per frame:

- filter to class `0` rows
- keep top `num_fly` by confidence
- initialize `ID = None`, `find = True`

Then remove suspicious oversized merged boxes:

- compute normalized area (scaled by runtime frame size)
- if area is much larger than mean (`Box_size_check_d`), check overlap with others
- likely merged boxes are dropped before ID matching, **except protected cases**:
  - **c4 wing-safe protection**: if class-4 overlaps body by >= 85%, keep that body
  - **c5 mount-aware protection**: if class-5 (mount region) suggests reliable one/two-fly structure, keep overlapping body rows

Current policy for large-box filtering:

1. mark oversized body candidates (`area / mean_area > Box_size_check_d`)
2. build protection set using class-4 and class-5 checks
3. apply `Clear_lst` only to oversized rows that are **not** protected

---

## 2) ID matching with two-step nearest pairing

Core matcher: `Dots_Sort(points1, points2)`

- computes full pairwise center-distance matrix (`cdist`)
- greedily picks globally shortest unmatched pairs
- returns index mapping between previous and current points

Two-step assignment logic:

- **Step 1**: match from previously reliable rows (`S_TMP_B.find == True`)
- **Step 2**: match remaining unmatched IDs from less-reliable rows (`find == False`)

Why two steps:

- it gives priority to stable tracks before trying harder/recovered objects

---

## 2b) Mount-aware ID correction (after matching)

When two flies are mounting, greedy nearest-center matching often assigns the **wrong** label to each of the two current detections (IDs crossed). After step 2, the script loads the **previous** frame’s full detection slice (`TB` at `prev_frame`, including class `5` mount regions).

For each **mount pair** found on the previous frame (same rules as c5 mount protection: bodies overlapping class `5` with either safe relative size or opposite body→nearest-head orientations; if more than two bodies overlap one c5, the top two by overlap are used):

- If **both** fly IDs from that pair appear on the current `TMP_B`, compare total center displacement in two ways: keep labels vs **swap** the two IDs on those rows.
- If swapping yields strictly lower sum of distances to the previous frame’s centers, the two `ID` values are exchanged (`_maybe_swap_mount_pair_ids`).

This uses the same geometric idea as the mount-relative recovery vector: prefer the assignment that best preserves each fly’s displacement from the last frame.

---

## 3) Box-size overlap correction

After provisional ID assignment:

- recompute per-row area
- for oversized assigned objects, call `Overlap_test(ob_ls, TMP_B)`
- if overlap issue is detected, recover prior size from `TB_cache` and shrink width/height (`*0.9`)

Goal:

- reduce ID corruption from merged detections (two flies in one large box)

---

## 4) Missing object recovery

If matched IDs are fewer than expected (`len(S_TMP_B)`):

- identify missing fly IDs (one at a time until the count is satisfied)

**Mount-relative recovery (preferred when it applies)**

If the missing fly belonged to a **mount pair** on the previous frame (`_iter_mount_pairs_prev_frame` on `S_TMP_B` and detections at `prev_frame`, including class `5`), and its **partner** fly ID is already matched on the current frame:

- predicted normalized center:  
  `partner_center_now + (lost_center_prev - partner_center_prev)`  
  (clamped to `[0, 1]` for x and y)
- width and height are taken from the lost fly’s last tracked row (`S_TMP_B`)
- row is appended with `find = False`
- `Obj_los_test` is **not** called for that missing ID

If the partner is also missing or the fly was not in a detected mount pair, fall back to `Obj_los_test(frame, ob_ls, cap, prev_frame)` as before.

**`Obj_los_test` recovery modes**

- `CroLst`: recover from cropped-image drift estimate
- `Overlap`: recover from overlap-based inference and cache history

Recovered row behavior (generic paths):

- inject recovered body row into current frame (`TMP_B`)
- default: keep body center at last known location (no drift shift applied), except mount-relative path above
- set `find = False` (less reliable, used in second matching tier later)
- re-run overlap adjustment safeguard if needed

This enables continuity through temporary detector dropouts or occlusion, with better behavior when mounting and one body drops out briefly.

---

## 5) Write output rows

Before writing:

- remove rows still lacking IDs (`TMP_B = TMP_B[TMP_B.ID.isna() == False]`)

Then append to tracked CSV.

Body data is also copied into `FLY_matrix[frame][fly]["body"]` for JSON emission and downstream head stage.

---

## 6) Update state/cache

After each frame:

- append `TMP_B` to `TB_cache`, keep last ~10 frames
- set `S_TMP_B = TMP_B`
- set `Dots_from = Dots_to`
- set `prev_frame = frame`
- keep `FLY_matrix` as sliding window of recent frames (last 10)

These caches are the memory that supports matching and recovery.

---

## QA metrics tied to body tracking

Relevant counters in `<output>.qa.json`:

- `match_pairs_step1`
- `match_pairs_step2`
- `missing_events`
- `missing_objects_total`
- `lost_recovery_crolst`
- `lost_recovery_overlap`
- `lost_recovery_total`
- `overlap_adjustments`
- `mount_id_swap_fixes`: times a mount pair’s two IDs were swapped after matching to minimize displacement from the previous frame
- `mount_relative_recovery`: missing bodies placed using partner-relative offset (mount context) instead of `Obj_los_test`
- `frames_tracked`

Interpretation:

- high `missing_events` / `missing_objects_total`: detector instability or frequent occlusion
- high `overlap_adjustments`: frequent merged-box situations
- high `lost_recovery_total`: tracker depends heavily on recovery logic
- non-zero `mount_id_swap_fixes` / `mount_relative_recovery`: mount heuristics engaged; inspect video if counts seem high relative to true mounting

---

## Practical risks and behavior limits

1. **Crowding sensitivity**: nearest-center matching can swap IDs when bodies cross tightly (mount-aware swap mitigates this only for class-5 mount pairs).
2. **Merged detection instability**: overlap correction is heuristic and may under/over-correct.
3. **Recovery dependency**: when gaps are long, fixed-position recovery can lag true motion; mount-relative recovery assumes the partner’s center is correct and the inter-fly offset is stable across one frame.
4. **State-window limits**: short cache windows can reduce robustness for long occlusions.

---

## Minimal pseudocode

```text
initialize IDs at start frame from top-confidence bodies

for each next frame:
  select top-confidence body detections
  mark oversized body boxes
  protect reliable oversized boxes via c4 and mount-aware c5 checks
  drop only unprotected oversized boxes

  step1 match: previous find==True -> current
  step2 match: previous find==False -> remaining current

  if previous frame had class-5 mount pair and both IDs exist on current frame:
    optionally swap the two IDs if that reduces total displacement vs previous centers

  adjust oversized assigned boxes using overlap + history cache

  if any IDs missing:
    for each missing id: if mount pair + partner matched -> predicted center from relative offset to partner
    else recover via Obj_los_test; keep last known center unless overlap branch
    inject recovered rows (find=False)
    run overlap adjustment check

  drop unassigned rows
  append frame rows to tracked CSV

  update FLY_matrix body entries
  append JSON chunk
  update TB_cache, S_TMP_B, Dots_from, prev_frame
```

---

## Real code excerpts

### Greedy nearest pairing (`Dots_Sort`)

```python
def Dots_Sort(points1, points2):
    if len(points1) == 0 or len(points2) == 0:
        return pd.DataFrame(columns=[0, 1])
    distances = cdist(points1, points2)
    sorted_indices = np.argsort(distances, axis=None)
    paired_points1 = set()
    paired_points2 = set()
    pairs = []
    for index in sorted_indices:
        i1, i2 = np.unravel_index(index, distances.shape)
        if i1 not in paired_points1 and i2 not in paired_points2:
            paired_points1.add(i1)
            paired_points2.add(i2)
            pairs.append((i1, i2))
    Dots = pd.DataFrame(pairs)
    return(Dots)
```

### Two-step ID assignment + missing recovery

```python
TMP_B = select_top_confidence_bodies(TMP[TMP[1] == 0], Num)
TMP_B['ID'] = None
TMP_B['find'] = True

Dots = Dots_Sort(Dots_from[S_TMP_B.find], Dots_to)
for i in range(len(Dots)):
    TMP_B.ID.iloc[Dots[1][i]] = S_TMP_B.ID[S_TMP_B.find].iloc[Dots[0][i]]

Dots = Dots_Sort(Dots_from[~S_TMP_B.find], Dots_to[TMP_B.ID.isna()])
mask = TMP_B[TMP_B.ID.isna()]
for i in range(len(Dots)):
    TMP_B.ID.iloc[TMP_B.index == mask.iloc[Dots[1][i]].name] = S_TMP_B.ID[~S_TMP_B.find].iloc[Dots[0][i]]

matched_count = TMP_B.ID.notna().sum()
if matched_count < len(S_TMP_B):
    ob_ls = S_TMP_B.ID[S_TMP_B.ID.isin(TMP_B.ID) == False].iloc[0]
    Lost = Obj_los_test(frame, ob_ls, cap, prev_frame)
    ...
```

### Cache/state update

```python
TB_cache = pd.concat([TB_cache, TMP_B])
TB_cache = TB_cache[TB_cache[0].isin(TB_cache[0].unique()[-10:])]

S_TMP_B = TMP_B
Dots_from = Dots_to
prev_frame = frame
```

---

## Related files

- `utils/Post_track.py`
- `documentation/tracking/head_bind.md`
- `documentation/tracking/Post_track.md`
