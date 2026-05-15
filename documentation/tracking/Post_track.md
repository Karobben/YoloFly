# `utils/Post_track.py` Tracking Design Notes

This document explains how `utils/Post_track.py` works today: inputs, execution modes, matching logic, recovery logic, and outputs.

---

## 1) Purpose

`Post_track.py` takes a per-frame detection table (typically from `detect_2.py`) and assigns stable fly IDs over time.

It produces:

- a tracked CSV (`--output`) with fly IDs appended
- a tracking JSON stream (`<output>_<start_frame>_.json`) containing per-frame body/head data

The script supports:

- **single-process tracking**
- **window-parallel tracking** (`--workers > 1`)
- **split-region tracking** (`--split-x` / `--split-y`)

---

## 2) Inputs and expected format

Required CLI args:

- `--input`: detection table (`.csv`/whitespace text, or `.npy`)
- `--output`: tracked CSV path
- `--video`: source video path

Important optional args:

- `--num-fly` (default `12`): expected tracked bodies per frame
- `--test` / `--test-frames`: process only the first N frames (default: all frames)
- `--initial-frame`: start tracking from this frame index (must exist in input detections)
- `--initial-results`: optional manually corrected detection file; rows at `--initial-frame` replace the input detections at that frame before tracking starts
  - accepted formats:
    - frame-first table: `frame class x y w h [conf]`
    - snapshot label file: `class x y w h [conf]` (script auto-assigns `frame = --initial-frame`)
- `--workers`, `--window-overlap`: window-parallel mode
- `--split-x`, `--split-y`, `--frame-width`, `--frame-height`: region-split mode
- `--head-reacquire-after` (default `8`): consecutive fallback-head frames before trying nearest-head reacquire
- `--head-reacquire-max-dist` (default `0.03`): normalized XY threshold for nearest-head reacquire
- `--qa-report`: optional QA JSON output path (default `<output>.qa.json`)
- `--log-level` (default `INFO`): structured log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--log-every` (default `1`): emit one per-frame summary log every N frames
- `--no-progress`: disable rich progress bars

Detection table columns (normalized coordinates):

- `0`: frame
- `1`: class (body is `0`, head is `1`; class `4` = wing region, class `5` = mount region — used for large-box protection and mount-pair logic, not as tracked bodies)
- `2,3,4,5`: `x_center`, `y_center`, `w`, `h`
- optional `6`: confidence (if missing, script inserts `1.0`)

---

## 3) Startup and validation

At startup, the script:

1. validates required arguments and input/video existence
2. loads the detection table via `load_detection_table()`
3. sorts frame indices, optionally starts from `--initial-frame`, then keeps only the first `--test` frames from that start if `--test` is provided
4. if `--initial-results` is set, replaces the input detections at the selected start frame with manual corrected detections from that file
5. opens video and reads runtime dimensions (`TRACK_FRAME_WIDTH`, `TRACK_FRAME_HEIGHT`)
6. initializes structured logs and progress bars for major loops (regions/windows/frames + pipeline steps) unless `--no-progress` is set

These runtime dimensions are then used in area/crop geometry logic.

---

## 4) Execution modes

## 4.1 Split-region mode (`run_split_region_tracking`)

Triggered when split boundaries are provided and internal flags are not set.

Flow:

1. Build region grid from split boundaries.
2. For each region, filter detections spatially.
3. Run a subprocess of `Post_track.py` for that region (`--internal-region`).
4. Remap region-local IDs to globally unique IDs.
5. Merge tracked CSV rows and JSON by frame.

This is useful when multiple arenas/dishes are in one frame.

## 4.2 Window-parallel mode (`run_windowed_tracking`)

Triggered when `--workers > 1` (and not internal window mode).

Flow:

1. Partition frames into overlapping windows (`build_windows`).
2. Run one subprocess per window (`--internal-window`) via `ThreadPoolExecutor`.
3. For each window result, map IDs using overlap with previous window (`stitch_id_map`).
4. Keep core frames from each window and merge into final CSV/JSON.

This speeds tracking while trying to keep ID continuity across windows.

## 4.3 Single-process mode

If split/window modes are not chosen, the script tracks frame-by-frame in one process.

---

## 5) Single-process tracking algorithm

Main state:

- `S_TMP_B`: previous tracked body rows
- `Dots_from`: previous body centers
- `TB_cache`: sliding history of recent tracked rows (last ~10 frames)
- `FLY_matrix`: per-frame dict of fly body/head for JSON output

Per frame:

1. **Select body detections**
   - filter class `0`
   - keep top `--num-fly` by confidence (`select_top_confidence_bodies`)

2. **Remove likely merged/oversized boxes**
   - compute box area
   - detect suspicious overlap boxes by polygon intersection ratios

3. **Two-step nearest matching**
   - step 1: assign IDs to rows predicted as clean (`find=True`)
   - step 2: fill remaining IDs from unmatched previous rows
   - matching uses greedy nearest pairing from pairwise distances (`Dots_Sort`)

4. **Mount-aware ID correction** (uses class `5` + heads on `prev_frame`)
   - after matching, enumerate mount pairs on the previous frame (`_iter_mount_pairs_prev_frame`, same safety rules as c5 large-box protection)
   - if both IDs of a pair are present on the current frame, optionally **swap** the two labels when that lowers the sum of center displacements from the previous frame (`_maybe_swap_mount_pair_ids`)

5. **Box-size correction**
   - `Overlap_test()` checks if one tracked box likely absorbed another fly
   - if so, pulls historical shape from cache and shrinks width/height

6. **Lost-object recovery**
   - if fewer IDs matched than expected, for each missing ID:
     - **Mount-relative path**: if the fly was in a mount pair on `prev_frame` and its partner is already matched this frame, insert a synthetic body row at `partner_now + (lost_prev - partner_prev)` (normalized, clamped), `find=False`, and skip `Obj_los_test` for that ID
     - **Otherwise** call `Obj_los_test()` (crop-mask ratio, SSIM, overlap, drift)
   - generic recovered rows use last known center from cache or previous state unless the mount-relative path applies

7. **Head assignment**
   - body table + class-1 head candidates go through `head_bind.main(...)`
   - when head match fails, head position is inherited by relative offset from previous frame
   - after `--head-reacquire-after` consecutive fallback frames for a fly, the script attempts nearest unassigned head reacquire (within `--head-reacquire-max-dist`)

8. **Write outputs**
   - append tracked rows to CSV
   - append `{frame: {fly_id: {body, head}}}` chunk to JSON stream
   - refresh caches and proceed
   - collect runtime QA counters and write a QA JSON report at the end

---

## 6) Output files

For `--output <path>`:

- **Tracked CSV**: `<path>`
  - original columns plus fly ID (`ID`) and helper columns used internally
- **Tracking JSON**: `<path>_<start_frame>_.json`
  - semicolon-separated JSON objects, one frame per object
- **QA report JSON**: `<path>.qa.json` (or `--qa-report <custom_path>`)
  - machine-readable counters and run metadata for quick health checks

Example JSON chunk shape:

```json
{ "1234": { "fly_0": { "body": [...], "head": [...] }, "fly_1": { ... } } };
```

Example QA fields:

```json
{
  "frames_total": 5400,
  "frames_tracked": 5400,
  "match_pairs_step1": 61234,
  "match_pairs_step2": 412,
  "missing_events": 93,
  "missing_objects_total": 127,
  "lost_recovery_total": 127,
  "overlap_adjustments": 208,
  "head_bind_success": 62000,
  "head_fallback": 2800,
  "head_reacquire_attempts": 190,
  "head_reacquire_success": 74,
  "mount_id_swap_fixes": 42,
  "mount_relative_recovery": 18,
  "runtime_seconds": 312.4
}
```

---

## 7) Internal flags

These are used only by subprocess recursion:

- `--internal-window`
- `--internal-region`

They prevent nested mode re-entry when the script launches itself for windows/regions.

---

## 8) Assumptions and caveats

- Expects normalized xywh coordinates in detection input.
- Expects enough body detections in the tracking start frame (`>= --num-fly`) after applying `--initial-results` override (if provided).
- Heuristic-heavy recovery (`Overlap_test`, `Obj_los_test`) depends on video quality and detection quality; mount-relative recovery reduces reliance on `Obj_los_test` when class `5` + partner geometry apply.
- Window/region stitching relies on overlap quality; poor overlap can cause ID swaps.
- The file still contains a large commented legacy block at the bottom (not executed).

---

## 9) Typical usage

Single-process:

```bash
python utils/Post_track.py -i csv/video.csv -o csv/video_tracked.csv -v /abs/path/video.mp4 -n 12
```

Window-parallel:

```bash
python utils/Post_track.py -i csv/video.csv -o csv/video_tracked.csv -v /abs/path/video.mp4 -n 12 --workers 8 --window-overlap 200
```

Single-process with QA and stricter head-reacquire tuning:

```bash
python utils/Post_track.py \
  -i csv/video.csv \
  -o csv/video_tracked.csv \
  -v /abs/path/video.mp4 \
  -n 12 \
  --head-reacquire-after 6 \
  --head-reacquire-max-dist 0.025 \
  --qa-report csv/video_tracked.qa.json
```

Resume from a manually corrected frame (override detections at start frame):

```bash
python utils/Post_track.py \
  -i csv/video.csv \
  -o csv/video_tracked.csv \
  -v /abs/path/video.mp4 \
  -n 12 \
  --initial-frame 1200 \
  --initial-results csv/video_manual_detect.csv
```

Split-region:

```bash
python utils/Post_track.py -i csv/video.csv -o csv/video_tracked.csv -v /abs/path/video.mp4 -n 24 --split-x 960
```

