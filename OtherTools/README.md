# OtherTools

Quality-assurance and manual-correction utilities for fly-tracking data.  
All scripts are run from the **project root** (the directory that contains `csv/` and `Video_list.csv`).

---

## Contents

| Script | Purpose |
|--------|---------|
| `video_annotate_from_tracking.py` | Render an annotated video overlaying fly IDs, head→body arrows, behavior-class boxes, and JSON/CSV mismatch highlights. |
| `high_risk_detection.py` | Detect high-risk frames: **LOSS** (missing flies), **GAIN** (extra detections), and **LEAP** (huge position jumps indicating ID switches). Optionally creates annotated preview clips. |
| `loss_detection.py` | *Deprecated* — now forwards all arguments to `high_risk_detection.py`. |
| `json_correct.py` | Apply manual corrections to a FLY_matrix tracking JSON: swap fly IDs or replace bad positions by inheriting from the previous frame. |
| `corrections_example.csv` | Example correction table for `json_correct.py`. |

---

## Prerequisite: `Video_list.csv`

All scripts read `Video_list.csv` from the project root.  
It is **tab-separated**, **no header**, with 7 columns:

| Col | Name          | Description |
|-----|---------------|-------------|
| 1   | `video_id`    | Identifier matching the filenames under `csv/` |
| 2   | `petri_pixel` | Plate diameter in pixels |
| 3   | `petri_mm`    | Plate diameter in mm |
| 4   | `frame_start` | First frame (inclusive) |
| 5   | `frame_end`   | Last frame (exclusive) |
| 6   | `num_flies`   | Expected number of flies |
| 7   | `abs_path`    | Absolute path to the video file |

Example row (columns separated by a tab `\t`):
```
adf6254_Movie_S3.mp4  1080  95  1  1826  13  /mnt/Data/Videos/adf6254_Movie_S3.mp4
```

---

## 1 · `video_annotate_from_tracking.py` — Annotated video

### What it draws on each frame

| Visual element | Meaning |
|----------------|---------|
| Colored arrow (body → head) | One arrow per fly; color follows the fly's ID in the palette |
| Colored circle at body center | Fly body position from JSON |
| Fly ID text at head box top-right | ID label, same color as arrow |
| Colored boxes (cls 2/3/4) | Behavior detections: grooming / chasing / flapping |
| **Red box + "LOSS"** | JSON fly with no matching CSV body (missed detection) |
| **Blue box + "GAIN"** | CSV body with no matching JSON fly (extra / duplicate detection) |
| `JSON:N \| CSV:M (+K merged)` | Per-frame count overlay (red when counts differ) |
| `Frame N` (top-right, black) | Current frame number |
| `Frame N [LOSS/GAIN]` (red + border) | Frame flagged as a loss or gain event |

### Basic usage

```bash
# Annotate the full configured frame range
python OtherTools/video_annotate_from_tracking.py -id adf6254_Movie_S3.mp4

# Test: only render the first second
python OtherTools/video_annotate_from_tracking.py -id adf6254_Movie_S3.mp4 --test

# Override the frame window (e.g. frames 800-900 only)
python OtherTools/video_annotate_from_tracking.py -id adf6254_Movie_S3.mp4 --fs 800 --fe 900

# Custom output path
python OtherTools/video_annotate_from_tracking.py -id adf6254_Movie_S3.mp4 -o /tmp/preview.mp4

# Hide behavior-class boxes (useful when inspecting fly IDs)
python OtherTools/video_annotate_from_tracking.py -id adf6254_Movie_S3.mp4 --no-behavior

# Mark a specific frame (e.g. frame 843 as LOSS event)
python OtherTools/video_annotate_from_tracking.py -id adf6254_Movie_S3.mp4 \
    --fs 810 --fe 876 --mark-frame 843 --mark-label LOSS
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `-id / --video_id` | required | Video id from `Video_list.csv` |
| `-vlist` | `Video_list.csv` | Path to `Video_list.csv` |
| `-o / --output` | `Video_post/Preview_…mp4` | Output video path |
| `--fs` | from Video_list | Override frame start |
| `--fe` | from Video_list | Override frame end |
| `--mark-frame` | — | Highlight a specific frame with a red border |
| `--mark-label` | `MARK` | Label shown on the marked frame (e.g. `LOSS`, `GAIN`) |
| `--no-behavior` | off | Skip drawing behavior-class boxes |
| `--show` | off | Display frames in a window while processing |
| `-t / --test` | off | Limit to the first second only |

---

## 2 · `high_risk_detection.py` — Detect high-risk events

### What it detects

| Event | Source | Definition |
|-------|--------|-----------|
| **LOSS** | CSV cls0 | Consecutive frames where effective body count < expected fly count |
| **GAIN** | CSV cls0 | Consecutive frames where effective body count > expected fly count |
| **LEAP** | JSON FLY_matrix | Frames where a fly's per-second displacement > `leap_mult × body_size` |

Merged detections (model collapsed 2+ flies into one large box) are automatically excluded from LOSS/GAIN counts.

**LEAP detection logic:**  
For each consecutive pair of frames, the per-frame displacement of each fly is measured in pixels and scaled to per-second speed (`dist_px × fps`).  
This is compared against the fly's own body size in pixels. If the ratio exceeds `--leap-mult` (default 1.0), the frame is flagged as a LEAP.  
A fly moving more than its own body length per second is physically unusual and is a reliable signal of an ID switch in the tracker.

### Basic usage

```bash
# Run all three detections
python OtherTools/high_risk_detection.py -id adf6254_Movie_S3.mp4

# Override expected fly count
python OtherTools/high_risk_detection.py -id adf6254_Movie_S3.mp4 -n 12

# Change leap threshold (2.0 = must move 2× body lengths/s to flag)
python OtherTools/high_risk_detection.py -id adf6254_Movie_S3.mp4 --leap-mult 2.0

# Skip leap detection (faster if JSON is very large)
python OtherTools/high_risk_detection.py -id adf6254_Movie_S3.mp4 --no-leap

# Generate preview clips for every event
python OtherTools/high_risk_detection.py -id adf6254_Movie_S3.mp4 --make-previews

# Custom output directory + test mode (first 3 previews only)
python OtherTools/high_risk_detection.py -id adf6254_Movie_S3.mp4 \
    --make-previews --outdir Video_post/HighRisk --preview-test
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `-id / --video_id` | required | Video id from `Video_list.csv` |
| `-n / --expected` | from Video_list | Override expected fly count |
| `-vlist` | `Video_list.csv` | Path to `Video_list.csv` |
| `--min-gap` | `10` | Min consecutive abnormal frames for LOSS/GAIN |
| `--leap-mult` | `1.0` | LEAP threshold in body-lengths per second |
| `--no-leap` | off | Skip LEAP detection |
| `--make-previews` | off | Create annotated preview clips for every event |
| `--outdir` | `Video_post/HighRiskCheck` | Directory for preview clips |
| `--preview-test` | off | Limit to first 3 previews per event type |
| `-p / --processes` | CPU count − 1 | Number of parallel workers for preview rendering |

### Preview clips

When `--make-previews` is set, for each event:

- A ±1-second window is annotated.
- The event frame gets a **red border** and the event label.
- JSON/CSV body-box mismatches are highlighted (red = missed, blue = extra).
- Output filename patterns:
  - `LOSSPreview_{video_id}_{fs}_{fe}.mp4`
  - `GAINPreview_{video_id}_{fs}_{fe}.mp4`
  - `LEAP_{fly_id}Preview_{video_id}_{fs}_{fe}.mp4`

### Recommended QC workflow

```
1. Run high_risk_detection.py   → get all event lists in console
2. Check LEAP events first      → they often indicate ID switches
3. Run with --make-previews     → watch clips to confirm issues
4. Note which frames / fly IDs need correction
5. Fill corrections.csv         → swap / delete_inherit rows
6. Apply json_correct.py        → fix the JSON
7. Re-run high_risk_detection.py → verify no remaining events
```

---

## 3 · `json_correct.py` — Manual tracking corrections

### When to use it

Use this script **after reviewing annotated preview videos** from `loss_detection.py`.  
Two correction types are available:

| Type | When to use |
|------|-------------|
| **`swap`** | The tracker confused two flies' identities from a certain frame onwards (IDs switched). |
| **`delete_inherit`** | A fly has a wild/wrong position for a range of frames (missed detection or bad re-ID). Fix by holding its last good position. |

### Correction table format

Create a CSV (or TSV or Excel) file with this header:

```
action, fly_id_1, fly_id_2, frame_start, frame_end, note
```

| Column | Required | Description |
|--------|----------|-------------|
| `action` | yes | `swap` or `delete_inherit` |
| `fly_id_1` | yes | First fly ID (both actions) |
| `fly_id_2` | swap only | Second fly ID (leave blank for delete_inherit) |
| `frame_start` | yes | First frame to apply (inclusive) |
| `frame_end` | yes* | Last frame to apply (inclusive). *For `swap`, can be blank = apply to end of video. |
| `note` | no | Free-text comment, ignored by the script |

Example (`corrections_example.csv`):

```csv
action,fly_id_1,fly_id_2,frame_start,frame_end,note
swap,fly_0,fly_3,300,,fly_0 and fly_3 swapped identities from frame 300 to end
swap,fly_1,fly_5,500,640,short swap between 500-640 then swap back
delete_inherit,fly_2,,820,850,fly_2 jumps to wrong position frames 820-850
delete_inherit,fly_7,,1100,1115,fly_7 briefly lost frames 1100-1115
```

### Recommended workflow

```
1. Run loss_detection.py  →  identify abnormal frame ranges
2. Watch the preview clips  →  decide which correction to apply
3. Fill in corrections_example.csv  →  add one row per issue
4. Dry-run first  →  confirm the right frames are targeted
5. Apply  →  overwrite the JSON in place (or to a new file)
6. Re-run video_annotate_from_tracking.py  →  verify the fix visually
```

### Step-by-step example

**Step 1** – Find anomalies:

```bash
python OtherTools/loss_detection.py -id adf6254_Movie_S3.mp4 \
    --make-previews --outdir Video_post/LossCheck
```

**Step 2** – Watch the preview clips in `Video_post/LossCheck/`.  
Suppose you see that from frame 843 fly_2 and fly_5 have swapped, and frames 1060–1075 show fly_0 in a bad position.

**Step 3** – Write `my_corrections.csv`:

```csv
action,fly_id_1,fly_id_2,frame_start,frame_end,note
swap,fly_2,fly_5,843,,ID switch at frame 843
delete_inherit,fly_0,,1060,1075,bad position frames 1060-1075
```

**Step 4** – Dry-run:

```bash
python OtherTools/json_correct.py \
    -id adf6254_Movie_S3.mp4 \
    -c my_corrections.csv \
    --dry-run
```

**Step 5** – Apply (overwrites JSON in place):

```bash
python OtherTools/json_correct.py \
    -id adf6254_Movie_S3.mp4 \
    -c my_corrections.csv
```

To write to a separate file instead:

```bash
python OtherTools/json_correct.py \
    -id adf6254_Movie_S3.mp4 \
    -c my_corrections.csv \
    --out csv/adf6254_corrected.json
```

**Step 6** – Re-verify:

```bash
python OtherTools/video_annotate_from_tracking.py \
    -id adf6254_Movie_S3.mp4 \
    --fs 830 --fe 900 --no-behavior
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `-id / --video_id` | — | Video id (used to locate the JSON under `csv/`) |
| `-c / --corrections` | required | Path to correction table (CSV / TSV / XLSX) |
| `-vlist` | `Video_list.csv` | Path to `Video_list.csv` |
| `--json-path` | — | Override: supply the JSON path directly |
| `--out` | overwrite in place | Output JSON path |
| `--dry-run` | off | Print what would change without writing |

---

## Detection class labels (reference)

| Class | Label | Box size | Notes |
|-------|-------|----------|-------|
| 0 | **body** | w≈0.066, h≈0.079 | Whole fly; used for tracking and counting |
| 1 | **head** | w≈0.017, h≈0.019 | Head only; ~17× smaller area than cls 0 |
| 2 | **grooming** | — | |
| 3 | **chasing** | — | |
| 4 | **flapping** | — | |
| 5 | **holding** | — | |

---

## Color palette (fly IDs)

Fly IDs are colored in strict order (`fly_0` → first color, `fly_1` → second, …):

| Index | Hex | Name |
|-------|-----|------|
| 0 | `#A6CEE3` | Light Blue |
| 1 | `#1F78B4` | Dark Blue |
| 2 | `#B2DF8A` | Light Green |
| 3 | `#33A02C` | Dark Green |
| 4 | `#FB9A99` | Light Red |
| 5 | `#E31A1C` | Dark Red |
| 6 | `#FDBF6F` | Light Orange |
| 7 | `#FF7F00` | Dark Orange |
| 8 | `#CAB2D6` | Light Purple |
| 9 | `#6A3D9A` | Dark Purple |
| 10 | `#FFFF99` | Light Yellow |
| 11 | `#B15928` | Dark Brown |
