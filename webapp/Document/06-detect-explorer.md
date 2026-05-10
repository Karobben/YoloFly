# Detect explorer (`/detect_explore`)

## Purpose

Legacy workflow: browse **`runs/detect`** experiment folders, open images or video frames, overlay detections from CSV or tracking JSON, and **edit YOLO-format label `.txt` files** saved under the run (typically `labels/`).

This path **does not** use `projects.json` or QuickRun.

## Files

- **HTML**: `templates/index.html`
- **JS**: `static/app.js` (and any chunks referenced by the template)

## Server roots

- **`RUNS_ROOT`** = `ROOT / "runs" / "detect"` — runs are listed relative to this tree.
- **`CSV_ROOT`** = `ROOT / "csv"` — alternate CSV discovery paths used by some APIs.

## Representative APIs

| Endpoint | Role |
|----------|------|
| `GET /api/runs` | List run folder names. |
| `GET /api/run_assets` | Assets under a run (images, videos, labels). |
| `GET /api/media` | Serve media from within a run. |
| `GET /api/video_frame` | Extract frame image from video in run. |
| `GET /api/label` | Load label text for editing. |
| `POST /api/label` | Save label content (`run`, `path`, `content`). |
| `GET /api/csv_files`, `/api/json_files` | Discover CSV/JSON alongside runs or csv root. |
| `GET /api/csv_preview`, `/api/csv_detections`, `/api/csv_index`, `/api/csv_frame_boxes` | Detection CSV browsing. |
| `GET /api/tracking_index`, `/api/tracking_frame` | Tracking JSON browsing. |

## Label format

YOLO normalized boxes: **`class xc yc w h [conf]`** (see original short README note in repo).

## Safety

**`POST /api/label`** writes only under **`_safe_join(run_dir, rel_save)`** after resolving **`run`** under **`RUNS_ROOT`**, preventing directory escape.
