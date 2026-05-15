# Project Manager (home page `/`)

## Template and script

- **HTML**: `templates/home.html`
- **JS**: `static/home.js`
- **Styles**: `static/style.css`

## Data model (`projects.json`)

Path: `webapp/projects.json`. Top-level shape:

```json
{ "projects": [ /* project objects */ ] }
```

Each **project** (after normalization in `_normalize_project_item`) contains:

| Field | Type | Notes |
|-------|------|--------|
| `name` | string | Unique; validated by `_is_valid_project_name`. |
| `lab_info` | string | Free text. |
| `currently` | string | Short status / notes. |
| `abstract` | string | Longer description. |
| `quickrun_output` | string | Default output directory hint for QuickRun (stored with project; pipeline still receives explicit `output_dir` from the modal). |
| `snapshot_output` | string | Optional path for **`detect_2.py --project`**: base directory for snapshot batch runs. Empty → `<YoloFly parent>/snapshot`. Relative paths resolve from the same workdir as FastView (`ROOT.parent`). |
| `videos` | array | List of video records (deduped by absolute path). |

Each **video** record (`_normalize_video_entry` / `_new_video_record`) includes at minimum:

| Field | Meaning |
|-------|---------|
| `path`, `absolute_path`, `filename` | Canonical stored path and display name. |
| `disk_pixel`, `disk_radius_mm` | Optional floats for assay geometry. |
| `frame_start`, `frame_end` | Optional integer frame range (`frame_start` defaults to 1). |
| `fly_count` | Optional integer. |
| `detailed_location` | Optional string (e.g. scan root vs subfolder). |
| `split_x`, `split_y` | Optional integers. |
| `total_frames` | Optional; may be filled by probing. |
| `video_width`, `video_height` | Optional pixel dimensions of the video stream (native resolution from OpenCV **`CAP_PROP_FRAME_WIDTH` / `HEIGHT`**), auto-filled when registering from directory/paths and when using **Detect frames & resolution** in the metadata modal. |

Legacy entries that are plain strings are normalized to a minimal record with path only.

## HTTP APIs (projects)

| Method | Path | Role |
|--------|------|------|
| GET | `/api/projects` | List summaries (`name`, `lab_info`, `video_count`). |
| POST | `/api/projects` | Create project (`name`, optional `lab_info`). |
| DELETE | `/api/projects` | Delete project; JSON body `{ "name": "…" }`. |
| GET | `/api/project?name=…` | Full detail payload (`_project_detail_payload`). |
| PUT | `/api/project/lab_info` | Update `lab_info`. |
| PUT | `/api/project/meta` | Update `currently`, `abstract`, `quickrun_output`, `snapshot_output`, `tracking_output`. |
| POST | `/api/project/videos` | Add videos: paths list and/or `directory` scan (`_collect_videos_from_directory`). |
| DELETE | `/api/project/videos` | Remove videos by path list. |
| PUT | `/api/project/video_meta` | Patch fields on one video by path. |
| POST | `/api/project/import_video_meta_tsv` | Bulk import TSV columns into matching videos. |
| POST | `/api/project/export_video_meta_tsv` | Export current video table as TSV. |
| GET | `/api/project/video_subclips` | Saved total-speed **subclips** for one registered video (see below). |
| POST | `/api/project/snapshot_batch` | Queue **`detect_2.py`** single-frame snapshot jobs (home **Snapshot** modal). |
| POST | `/api/project/tracking_batch` | Queue **`detect_2.py`** tracking/detection jobs (main rows or subclips) into QuickRun session storage. |
| POST | `/api/project/post_track_batch` | Queue **`utils/Post_track.py`** jobs on existing tracking output folders (see batch bar **Tracking** below). |

TSV export/import includes **`video_width`** and **`video_height`** as the last two columns (optional on import). Column order matches the home page note under the TSV path field.

## Toolbar naming vs table columns

The **batch bar** uses short labels that do not match the table headers one-to-one:

| Batch bar button | Under the hood | API |
|------------------|----------------|-----|
| **Detection** | Runs **`detect_2.py`** in tracking mode (writes per-frame CSV/JSON under the project **Tracking batch base**). | `POST /api/project/tracking_batch` |
| **Tracking** | Runs **`utils/Post_track.py`** on an **already finished** detection output folder for that row (post-process IDs / heads; writes Post_track JSON beside the CSV). | `POST /api/project/post_track_batch` |

The **Registered videos** table still uses the older labels **Detection** / **Tracking** for the two status columns: those columns reflect **indexed artifacts** (see **`GET /api/quickrun/results_for_video`** and `documentation/webtools/04-results-and-artifacts.md`), not the batch button names.

## Registered videos table

When a project has videos, the home page shows a scrollable table (`home.html` / `home.js`).

**Columns (left to right)**

| Column | Content |
|--------|---------|
| *(checkbox)* | Select row for **batch delete**, **QuickRun**, or **Snapshot (1 frame)** (main rows and subclip rows when indexed). |
| **Video** | Filename (full path in tooltip). |
| **Clips** | Number of saved subclips for this video (main rows). |
| **Detected Flies** | Snapshot class counts (`class0/class1`) from indexed snapshot labels; clickable to open that snapshot in `detect_explore`. |
| **Detection** | Whether **detect_2-style** outputs exist for this row **under the project’s configured tracking result directory** (`tracking_output` / **Tracking batch base** in Meta). Status comes from **`GET /api/quickrun/results_for_video`** (artifacts scoped to that directory when `project` is passed). |
| **Tracking** | Whether **Post_track** output exists for this row (same artifact API and directory scope as **Detection**); **`Open`** jumps to **`detect_explore`** with that detection folder + video path. |
| **Ø px**, **R mm** | Optional assay geometry from video meta. |
| **Frame start**, **Frame end** | Optional analysis window (`frame_start` before `frame_end`, consistent with the Edit meta modal and TSV import). |
| **Flies** | Optional fly count. |
| **Split** | Optional `split_x`×`split_y` display. |
| **Actions** | **Play** (modal player; when a frame window is set, the app reloads the stream with an HTML5 **Media Fragment** `#t=start,end` so playback targets **only that time range**—starts at the clip start without manually scrubbing from time 0—then pauses at the end; falls back to seek + guard if the browser ignores fragments; **`/api/video_info_by_path`** supplies FPS / frame count), **Edit** (metadata modal), **Results** (`/video-results` with `video_path` and `project`). |

**Batch bar**

- **Select all videos** toggles only checkboxes on **main** video rows (class `.video-select-cb`).
- **Select all subclips** toggles every subclip checkbox (class `.video-subclip-select-cb`) once subclip rows have loaded. If there are no subclip rows yet, the control has no effect.
- **Delete selected** removes the **parent video(s)** from the project for: any checked main row, and any row where a **subclip** is checked (using that subclip’s parent `video_path`, deduplicated). Subclips themselves are not separate project members; you are still removing whole registered videos.
- **QuickRun** uses the **union** of paths from checked main videos and **parent paths** of checked subclips (deduplicated). With nothing checked, behaviour is unchanged: QuickRun runs on **all** videos in the project.
- **Snapshot (1 frame)** opens the **Snapshot batch** modal (weights, conf, image size, snapshot output base, `--quiet` / `--exist-ok`, **Rerun even if snapshot outputs exist**, and a **command preview**). On **Start snapshot batch**, it calls **`POST /api/project/snapshot_batch`** with the same **`items`** shape as other batches, then redirects to **Running progress** (`/quickrun?session=…`). For each **checked main video**, **`--snapshot-frame`** is that row’s **`frame_start`** (minimum 1). For each **checked subclip** (with indexed `source_csv` from `total_speed_clips`), the frame is taken from that clip’s saved **`start`** row. Passes **`--project <resolved snapshot base>`** (from Meta or modal override) and **`--name …`** so outputs land under **Project meta → Snapshot batch base** (`snapshot_output`) unless overridden in the modal. Uses modal weights (default same as QuickRun), **`--conf-thres`**, **`--img-size`**, optional **`--quiet`** / **`--exist-ok`**. **`preview_only`** is used internally for the command preview text area.
- **Detection** (toolbar) opens the **Tracking Batch** modal for **`detect_2.py`** and, on **Start detection**, submits **`POST /api/project/tracking_batch`** for selected videos/subclips. Modal parameters include:
  - Output base (`tracking_output` / `--project`), weights, `--device`, conf threshold, image size.
  - Frame overrides (`--tar-tr-start`, `--frame-start`, `--frame-end`).
  - Optional per-target overrides (`--name`, `--tracking-dir`, `--init-label-path`) for **single-target** runs only.
  - Boolean section toggle for detect optional flags (when enabled: `--bh-count`, `--tar-track` + `--tar-tr-start`, `--head-bind`; default disabled).
  - `--quiet`, `--exist-ok`, and rerun behavior.
  - Snapshot-init options (`use_snapshot_init`, `allow_missing_init`).
  - If `device` is a comma-separated list like `0,1,2`, tracking jobs run in parallel with one active job per device.
  - **Command preview**: the modal calls the same endpoint with **`preview_only: true`** to show the shell command. Snapshot label resolution for **`--init-label-path`** is **skipped in preview** so the UI stays responsive; the preview may omit that flag even when “use snapshot init” is checked. The **real** run resolves the label when you click **Start detection**.
- **Tracking** (toolbar) opens the **Tracking Batch (Post_track)** modal for **`utils/Post_track.py`**. On **Start tracking**, it submits **`POST /api/project/post_track_batch`** for the same checked **`items`** shape as snapshot/detection (`video` or `subclip` with `source_csv` + `clip_id` when applicable). Flow per target:
  1. Resolve the video path and confirm it is in **`projects.json`**.
  2. Resolve the **latest detection output directory** for that row (main video: `_latest_tracking_output_dir_for_video`; subclip: `_latest_tracking_output_dir_for_subclip` using the clip id and **`source_csv`**).
  3. Read **`_tracking_output_manifest`** in that folder to find the **`detect_2`** CSV path and starting frame.
  4. Optionally resolve a snapshot label file for **`--initial-results`** when `use_snapshot_init` is true (again **omitted from preview-only** responses for speed).
  5. Enqueue one QuickRun job per target that runs **`Post_track.py`** (`-i` CSV, `-o` stem under the same folder, `-v` video, `-n`, `--workers`, `--initial-frame`, optional `--initial-results`). Intermediate CSV cleanup and JSON placement follow **`_quickrun_fill_job_outputs_done`** for `job_kind: post_track`.
  6. The browser redirects to **`/quickrun?session=…`** to watch logs.
  - Modal fields: optional **`num_fly`**, **`post_track_workers`**, **`use_snapshot_init`**, **`rerun`**.
  - **Requires** a prior **Detection** (`detect_2`) run so the tracking folder and CSV exist; otherwise the API returns an error before queueing.

## Project meta dialog

Besides **Currently**, **Abstract**, and **QuickRun output**, you can set:

- **Snapshot batch base** (`snapshot_output`) as described above.
- **Tracking batch base** (`tracking_output`) — default base directory for **`detect_2`** outputs (toolbar **Detection**) and the scope used when the home table resolves **Detection** / **Tracking** status under **`GET /api/quickrun/results_for_video?project=…`**. Post_track (toolbar **Tracking**) reads inputs from the **latest** such folder per row.

## Subclips (Total speed — interactive plot)

Under each **main** video row, the client loads **`GET /api/project/video_subclips?name=<project>&video_path=<absolute path>`** and renders **zero or more subclip rows** plus optional status rows (loading, error, or “no subclips yet”).

**Purpose**

Subclips are the same persisted bands as on **`/total-speed-plot`**: frame ranges and names stored in SQLite (`total_speed_clips` in the QuickRun DB), keyed by the canonical **`total_speed.csv`** path. The home page lists them so you can see saved clips without opening the plot.

**Row layout**

Each subclip uses the **same nine-column structure** as the parent video, with a slightly different background color (see `.video-subclip-row` in `style.css`):

- **Checkbox column** — selects that subclip for **Delete selected** and **QuickRun** (via its parent video path; see batch bar above).
- **Video** — Subclip **name** (or `Clip <id>`); a second line shows the parent filename (`↳ …`) and a **Plot** link to **`/total-speed-plot?path=<source_csv>&video_path=…&project=…`** when `source_csv` is known.
- **Ø px**, **R mm**, **Flies**, **Split** — Copied from the **parent** video meta (same assay context).
- **Frame start**, **Frame end** — From the subclip’s saved range (rounded for display).
- **Actions** — **Play** uses the **subclip’s** saved frame range on the same parent file; **Edit** and **Results** behave like the parent row (same `video_path` and video record for Edit).

**When subclips appear**

The server resolves `total_speed.csv` paths from the **`quickrun_video_artifacts`** index (and syncs completed jobs like **`/api/quickrun/results_for_video`**), normalizes paths to match clip storage, and can fall back to CSV filenames that start with **`<video_filename>_`** (FastView-style outputs). If nothing matches or no clips exist, the UI shows the empty-state message.

## UI flows (conceptual)

1. User selects or creates a project — client loads `/api/project` for detail.
2. **Lab info** and **Meta** panels map to `lab_info` and `currently` / `abstract` / `quickrun_output`.
3. **Videos**: directory scan or paste paths; main table rows open **Play** (video), **Edit** (meta modal), **Results** (navigates to `/video-results` with query params). Subclip rows reuse the same three actions for the **parent** video.
4. **QuickRun** opens a modal; submitting calls **`POST /api/quickrun/start`** then redirects to the **Running progress** page (`/quickrun?session=…`).
5. Typical processing order on `/`: **Snapshot (1 frame)** → toolbar **Detection** (`detect_2`) → toolbar **Tracking** (`Post_track`). The table **Detection** / **Tracking** columns then update from **`/api/quickrun/results_for_video`** (with `project` set) as artifacts are indexed.

## Measurement overlay

Home page may show a measurement UI tied to video geometry fields (`disk_pixel`, etc.) — behaviour is implemented in `home.js` and templates; coordinates are persisted via video meta APIs.
