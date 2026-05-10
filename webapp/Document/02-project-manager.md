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
| PUT | `/api/project/meta` | Update `currently`, `abstract`, `quickrun_output`. |
| POST | `/api/project/videos` | Add videos: paths list and/or `directory` scan (`_collect_videos_from_directory`). |
| DELETE | `/api/project/videos` | Remove videos by path list. |
| PUT | `/api/project/video_meta` | Patch fields on one video by path. |
| POST | `/api/project/import_video_meta_tsv` | Bulk import TSV columns into matching videos. |
| POST | `/api/project/export_video_meta_tsv` | Export current video table as TSV. |
| GET | `/api/project/video_subclips` | Saved total-speed **subclips** for one registered video (see below). |

TSV export/import includes **`video_width`** and **`video_height`** as the last two columns (optional on import). Column order matches the home page note under the TSV path field.

## Registered videos table

When a project has videos, the home page shows a scrollable table (`home.html` / `home.js`).

**Columns (left to right)**

| Column | Content |
|--------|---------|
| *(checkbox)* | Select row for **batch delete**, **QuickRun**, or **Snapshot (1 frame)** (main rows and subclip rows when indexed). |
| **Video** | Filename (full path in tooltip). |
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
- **Snapshot (1 frame)** calls **`POST /api/project/snapshot_batch`**, starts a **QuickRun-style session**, then redirects to **Running progress** (`/quickrun?session=…`). For each **checked main video**, **`--snapshot-frame`** is that row’s **`frame_start`** (minimum 1). For each **checked subclip** (with indexed `source_csv` from `total_speed_clips`), the frame is taken from that clip’s saved **`start`** row. Passes **`--project <resolved snapshot base>`** and **`--name …`** so outputs land under **Project meta → Snapshot batch base** (`snapshot_output`). Uses default weights (same as QuickRun), `--conf-thres 0.4`, `--img-size 1280`, `--quiet`, **`--exist-ok`**. Per-job logs and **`snapshot_save_dir`** appear on Running progress when each job finishes.

## Project meta dialog

Besides **Currently**, **Abstract**, and **QuickRun output**, you can set **Snapshot batch base** (`snapshot_output`) as described above.

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

## Measurement overlay

Home page may show a measurement UI tied to video geometry fields (`disk_pixel`, etc.) — behaviour is implemented in `home.js` and templates; coordinates are persisted via video meta APIs.
