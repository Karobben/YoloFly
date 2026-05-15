# Results and artifacts

## Video results page

| URL | Files |
|-----|--------|
| `/video-results` | `templates/video_results.html`, `static/video_results.js` |

Typical query parameters:

- **`video_path`** — Absolute path to the video (must resolve consistently with stored project paths).
- **`project`** (optional) — Filters QuickRun history rows to sessions for that project name.

## API

**`GET /api/quickrun/results_for_video`**

- Requires **`video_path`**.
- Canonicalizes path via **`_resolve_stored_video_path`** with fallbacks.
- Optional **`project`** query must be a valid project name if provided.
- Optional **`clip_id`** scopes artifacts to one subclip row when positive.
- Syncs completed jobs into **`quickrun_video_artifacts`** for that video (scoped by **`project`** and **`clip_id`** when given).
- Returns **`artifacts`**: deduplicated indexed paths for that video. When **`project`** is set, paths are **restricted to the project’s configured tracking result directory** (`tracking_output` / resolved tracking base) so the home table **Detection** / **Tracking** columns only reflect outputs in the designated result tree.
- Returns **`failed_runs`**: recent failed QuickRun jobs for troubleshooting.

The home page uses this endpoint (via `home.js`) to paint the **Detection** and **Tracking** status cells; it does **not** return a full session job listing (use **`/video-results`** / session APIs for history).

## Output metadata (`_quickrun_compute_outputs`)

After a successful job, each job’s **`outputs`** object includes (when resolvable):

| Key | Meaning |
|-----|---------|
| `video_file` | Absolute video path. |
| `detection_csv` | Path under **`FASTVIEW_WORKDIR/csv/`** if file exists (`csv_name_for` + `frame_skip`). |
| `track_stem` | Stem next to detection CSV used for track JSON discovery. |
| `tracked_json` | Resolved tracking JSON path if present. |
| `plots` | List of `{ path, exists, label }` from **`expected_plot_paths`**. |
| `total_speed_csv` | **`{tracked_json.name}.total_speed.csv`** beside plots if present. |
| `output_directory` | Resolved pipeline output directory. |
| `weights_resolved` | Resolved weights path when possible. |
| `resolve_error` | String if importing FastView helpers failed. |

The UI uses these paths for **Open** links (often via **`/api/local_file`**) and **View table** for CSV/TSV → **`/csv-table?path=…`**.

## Serving local files

**`GET /api/local_file?path=…`**

- Uses **`_safe_path_any`** — path must stay under **`ALLOWED_PATH_ROOTS`**.
- Sends file with MIME guessed from extension (images, CSV, JSON, txt, log, etc.).

Video streaming helpers include **`/api/media`**, **`/api/video_file_by_path`**, **`/api/video_frame_by_path`** — each applies path safety appropriate to their implementation (see `yolo_review_app.py`).

## Relationship to Project Manager

From the home videos table, **Results** navigates to **`/video-results`** with the row’s video path and project name so the user sees **all QuickRun jobs** tied to that file without hunting session IDs.

**Subclips on the home page** use the same artifact discovery as this results flow (including syncing completed jobs into **`quickrun_video_artifacts`**) and list clips stored for each video’s **`total_speed.csv`** outputs — see **`documentation/webtools/02-project-manager.md`** → *Subclips (Total speed — interactive plot)* and **`GET /api/project/video_subclips`** in **`documentation/webtools/07-api-reference.md`**.
