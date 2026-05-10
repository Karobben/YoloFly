# QuickRun and FastView integration

## User-facing pages

| URL | Files |
|-----|--------|
| `/quickrun` (page title **Running progress**) | `templates/quickrun.html`, `static/quickrun.js` |

Query parameter **`session=<32-char hex>`** focuses polling on one session (returned by start API). The same UI is used to monitor multiple kinds of runs submitted to this flow.

## Starting a session

**`POST /api/quickrun/start`** (JSON body):

- **`name`**: Project name (must exist).
- **`video_paths`** (optional): List of paths; if omitted, **all** project videos are queued (after TSV row construction).
- **Pipeline parameters** (parsed by `_parse_quick_run_pipeline_params`), including:

| Key | Role |
|-----|------|
| `workers`, `window_overlap`, `speed_window`, `frame_skip`, `imgsz` | Integers with min/max clamps. |
| `limit` | Max number of videos to process (`0` = no cap). |
| `conf_thres`, `iou_thres` | Floats in `[0, 1]`. |
| `output_dir`, `weights`, `device` | Strings (sanitized length). |
| `skip_detect`, `skip_track`, `skip_visualize`, `rerun` | Booleans passed through to FastView CLI semantics. |

Server steps:

1. Resolves video rows via `_quickrun_video_entries` (same columns as FastView TSV).
2. Creates **`session_id`** = UUID hex; directory **`/tmp/yolofly_quickrun_sessions/<sid>/`** with `0000.tsv`, `0000.log`, … per job.
3. Persists session + jobs to **`quickrun.sqlite3`** (`_quickrun_insert_session`).
4. Starts **`_quickrun_run_session_worker`** in a daemon thread.

Response includes `session_id`, `job_count`, and **`quickrun_url`**.

## Worker behaviour (`_quickrun_run_session_worker`)

For each job in order:

1. Marks job **`processing`**, records **`started_at`**, runs subprocess with command from **`_build_quickrun_cmd`** targeting **`FastView/fastview_pipeline.py`** with a **single-row** TSV.
2. Captures stdout/stderr to the job’s **`log_path`**.
3. On exit: **`done`** if exit code 0 and no error message; else **`failed`**. Fills **`outputs`** via **`_quickrun_compute_outputs`** when successful (uses imported FastView helpers for CSV name, track JSON, plot paths, `total_speed.csv`).
4. Updates **`log_tail`** (last N lines) for UI polling.

Session **`session_status`** and **`finished_at`** are updated when all jobs complete.

## SQLite schema (summary)

- **`quickrun_sessions`**: `id`, `project`, `created_at`, `finished_at`, `session_status`, `pipeline_params_json`, `workdir`, `fatal_error`.
- **`quickrun_jobs`**: Per-video rows keyed by `(session_id, job_key)` with paths, status, snapshots, log paths, JSON **`outputs_json`**, etc.

Indexes: `(session_id, sort_order)`, `(video_path)` for history and per-video queries.

## History and deletion

- **`GET /api/quickrun/history?limit=`** — Recent sessions (limit capped, e.g. max 200).
- **`DELETE /api/quickrun/session/<sid>`** — Deletes DB rows (CASCADE jobs) and **`rmtree`** session folder under `QUICKRUN_SESSIONS_ROOT`. Does **not** delete detection CSVs, track JSON, or plots under `FASTVIEW_WORKDIR`.

## Legacy endpoint

**`POST /api/project/quick_run_fastview`** — Older “fire and forget” style: writes a single **`/tmp/yolofly_quickrun_<uuid>.tsv`**, runs pipeline in background **without** the SQLite session UI. Prefer **`/api/quickrun/start`** for integrated monitoring.

## FastView script contract

The worker expects **`ROOT/FastView/fastview_pipeline.py`** to exist. Working directory for the subprocess is **`FASTVIEW_WORKDIR`** (`ROOT.parent`). Relative **`output_dir`** and **`weights`** resolve against that workdir inside FastView code paths.

For pipeline stages and skip/rerun flags, see **`FastView/README.md`** and **`FastView/fastview_pipeline.py`** in the repo.
