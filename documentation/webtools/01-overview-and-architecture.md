# Overview and architecture

## Purpose

The web app is a thin Flask layer around four concerns:

1. **Project registry** — JSON-backed list of videos and lab/meta fields.
2. **QuickRun** — Queue FastView pipeline jobs with SQLite persistence and `/tmp` session artifacts.
3. **Browsing** — Serve logs, CSV paths, plots, and arbitrary allowed-path files.
4. **Detect explorer** — Legacy UI over `runs/detect` for frame/label editing.

## Tech stack

| Layer | Location |
|-------|-----------|
| Server | `webapp/yolo_review_app.py` |
| Templates | `webapp/templates/*.html` |
| Client JS | `webapp/static/*.js`, `webapp/static/style.css` |

Python imports include **Flask**, **OpenCV** (`cv2`) for video probing, **SQLite** for QuickRun state.

## Important constants

Defined near the top of `yolo_review_app.py`:

| Symbol | Meaning |
|--------|---------|
| `ROOT` | Repository root (`YoloFly/`). |
| `FASTVIEW_WORKDIR` | `ROOT.parent` — cwd for FastView subprocesses and resolution of relative `output_dir` / weights paths. |
| `QUICKRUN_SESSIONS_ROOT` | `/tmp/yolofly_quickrun_sessions` — per-session folders with numbered `.tsv` and `.log` files. |
| `QUICKRUN_DB_PATH` | `webapp/quickrun.sqlite3` |
| `PROJECTS_DB_PATH` | `webapp/projects.json` |
| `RUNS_ROOT` | `runs/detect` under `ROOT` |
| `CSV_ROOT` | `csv/` under `ROOT` (used by detect-explorer APIs) |
| `ALLOWED_PATH_ROOTS` | `[ROOT.resolve(), ROOT.parent.resolve()]` — roots enforced by `_safe_path_any`. |

## Concurrency model

- **QuickRun worker**: Each session starts a **`threading.Thread`** (`daemon=True`) running `_quickrun_run_session_worker`. Jobs run **sequentially** (one subprocess at a time per session).
- **SQLite**: Connections use `timeout=60.0`; writes go through `_QUICKRUN_SESSION_LOCK`. WAL mode is enabled on the QuickRun DB.

## Caching

- `CSV_INDEX_CACHE` and `TRACK_INDEX_CACHE` memoize parsed CSV / tracking JSON indexes for the detect-explorer-style APIs (mtime/size keyed).

## Related reading

- [02-project-manager.md](02-project-manager.md)
- [03-quickrun-and-fastview.md](03-quickrun-and-fastview.md)
- [08-data-storage-and-security.md](08-data-storage-and-security.md)
