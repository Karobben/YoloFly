# Data storage and security

## Files written by the app

| Path | Content |
|------|---------|
| `webapp/projects.json` | All projects and video registry (JSON). |
| `webapp/quickrun.sqlite3` (+ `-wal`, `-shm`) | QuickRun sessions and jobs. |
| `/tmp/yolofly_quickrun_sessions/<session_id>/` | Per-job `NNNN.tsv` and `NNNN.log`. |
| `/tmp/yolofly_quickrun_<uuid>.tsv` (legacy) | Temporary list file for `quick_run_fastview`. |

Pipeline outputs (CSV under **`FASTVIEW_WORKDIR/csv`**, plots, `total_speed.csv`, track JSON) are written by **FastView subprocesses**, not Flask directly.

## Git hygiene

`.gitignore` should exclude local databases and SQLite sidecars (e.g. `webapp/quickrun.sqlite3` and WAL/SHM) so sessions are not committed.

## Path safety

### `_safe_join(base, rel)`

Used for **`runs/detect`** relative paths. Resolved path must **start with** `base.resolve()` or **`ValueError`** (“Path escapes allowed directory”).

### `_safe_path_any(path_str)`

Used for arbitrary absolute (or repo-relative) reads such as **`/api/local_file`** and **`/api/csv_table`**:

1. Relative paths are resolved under **`ROOT`**.
2. Absolute paths are resolved normally.
3. Final path must **start with** one of **`ALLOWED_PATH_ROOTS`**: **`ROOT`** and **`ROOT.parent`**.

Anything else → **400** with “outside allowed roots”.

### Implications

- Users cannot read arbitrary system files outside the repo parent.
- Deployments that need additional mounts must extend **`ALLOWED_PATH_ROOTS`** in code (there is no env toggle in-repo unless added later).

## Thread safety

QuickRun SQLite access is serialized with **`_QUICKRUN_SESSION_LOCK`** for migrations/inserts/updates consistent with the worker thread.

## Operational notes

- **Daemon threads**: If the Flask process is killed hard, in-flight subprocesses may be orphaned; logs remain under `/tmp` until cleaned.
- **Disk**: Large **`max_rows`** on **`/api/csv_table`** can increase memory use on the server for that request.
