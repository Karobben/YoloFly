# YoloFly Web Application

Flask application (`yolo_review_app.py`) that provides a **Project Manager** for videos and metadata, **QuickRun** integration with the FastView pipeline, **results browsing**, **CSV table viewing**, and a legacy **Detect explorer** for YOLO run folders.

## Run the server

From the repository root:

```bash
cd /path/to/YoloFly
python webapp/yolo_review_app.py
```

Default bind: **`http://0.0.0.0:8000`** (see `app.run` at the bottom of `yolo_review_app.py`). Open **`http://127.0.0.1:8000`** locally.

Static files live under `webapp/static/`; HTML templates under `webapp/templates/`.

---

## Quick start (project pipeline)

This is the recommended order for daily use:

1. **Project build**
   - Open `/` and create a project.
   - (Optional) open **Meta** and set default output folders (`quickrun_output`, `snapshot_output`, `tracking_output`).
2. **Load videos**
   - Add videos by directory or by absolute paths.
   - Confirm each row appears in **Registered videos**.
3. **Meta info**
   - Use **Edit** on each video row to fill key fields such as `frame_start`, `frame_end`, `fly_count`, `disk_pixel`, `disk_radius_mm`.
4. **Snapshot create and correction**
   - Run **Snapshot (1 frame)** from the batch modal (set **Rerun** to overwrite existing snapshot outputs), then click **Detected Flies** in the table (or open snapshot from results) to jump to `detect_explore` and correct labels if needed.
5. **Detection** (toolbar): run **`detect_2.py`** tracking batch (CSV/JSON under the project **Tracking batch base**).
6. **Tracking** (toolbar): run **`Post_track.py`** batch to produce post-track JSON in the same detection output folder; use the table **Tracking** column **Open** to jump to **`detect_explore`**.

Toolbar **Detection** / **Tracking** map to **`POST /api/project/tracking_batch`** and **`POST /api/project/post_track_batch`** respectively (see `documentation/webtools/02-project-manager.md`).

---

## End-to-end workflow (main pipeline)

The application is organized around **projects** (datasets of registered videos). Optional processing runs **outside** the Flask process as subprocesses; the app **records state** in JSON and SQLite and **surfaces paths** for browsing.

```mermaid
flowchart TB
  subgraph PM[Project Manager /]
    PJSON[(projects.json)]
    CRUD[Create project · Lab info · Meta]
    VIDS[Register videos · Per-video meta · Subclips · TSV import/export]
    CRUD --> PJSON
    VIDS --> PJSON
  end

  subgraph QR[QuickRun]
    MODAL[QuickRun modal · Pipeline params]
    APISTART[POST /api/quickrun/start]
    DB[(quickrun.sqlite3)]
    TMP[/tmp/yolofly_quickrun_sessions/]
    WORKER[Background worker · One job per video]
    PIPE[FastView/fastview_pipeline.py]
    MODAL --> APISTART
    APISTART --> DB
    APISTART --> TMP
    APISTART --> WORKER
    WORKER --> PIPE
    PIPE --> ART[CSV · Track JSON · Plots · total_speed CSV]
  end

  subgraph VIEW[Viewing]
    QPAGE[/quickrun · Running progress]
    GPU[/gpu-monitor · GPU utilization + kills]
    VRES[/video-results · Per-video QuickRun history]
    CSV[/csv-table · Tabular CSV viewer]
    TSP[/total-speed-plot · Interactive speed + saved clips]
    QPAGE --> DB
    GPU --> DB
    VRES --> DB
    VRES --> TSP
    CSV --> ART
  end

  PM --> QR
  QR --> VIEW
  VRES --> CSV
```

### Logic between sections (concise)

| Step | Where | What happens |
|------|--------|----------------|
| 1 | **Project Manager** (`/`) | User maintains projects in `projects.json`: name, lab info, optional **project meta** (currently / abstract / default QuickRun output dir), and a list of videos with paths and metadata. The videos table shows **frame start/end**, and **subclip rows** under each video mirror clips saved on **`/total-speed-plot`** (same Play / Edit / Results as the parent). |
| 2 | **QuickRun / Snapshot / Tracking** | User opens the modal from the project detail panel and starts a **session**. FastView sessions use one TSV per video under `/tmp/yolofly_quickrun_sessions/<session_id>/`; snapshot/tracking sessions enqueue one detect_2 job per selected target. Tracking supports **`device`** strings like `0,1,2` to run one active tracking job per device in parallel. |
| 3 | **Running progress** (`/quickrun`) | Polls **`GET /api/quickrun/session/<id>`** for job status, log tails, and resolved output paths for FastView/snapshot/tracking runs. **History** lists past sessions; **Delete** removes DB rows and the session’s temp folder (not pipeline outputs on disk). |
| 4 | **GPU monitor** (`/gpu-monitor`) | Live GPU utilization/processes from `nvidia-smi`, mapped to currently running quickrun jobs when PIDs match. Supports kill actions for detect_2 jobs and external GPU processes. |
| 5 | **Video results** (`/video-results`) | **`GET /api/quickrun/results_for_video`** lists completed/failed jobs for a canonical video path (optional **project** filter). Shows links to artifacts and **View table** for CSV files. |
| 6 | **CSV table view** (`/csv-table`) | **`GET /api/csv_table`** reads an allowed-path CSV/TSV and returns bounded rows as JSON for a full-page scrollable table. |
| 7 | **Detect explorer** (`/detect_explore`) | Separate workflow: browse **`runs/detect`**, open images/video frames, edit YOLO labels under the run folder (does not use `projects.json`). |

Detailed behaviour for each block lives under **`Document/`** (see below).

---

## Documentation map

| Document | Contents |
|----------|-----------|
| [Document/01-overview-and-architecture.md](Document/01-overview-and-architecture.md) | Tech stack, paths, security roots, thread model |
| [Document/02-project-manager.md](Document/02-project-manager.md) | Projects, videos table (frame range columns), total-speed **subclips**, meta modals, TSV |
| [Document/03-quickrun-and-fastview.md](Document/03-quickrun-and-fastview.md) | Sessions, DB schema, pipeline, skip/rerun behaviour |
| [Document/04-results-and-artifacts.md](Document/04-results-and-artifacts.md) | Results page, output paths, `local_file` |
| [Document/05-csv-table-viewer.md](Document/05-csv-table-viewer.md) | Full-page table viewer API and limits |
| [Document/06-detect-explorer.md](Document/06-detect-explorer.md) | Legacy run browser and label editing |
| [Document/07-api-reference.md](Document/07-api-reference.md) | Route and endpoint checklist |
| [Document/08-data-storage-and-security.md](Document/08-data-storage-and-security.md) | Files, DB, gitignore, path rules |

---

## Quick page index

| URL | Role |
|-----|------|
| `/` | Project Manager |
| `/quickrun` | Running progress & history (`?session=` for one session) |
| `/gpu-monitor` | Live GPU/process monitor and kill actions |
| `/video-results` | QuickRun outputs for one video (`video_path`, optional `project`) |
| `/csv-table` | Table view for one file (`path`, optional `video_path`/`project` for back link) |
| `/total-speed-plot` | Interactive total-speed chart and persisted clip bands (`path` = CSV, optional `video_path` / `project`) |
| `/detect_explore` | YOLO run explorer |

---

## Related repository components (outside `webapp/`)

- **`FastView/fastview_pipeline.py`** — orchestrates `detect_2.py`, `utils/Post_track.py`, and `FastView/visualize_tracks.py` per video list; working directory for subprocesses is **`ROOT.parent`** (same as `FASTVIEW_WORKDIR` in the Flask app).
- **Detection CSVs** — typically under `<WORKDIR>/csv/` relative to that parent.
- **Plots and `total_speed.csv`** — produced under the pipeline **`-o` / `--output-dir`** (resolved relative to `WORKDIR`).

For deeper FastView behaviour, see `FastView/README.md` in the repo root.
