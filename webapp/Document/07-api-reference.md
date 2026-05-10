# API reference (checklist)

All routes are defined in **`webapp/yolo_review_app.py`**. This document is a **checklist**; request/response shapes follow handler implementations.

## Pages (GET, HTML)

| Path |
|------|
| `/` |
| `/quickrun` |
| `/csv-table` |
| `/video-results` |
| `/total-speed-plot` |
| `/detect_explore` |

## Projects

| Method | Path |
|--------|------|
| GET | `/api/projects` |
| POST | `/api/projects` |
| DELETE | `/api/projects` |
| GET | `/api/project` |
| PUT | `/api/project/lab_info` |
| PUT | `/api/project/meta` |
| POST | `/api/project/snapshot_batch` |
| POST | `/api/project/videos` |
| DELETE | `/api/project/videos` |
| PUT | `/api/project/video_meta` |
| POST | `/api/project/import_video_meta_tsv` |
| POST | `/api/project/export_video_meta_tsv` |
| POST | `/api/project/quick_run_fastview` |
| GET | `/api/project/video_subclips` |

Query params for **`/api/project/video_subclips`**: **`name`** (project), **`video_path`**. Validates membership in the project; runs **`_quickrun_sync_artifacts_from_jobs_for_video`**, then returns **`{ ok, video_path, clips }`** where each clip matches **`/api/total_speed_clips`** shape plus optional **`source_csv`** for linking to the plot page.

**`POST /api/project/snapshot_batch`** — JSON body **`name`**, and either **`items`** (preferred) or legacy **`video_paths`**. Each **`items`** entry is **`{ type: "video", video_path }`** or **`{ type: "subclip", video_path, source_csv, clip_id }`**. Optional **`weights`**, **`conf_thres`**, **`img_size`**. Enqueues a QuickRun session with **`session_kind: "snapshot"`** and returns **`{ session_id, job_count, quickrun_url, … }`**; the client should open **`quickrun_url`** (Running progress) for per-job logs. Main videos use **`frame_start`**; subclips use the clip’s **`start`** from **`source_csv`**. **`--project`** = resolved **`snapshot_output`** (see **`Document/02-project-manager.md`**).

## Total speed plot (clips)

| Method | Path |
|--------|------|
| GET | `/api/total_speed_clips` |
| POST | `/api/total_speed_clips` |
| PUT | `/api/total_speed_clips/<cid>` |
| DELETE | `/api/total_speed_clips/<cid>` |

Used by **`/total-speed-plot`** (`templates/total_speed_plot.html`, `static/total_speed_plot.js`). List/create/update/delete require a canonical **`path`** query/body field pointing at the **`*.total_speed.csv`** file. Rows live in **`total_speed_clips`** (QuickRun SQLite).

## QuickRun

| Method | Path |
|--------|------|
| POST | `/api/quickrun/start` |
| GET | `/api/quickrun/session/<sid>` |
| GET | `/api/quickrun/history` |
| DELETE | `/api/quickrun/session/<sid>` |
| GET | `/api/quickrun/results_for_video` |

## Detect explorer / assets

| Method | Path |
|--------|------|
| GET | `/api/runs` |
| GET | `/api/run_assets` |
| GET | `/api/media` |
| GET | `/api/video_frame` |
| GET | `/api/label` |
| POST | `/api/label` |
| GET | `/api/csv_files` |
| GET | `/api/json_files` |
| GET | `/api/csv_preview` |
| GET | `/api/csv_detections` |
| GET | `/api/csv_index` |
| GET | `/api/csv_frame_boxes` |
| GET | `/api/tracking_index` |
| GET | `/api/tracking_frame` |

## Paths and media (shared)

| Method | Path |
|--------|------|
| GET | `/api/video_frame_by_path` |
| GET | `/api/video_file_by_path` |
| GET | `/api/local_file` |
| GET | `/api/csv_table` |
| GET | `/api/video_info_by_path` |

**`GET /api/video_info_by_path?video_path=…`** — JSON includes **`frame_count`**, **`fps`**, and **`width`** / **`height`** when OpenCV reports positive stream dimensions.

## Session ID format

QuickRun **`sid`** must match **`[0-9a-f]{32}`** for session get/delete APIs.
