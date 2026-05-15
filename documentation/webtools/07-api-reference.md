# API reference (checklist)

All routes are defined in **`webapp/yolo_review_app.py`**. This document is a **checklist**; request/response shapes follow handler implementations.

## Pages (GET, HTML)

| Path |
|------|
| `/` |
| `/quickrun` |
| `/gpu-monitor` |
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
| POST | `/api/project/tracking_batch` |
| POST | `/api/project/post_track_batch` |
| POST | `/api/project/videos` |
| DELETE | `/api/project/videos` |
| PUT | `/api/project/video_meta` |
| POST | `/api/project/import_video_meta_tsv` |
| POST | `/api/project/export_video_meta_tsv` |
| POST | `/api/project/quick_run_fastview` |
| GET | `/api/project/video_subclips` |

Query params for **`/api/project/video_subclips`**: **`name`** (project), **`video_path`**. Validates membership in the project; runs **`_quickrun_sync_artifacts_from_jobs_for_video`**, then returns **`{ ok, video_path, clips }`** where each clip matches **`/api/total_speed_clips`** shape plus optional **`source_csv`** for linking to the plot page.

**`POST /api/project/snapshot_batch`** — JSON body **`name`**, **`items`** (same shape as tracking batch) or legacy **`video_paths`**. Optional **`weights`**, **`conf_thres`**, **`img_size`**, **`snapshot_output`** (batch-only override of project snapshot base), **`quiet`**, **`exist_ok`**, **`rerun`**. Boolean **`preview_only`** returns **`commands`** without queueing. Enqueues **`detect_2.py`** snapshot jobs in a QuickRun session; see **`documentation/webtools/02-project-manager.md`**.

**`POST /api/project/tracking_batch`** — JSON body **`name`** and **`items`** (same shape as snapshot) or legacy **`video_paths`**. Supports optional tracking params including:

- `weights`, `device`, `conf_thres`, `img_size`, `tracking_output`
- `tar_tr_start_override`, `frame_start_override`, `frame_end_override`
- `run_name_override`, `tracking_dir_override`, `init_label_path_override` (single-target only)
- `tracking_detect_flags` (section toggle: when true, adds `--bh-count`, `--tar-track` + `--tar-tr-start`, `--head-bind`)
- `quiet`, `exist_ok`, `use_snapshot_init`, `allow_missing_init`, `rerun`

Returns QuickRun session metadata (`session_id`, `job_count`, `quickrun_url`). Boolean **`preview_only`** returns **`commands`** without queueing; snapshot **`--init-label-path`** resolution may be skipped in preview for responsiveness (see **`documentation/webtools/02-project-manager.md`**).

**`POST /api/project/post_track_batch`** — JSON body **`name`**, **`items`** (same `video` / `subclip` shape as snapshot batch: subclips include **`source_csv`** and **`clip_id`**). Optional **`num_fly`**, **`post_track_workers`**, **`use_snapshot_init`**, **`rerun`**. Boolean **`preview_only`** returns **`commands`** (argv + shell string) without queueing; snapshot init path may be omitted in preview for speed. Non-preview: resolves each row’s latest **`detect_2`** output directory under the project tracking base, reads the CSV via **`_tracking_output_manifest`**, then enqueues **`utils/Post_track.py`** jobs in a QuickRun session (`session_kind: "post_track"`). See **`documentation/webtools/02-project-manager.md`** → batch bar **Tracking**.

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
| POST | `/api/quickrun/kill_job` |
| GET | `/api/quickrun/history` |
| DELETE | `/api/quickrun/session/<sid>` |
| GET | `/api/quickrun/results_for_video` |

## GPU monitor

| Method | Path |
|--------|------|
| GET | `/api/gpu/monitor` |
| POST | `/api/gpu/kill_pid` |

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
