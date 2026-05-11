#!/usr/bin/env python3
"""
Server-backed YOLO review/edit app.

Features:
- Browse runs under runs/detect
- Open run images or raw videos
- View a specific video frame
- Load/edit/save YOLO label txt files
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from flask import Flask, jsonify, request, send_file, send_from_directory


ROOT = Path(__file__).resolve().parents[1]
FASTVIEW_WORKDIR = ROOT.parent
QUICKRUN_SESSIONS_ROOT = Path("/tmp/yolofly_quickrun_sessions")
QUICKRUN_DB_PATH = ROOT / "webapp" / "quickrun.sqlite3"
_FASTVIEW_PIPELINE_MOD: Any = None
_QUICKRUN_SESSION_LOCK = threading.Lock()
RUNS_ROOT = ROOT / "runs" / "detect"
CSV_ROOT = ROOT / "csv"
PROJECTS_DB_PATH = ROOT / "webapp" / "projects.json"
ALLOWED_PATH_ROOTS = [ROOT.resolve(), ROOT.parent.resolve()]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

app = Flask(__name__, static_folder="static", template_folder="templates")
CSV_INDEX_CACHE: Dict[str, dict] = {}
TRACK_INDEX_CACHE: Dict[str, dict] = {}


def _safe_join(base: Path, rel: str) -> Path:
    p = (base / rel).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise ValueError("Path escapes allowed directory.")
    return p


def _safe_path_any(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.resolve()
    if not any(str(p).startswith(str(base)) for base in ALLOWED_PATH_ROOTS):
        raise ValueError("Path is outside allowed roots.")
    return p


def _safe_csv_or_any(path_str: str) -> Path:
    """
    CSV/JSON helper: allow legacy relative paths under CSV_ROOT, or absolute paths under ALLOWED_PATH_ROOTS.
    """
    raw = str(path_str or "").strip()
    if not raw:
        raise ValueError("path is required")
    p0 = Path(raw)
    if p0.is_absolute():
        return _safe_path_any(raw)
    return _safe_join(CSV_ROOT, raw)


def _list_files(base: Path, exts: set[str]) -> List[str]:
    if not base.exists():
        return []
    out = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(str(p.relative_to(base)).replace("\\", "/"))
    return sorted(out)


def _parse_result_row(line: str) -> list[str]:
    t = line.strip()
    if not t:
        return []
    if "," in t:
        return [c.strip() for c in t.split(",") if c.strip() != ""]
    return [c for c in re.split(r"\s+", t) if c != ""]


def _build_csv_index(f: Path) -> dict:
    stat = f.stat()
    key = str(f)
    cached = CSV_INDEX_CACHE.get(key)
    if cached and cached["mtime"] == stat.st_mtime and cached["size"] == stat.st_size:
        return cached

    by_frame: Dict[int, list[dict]] = {}
    class_counts: Dict[int, int] = {}
    count = 0
    with f.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            row = _parse_result_row(line)
            if len(row) < 6:
                continue
            try:
                nums = [float(x) for x in row[:7]]
            except ValueError:
                continue
            frame = int(round(nums[0]))
            cls = int(round(nums[1]))
            det = {
                "frame": frame,
                "cls": cls,
                "xc": nums[2],
                "yc": nums[3],
                "w": nums[4],
                "h": nums[5],
                "conf": nums[6] if len(nums) >= 7 else None,
            }
            by_frame.setdefault(frame, []).append(det)
            class_counts[cls] = class_counts.get(cls, 0) + 1
            count += 1

    frames = sorted(by_frame.keys())
    cached = {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "by_frame": by_frame,
        "class_counts": class_counts,
        "count": count,
        "frame_min": frames[0] if frames else 1,
        "frame_max": frames[-1] if frames else 1,
        "classes": sorted(class_counts.keys()),
    }
    CSV_INDEX_CACHE[key] = cached
    return cached


def _build_tracking_index(f: Path) -> dict:
    stat = f.stat()
    key = str(f)
    cached = TRACK_INDEX_CACHE.get(key)
    if cached and cached["mtime"] == stat.st_mtime and cached["size"] == stat.st_size:
        return cached

    by_frame: Dict[int, dict] = {}
    buf = ""
    with f.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(";")
            buf = parts.pop()
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                try:
                    obj = json.loads(part)
                except json.JSONDecodeError:
                    continue
                for fr, flies in obj.items():
                    try:
                        frame = int(fr)
                    except ValueError:
                        continue
                    by_frame[frame] = flies
        tail = buf.strip()
        if tail:
            try:
                obj = json.loads(tail)
                for fr, flies in obj.items():
                    try:
                        frame = int(fr)
                    except ValueError:
                        continue
                    by_frame[frame] = flies
            except json.JSONDecodeError:
                pass

    frames = sorted(by_frame.keys())
    cached = {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "by_frame": by_frame,
        "frame_min": frames[0] if frames else 1,
        "frame_max": frames[-1] if frames else 1,
        "count": len(frames),
    }
    TRACK_INDEX_CACHE[key] = cached
    return cached


def _resolve_label_path(run_dir: Path, rel: str) -> Path:
    """Resolve label path with compatibility fallback to run_dir/labels."""
    p = _safe_join(run_dir, rel)
    if p.exists():
        return p
    # If caller passed bare filename, try labels/<filename>.
    if "/" not in rel and "\\" not in rel:
        alt = _safe_join(run_dir, f"labels/{rel}")
        if alt.exists():
            return alt
    return p


def _list_runs() -> List[str]:
    if not RUNS_ROOT.exists():
        return []
    return sorted([p.name for p in RUNS_ROOT.iterdir() if p.is_dir()])


def _is_valid_project_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+", name or ""))


def _resolve_stored_video_path(p: str) -> str:
    """Canonical absolute path for storing `path` / `absolute_path` (matches _safe_path_any rules)."""
    try:
        return str(_safe_path_any(p).resolve())
    except ValueError:
        pp = Path(p)
        if pp.is_absolute():
            return str(pp.resolve())
        return str((ROOT / pp).resolve())


def _video_entry_canon_key(entry: dict) -> str:
    """Stable key for matching a stored video entry."""
    p = str(entry.get("absolute_path") or entry.get("path") or "").strip()
    if not p:
        return ""
    return _resolve_stored_video_path(p)


def _tsv_detailed_location_is_absolute(loc: str) -> bool:
    """Relative paths in TSV must not overwrite stored absolute detailed_location."""
    t = str(loc).strip()
    if not t:
        return False
    if t.startswith("/"):
        return True
    if len(t) >= 3 and t[0].isalpha() and t[1] == ":" and t[2] in "/\\":
        return True
    if t.startswith("\\\\"):
        return True
    return False


def _apply_video_meta_tsv_row(entry: dict, cells: List[str]) -> None:
    """Merge one TSV row into a project video entry (columns per user spec)."""
    if len(cells) < 7:
        return
    if len(cells) > 1 and cells[1].strip():
        v = _optional_float(cells[1])
        if v is not None:
            entry["disk_pixel"] = v
    if len(cells) > 2 and cells[2].strip():
        v = _optional_float(cells[2])
        if v is not None:
            entry["disk_radius_mm"] = v
    if len(cells) > 3 and cells[3].strip():
        v = _optional_int(cells[3])
        if v is not None:
            entry["frame_start"] = v
    if len(cells) > 4:
        fe_raw = cells[4].strip()
        if fe_raw and fe_raw != "-1":
            v = _optional_int(cells[4])
            if v is not None:
                entry["frame_end"] = v
    if len(cells) > 5 and cells[5].strip():
        v = _optional_int(cells[5])
        if v is not None:
            entry["fly_count"] = v
    if len(cells) > 6 and cells[6].strip():
        loc = cells[6].strip()
        if _tsv_detailed_location_is_absolute(loc):
            entry["detailed_location"] = loc
    if len(cells) > 7 and cells[7].strip():
        v = _optional_int(cells[7])
        if v is not None:
            entry["split_x"] = v
    if len(cells) > 8 and cells[8].strip():
        v = _optional_int(cells[8])
        if v is not None:
            entry["split_y"] = v
    if len(cells) > 9 and cells[9].strip():
        v = _optional_int(cells[9])
        if v is not None:
            entry["video_width"] = v
    if len(cells) > 10 and cells[10].strip():
        v = _optional_int(cells[10])
        if v is not None:
            entry["video_height"] = v


def _fmt_tsv_cell(val) -> str:
    if val is None:
        return ""
    return str(val)


def _new_video_record(canonical_abs_path: str) -> dict:
    """Default record for one video (path + editable metadata)."""
    return {
        "path": canonical_abs_path,
        "absolute_path": canonical_abs_path,
        "disk_pixel": None,
        "disk_radius_mm": None,
        "frame_start": 1,
        "frame_end": None,
        "fly_count": None,
        "detailed_location": "",
        "split_x": None,
        "split_y": None,
        "total_frames": None,
        "video_width": None,
        "video_height": None,
    }


def _optional_int(val) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _optional_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _probe_video_stream_info(video_path_str: str) -> dict | None:
    """OpenCV: frame_count, intrinsic width/height when the file opens; one capture pass."""
    try:
        vid = _safe_path_any(video_path_str)
    except ValueError:
        return None
    if not vid.exists() or not vid.is_file():
        return None
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)))
        h = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)))
    finally:
        cap.release()
    out: dict[str, int] = {}
    if n > 0:
        out["frame_count"] = n
    if w > 0:
        out["video_width"] = w
    if h > 0:
        out["video_height"] = h
    return out if out else None


def _probe_total_frames(video_path_str: str) -> int | None:
    """Return OpenCV frame count if file is readable, else None."""
    info = _probe_video_stream_info(video_path_str)
    if not info:
        return None
    n = info.get("frame_count")
    return int(n) if n else None


def _normalize_video_entry(raw) -> dict | None:
    """Accept legacy string path or full object; return canonical dict."""
    if isinstance(raw, str):
        p = str(raw).strip()
        if not p:
            return None
        canon = _resolve_stored_video_path(p)
        return _new_video_record(canon)
    if not isinstance(raw, dict):
        return None
    p = str(raw.get("path", "")).strip()
    if not p:
        return None
    canon = _resolve_stored_video_path(p)
    rec = _new_video_record(canon)
    rec["disk_pixel"] = _optional_float(raw.get("disk_pixel"))
    rec["disk_radius_mm"] = _optional_float(raw.get("disk_radius_mm"))
    fs = _optional_int(raw.get("frame_start"))
    rec["frame_start"] = 1 if fs is None else fs
    rec["frame_end"] = _optional_int(raw.get("frame_end"))
    rec["fly_count"] = _optional_int(raw.get("fly_count"))
    rec["detailed_location"] = str(raw.get("detailed_location", "") or "").strip()
    rec["split_x"] = _optional_int(raw.get("split_x"))
    rec["split_y"] = _optional_int(raw.get("split_y"))
    rec["total_frames"] = _optional_int(raw.get("total_frames"))
    rec["video_width"] = _optional_int(raw.get("video_width"))
    rec["video_height"] = _optional_int(raw.get("video_height"))
    return rec


def _normalize_project_item(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name", "")).strip()
    if not _is_valid_project_name(name):
        return None
    lab_info = str(item.get("lab_info", "") or "").strip()
    videos = item.get("videos", [])
    if not isinstance(videos, list):
        videos = []
    clean_videos = []
    seen = set()
    for v in videos:
        normalized = _normalize_video_entry(v)
        if not normalized:
            continue
        path_k = normalized["absolute_path"]
        if path_k in seen:
            continue
        seen.add(path_k)
        clean_videos.append(normalized)
    currently = str(item.get("currently", "") or "").strip()
    abstract = str(item.get("abstract", "") or "").strip()
    quickrun_output = str(item.get("quickrun_output", "") or "").strip()
    snapshot_output = str(item.get("snapshot_output", "") or "").strip()
    tracking_output = str(item.get("tracking_output", "") or "").strip()
    return {
        "name": name,
        "lab_info": lab_info,
        "currently": currently,
        "abstract": abstract,
        "quickrun_output": quickrun_output,
        "snapshot_output": snapshot_output,
        "tracking_output": tracking_output,
        "videos": clean_videos,
    }


def _read_projects_db() -> List[dict]:
    if not PROJECTS_DB_PATH.exists():
        return []
    try:
        data = json.loads(PROJECTS_DB_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    projects = data.get("projects", [])
    if not isinstance(projects, list):
        return []
    out = []
    seen = set()
    for item in projects:
        normalized = _normalize_project_item(item)
        if not normalized:
            continue
        name = normalized["name"]
        if name in seen:
            continue
        seen.add(name)
        out.append(normalized)
    return out


def _write_projects_db(projects: List[dict]) -> None:
    PROJECTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROJECTS_DB_PATH.write_text(
        json.dumps({"projects": projects}, indent=2),
        encoding="utf-8",
    )


def _project_summary(project: dict) -> dict:
    return {
        "name": project["name"],
        "lab_info": project["lab_info"],
        "video_count": len(project["videos"]),
    }


def _project_detail_payload(project: dict) -> dict:
    """Full project document for API responses (includes editable meta)."""
    return {
        "name": project["name"],
        "lab_info": project["lab_info"],
        "currently": project.get("currently") or "",
        "abstract": project.get("abstract") or "",
        "quickrun_output": project.get("quickrun_output") or "",
        "snapshot_output": project.get("snapshot_output") or "",
        "tracking_output": project.get("tracking_output") or "",
        "videos": project["videos"],
        "video_count": len(project["videos"]),
    }


def _project_long_text(raw: object, field: str, max_len: int = 20000) -> tuple[str, Optional[str]]:
    s = str(raw if raw is not None else "")
    if "\x00" in s:
        return "", f"{field} contains invalid characters"
    if len(s) > max_len:
        return "", f"{field} is too long ({max_len} characters max)"
    return s.strip(), None


def _collect_videos_from_directory(path: Path) -> List[dict]:
    """List video files under path with parent folder as detailed_location text."""
    if not path.exists() or not path.is_dir():
        raise ValueError("directory not found")
    root = path.resolve()
    entries: List[dict] = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            abs_p = str(p.resolve())
            parent = str(p.parent.resolve())
            if parent != str(root):
                detailed_location = f"{root} | {parent}"
            else:
                detailed_location = str(root)
            entries.append({"path": abs_p, "detailed_location": detailed_location})
    return sorted(entries, key=lambda e: e["path"])


@app.get("/")
def index():
    return send_from_directory(app.template_folder, "home.html")


@app.get("/quickrun")
def quickrun_page():
    return send_from_directory(app.template_folder, "quickrun.html")


@app.get("/csv-table")
def csv_table_page():
    return send_from_directory(app.template_folder, "csv_table.html")


@app.get("/total-speed-plot")
def total_speed_plot_page():
    return send_from_directory(app.template_folder, "total_speed_plot.html")


@app.get("/video-results")
def video_results_page():
    return send_from_directory(app.template_folder, "video_results.html")


@app.get("/detect_explore")
def detect_explore():
    return send_from_directory(app.template_folder, "index.html")


@app.get("/api/runs")
def api_runs():
    runs = _list_runs()
    return jsonify({"runs": runs})


@app.get("/api/projects")
def api_projects():
    projects = _read_projects_db()
    return jsonify({"projects": [_project_summary(p) for p in projects]})


@app.post("/api/projects")
def api_create_project():
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    lab_info = str(data.get("lab_info", "") or "").strip()
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    projects = _read_projects_db()
    if any(p["name"] == name for p in projects):
        return jsonify({"error": "project already exists"}), 409
    projects.append({
        "name": name,
        "lab_info": lab_info,
        "currently": "",
        "abstract": "",
        "quickrun_output": "",
        "snapshot_output": "",
        "tracking_output": "traking",
        "videos": [],
    })
    _write_projects_db(projects)
    return jsonify({"ok": True, "project": name})


@app.delete("/api/projects")
def api_delete_project():
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    projects = _read_projects_db()
    kept = [p for p in projects if p["name"] != name]
    if len(kept) == len(projects):
        return jsonify({"error": "project not found"}), 404
    _write_projects_db(kept)
    return jsonify({"ok": True, "deleted": name})


@app.put("/api/projects/reorder")
def api_projects_reorder():
    """Persist project list order (JSON array order in projects.json). First row = top priority."""
    data = request.get_json(force=True)
    order = data.get("order")
    if not isinstance(order, list) or not order:
        return jsonify({"error": "order must be a non-empty list of project names"}), 400
    order_names: List[str] = []
    for raw in order:
        n = str(raw or "").strip()
        if not _is_valid_project_name(n):
            return jsonify({"error": "invalid project name in order"}), 400
        order_names.append(n)
    projects = _read_projects_db()
    current_names = [p["name"] for p in projects]
    if len(order_names) != len(current_names):
        return jsonify({"error": "order length must match number of projects"}), 400
    if set(order_names) != set(current_names):
        return jsonify({"error": "order must list each project exactly once"}), 400
    by_name = {p["name"]: p for p in projects}
    reordered = [by_name[n] for n in order_names]
    _write_projects_db(reordered)
    return jsonify({"ok": True, "projects": [_project_summary(p) for p in reordered]})


@app.get("/api/project")
def api_get_project():
    name = str(request.args.get("name", "")).strip()
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    project = next((p for p in _read_projects_db() if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404
    return jsonify({"project": _project_detail_payload(project)})


@app.get("/api/project/video_subclips")
def api_project_video_subclips():
    """Total-speed plot clips for one registered video (same rows as /api/total_speed_clips per CSV)."""
    name = str(request.args.get("name", "")).strip()
    raw_vp = str(request.args.get("video_path", "")).strip()
    if not raw_vp:
        return jsonify({"error": "video_path is required"}), 400
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    project = next((p for p in _read_projects_db() if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404
    try:
        canon_vp = _resolve_stored_video_path(raw_vp)
    except (ValueError, OSError):
        return jsonify({"error": "invalid video_path"}), 400
    videos = project.get("videos") or []
    if not any(_video_entry_canon_key(v) == canon_vp for v in videos):
        return jsonify({"error": "video not in project"}), 404
    # Same sync as /api/quickrun/results_for_video so artifacts exist before we read CSV paths.
    _quickrun_sync_artifacts_from_jobs_for_video(canon_vp)
    clips = _video_subclips_for_video(canon_vp, project_name=name)
    return jsonify({"ok": True, "video_path": canon_vp, "clips": clips})


@app.get("/api/project/total_speed_plot_url")
def api_project_total_speed_plot_url():
    """Resolve CSV path + query params for /total-speed-plot (first matching artifact or subclip CSV)."""
    name = str(request.args.get("name", "")).strip()
    raw_vp = str(request.args.get("video_path", "")).strip()
    raw_clip = str(request.args.get("clip_id", "")).strip()
    if not raw_vp:
        return jsonify({"error": "video_path is required"}), 400
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    project = next((p for p in _read_projects_db() if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404
    try:
        canon_vp = _resolve_stored_video_path(raw_vp)
    except (ValueError, OSError):
        return jsonify({"error": "invalid video_path"}), 400
    videos = project.get("videos") or []
    if not any(_video_entry_canon_key(v) == canon_vp for v in videos):
        return jsonify({"error": "video not in project"}), 404

    _quickrun_sync_artifacts_from_jobs_for_video(canon_vp)

    clip_id_int: Optional[int] = None
    if raw_clip:
        try:
            clip_id_int = int(raw_clip)
        except ValueError:
            return jsonify({"error": "invalid clip_id"}), 400

    if clip_id_int is not None:
        clips = _video_subclips_for_video(canon_vp, project_name=name)
        csv_path = ""
        for c in clips:
            if int(c["id"]) == clip_id_int:
                sc = c.get("source_csv")
                if sc:
                    csv_path = str(sc).strip()
                break
        if not csv_path:
            return jsonify({"error": "no total speed CSV for this subclip"}), 404
        return jsonify(
            {"ok": True, "path": csv_path, "video_path": canon_vp, "clip_id": clip_id_int},
        )

    paths = _video_total_speed_csv_paths_for_video(canon_vp, name)
    if not paths:
        return jsonify({"error": "no total speed CSV for this video"}), 404
    return jsonify({"ok": True, "path": paths[0], "video_path": canon_vp, "clip_id": None})


def _count_yolo_label_classes_01(text: str) -> Tuple[int, int]:
    """Count detections with YOLO class id 0 and 1 in label file text."""
    c0 = 0
    c1 = 0
    for ln in text.splitlines():
        t = ln.strip()
        if not t or t.startswith("#"):
            continue
        parts = t.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except (ValueError, TypeError):
            continue
        if cls == 0:
            c0 += 1
        elif cls == 1:
            c1 += 1
    return c0, c1


def _snapshot_folder_frame_token(folder_name: str) -> Optional[int]:
    m = re.search(r"_f(\d+)_", folder_name)
    if not m:
        return None
    return int(m.group(1))


def _snapshot_folder_clip_id(folder_name: str) -> Optional[int]:
    """detect_2 snapshot folder: …_f{frame}_c{clip_id}_{hash}"""
    m = re.search(r"_c(\d+)_", folder_name)
    if not m:
        return None
    return int(m.group(1))


def _quickrun_list_snapshot_output_dirs(canon_path: str, project_name: str) -> List[dict]:
    """Snapshot session output directories for the main video row (not subclip-scoped)."""
    _quickrun_ensure_db()
    rows_raw: List[sqlite3.Row] = []
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            rows_raw = conn.execute(
                """
                SELECT artifact_path, finished_at, pipeline_params_json, clip_scope
                FROM quickrun_video_artifacts
                WHERE video_path = ? AND project = ? AND artifact_kind = ?
                """,
                (canon_path, project_name, "output_directory"),
            ).fetchall()
        finally:
            conn.close()
    out: List[dict] = []
    for r in rows_raw:
        try:
            cs = int(r["clip_scope"])
        except (KeyError, TypeError, ValueError):
            cs = -1
        if cs != -1:
            continue
        try:
            pp = json.loads(r["pipeline_params_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if pp.get("session_kind") != "snapshot":
            continue
        out.append({
            "path": str(r["artifact_path"]),
            "finished_at": r["finished_at"],
        })
    return out


def _pick_snapshot_dir_for_video_row(
    canon_path: str,
    project_name: str,
    frame_start: Optional[int],
) -> Optional[Path]:
    rows = _quickrun_list_snapshot_output_dirs(canon_path, project_name)
    if not rows:
        return None
    if frame_start is not None:
        matched = [
            r for r in rows
            if _snapshot_folder_frame_token(Path(r["path"]).name) == int(frame_start)
        ]
        if matched:
            rows = matched
    rows.sort(key=lambda x: str(x.get("finished_at") or ""), reverse=True)
    for r in rows:
        p = Path(r["path"])
        try:
            if p.is_dir():
                return p
        except OSError:
            continue
    return None


def _quickrun_list_snapshot_output_dirs_for_clip(
    canon_path: str,
    project_name: str,
    clip_id: int,
) -> List[dict]:
    """Snapshot output dirs indexed for one subclip (clip_scope == clip DB id)."""
    _quickrun_ensure_db()
    rows_raw: List[sqlite3.Row] = []
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            rows_raw = conn.execute(
                """
                SELECT artifact_path, finished_at, pipeline_params_json, clip_scope
                FROM quickrun_video_artifacts
                WHERE video_path = ? AND project = ? AND artifact_kind = ?
                """,
                (canon_path, project_name, "output_directory"),
            ).fetchall()
        finally:
            conn.close()
    out: List[dict] = []
    cid = int(clip_id)
    for r in rows_raw:
        try:
            cs = int(r["clip_scope"])
        except (KeyError, TypeError, ValueError):
            cs = -1
        if cs != cid:
            continue
        try:
            pp = json.loads(r["pipeline_params_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if pp.get("session_kind") != "snapshot":
            continue
        out.append({
            "path": str(r["artifact_path"]),
            "finished_at": r["finished_at"],
        })
    return out


def _pick_snapshot_dir_for_subclip(
    canon_path: str,
    project_name: str,
    clip_id: int,
    clip_start_frame: Optional[int],
) -> Optional[Path]:
    rows = _quickrun_list_snapshot_output_dirs_for_clip(
        canon_path, project_name, clip_id,
    )
    if not rows:
        return None
    if clip_start_frame is not None:
        want_f = int(clip_start_frame)
        matched = []
        for r in rows:
            name = Path(r["path"]).name
            if _snapshot_folder_clip_id(name) == int(clip_id):
                ft = _snapshot_folder_frame_token(name)
                if ft == want_f:
                    matched.append(r)
        if matched:
            rows = matched
    rows.sort(key=lambda x: str(x.get("finished_at") or ""), reverse=True)
    for r in rows:
        p = Path(r["path"])
        try:
            if p.is_dir():
                return p
        except OSError:
            continue
    return None


def _snapshot_label_counts_from_output_dir(dpath: Path, fly_n: Optional[int]) -> dict:
    """Read manifest + label; set matches_flies if sum==flies OR (c0==c1==flies), e.g. 24/24 vs Flies 24."""
    man = _snapshot_output_manifest(dpath)
    if not man.get("ok") or not man.get("raw_image_abs"):
        return {
            "has_snapshot": False,
            "class0": None,
            "class1": None,
            "total": None,
            "matches_flies": None,
            "fly_count": fly_n,
        }
    lab = man.get("label_abs_path")
    if not lab:
        return {
            "has_snapshot": False,
            "class0": None,
            "class1": None,
            "total": None,
            "matches_flies": None,
            "fly_count": fly_n,
        }
    lf = Path(lab)
    if not lf.is_file():
        c0, c1 = 0, 0
    else:
        try:
            text = lf.read_text(encoding="utf-8")
        except OSError:
            return {
                "has_snapshot": False,
                "class0": None,
                "class1": None,
                "total": None,
                "matches_flies": None,
                "fly_count": fly_n,
            }
        c0, c1 = _count_yolo_label_classes_01(text)
    total = int(c0 + c1)
    matches = False
    if fly_n is not None:
        try:
            fn = int(fly_n)
        except (TypeError, ValueError):
            fn = None
        if fn is not None:
            matches = total == fn or (int(c0) == fn and int(c1) == fn)
    return {
        "has_snapshot": True,
        "class0": int(c0),
        "class1": int(c1),
        "total": total,
        "matches_flies": matches,
        "fly_count": fly_n,
    }


def _snapshot_label_counts_for_subclip_row(
    canon_path: str,
    project_name: str,
    clip_id: int,
    fly_n: Optional[int],
    clip_start_frame: Optional[int],
) -> dict:
    _quickrun_sync_artifacts_from_jobs_for_video(canon_path)
    dpath = _pick_snapshot_dir_for_subclip(
        canon_path, project_name, clip_id, clip_start_frame,
    )
    if dpath is None:
        return {
            "has_snapshot": False,
            "class0": None,
            "class1": None,
            "total": None,
            "matches_flies": None,
            "fly_count": fly_n,
        }
    return _snapshot_label_counts_from_output_dir(dpath, fly_n)


def _snapshot_label_counts_for_video_entry(ent: dict, project_name: str) -> dict:
    canon = _video_entry_canon_key(ent)
    if not canon:
        return {
            "has_snapshot": False,
            "class0": None,
            "class1": None,
            "total": None,
            "matches_flies": None,
            "fly_count": None,
        }
    fly_n = _optional_int(ent.get("fly_count"))
    fs = _optional_int(ent.get("frame_start"))
    _quickrun_sync_artifacts_from_jobs_for_video(canon)
    dpath = _pick_snapshot_dir_for_video_row(canon, project_name, fs)
    if dpath is None:
        return {
            "has_snapshot": False,
            "class0": None,
            "class1": None,
            "total": None,
            "matches_flies": None,
            "fly_count": fly_n,
        }
    return _snapshot_label_counts_from_output_dir(dpath, fly_n)


@app.get("/api/project/snapshot_label_counts")
def api_project_snapshot_label_counts():
    """Per-video and per-subclip class 0/1 counts from snapshot labels vs Flies (parent meta)."""
    name = str(request.args.get("name", "")).strip()
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    projects = _read_projects_db()
    project = next((p for p in projects if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404
    out: Dict[str, dict] = {}
    clip_counts: Dict[str, Dict[str, dict]] = {}
    for ent in project.get("videos") or []:
        if not isinstance(ent, dict):
            continue
        canon = _video_entry_canon_key(ent)
        if not canon:
            continue
        out[canon] = _snapshot_label_counts_for_video_entry(ent, name)
        fly_n = _optional_int(ent.get("fly_count"))
        clip_counts[canon] = {}
        try:
            clips = _video_subclips_for_video(canon, project_name=name)
        except (OSError, ValueError):
            clips = []
        for clip in clips:
            cid = int(clip["id"])
            try:
                clip_frame = int(round(float(clip["start"])))
            except (TypeError, ValueError):
                clip_frame = None
            clip_counts[canon][str(cid)] = _snapshot_label_counts_for_subclip_row(
                canon, name, cid, fly_n, clip_frame,
            )
    return jsonify({"ok": True, "counts": out, "clip_counts": clip_counts})


@app.put("/api/project/lab_info")
def api_update_lab_info():
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    lab_info = str(data.get("lab_info", "") or "").strip()
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    projects = _read_projects_db()
    found = False
    updated = None
    for p in projects:
        if p["name"] != name:
            continue
        p["lab_info"] = lab_info
        found = True
        updated = p
        break
    if not found or not updated:
        return jsonify({"error": "project not found"}), 404
    _write_projects_db(projects)
    return jsonify({"ok": True, "project": _project_summary(updated)})


def _resolve_snapshot_output_base(project: dict) -> Path:
    """Where detect_2 --project points: workdir-relative folder (default YoloFly parent / snapshot)."""
    raw = str(project.get("snapshot_output") or "").strip()
    wdir = FASTVIEW_WORKDIR.resolve()
    if not raw:
        p = (wdir / "snapshot").resolve()
    else:
        pp = Path(raw)
        if pp.is_absolute():
            p = pp.expanduser().resolve()
        else:
            p = (wdir / raw).resolve()
    if not any(str(p).startswith(str(b)) for b in ALLOWED_PATH_ROOTS):
        raise ValueError("Snapshot output directory is outside allowed roots.")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_tracking_output_base(project: dict) -> Path:
    """Where tracking batch outputs are saved (default: workdir/traking)."""
    raw = str(project.get("tracking_output") or "").strip()
    wdir = FASTVIEW_WORKDIR.resolve()
    if not raw:
        p = (wdir / "traking").resolve()
    else:
        pp = Path(raw)
        if pp.is_absolute():
            p = pp.expanduser().resolve()
        else:
            p = (wdir / raw).resolve()
    if not any(str(p).startswith(str(b)) for b in ALLOWED_PATH_ROOTS):
        raise ValueError("Tracking output directory is outside allowed roots.")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _snapshot_run_folder_name(video_path: str, frame: int, clip_id: Optional[int] = None) -> str:
    stem = Path(video_path).stem
    stem_safe = re.sub(r"[^\w\-.]+", "_", stem).strip("_")[:80] or "video"
    key = f"{Path(video_path).resolve()}|{frame}|{clip_id if clip_id is not None else ''}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    if clip_id is not None:
        return f"{stem_safe}_f{frame}_c{clip_id}_{h}"
    return f"{stem_safe}_f{frame}_{h}"


def _snapshot_frame_for_clip(source_csv: str, clip_id: int) -> Optional[int]:
    """1-based frame index for detect_2 --snapshot-frame from total_speed_clips row."""
    try:
        canon_csv = _total_speed_clips_csv_canonical(source_csv)
    except ValueError:
        return None
    for c in _total_speed_clips_fetch_all(canon_csv):
        if int(c["id"]) == int(clip_id):
            s0 = float(c["start"])
            return max(1, int(round(s0)))
    return None


def _clip_frame_window_for_clip(source_csv: str, clip_id: int) -> Optional[tuple[int, int]]:
    """1-based inclusive [start, end] frame window for one clip id."""
    try:
        canon_csv = _total_speed_clips_csv_canonical(source_csv)
    except ValueError:
        return None
    for c in _total_speed_clips_fetch_all(canon_csv):
        if int(c["id"]) == int(clip_id):
            s0 = max(1, int(round(float(c["start"]))))
            e0 = max(s0, int(round(float(c["end"]))))
            return (s0, e0)
    return None


def _tracking_run_folder_name(
    video_path: str,
    frame_start: int,
    frame_end: Optional[int],
    clip_id: Optional[int] = None,
) -> str:
    stem = Path(video_path).stem
    stem_safe = re.sub(r"[^\w\-.]+", "_", stem).strip("_")[:80] or "video"
    end_s = "" if frame_end is None else f"-{int(frame_end)}"
    key = f"{Path(video_path).resolve()}|{int(frame_start)}|{int(frame_end) if frame_end is not None else ''}|{clip_id if clip_id is not None else ''}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    if clip_id is not None:
        return f"{stem_safe}_trk_f{int(frame_start)}{end_s}_c{clip_id}_{h}"
    return f"{stem_safe}_trk_f{int(frame_start)}{end_s}_{h}"


def _snapshot_init_label_for_tracking(
    canon_path: str,
    project_name: str,
    frame_start: int,
    clip_id: Optional[int] = None,
) -> Optional[str]:
    """Best-effort label path from snapshot output matching tracking start frame."""
    dpath: Optional[Path]
    if clip_id is None:
        dpath = _pick_snapshot_dir_for_video_row(canon_path, project_name, frame_start)
    else:
        dpath = _pick_snapshot_dir_for_subclip(canon_path, project_name, int(clip_id), frame_start)
    if dpath is None:
        return None
    man = _snapshot_output_manifest(dpath)
    lab = man.get("label_abs_path")
    if not lab or not man.get("label_exists"):
        return None
    try:
        lp = Path(str(lab)).resolve()
    except OSError:
        return None
    if not lp.is_file():
        return None
    return str(lp)


def _build_snapshot_detect_cmd(entry: dict, pipeline_params: dict) -> List[str]:
    """Shell command for one detect_2 snapshot job (matches snapshot_batch API)."""
    pp = pipeline_params
    return [
        sys.executable,
        str(ROOT / "detect_2.py"),
        "--weights", str(pp["weights"]),
        "--source", str(entry["path"]),
        "--conf-thres", str(pp["conf_thres"]),
        "--img-size", str(pp["img_size"]),
        "--quiet",
        "--snapshot-frame", str(entry["snapshot_frame"]),
        "--project", str(pp["snapshot_project_base"]),
        "--name", str(entry["snapshot_run_name"]),
        "--exist-ok",
    ]


def _build_tracking_detect_cmd(entry: dict, pipeline_params: dict) -> List[str]:
    """Shell command for one detect_2 tracking job."""
    pp = pipeline_params
    cmd = [
        sys.executable,
        str(ROOT / "detect_2.py"),
        "--weights", str(pp["weights"]),
        "--source", str(entry["path"]),
        "--conf-thres", str(pp["conf_thres"]),
        "--img-size", str(pp["img_size"]),
        "--quiet",
        "--tar-track",
        "--tar-tr-start", str(entry["tracking_frame_start"]),
        "--frame-start", str(entry["tracking_frame_start"]),
        "--project", str(pp["tracking_project_base"]),
        "--name", str(entry["tracking_run_name"]),
        "--tracking-dir", str(entry["tracking_save_dir"]),
        "--exist-ok",
    ]
    fe = entry.get("tracking_frame_end")
    if fe is not None:
        cmd.extend(["--frame-end", str(fe)])
    init_label = str(entry.get("tracking_init_label") or "").strip()
    if init_label:
        cmd.extend(["--init-label-path", init_label])
    return cmd


@app.put("/api/project/meta")
def api_put_project_meta():
    """Update project-level meta including QuickRun and snapshot output dirs."""
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    currently, err = _project_long_text(data.get("currently"), "currently")
    if err:
        return jsonify({"error": err}), 400
    abstract, err = _project_long_text(data.get("abstract"), "abstract")
    if err:
        return jsonify({"error": err}), 400
    qro_raw = data.get("quickrun_output", "")
    quickrun_output, err = _quickrun_sanitize_arg_str(qro_raw, "quickrun_output")
    if err:
        return jsonify({"error": err}), 400
    snap_raw = data.get("snapshot_output", "")
    snapshot_output, err = _quickrun_sanitize_arg_str(snap_raw, "snapshot_output")
    if err:
        return jsonify({"error": err}), 400
    trk_raw = data.get("tracking_output", "traking")
    tracking_output, err = _quickrun_sanitize_arg_str(trk_raw, "tracking_output")
    if err:
        return jsonify({"error": err}), 400
    projects = _read_projects_db()
    project = next((p for p in projects if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404
    project["currently"] = currently
    project["abstract"] = abstract
    project["quickrun_output"] = quickrun_output
    project["snapshot_output"] = snapshot_output
    project["tracking_output"] = tracking_output or "traking"
    _write_projects_db(projects)
    return jsonify({"ok": True, "project": _project_detail_payload(project)})


@app.post("/api/project/snapshot_batch")
def api_project_snapshot_batch():
    """
    Queue detect_2.py snapshot jobs (one frame each) like QuickRun: same Running progress UI.
    Request body may include:
    - items: [ { "type": "video", "video_path": "..." },
               { "type": "subclip", "video_path": "...", "source_csv": "...", "clip_id": N } ]
    - or legacy video_paths: [ ... ] (main videos only; uses frame_start).
    """
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    weights = str(data.get("weights", "") or "").strip() or _QUICKRUN_DEFAULT_WEIGHTS
    conf_raw = data.get("conf_thres")
    try:
        conf_thres = float(conf_raw) if conf_raw is not None and str(conf_raw).strip() != "" else 0.4
    except (TypeError, ValueError):
        return jsonify({"error": "conf_thres must be a number"}), 400
    imgsz_raw = data.get("img_size")
    try:
        imgsz = int(imgsz_raw) if imgsz_raw is not None and str(imgsz_raw).strip() != "" else 1280
    except (TypeError, ValueError):
        return jsonify({"error": "img_size must be an integer"}), 400
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    wpath, werr = _quickrun_sanitize_arg_str(weights, "weights")
    if werr or not wpath:
        return jsonify({"error": werr or "invalid weights"}), 400

    projects = _read_projects_db()
    project = next((p for p in projects if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404
    try:
        project_base = _resolve_snapshot_output_base(project)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not (ROOT / "detect_2.py").is_file():
        return jsonify({"error": "detect_2.py not found"}), 404

    work_items: List[dict] = []
    items_raw = data.get("items")
    if isinstance(items_raw, list) and items_raw:
        for it in items_raw:
            if not isinstance(it, dict):
                continue
            typ = str(it.get("type", "")).strip().lower()
            if typ == "video":
                raw_vp = str(it.get("video_path", "")).strip()
                if not raw_vp:
                    continue
                try:
                    vp = _resolve_stored_video_path(raw_vp)
                except (ValueError, OSError):
                    return jsonify({"error": f"invalid video_path: {raw_vp}"}), 400
                if not any(_video_entry_canon_key(v) == vp for v in project.get("videos") or []):
                    return jsonify({"error": f"video not in project: {vp}"}), 400
                work_items.append({"kind": "video", "video_path": vp})
            elif typ == "subclip":
                raw_vp = str(it.get("video_path", "")).strip()
                csv_p = str(it.get("source_csv", "")).strip()
                if not raw_vp or not csv_p:
                    return jsonify({"error": "subclip requires video_path and source_csv"}), 400
                try:
                    vp = _resolve_stored_video_path(raw_vp)
                except (ValueError, OSError):
                    return jsonify({"error": f"invalid video_path: {raw_vp}"}), 400
                if not any(_video_entry_canon_key(v) == vp for v in project.get("videos") or []):
                    return jsonify({"error": f"video not in project: {vp}"}), 400
                try:
                    cid = int(it["clip_id"])
                except (KeyError, TypeError, ValueError):
                    return jsonify({"error": "subclip requires integer clip_id"}), 400
                fr = _snapshot_frame_for_clip(csv_p, cid)
                if fr is None:
                    return jsonify({"error": f"clip id {cid} not found for source_csv"}), 400
                work_items.append({
                    "kind": "subclip",
                    "video_path": vp,
                    "source_csv": csv_p,
                    "clip_id": cid,
                    "frame": fr,
                })
    else:
        raw_paths = data.get("video_paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            return jsonify({"error": "items or video_paths must be a non-empty list"}), 400
        for raw in raw_paths:
            ps = str(raw).strip()
            if not ps:
                continue
            try:
                vp = _resolve_stored_video_path(ps)
            except (ValueError, OSError):
                return jsonify({"error": f"invalid video_path: {ps}"}), 400
            if not any(_video_entry_canon_key(v) == vp for v in project.get("videos") or []):
                return jsonify({"error": f"video not in project: {vp}"}), 400
            work_items.append({"kind": "video", "video_path": vp})

    if not work_items:
        return jsonify({"error": "no valid snapshot targets"}), 400

    sid = uuid.uuid4().hex
    sdir = QUICKRUN_SESSIONS_ROOT / sid
    try:
        sdir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500

    rerun_snap = _quickrun_bool_param(data, "rerun", False)
    pipeline_params = {
        "session_kind": "snapshot",
        "weights": wpath,
        "conf_thres": conf_thres,
        "img_size": imgsz,
        "snapshot_project_base": str(project_base),
        "rerun": rerun_snap,
    }

    jobs: List[dict] = []
    for i, wi in enumerate(work_items):
        vp = wi["video_path"]
        fname = Path(vp).name
        if wi["kind"] == "video":
            ent = next(
                (e for e in project.get("videos") or [] if _video_entry_canon_key(e) == vp),
                None,
            )
            fs = _optional_int(ent.get("frame_start")) if ent else None
            frame_n = max(1, int(fs)) if fs is not None else 1
            run_name = _snapshot_run_folder_name(vp, frame_n)
            label = fname
            clip_id = None
        else:
            frame_n = int(wi["frame"])
            cid = int(wi["clip_id"])
            run_name = _snapshot_run_folder_name(vp, frame_n, cid)
            label = f"{fname} · clip {cid}"

        ent_match = next(
            (e for e in project.get("videos") or [] if _video_entry_canon_key(e) == vp),
            None,
        )
        assert ent_match is not None
        snap = _video_entry_snapshot(ent_match)
        snap["job_kind"] = "snapshot"
        snap["snapshot_frame"] = frame_n
        snap["snapshot_run_name"] = run_name
        if wi["kind"] == "subclip":
            snap["subclip_source_csv"] = wi["source_csv"]
            snap["subclip_clip_id"] = wi["clip_id"]

        placeholder_tsv = sdir / f"{i:04d}.tsv"
        try:
            placeholder_tsv.write_text("", encoding="utf-8")
        except OSError as exc:
            return jsonify({"error": str(exc)}), 500

        jobs.append({
            "id": f"s{i}",
            "video_path": vp,
            "video_label": label,
            "status": "queued",
            "entry_snapshot": snap,
            "tsv_path": str(placeholder_tsv.resolve()),
            "log_path": str((sdir / f"{i:04d}.log").resolve()),
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "error_message": None,
            "log_tail": "",
            "outputs": None,
        })

    sess = {
        "id": sid,
        "project": name,
        "created_at": _utc_now_iso(),
        "finished_at": None,
        "session_status": "running",
        "pipeline_params": pipeline_params,
        "workdir": str(FASTVIEW_WORKDIR.resolve()),
        "jobs": jobs,
    }
    _quickrun_insert_session(sess)
    threading.Thread(target=_quickrun_run_session_worker, args=(sid,), daemon=True).start()

    return jsonify({
        "ok": True,
        "session_id": sid,
        "job_count": len(jobs),
        "quickrun_url": f"/quickrun?session={sid}",
        "snapshot_project_base": str(project_base),
    })


@app.post("/api/project/tracking_batch")
def api_project_tracking_batch():
    """
    Queue detect_2.py tracking jobs (one per selected video/subclip).
    Optional snapshot bootstrap labels are loaded for first tracked frame when available.
    """
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    weights = str(data.get("weights", "") or "").strip() or _QUICKRUN_DEFAULT_WEIGHTS
    conf_raw = data.get("conf_thres")
    try:
        conf_thres = float(conf_raw) if conf_raw is not None and str(conf_raw).strip() != "" else 0.4
    except (TypeError, ValueError):
        return jsonify({"error": "conf_thres must be a number"}), 400
    imgsz_raw = data.get("img_size")
    try:
        imgsz = int(imgsz_raw) if imgsz_raw is not None and str(imgsz_raw).strip() != "" else 1280
    except (TypeError, ValueError):
        return jsonify({"error": "img_size must be an integer"}), 400
    use_snapshot_init = _quickrun_bool_param(data, "use_snapshot_init", True)
    allow_missing_init = _quickrun_bool_param(data, "allow_missing_init", False)

    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    wpath, werr = _quickrun_sanitize_arg_str(weights, "weights")
    if werr or not wpath:
        return jsonify({"error": werr or "invalid weights"}), 400
    projects = _read_projects_db()
    project = next((p for p in projects if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404
    try:
        project_base = _resolve_tracking_output_base(project)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not (ROOT / "detect_2.py").is_file():
        return jsonify({"error": "detect_2.py not found"}), 404

    by_path = {
        _video_entry_canon_key(v): v
        for v in (project.get("videos") or [])
        if _video_entry_canon_key(v)
    }
    work_items: List[dict] = []
    items_raw = data.get("items")
    if isinstance(items_raw, list) and items_raw:
        for it in items_raw:
            if not isinstance(it, dict):
                continue
            typ = str(it.get("type", "")).strip().lower()
            if typ == "video":
                raw_vp = str(it.get("video_path", "")).strip()
                if not raw_vp:
                    continue
                try:
                    vp = _resolve_stored_video_path(raw_vp)
                except (ValueError, OSError):
                    return jsonify({"error": f"invalid video_path: {raw_vp}"}), 400
                ent = by_path.get(vp)
                if ent is None:
                    return jsonify({"error": f"video not in project: {vp}"}), 400
                fs = _optional_int(ent.get("frame_start"))
                fe = _optional_int(ent.get("frame_end"))
                fsn = max(1, int(fs)) if fs is not None else 1
                fen = int(fe) if fe is not None and int(fe) >= fsn else None
                work_items.append({
                    "kind": "video",
                    "video_path": vp,
                    "frame_start": fsn,
                    "frame_end": fen,
                })
            elif typ == "subclip":
                raw_vp = str(it.get("video_path", "")).strip()
                csv_p = str(it.get("source_csv", "")).strip()
                if not raw_vp or not csv_p:
                    return jsonify({"error": "subclip requires video_path and source_csv"}), 400
                try:
                    vp = _resolve_stored_video_path(raw_vp)
                except (ValueError, OSError):
                    return jsonify({"error": f"invalid video_path: {raw_vp}"}), 400
                if vp not in by_path:
                    return jsonify({"error": f"video not in project: {vp}"}), 400
                try:
                    cid = int(it["clip_id"])
                except (KeyError, TypeError, ValueError):
                    return jsonify({"error": "subclip requires integer clip_id"}), 400
                win = _clip_frame_window_for_clip(csv_p, cid)
                if win is None:
                    return jsonify({"error": f"clip id {cid} not found for source_csv"}), 400
                fsn, fen = win
                work_items.append({
                    "kind": "subclip",
                    "video_path": vp,
                    "source_csv": csv_p,
                    "clip_id": cid,
                    "frame_start": fsn,
                    "frame_end": fen,
                })
    else:
        raw_paths = data.get("video_paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            return jsonify({"error": "items or video_paths must be a non-empty list"}), 400
        for raw in raw_paths:
            ps = str(raw).strip()
            if not ps:
                continue
            try:
                vp = _resolve_stored_video_path(ps)
            except (ValueError, OSError):
                return jsonify({"error": f"invalid video_path: {ps}"}), 400
            ent = by_path.get(vp)
            if ent is None:
                return jsonify({"error": f"video not in project: {vp}"}), 400
            fs = _optional_int(ent.get("frame_start"))
            fe = _optional_int(ent.get("frame_end"))
            fsn = max(1, int(fs)) if fs is not None else 1
            fen = int(fe) if fe is not None and int(fe) >= fsn else None
            work_items.append({
                "kind": "video",
                "video_path": vp,
                "frame_start": fsn,
                "frame_end": fen,
            })

    if not work_items:
        return jsonify({"error": "no valid tracking targets"}), 400

    missing_init: List[dict] = []
    for wi in work_items:
        init_label: Optional[str] = None
        if use_snapshot_init:
            init_label = _snapshot_init_label_for_tracking(
                wi["video_path"],
                name,
                int(wi["frame_start"]),
                int(wi["clip_id"]) if wi.get("kind") == "subclip" else None,
            )
            if not init_label:
                item_desc = {
                    "kind": wi["kind"],
                    "video_path": wi["video_path"],
                    "frame_start": int(wi["frame_start"]),
                }
                if wi.get("clip_id") is not None:
                    item_desc["clip_id"] = int(wi["clip_id"])
                missing_init.append(item_desc)
        wi["tracking_init_label"] = init_label
    if use_snapshot_init and missing_init and not allow_missing_init:
        return jsonify({
            "error": (
                f"{len(missing_init)} target(s) do not have snapshot labels for the tracking start frame. "
                "Continue to use model output for those targets?"
            ),
            "missing_init": missing_init,
            "missing_count": len(missing_init),
        }), 409

    sid = uuid.uuid4().hex
    sdir = QUICKRUN_SESSIONS_ROOT / sid
    try:
        sdir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500

    rerun_tracking = _quickrun_bool_param(data, "rerun", False)
    pipeline_params = {
        "session_kind": "tracking",
        "weights": wpath,
        "conf_thres": conf_thres,
        "img_size": imgsz,
        "tracking_project_base": str(project_base),
        "rerun": rerun_tracking,
        "use_snapshot_init": use_snapshot_init,
    }

    jobs: List[dict] = []
    for i, wi in enumerate(work_items):
        vp = wi["video_path"]
        fname = Path(vp).name
        fsn = int(wi["frame_start"])
        fen = int(wi["frame_end"]) if wi.get("frame_end") is not None else None
        cid = int(wi["clip_id"]) if wi.get("kind") == "subclip" else None
        run_name = _tracking_run_folder_name(vp, fsn, fen, cid)
        save_dir = (project_base / run_name).resolve()
        label = f"{fname} · clip {cid}" if cid is not None else fname

        ent_match = by_path.get(vp)
        assert ent_match is not None
        snap = _video_entry_snapshot(ent_match)
        snap["job_kind"] = "tracking"
        snap["tracking_frame_start"] = fsn
        snap["tracking_frame_end"] = fen
        snap["tracking_run_name"] = run_name
        snap["tracking_save_dir"] = str(save_dir)
        if wi.get("tracking_init_label"):
            snap["tracking_init_label"] = str(wi["tracking_init_label"])
        if wi["kind"] == "subclip":
            snap["subclip_source_csv"] = wi["source_csv"]
            snap["subclip_clip_id"] = int(wi["clip_id"])

        placeholder_tsv = sdir / f"{i:04d}.tsv"
        try:
            placeholder_tsv.write_text("", encoding="utf-8")
        except OSError as exc:
            return jsonify({"error": str(exc)}), 500
        jobs.append({
            "id": f"t{i}",
            "video_path": vp,
            "video_label": label,
            "status": "queued",
            "entry_snapshot": snap,
            "tsv_path": str(placeholder_tsv.resolve()),
            "log_path": str((sdir / f"{i:04d}.log").resolve()),
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "error_message": None,
            "log_tail": "",
            "outputs": None,
        })

    sess = {
        "id": sid,
        "project": name,
        "created_at": _utc_now_iso(),
        "finished_at": None,
        "session_status": "running",
        "pipeline_params": pipeline_params,
        "workdir": str(FASTVIEW_WORKDIR.resolve()),
        "jobs": jobs,
    }
    _quickrun_insert_session(sess)
    threading.Thread(target=_quickrun_run_session_worker, args=(sid,), daemon=True).start()
    return jsonify({
        "ok": True,
        "session_id": sid,
        "job_count": len(jobs),
        "quickrun_url": f"/quickrun?session={sid}",
        "tracking_project_base": str(project_base),
        "missing_init_used_model_output": len(missing_init) if use_snapshot_init else 0,
    })


@app.post("/api/project/videos")
def api_add_project_videos():
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    mode = str(data.get("mode", "paths")).strip().lower()
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    projects = _read_projects_db()
    project = next((p for p in projects if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404

    existing_paths = {e.get("absolute_path") or e["path"] for e in project["videos"]}
    added = 0

    if mode == "directory":
        directory = str(data.get("directory", "")).strip()
        if not directory:
            return jsonify({"error": "directory is required"}), 400
        try:
            dir_path = _safe_path_any(directory)
            dir_entries = _collect_videos_from_directory(dir_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        for ent in dir_entries:
            v = ent["path"]
            canon = _resolve_stored_video_path(v)
            if canon in existing_paths:
                continue
            existing_paths.add(canon)
            rec = _new_video_record(canon)
            rec["detailed_location"] = str(ent.get("detailed_location") or "").strip()
            info = _probe_video_stream_info(canon)
            if info:
                if info.get("frame_count"):
                    rec["total_frames"] = info["frame_count"]
                    rec["frame_end"] = info["frame_count"]
                if info.get("video_width"):
                    rec["video_width"] = info["video_width"]
                if info.get("video_height"):
                    rec["video_height"] = info["video_height"]
            project["videos"].append(rec)
            added += 1
    else:
        paths = data.get("paths", [])
        if not isinstance(paths, list):
            return jsonify({"error": "paths must be a list"}), 400
        new_videos: List[str] = []
        for raw in paths:
            p = str(raw).strip()
            if not p:
                continue
            try:
                path_obj = _safe_path_any(p)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            if not path_obj.exists() or not path_obj.is_file():
                continue
            if path_obj.suffix.lower() not in VIDEO_EXTS:
                continue
            new_videos.append(str(path_obj.resolve()))
        for v in new_videos:
            canon = _resolve_stored_video_path(v)
            if canon in existing_paths:
                continue
            existing_paths.add(canon)
            rec = _new_video_record(canon)
            info = _probe_video_stream_info(canon)
            if info:
                if info.get("frame_count"):
                    rec["total_frames"] = info["frame_count"]
                if info.get("video_width"):
                    rec["video_width"] = info["video_width"]
                if info.get("video_height"):
                    rec["video_height"] = info["video_height"]
            project["videos"].append(rec)
            added += 1
    _write_projects_db(projects)
    return jsonify({
        "ok": True,
        "added": added,
        "project": _project_detail_payload(project),
    })


@app.delete("/api/project/videos")
def api_delete_project_videos():
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    paths = data.get("video_paths", [])
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    if not isinstance(paths, list) or not paths:
        return jsonify({"error": "video_paths must be a non-empty list"}), 400
    projects = _read_projects_db()
    project = next((p for p in projects if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404
    remove: set[str] = set()
    for raw in paths:
        p = str(raw).strip()
        if not p:
            continue
        remove.add(_resolve_stored_video_path(p))
    if not remove:
        return jsonify({"error": "no valid paths to delete"}), 400
    before = len(project["videos"])
    project["videos"] = [e for e in project["videos"] if _video_entry_canon_key(e) not in remove]
    deleted = before - len(project["videos"])
    _write_projects_db(projects)
    return jsonify({
        "ok": True,
        "removed": deleted,
        "project": _project_detail_payload(project),
    })


def _tsv_row_looks_like_header(cells: List[str]) -> bool:
    if not cells:
        return False
    h = cells[0].strip().lower()
    return h in ("video name", "filename", "file", "name", "video")


@app.post("/api/project/import_video_meta_tsv")
def api_import_video_meta_tsv():
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    tsv_path = str(data.get("tsv_path", "")).strip()
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    if not tsv_path:
        return jsonify({"error": "tsv_path is required"}), 400
    try:
        fpath = _safe_path_any(tsv_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not fpath.is_file():
        return jsonify({"error": "tsv file not found"}), 404
    projects = _read_projects_db()
    project = next((p for p in projects if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404

    rows_total = 0
    rows_skipped_short = 0
    entries_updated = 0
    unknown_video_names: List[str] = []

    with fpath.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row or not any(str(c).strip() for c in row):
                continue
            rows_total += 1
            if _tsv_row_looks_like_header(row):
                continue
            if len(row) < 7:
                rows_skipped_short += 1
                continue
            video_name = str(row[0]).strip()
            if not video_name:
                rows_skipped_short += 1
                continue
            key = video_name.lower()
            matches = [
                e for e in project["videos"]
                if Path(str(e.get("absolute_path") or e.get("path") or "")).name.lower() == key
            ]
            if not matches:
                unknown_video_names.append(video_name)
                continue
            for e in matches:
                _apply_video_meta_tsv_row(e, row)
                entries_updated += 1

    _write_projects_db(projects)
    return jsonify({
        "ok": True,
        "rows_total": rows_total,
        "rows_skipped_short": rows_skipped_short,
        "entries_updated": entries_updated,
        "unknown_video_names": unknown_video_names,
        "unknown_count": len(unknown_video_names),
        "project": _project_detail_payload(project),
    })


@app.post("/api/project/export_video_meta_tsv")
def api_export_video_meta_tsv():
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    tsv_path = str(data.get("tsv_path", "")).strip()
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    if not tsv_path:
        return jsonify({"error": "tsv_path is required"}), 400
    try:
        fpath = _safe_path_any(tsv_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    projects = _read_projects_db()
    project = next((p for p in projects if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404

    lines: List[str] = []
    for e in project["videos"]:
        p = str(e.get("absolute_path") or e.get("path") or "")
        file_name = Path(p).name if p else ""
        loc = str(e.get("detailed_location") or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")
        row_cells = [
            file_name,
            _fmt_tsv_cell(e.get("disk_pixel")),
            _fmt_tsv_cell(e.get("disk_radius_mm")),
            _fmt_tsv_cell(e.get("frame_start")),
            _fmt_tsv_cell(e.get("frame_end")),
            _fmt_tsv_cell(e.get("fly_count")),
            loc,
            _fmt_tsv_cell(e.get("split_x")),
            _fmt_tsv_cell(e.get("split_y")),
            _fmt_tsv_cell(e.get("video_width")),
            _fmt_tsv_cell(e.get("video_height")),
        ]
        lines.append("\t".join(row_cells))

    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return jsonify({
        "ok": True,
        "written": str(fpath),
        "row_count": len(lines),
        "project": _project_summary(project),
    })


def _fmt_quickrun_cell(val) -> str:
    if val is None:
        return ""
    return str(val)


def _quickrun_video_entries(project: dict, filter_paths: Optional[List[str]]) -> List[dict]:
    videos = project.get("videos") or []
    if not filter_paths:
        return list(videos)
    want: set[str] = set()
    for p in filter_paths:
        ps = str(p).strip()
        if ps:
            want.add(_resolve_stored_video_path(ps))
    return [e for e in videos if _video_entry_canon_key(e) in want]


def _write_fastview_pipeline_tsv(entries: List[dict], out_path: Path) -> None:
    """TSV rows compatible with FastView/fastview_pipeline.read_video_list (≥7 cols; video path = col 7)."""
    lines: List[str] = []
    for e in entries:
        p = str(e.get("absolute_path") or e.get("path") or "").strip()
        if not p:
            continue
        canon = _resolve_stored_video_path(p)
        name = Path(canon).name
        flies = e.get("fly_count")
        if flies is None:
            flies = 1
        try:
            flies_i = int(flies)
        except (TypeError, ValueError):
            flies_i = 1
        row_cells = [
            name,
            _fmt_quickrun_cell(e.get("disk_pixel")),
            _fmt_quickrun_cell(e.get("disk_radius_mm")),
            _fmt_quickrun_cell(e.get("frame_start")),
            _fmt_quickrun_cell(e.get("frame_end")),
            str(flies_i),
            canon,
            _fmt_quickrun_cell(e.get("split_x")),
            _fmt_quickrun_cell(e.get("split_y")),
        ]
        lines.append("\t".join(row_cells))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


_QUICKRUN_DEFAULT_WEIGHTS = (
    "YoloFly/runs/train/2022_05_11_p633_1280_5l_e700_b128/weights/best.pt"
)
_QUICKRUN_DEFAULT_OUTPUT_DIR = "QuickTestForAging"


def _quickrun_sanitize_arg_str(raw: object, field: str, max_len: int = 2048) -> tuple[Optional[str], Optional[str]]:
    s = str(raw if raw is not None else "").strip()
    if len(s) > max_len:
        return None, f"{field} is too long"
    if "\n" in s or "\r" in s or "\x00" in s:
        return None, f"{field} contains invalid characters"
    return s, None


def _quickrun_int_param(q: dict, key: str, default: int, *, min_v: Optional[int] = None, max_v: Optional[int] = None) -> tuple[Optional[int], Optional[str]]:
    if key not in q:
        return default, None
    val = q[key]
    if isinstance(val, bool):
        return None, f"{key} must be an integer"
    try:
        n = int(val)
    except (TypeError, ValueError):
        return None, f"{key} must be an integer"
    if min_v is not None and n < min_v:
        return None, f"{key} must be >= {min_v}"
    if max_v is not None and n > max_v:
        return None, f"{key} must be <= {max_v}"
    return n, None


def _quickrun_float_param(q: dict, key: str, default: float, *, min_v: Optional[float] = None, max_v: Optional[float] = None) -> tuple[Optional[float], Optional[str]]:
    if key not in q:
        return default, None
    val = q[key]
    if isinstance(val, bool):
        return None, f"{key} must be a number"
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None, f"{key} must be a number"
    if min_v is not None and x < min_v:
        return None, f"{key} must be >= {min_v}"
    if max_v is not None and x > max_v:
        return None, f"{key} must be <= {max_v}"
    return x, None


def _quickrun_bool_param(q: dict, key: str, default: bool = False) -> bool:
    if key not in q:
        return default
    return bool(q[key])


def _parse_quick_run_pipeline_params(data: dict) -> tuple[Optional[dict], Optional[str]]:
    workers, err = _quickrun_int_param(data, "workers", 64, min_v=1, max_v=4096)
    if err:
        return None, err
    window_overlap, err = _quickrun_int_param(data, "window_overlap", 200, min_v=0, max_v=1_000_000)
    if err:
        return None, err
    speed_window, err = _quickrun_int_param(data, "speed_window", 300, min_v=1, max_v=1_000_000)
    if err:
        return None, err
    frame_skip, err = _quickrun_int_param(data, "frame_skip", 30, min_v=1, max_v=1_000_000)
    if err:
        return None, err
    imgsz, err = _quickrun_int_param(data, "imgsz", 640, min_v=32, max_v=8192)
    if err:
        return None, err
    limit_n, err = _quickrun_int_param(data, "limit", 0, min_v=0, max_v=1_000_000)
    if err:
        return None, err
    conf_thres, err = _quickrun_float_param(data, "conf_thres", 0.3, min_v=0.0, max_v=1.0)
    if err:
        return None, err
    iou_thres, err = _quickrun_float_param(data, "iou_thres", 0.45, min_v=0.0, max_v=1.0)
    if err:
        return None, err

    output_dir, err = _quickrun_sanitize_arg_str(data.get("output_dir", _QUICKRUN_DEFAULT_OUTPUT_DIR), "output_dir")
    if err:
        return None, err
    if not output_dir:
        output_dir = _QUICKRUN_DEFAULT_OUTPUT_DIR

    weights, err = _quickrun_sanitize_arg_str(data.get("weights", _QUICKRUN_DEFAULT_WEIGHTS), "weights")
    if err:
        return None, err
    if not weights:
        weights = _QUICKRUN_DEFAULT_WEIGHTS

    device_s, err = _quickrun_sanitize_arg_str(data.get("device", ""), "device")
    if err:
        return None, err

    skip_detect = _quickrun_bool_param(data, "skip_detect", False)
    skip_track = _quickrun_bool_param(data, "skip_track", False)
    skip_visualize = _quickrun_bool_param(data, "skip_visualize", False)
    rerun = _quickrun_bool_param(data, "rerun", False)

    return {
        "workers": workers,
        "window_overlap": window_overlap,
        "speed_window": speed_window,
        "frame_skip": frame_skip,
        "imgsz": imgsz,
        "limit_n": limit_n,
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "output_dir": output_dir,
        "weights": weights,
        "device_s": device_s,
        "skip_detect": skip_detect,
        "skip_track": skip_track,
        "skip_visualize": skip_visualize,
        "rerun": rerun,
    }, None


def _build_quickrun_cmd(script: Path, tmp_tsv: Path, p: dict) -> List[str]:
    cmd = [
        sys.executable,
        str(script.resolve()),
        "--video-list",
        str(tmp_tsv.resolve()),
        "--weights",
        p["weights"],
        "--frame-skip",
        str(p["frame_skip"]),
        "--workers",
        str(p["workers"]),
        "--window-overlap",
        str(p["window_overlap"]),
        "--conf-thres",
        str(p["conf_thres"]),
        "--iou-thres",
        str(p["iou_thres"]),
        "--imgsz",
        str(p["imgsz"]),
        "--speed-window",
        str(p["speed_window"]),
        "-o",
        p["output_dir"],
    ]
    if p["device_s"]:
        cmd.extend(["--device", p["device_s"]])
    if p["limit_n"] > 0:
        cmd.extend(["--limit", str(p["limit_n"])])
    if p["skip_detect"]:
        cmd.append("--skip-detect")
    if p["skip_track"]:
        cmd.append("--skip-track")
    if p["skip_visualize"]:
        cmd.append("--skip-visualize")
    if p["rerun"]:
        cmd.append("--rerun")
    return cmd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_fastview_pipeline_mod():
    global _FASTVIEW_PIPELINE_MOD
    if _FASTVIEW_PIPELINE_MOD is not None:
        return _FASTVIEW_PIPELINE_MOD
    path = ROOT / "FastView" / "fastview_pipeline.py"
    spec = importlib.util.spec_from_file_location("yolofly_fastview_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("cannot load fastview_pipeline")
    spec.loader.exec_module(mod)
    _FASTVIEW_PIPELINE_MOD = mod
    return mod


def _video_entry_snapshot(e: dict) -> dict:
    p = str(e.get("absolute_path") or e.get("path") or "").strip()
    canon = _resolve_stored_video_path(p)
    loc = e.get("detailed_location")
    if isinstance(loc, str):
        loc = loc.strip() or None
    else:
        loc = None
    return {
        "path": canon,
        "filename": Path(canon).name,
        "disk_pixel": e.get("disk_pixel"),
        "disk_radius_mm": e.get("disk_radius_mm"),
        "frame_start": e.get("frame_start"),
        "frame_end": e.get("frame_end"),
        "fly_count": e.get("fly_count"),
        "detailed_location": loc,
        "split_x": e.get("split_x"),
        "split_y": e.get("split_y"),
        "total_frames": e.get("total_frames"),
        "video_width": e.get("video_width"),
        "video_height": e.get("video_height"),
    }


def _read_log_tail(path: Path, max_lines: int = 80) -> str:
    if not path.is_file():
        return ""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = raw.splitlines()
    if len(lines) <= max_lines:
        return raw.strip()
    return "\n".join(lines[-max_lines:]).strip()


def _quickrun_compute_outputs(video_abs: str, p: dict) -> dict:
    out: dict[str, Any] = {"video_file": video_abs}
    try:
        fv = _get_fastview_pipeline_mod()
    except Exception as exc:
        out["resolve_error"] = str(exc)
        return out

    workdir = FASTVIEW_WORKDIR
    video = Path(video_abs).expanduser().resolve()
    frame_skip = int(p["frame_skip"])
    csv_name = fv.csv_name_for(video, frame_skip)
    csv_path = workdir / "csv" / csv_name
    track_output = csv_path.with_suffix("")
    tracked_json = None
    if csv_path.exists():
        tracked_json = fv.resolve_existing_track_json(track_output, csv_path)
    out_dir = fv.resolve_workdir_path(p["output_dir"])

    plots_detail: List[dict] = []
    if tracked_json is not None:
        for plot in fv.expected_plot_paths(tracked_json, out_dir):
            plots_detail.append({
                "path": str(plot.resolve()),
                "exists": plot.is_file(),
                "label": plot.name,
            })

    out["detection_csv"] = str(csv_path.resolve()) if csv_path.exists() else None
    out["track_stem"] = str(track_output.resolve())
    out["tracked_json"] = str(tracked_json.resolve()) if tracked_json is not None else None
    out["plots"] = plots_detail
    out["total_speed_csv"] = None
    if tracked_json is not None:
        ts_csv = out_dir / f"{tracked_json.name}.total_speed.csv"
        if ts_csv.is_file():
            out["total_speed_csv"] = str(ts_csv.resolve())
    out["output_directory"] = str(out_dir.resolve())
    out["weights_resolved"] = None
    try:
        wpath = fv.resolve_existing_path(p["weights"])
        out["weights_resolved"] = str(wpath.resolve())
    except Exception:
        pass
    return out


def _quickrun_ensure_db() -> None:
    QUICKRUN_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS quickrun_sessions (
                  id TEXT PRIMARY KEY,
                  project TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  finished_at TEXT,
                  session_status TEXT NOT NULL,
                  pipeline_params_json TEXT NOT NULL,
                  workdir TEXT NOT NULL,
                  fatal_error TEXT
                );
                CREATE TABLE IF NOT EXISTS quickrun_jobs (
                  session_id TEXT NOT NULL,
                  sort_order INTEGER NOT NULL,
                  job_key TEXT NOT NULL,
                  video_path TEXT NOT NULL,
                  video_label TEXT NOT NULL,
                  status TEXT NOT NULL,
                  entry_snapshot_json TEXT NOT NULL,
                  tsv_path TEXT NOT NULL,
                  log_path TEXT NOT NULL,
                  pid INTEGER,
                  started_at TEXT,
                  finished_at TEXT,
                  exit_code INTEGER,
                  error_message TEXT,
                  log_tail TEXT,
                  outputs_json TEXT,
                  PRIMARY KEY (session_id, job_key),
                  FOREIGN KEY (session_id) REFERENCES quickrun_sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_quickrun_jobs_session_order
                  ON quickrun_jobs(session_id, sort_order);
                CREATE INDEX IF NOT EXISTS idx_quickrun_jobs_video_path
                  ON quickrun_jobs(video_path);
                CREATE TABLE IF NOT EXISTS quickrun_video_artifacts (
                  video_path TEXT NOT NULL,
                  artifact_path TEXT NOT NULL,
                  artifact_kind TEXT NOT NULL,
                  artifact_label TEXT NOT NULL DEFAULT '',
                  project TEXT NOT NULL,
                  session_id TEXT,
                  job_key TEXT,
                  finished_at TEXT,
                  pipeline_params_json TEXT,
                  clip_scope INTEGER NOT NULL DEFAULT -1,
                  PRIMARY KEY (video_path, artifact_path, clip_scope)
                );
                CREATE INDEX IF NOT EXISTS idx_qva_video ON quickrun_video_artifacts(video_path);
                CREATE INDEX IF NOT EXISTS idx_qva_video_project
                  ON quickrun_video_artifacts(video_path, project);
                CREATE TABLE IF NOT EXISTS total_speed_clips (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  csv_path TEXT NOT NULL,
                  name TEXT NOT NULL DEFAULT '',
                  start_frame REAL NOT NULL,
                  end_frame REAL NOT NULL,
                  color_idx INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_total_speed_clips_csv
                  ON total_speed_clips(csv_path);
                """
            )
            conn.commit()
            try:
                _quickrun_migrate_artifacts_clip_scope_once(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()


def _quickrun_migrate_artifacts_clip_scope_once(conn: sqlite3.Connection) -> None:
    """Add clip_scope + widen PK for legacy DBs that only had (video_path, artifact_path)."""
    cur = conn.execute("PRAGMA table_info(quickrun_video_artifacts)")
    cols = {str(r[1]) for r in cur.fetchall()}
    if not cols:
        return
    if "clip_scope" in cols:
        return
    conn.executescript(
        """
        CREATE TABLE quickrun_video_artifacts_new (
          video_path TEXT NOT NULL,
          artifact_path TEXT NOT NULL,
          artifact_kind TEXT NOT NULL,
          artifact_label TEXT NOT NULL DEFAULT '',
          project TEXT NOT NULL,
          session_id TEXT,
          job_key TEXT,
          finished_at TEXT,
          pipeline_params_json TEXT,
          clip_scope INTEGER NOT NULL DEFAULT -1,
          PRIMARY KEY (video_path, artifact_path, clip_scope)
        );
        INSERT INTO quickrun_video_artifacts_new (
          video_path, artifact_path, artifact_kind, artifact_label, project,
          session_id, job_key, finished_at, pipeline_params_json, clip_scope
        )
        SELECT video_path, artifact_path, artifact_kind, artifact_label, project,
               session_id, job_key, finished_at, pipeline_params_json, -1
        FROM quickrun_video_artifacts;
        DROP TABLE quickrun_video_artifacts;
        ALTER TABLE quickrun_video_artifacts_new RENAME TO quickrun_video_artifacts;
        CREATE INDEX IF NOT EXISTS idx_qva_video ON quickrun_video_artifacts(video_path);
        CREATE INDEX IF NOT EXISTS idx_qva_video_project
          ON quickrun_video_artifacts(video_path, project);
        """
    )


def _total_speed_clips_csv_canonical(path_str: str) -> str:
    p = _safe_path_any(path_str)
    return str(p.resolve())


def _total_speed_clips_fetch_all(canon_csv: str) -> List[dict]:
    _quickrun_ensure_db()
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            rows = conn.execute(
                """
                SELECT id, name, start_frame, end_frame, color_idx
                FROM total_speed_clips
                WHERE csv_path = ?
                ORDER BY id ASC
                """,
                (canon_csv,),
            ).fetchall()
        finally:
            conn.close()
    out: List[dict] = []
    for r in rows:
        out.append({
            "id": int(r["id"]),
            "name": str(r["name"] or ""),
            "start": float(r["start_frame"]),
            "end": float(r["end_frame"]),
            "colorIdx": int(r["color_idx"]),
        })
    return out


def _total_speed_clips_insert(
    canon_csv: str, name: str, start: float, end: float, color_idx: int,
) -> int:
    _quickrun_ensure_db()
    created = _utc_now_iso()
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            cur = conn.execute(
                """
                INSERT INTO total_speed_clips (
                  csv_path, name, start_frame, end_frame, color_idx, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    canon_csv,
                    name[:200],
                    float(start),
                    float(end),
                    int(color_idx) % 256,
                    created,
                ),
            )
            conn.commit()
            rid = int(cur.lastrowid)
        finally:
            conn.close()
    return rid


def _total_speed_clips_update(
    canon_csv: str, cid: int, name: str, start: float, end: float,
) -> bool:
    _quickrun_ensure_db()
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            cur = conn.execute(
                """
                UPDATE total_speed_clips
                SET name = ?, start_frame = ?, end_frame = ?
                WHERE id = ? AND csv_path = ?
                """,
                (name[:200], float(start), float(end), int(cid), canon_csv),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def _total_speed_clips_delete(canon_csv: str, cid: int) -> bool:
    _quickrun_ensure_db()
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            cur = conn.execute(
                "DELETE FROM total_speed_clips WHERE id = ? AND csv_path = ?",
                (int(cid), canon_csv),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def _canonical_total_speed_csv_path(raw: str) -> Optional[str]:
    """Match paths used in total_speed_clips (see _total_speed_clips_csv_canonical)."""
    try:
        return _total_speed_clips_csv_canonical(raw)
    except (ValueError, OSError):
        try:
            return str(Path(raw).expanduser().resolve())
        except OSError:
            return None


def _video_total_speed_csv_paths_for_video(
    canon_video_path: str,
    project_name: Optional[str] = None,
) -> List[str]:
    """
    Distinct total_speed.csv paths for this video: QuickRun artifact index, optionally scoped
    by project, plus a filename-prefix fallback so clips match the interactive plot even if
    artifact rows were never written (CSV basename starts with '<video_filename>_').

    Paths are normalized the same way as /api/total_speed_clips keys.
    """
    _quickrun_ensure_db()
    ordered: List[str] = []
    seen: set[str] = set()

    def add_raw_paths(raw_iter: List[str]) -> None:
        for raw in raw_iter:
            rp = str(raw).strip()
            if not rp:
                continue
            canon = _canonical_total_speed_csv_path(rp)
            if not canon:
                continue
            if canon not in seen:
                seen.add(canon)
                ordered.append(canon)

    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        try:
            conn.execute("PRAGMA foreign_keys=ON")

            def artifact_rows(for_project: Optional[str]) -> List[sqlite3.Row]:
                if for_project and _is_valid_project_name(for_project):
                    return list(conn.execute(
                        """
                        SELECT DISTINCT artifact_path FROM quickrun_video_artifacts
                        WHERE video_path = ? AND artifact_kind = ?
                          AND project = ?
                        ORDER BY artifact_path
                        """,
                        (canon_video_path, "total_speed_csv", for_project),
                    ).fetchall())
                return list(conn.execute(
                    """
                    SELECT DISTINCT artifact_path FROM quickrun_video_artifacts
                    WHERE video_path = ? AND artifact_kind = ?
                    ORDER BY artifact_path
                    """,
                    (canon_video_path, "total_speed_csv"),
                ).fetchall())

            rows = artifact_rows(project_name)
            if not rows and project_name and _is_valid_project_name(project_name):
                rows = artifact_rows(None)
            add_raw_paths([str(r[0]) for r in rows])

            vfile = Path(canon_video_path).name
            prefix = vfile + "_"
            clip_rows = conn.execute(
                "SELECT DISTINCT csv_path FROM total_speed_clips ORDER BY csv_path",
            ).fetchall()
            guess: List[str] = []
            for (csv_p,) in clip_rows:
                cn = _canonical_total_speed_csv_path(str(csv_p))
                if cn and Path(cn).name.startswith(prefix):
                    guess.append(str(csv_p))
            add_raw_paths(guess)
        finally:
            conn.close()

    return ordered


def _video_subclips_for_video(
    canon_video_path: str,
    project_name: Optional[str] = None,
) -> List[dict]:
    """Saved clips from Total speed — interactive plot (/api/total_speed_clips) for this video."""
    paths = _video_total_speed_csv_paths_for_video(canon_video_path, project_name)
    tracking_dirs: Dict[int, str] = {}
    _quickrun_ensure_db()
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")

            def q_rows(for_project: Optional[str]) -> List[sqlite3.Row]:
                if for_project and _is_valid_project_name(for_project):
                    return list(conn.execute(
                        """
                        SELECT clip_scope, artifact_path
                        FROM quickrun_video_artifacts
                        WHERE video_path = ?
                          AND artifact_kind = 'output_directory'
                          AND clip_scope > 0
                          AND project = ?
                          AND lower(COALESCE(artifact_label, '')) LIKE '%tracking%'
                        ORDER BY COALESCE(finished_at, '') DESC, rowid DESC
                        """,
                        (canon_video_path, for_project),
                    ).fetchall())
                return list(conn.execute(
                    """
                    SELECT clip_scope, artifact_path
                    FROM quickrun_video_artifacts
                    WHERE video_path = ?
                      AND artifact_kind = 'output_directory'
                      AND clip_scope > 0
                      AND lower(COALESCE(artifact_label, '')) LIKE '%tracking%'
                    ORDER BY COALESCE(finished_at, '') DESC, rowid DESC
                    """,
                    (canon_video_path,),
                ).fetchall())

            rows = q_rows(project_name)
            if not rows and project_name and _is_valid_project_name(project_name):
                rows = q_rows(None)
            for r in rows:
                try:
                    cid = int(r["clip_scope"])
                except (TypeError, ValueError):
                    continue
                if cid in tracking_dirs:
                    continue
                p = str(r["artifact_path"] or "").strip()
                if not p:
                    continue
                try:
                    dp = Path(p).resolve()
                except OSError:
                    continue
                if not dp.is_dir():
                    continue
                tracking_dirs[cid] = str(dp)
        finally:
            conn.close()

    out: List[dict] = []
    seen: set[int] = set()
    for csv_p in paths:
        for c in _total_speed_clips_fetch_all(csv_p):
            cid = int(c["id"])
            if cid in seen:
                continue
            seen.add(cid)
            row = dict(c)
            row["source_csv"] = csv_p
            row["tracking_output_dir"] = tracking_dirs.get(cid)
            row["has_tracking"] = bool(row["tracking_output_dir"])
            out.append(row)
    out.sort(key=lambda x: (float(x["start"]), int(x["id"])))
    return out


_QUICKRUN_ARTIFACT_KIND_ORDER = {
    "detection_csv": 0,
    "tracked_json": 1,
    "total_speed_csv": 2,
    "weights": 3,
    "output_directory": 4,
    "plot": 5,
}


def _quickrun_normalize_artifact_path(path_str: Optional[str]) -> Optional[str]:
    if path_str is None:
        return None
    raw = str(path_str).strip()
    if not raw:
        return None
    try:
        return str(Path(raw).expanduser().resolve())
    except OSError:
        return raw


def _quickrun_clip_scope_from_entry(entry: Optional[dict]) -> int:
    """-1 = main video row; positive DB id = one subclip’s artifacts only."""
    if not isinstance(entry, dict):
        return -1
    if entry.get("subclip_clip_id") is not None:
        try:
            return int(entry["subclip_clip_id"])
        except (TypeError, ValueError):
            return -1
    return -1


def _quickrun_outputs_to_artifact_entries(outputs: dict) -> List[tuple[str, str, str]]:
    """Return (kind, normalized_path, label) for indexing."""
    out: List[tuple[str, str, str]] = []
    if not isinstance(outputs, dict):
        return out

    def add(kind: str, raw: Optional[str], label: str) -> None:
        np = _quickrun_normalize_artifact_path(raw)
        if np:
            out.append((kind, np, label))

    add("detection_csv", outputs.get("detection_csv"), "Detection CSV")
    add("tracked_json", outputs.get("tracked_json"), "Tracked JSON")
    add("total_speed_csv", outputs.get("total_speed_csv"), "Total speed table (CSV)")
    add("weights", outputs.get("weights_resolved"), "Weights")
    add("output_directory", outputs.get("output_directory"), "Output directory")
    if outputs.get("job_kind") == "snapshot" and outputs.get("snapshot_save_dir"):
        add("output_directory", outputs.get("snapshot_save_dir"), "Snapshot output")
    if outputs.get("job_kind") == "tracking" and outputs.get("tracking_save_dir"):
        add("output_directory", outputs.get("tracking_save_dir"), "Tracking output")
    for p in outputs.get("plots") or []:
        if isinstance(p, dict):
            lab = str(p.get("label") or "").strip() or "Plot"
            add("plot", p.get("path"), lab)
    return out


def _quickrun_upsert_video_artifacts_conn(
    conn: sqlite3.Connection,
    video_path: str,
    project: str,
    session_id: str,
    job_key: str,
    finished_at: Optional[str],
    pipeline_params: dict,
    outputs: dict,
    clip_scope: int = -1,
) -> None:
    entries = _quickrun_outputs_to_artifact_entries(outputs)
    if not entries:
        return
    params_json = json.dumps(pipeline_params)
    sql = """
        INSERT INTO quickrun_video_artifacts (
          video_path, artifact_path, artifact_kind, artifact_label, project,
          session_id, job_key, finished_at, pipeline_params_json, clip_scope
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_path, artifact_path, clip_scope) DO UPDATE SET
          artifact_kind = excluded.artifact_kind,
          artifact_label = excluded.artifact_label,
          project = excluded.project,
          session_id = excluded.session_id,
          job_key = excluded.job_key,
          finished_at = excluded.finished_at,
          pipeline_params_json = excluded.pipeline_params_json
    """
    for kind, apath, label in entries:
        conn.execute(
            sql,
            (
                video_path,
                apath,
                kind,
                label,
                project,
                session_id,
                job_key,
                finished_at,
                params_json,
                int(clip_scope),
            ),
        )


def _quickrun_persist_video_artifacts_for_job(
    video_path: str,
    project: str,
    session_id: str,
    job_key: str,
    finished_at: Optional[str],
    pipeline_params: dict,
    outputs: Optional[dict],
    entry_snapshot: Optional[dict] = None,
) -> None:
    if not outputs or not isinstance(outputs, dict):
        return
    clip_scope = _quickrun_clip_scope_from_entry(entry_snapshot)
    _quickrun_ensure_db()
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            _quickrun_upsert_video_artifacts_conn(
                conn,
                video_path,
                project,
                session_id,
                job_key,
                finished_at,
                pipeline_params,
                outputs,
                clip_scope,
            )
            conn.commit()
        finally:
            conn.close()


def _quickrun_sync_artifacts_from_jobs_for_video(canon_path: str) -> None:
    """Copy outputs from completed jobs into quickrun_video_artifacts (idempotent)."""
    _quickrun_ensure_db()
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            rows = conn.execute(
                """
                SELECT j.outputs_json, j.entry_snapshot_json, j.video_path, j.job_key,
                       j.finished_at,
                       s.id AS session_id, s.project, s.pipeline_params_json
                FROM quickrun_jobs j
                JOIN quickrun_sessions s ON s.id = j.session_id
                WHERE j.video_path = ? AND j.status = 'done'
                  AND COALESCE(j.outputs_json, '') != ''
                ORDER BY COALESCE(j.finished_at, j.started_at, '') ASC,
                         COALESCE(j.started_at, '') ASC
                """,
                (canon_path,),
            ).fetchall()
            for r in rows:
                try:
                    outs = json.loads(r["outputs_json"])
                except (json.JSONDecodeError, TypeError):
                    continue
                try:
                    pparams = json.loads(r["pipeline_params_json"])
                except (json.JSONDecodeError, TypeError):
                    pparams = {}
                ent: dict = {}
                try:
                    ent = json.loads(r["entry_snapshot_json"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    ent = {}
                cs = _quickrun_clip_scope_from_entry(ent)
                _quickrun_upsert_video_artifacts_conn(
                    conn,
                    str(r["video_path"]),
                    str(r["project"]),
                    str(r["session_id"]),
                    str(r["job_key"]),
                    r["finished_at"],
                    pparams,
                    outs,
                    cs,
                )
            conn.commit()
        finally:
            conn.close()


def _quickrun_fetch_video_artifacts(
    canon_path: str,
    project_name: Optional[str],
    clip_scope: int = -1,
) -> List[dict]:
    """Unique paths per video from quickrun_video_artifacts (survives session deletion)."""
    _quickrun_ensure_db()
    use_project = bool(project_name and _is_valid_project_name(project_name))
    clip_scope = int(clip_scope)
    rows: List[sqlite3.Row] = []
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            if use_project:
                rows = conn.execute(
                    """
                    SELECT rowid AS rid, artifact_path, artifact_kind, artifact_label, project,
                           session_id, job_key, finished_at, pipeline_params_json, clip_scope
                    FROM quickrun_video_artifacts
                    WHERE video_path = ? AND project = ? AND clip_scope = ?
                    """,
                    (canon_path, project_name, clip_scope),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT rowid AS rid, artifact_path, artifact_kind, artifact_label, project,
                           session_id, job_key, finished_at, pipeline_params_json, clip_scope
                    FROM quickrun_video_artifacts
                    WHERE video_path = ? AND clip_scope = ?
                    """,
                    (canon_path, clip_scope),
                ).fetchall()
            stale_rowids: List[int] = []
            for r in rows:
                apath = str(r["artifact_path"])
                try:
                    p = Path(apath)
                    exists = p.is_file() or p.is_dir()
                except OSError:
                    exists = False
                if not exists:
                    try:
                        stale_rowids.append(int(r["rid"]))
                    except (TypeError, ValueError):
                        pass
            if stale_rowids:
                marks = ",".join("?" for _ in stale_rowids)
                conn.execute(
                    f"DELETE FROM quickrun_video_artifacts WHERE rowid IN ({marks})",
                    tuple(stale_rowids),
                )
                conn.commit()
        finally:
            conn.close()

    def kind_rank(k: str) -> int:
        return _QUICKRUN_ARTIFACT_KIND_ORDER.get(k, 99)

    out: List[dict] = []
    for r in rows:
        apath = str(r["artifact_path"])
        try:
            p = Path(apath)
            exists = p.is_file() or p.is_dir()
        except OSError:
            exists = False
        if not exists:
            continue
        kind = str(r["artifact_kind"])
        params: dict = {}
        raw_p = r["pipeline_params_json"]
        if raw_p:
            try:
                params = json.loads(raw_p)
            except (json.JSONDecodeError, TypeError):
                params = {}
        out.append({
            "path": apath,
            "kind": kind,
            "label": str(r["artifact_label"] or _default_label_for_artifact_kind(kind)),
            "exists": exists,
            "project": str(r["project"]),
            "session_id": r["session_id"],
            "job_key": r["job_key"],
            "finished_at": r["finished_at"],
            "pipeline_params": params,
            "clip_scope": int(r["clip_scope"]) if r["clip_scope"] is not None else -1,
            "_sort_kind": kind_rank(kind),
        })
    out.sort(key=lambda x: (x["_sort_kind"], x["path"]))
    for x in out:
        del x["_sort_kind"]
    return out


def _default_label_for_artifact_kind(kind: str) -> str:
    return {
        "detection_csv": "Detection CSV",
        "tracked_json": "Tracked JSON",
        "total_speed_csv": "Total speed table (CSV)",
        "weights": "Weights",
        "output_directory": "Output directory",
        "plot": "Plot",
    }.get(kind, kind)


def _quickrun_failed_jobs_for_video(
    canon_path: str,
    project_name: Optional[str],
    clip_scope: int = -1,
) -> List[dict]:
    """Failed jobs still tied to an existing session (for troubleshooting)."""
    _quickrun_ensure_db()
    limit = 40
    use_project = bool(project_name and _is_valid_project_name(project_name))
    clip_scope = int(clip_scope)
    scope_sql = (
        " AND json_extract(j.entry_snapshot_json, '$.subclip_clip_id') IS NULL"
        if clip_scope == -1
        else " AND json_extract(j.entry_snapshot_json, '$.subclip_clip_id') = ?"
    )
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            base_sql = """
                SELECT j.session_id, j.job_key, j.status, j.error_message,
                       j.finished_at, j.exit_code, j.log_path,
                       s.project, s.created_at AS session_created_at
                FROM quickrun_jobs j
                JOIN quickrun_sessions s ON s.id = j.session_id
                WHERE j.video_path = ? AND j.status = 'failed'
            """ + scope_sql
            if use_project:
                q = base_sql + " AND s.project = ? ORDER BY s.created_at DESC LIMIT ?"
                params: Tuple[Any, ...] = (canon_path,)
                if clip_scope != -1:
                    params = params + (clip_scope,)
                params = params + (project_name, limit)
                rows = conn.execute(q, params).fetchall()
            else:
                q = base_sql + " ORDER BY s.created_at DESC LIMIT ?"
                params = (canon_path,)
                if clip_scope != -1:
                    params = params + (clip_scope,)
                params = params + (limit,)
                rows = conn.execute(q, params).fetchall()
        finally:
            conn.close()

    out: List[dict] = []
    for r in rows:
        out.append({
            "session_id": r["session_id"],
            "job_key": r["job_key"],
            "job_status": r["status"],
            "error_message": r["error_message"],
            "finished_at": r["finished_at"],
            "exit_code": r["exit_code"],
            "log_path": r["log_path"],
            "project": r["project"],
            "session_created_at": r["session_created_at"],
        })
    return out


def _quickrun_delete_session_artifacts(sid: str) -> None:
    """Remove per-session TSV/log directory under QUICKRUN_SESSIONS_ROOT (best effort)."""
    d = QUICKRUN_SESSIONS_ROOT / sid
    if d.is_dir():
        try:
            shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def _quickrun_list_history(limit: int = 50) -> List[dict]:
    _quickrun_ensure_db()
    limit = max(1, min(int(limit), 200))
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            rows = conn.execute(
                """
                SELECT s.id, s.project, s.created_at, s.finished_at, s.session_status,
                  COUNT(j.job_key) AS job_total,
                  COALESCE(SUM(CASE WHEN j.status = 'done' THEN 1 ELSE 0 END), 0) AS job_done,
                  COALESCE(SUM(CASE WHEN j.status = 'failed' THEN 1 ELSE 0 END), 0) AS job_failed,
                  COALESCE(SUM(CASE WHEN j.status = 'processing' THEN 1 ELSE 0 END), 0) AS job_processing,
                  COALESCE(SUM(CASE WHEN j.status = 'queued' THEN 1 ELSE 0 END), 0) AS job_queued
                FROM quickrun_sessions s
                LEFT JOIN quickrun_jobs j ON j.session_id = s.id
                GROUP BY s.id, s.project, s.created_at, s.finished_at, s.session_status
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()

    out: List[dict] = []
    for r in rows:
        out.append({
            "id": r["id"],
            "project": r["project"],
            "created_at": r["created_at"],
            "finished_at": r["finished_at"],
            "session_status": r["session_status"],
            "job_total": int(r["job_total"] or 0),
            "job_done": int(r["job_done"] or 0),
            "job_failed": int(r["job_failed"] or 0),
            "job_processing": int(r["job_processing"] or 0),
            "job_queued": int(r["job_queued"] or 0),
        })
    return out


def _quickrun_insert_session(sess: dict) -> None:
    """Persist a new session and all jobs (initial QuickRun submit)."""
    _quickrun_ensure_db()
    sid = sess["id"]
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                """
                INSERT INTO quickrun_sessions (
                  id, project, created_at, finished_at, session_status,
                  pipeline_params_json, workdir, fatal_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    sess["project"],
                    sess["created_at"],
                    sess.get("finished_at"),
                    sess["session_status"],
                    json.dumps(sess["pipeline_params"]),
                    sess["workdir"],
                    sess.get("fatal_error"),
                ),
            )
            for i, job in enumerate(sess["jobs"]):
                conn.execute(
                    """
                    INSERT INTO quickrun_jobs (
                      session_id, sort_order, job_key, video_path, video_label, status,
                      entry_snapshot_json, tsv_path, log_path, pid, started_at, finished_at,
                      exit_code, error_message, log_tail, outputs_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sid,
                        i,
                        job["id"],
                        job["video_path"],
                        job["video_label"],
                        job["status"],
                        json.dumps(job["entry_snapshot"]),
                        job["tsv_path"],
                        job["log_path"],
                        job.get("pid"),
                        job.get("started_at"),
                        job.get("finished_at"),
                        job.get("exit_code"),
                        job.get("error_message"),
                        job.get("log_tail") or "",
                        json.dumps(job["outputs"]) if job.get("outputs") is not None else None,
                    ),
                )
            conn.commit()
        finally:
            conn.close()


def _load_quickrun_session(sid: str) -> Optional[dict]:
    _quickrun_ensure_db()
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            srow = conn.execute(
                "SELECT * FROM quickrun_sessions WHERE id = ?", (sid,)
            ).fetchone()
            if srow is None:
                return None
            jrows = conn.execute(
                "SELECT * FROM quickrun_jobs WHERE session_id = ? ORDER BY sort_order",
                (sid,),
            ).fetchall()
        finally:
            conn.close()

    jobs: List[dict] = []
    for jr in jrows:
        out_raw = jr["outputs_json"]
        jobs.append({
            "id": jr["job_key"],
            "video_path": jr["video_path"],
            "video_label": jr["video_label"],
            "status": jr["status"],
            "entry_snapshot": json.loads(jr["entry_snapshot_json"]),
            "tsv_path": jr["tsv_path"],
            "log_path": jr["log_path"],
            "pid": jr["pid"],
            "started_at": jr["started_at"],
            "finished_at": jr["finished_at"],
            "exit_code": jr["exit_code"],
            "error_message": jr["error_message"],
            "log_tail": jr["log_tail"] or "",
            "outputs": json.loads(out_raw) if out_raw else None,
        })
    return {
        "id": srow["id"],
        "project": srow["project"],
        "created_at": srow["created_at"],
        "finished_at": srow["finished_at"],
        "session_status": srow["session_status"],
        "pipeline_params": json.loads(srow["pipeline_params_json"]),
        "workdir": srow["workdir"],
        "fatal_error": srow["fatal_error"],
        "jobs": jobs,
    }


def _save_quickrun_session(sess: dict) -> None:
    """Update session header and every job row from the in-memory session dict."""
    _quickrun_ensure_db()
    sid = sess["id"]
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                """
                UPDATE quickrun_sessions SET
                  finished_at = ?, session_status = ?, fatal_error = ?
                WHERE id = ?
                """,
                (
                    sess.get("finished_at"),
                    sess["session_status"],
                    sess.get("fatal_error"),
                    sid,
                ),
            )
            for i, job in enumerate(sess["jobs"]):
                conn.execute(
                    """
                    UPDATE quickrun_jobs SET
                      sort_order = ?, video_path = ?, video_label = ?, status = ?,
                      entry_snapshot_json = ?, tsv_path = ?, log_path = ?,
                      pid = ?, started_at = ?, finished_at = ?,
                      exit_code = ?, error_message = ?, log_tail = ?, outputs_json = ?
                    WHERE session_id = ? AND job_key = ?
                    """,
                    (
                        i,
                        job["video_path"],
                        job["video_label"],
                        job["status"],
                        json.dumps(job["entry_snapshot"]),
                        job["tsv_path"],
                        job["log_path"],
                        job.get("pid"),
                        job.get("started_at"),
                        job.get("finished_at"),
                        job.get("exit_code"),
                        job.get("error_message"),
                        job.get("log_tail") or "",
                        json.dumps(job["outputs"]) if job.get("outputs") is not None else None,
                        sid,
                        job["id"],
                    ),
                )
            conn.commit()
        finally:
            conn.close()


def _quickrun_fail_session(sess: dict, msg: str) -> None:
    sess["session_status"] = "failed"
    sess["finished_at"] = _utc_now_iso()
    sess["fatal_error"] = msg
    for j in sess["jobs"]:
        if j["status"] == "queued":
            j["status"] = "failed"
            j["error_message"] = msg
            j["finished_at"] = _utc_now_iso()
    _save_quickrun_session(sess)


def _quickrun_snapshot_outputs_exist(entry: dict, params: dict) -> bool:
    """True if snapshot folder exists and raw PNG from manifest is on disk."""
    try:
        base = Path(params["snapshot_project_base"])
        run_name = entry.get("snapshot_run_name")
        if not run_name:
            return False
        save_dir = (base / str(run_name)).resolve()
    except (TypeError, OSError, ValueError):
        return False
    if not save_dir.is_dir():
        return False
    try:
        man = _snapshot_output_manifest(save_dir)
    except Exception:
        return False
    raw = man.get("raw_image_abs")
    if not raw:
        return False
    try:
        return Path(raw).is_file()
    except OSError:
        return False


def _quickrun_fastview_detection_csv_exists(video_abs: str, params: dict) -> bool:
    """True if FastView detection CSV for this video/settings already exists."""
    outs = _quickrun_compute_outputs(video_abs, params)
    if outs.get("resolve_error"):
        return False
    dc = outs.get("detection_csv")
    if not dc:
        return False
    try:
        return Path(dc).is_file()
    except OSError:
        return False


def _quickrun_tracking_outputs_exist(entry: dict, params: dict) -> bool:
    """True if tracking output folder has at least one JSON artifact."""
    try:
        base = Path(params["tracking_project_base"])
        run_name = str(entry.get("tracking_run_name") or "").strip()
        if not run_name:
            return False
        save_dir = (base / run_name).resolve()
    except (TypeError, OSError, ValueError):
        return False
    if not save_dir.is_dir():
        return False
    try:
        return any(save_dir.glob("*.json"))
    except OSError:
        return False


def _quickrun_job_outputs_already_exist(sess: dict, job: dict) -> bool:
    entry = job["entry_snapshot"]
    params = sess["pipeline_params"]
    if entry.get("job_kind") == "snapshot":
        return _quickrun_snapshot_outputs_exist(entry, params)
    if entry.get("job_kind") == "tracking":
        return _quickrun_tracking_outputs_exist(entry, params)
    vp = entry.get("path")
    if not vp:
        return False
    return _quickrun_fastview_detection_csv_exists(str(vp), params)


def _quickrun_fill_job_outputs_done(
    sess: dict,
    sid: str,
    job: dict,
    skipped_existing: bool = False,
) -> None:
    """Populate job outputs and index artifacts (shared by subprocess success and skip-existing)."""
    ent = job["entry_snapshot"]
    if ent.get("job_kind") == "snapshot":
        pp = sess["pipeline_params"]
        base = Path(pp["snapshot_project_base"])
        save_dir = (base / ent["snapshot_run_name"]).resolve()
        job["outputs"] = {
            "job_kind": "snapshot",
            "snapshot_save_dir": str(save_dir),
        }
        if skipped_existing:
            job["outputs"]["skipped_existing"] = True
        _quickrun_persist_video_artifacts_for_job(
            str(ent["path"]),
            sess["project"],
            sid,
            job["id"],
            job["finished_at"],
            sess["pipeline_params"],
            job["outputs"],
            ent,
        )
    elif ent.get("job_kind") == "tracking":
        pp = sess["pipeline_params"]
        save_dir = Path(str(ent["tracking_save_dir"])).resolve()
        video_name = Path(str(ent["path"])).name
        expected_json = save_dir / f"{video_name}_{int(ent['tracking_frame_start'])}_.json"
        tracked_json: Optional[Path] = expected_json if expected_json.is_file() else None
        if tracked_json is None:
            try:
                jfiles = sorted(save_dir.glob("*.json"))
                tracked_json = jfiles[0] if jfiles else None
            except OSError:
                tracked_json = None
        det_csv = save_dir / video_name
        if not det_csv.suffix.lower() == ".csv":
            det_csv = save_dir / f"{video_name}.csv"
        job["outputs"] = {
            "job_kind": "tracking",
            "tracking_save_dir": str(save_dir),
            "tracked_json": str(tracked_json.resolve()) if tracked_json and tracked_json.is_file() else None,
            "detection_csv": str(det_csv.resolve()) if det_csv.is_file() else None,
            "weights_resolved": str(pp.get("weights", "")) if pp.get("weights") else None,
        }
        if skipped_existing:
            job["outputs"]["skipped_existing"] = True
        _quickrun_persist_video_artifacts_for_job(
            str(ent["path"]),
            sess["project"],
            sid,
            job["id"],
            job["finished_at"],
            sess["pipeline_params"],
            job["outputs"],
            ent,
        )
    else:
        vp = ent["path"]
        job["outputs"] = _quickrun_compute_outputs(vp, sess["pipeline_params"])
        if skipped_existing:
            job["outputs"]["skipped_existing"] = True
        _quickrun_persist_video_artifacts_for_job(
            vp,
            sess["project"],
            sid,
            job["id"],
            job["finished_at"],
            sess["pipeline_params"],
            job["outputs"],
            ent,
        )


def _quickrun_run_session_worker(sid: str) -> None:
    sess0 = _load_quickrun_session(sid)
    if sess0 is None:
        return
    sk = sess0["pipeline_params"].get("session_kind", "fastview")
    if sk in ("snapshot", "tracking"):
        if not (ROOT / "detect_2.py").is_file():
            _quickrun_fail_session(sess0, "detect_2.py not found")
            return
    else:
        script_chk = ROOT / "FastView" / "fastview_pipeline.py"
        if not script_chk.is_file():
            _quickrun_fail_session(sess0, "FastView/fastview_pipeline.py not found")
            return

    script = ROOT / "FastView" / "fastview_pipeline.py"

    while True:
        sess = _load_quickrun_session(sid)
        if sess is None:
            return

        idx: Optional[int] = None
        for i, job in enumerate(sess["jobs"]):
            if job["status"] == "queued":
                idx = i
                break

        if idx is None:
            sess["session_status"] = "complete"
            sess["finished_at"] = _utc_now_iso()
            _save_quickrun_session(sess)
            return

        job = sess["jobs"][idx]
        job["status"] = "processing"
        job["started_at"] = _utc_now_iso()
        job["error_message"] = None
        job["exit_code"] = None
        job["log_tail"] = ""
        job["outputs"] = None
        _save_quickrun_session(sess)

        log_path = Path(job["log_path"])
        params = sess["pipeline_params"]
        entry = job["entry_snapshot"]

        skip_existing = (
            not params.get("rerun")
            and _quickrun_job_outputs_already_exist(sess, job)
        )
        if skip_existing:
            msg = (
                "Skipped: outputs already exist on disk for this job. "
                "Enable “Rerun even if outputs exist” in QuickRun to force a fresh run.\n"
            )
            try:
                log_path.write_text(msg, encoding="utf-8")
            except OSError:
                pass
            job["pid"] = None
            job["finished_at"] = _utc_now_iso()
            job["exit_code"] = 0
            job["error_message"] = None
            job["log_tail"] = msg.strip()
            job["status"] = "done"
            _quickrun_fill_job_outputs_done(sess, sid, job, skipped_existing=True)
            _save_quickrun_session(sess)
            continue

        if entry.get("job_kind") == "snapshot":
            cmd = _build_snapshot_detect_cmd(entry, params)
        elif entry.get("job_kind") == "tracking":
            cmd = _build_tracking_detect_cmd(entry, params)
        else:
            tsv = Path(job["tsv_path"])
            cmd = _build_quickrun_cmd(script, tsv, params)

        rc = 0
        try:
            with log_path.open("w", encoding="utf-8") as log_f:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(FASTVIEW_WORKDIR.resolve()),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                job["pid"] = proc.pid
                _save_quickrun_session(sess)
                rc = proc.wait()
        except Exception as exc:
            rc = -1
            sess_err = _load_quickrun_session(sid)
            if sess_err:
                jerr = sess_err["jobs"][idx]
                jerr["error_message"] = str(exc)
                jerr["pid"] = None
                _save_quickrun_session(sess_err)

        sess = _load_quickrun_session(sid)
        if sess is None:
            return
        job = sess["jobs"][idx]
        job["pid"] = None
        job["finished_at"] = _utc_now_iso()
        job["exit_code"] = rc
        job["log_tail"] = _read_log_tail(Path(job["log_path"]), 80)

        if rc == 0 and not job.get("error_message"):
            job["status"] = "done"
            _quickrun_fill_job_outputs_done(sess, sid, job, skipped_existing=False)
        else:
            job["status"] = "failed"
            if not job.get("error_message"):
                job["error_message"] = f"pipeline exited with code {rc}"

        _save_quickrun_session(sess)


@app.post("/api/quickrun/start")
def api_quickrun_start():
    """Queue one FastView pipeline job per video and open `/quickrun?session=…` to track progress."""
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    filter_paths = data.get("video_paths")
    if filter_paths is not None and not isinstance(filter_paths, list):
        return jsonify({"error": "video_paths must be a list or omitted"}), 400
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    projects = _read_projects_db()
    project = next((p for p in projects if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404

    fps_list = [str(x).strip() for x in (filter_paths or []) if str(x).strip()]
    entries = _quickrun_video_entries(project, fps_list if fps_list else None)
    if not entries:
        return jsonify({"error": "no videos to process"}), 400

    params, err = _parse_quick_run_pipeline_params(data)
    if err:
        return jsonify({"error": err}), 400
    assert params is not None

    limit_n = int(params["limit_n"])
    if limit_n > 0:
        entries = entries[:limit_n]

    script = ROOT / "FastView" / "fastview_pipeline.py"
    if not script.is_file():
        return jsonify({"error": "FastView/fastview_pipeline.py not found"}), 404

    sid = uuid.uuid4().hex
    sdir = QUICKRUN_SESSIONS_ROOT / sid
    try:
        sdir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500

    jobs: List[dict] = []
    for i, ent in enumerate(entries):
        tsv_path = sdir / f"{i:04d}.tsv"
        log_path = sdir / f"{i:04d}.log"
        try:
            _write_fastview_pipeline_tsv([ent], tsv_path)
        except OSError as exc:
            return jsonify({"error": str(exc)}), 500
        snap = _video_entry_snapshot(ent)
        jobs.append({
            "id": f"j{i}",
            "video_path": snap["path"],
            "video_label": snap["filename"],
            "status": "queued",
            "entry_snapshot": snap,
            "tsv_path": str(tsv_path.resolve()),
            "log_path": str(log_path.resolve()),
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "error_message": None,
            "log_tail": "",
            "outputs": None,
        })

    params_jobs = {**params, "limit_n": 0}
    sess = {
        "id": sid,
        "project": name,
        "created_at": _utc_now_iso(),
        "finished_at": None,
        "session_status": "running",
        "pipeline_params": params_jobs,
        "workdir": str(FASTVIEW_WORKDIR.resolve()),
        "jobs": jobs,
    }
    _quickrun_insert_session(sess)

    threading.Thread(target=_quickrun_run_session_worker, args=(sid,), daemon=True).start()

    return jsonify({
        "ok": True,
        "session_id": sid,
        "job_count": len(jobs),
        "quickrun_url": f"/quickrun?session={sid}",
    })


@app.get("/api/quickrun/session/<sid>")
def api_quickrun_get_session(sid: str):
    if not re.fullmatch(r"[0-9a-f]{32}", sid):
        return jsonify({"error": "invalid session id"}), 400
    sess = _load_quickrun_session(sid)
    if not sess:
        return jsonify({"error": "session not found"}), 404

    jobs_out: List[dict] = []
    for j in sess["jobs"]:
        jc = dict(j)
        if jc["status"] == "processing":
            lp = Path(jc["log_path"])
            jc["log_tail"] = _read_log_tail(lp, 80)
        jobs_out.append(jc)

    payload = dict(sess)
    payload["jobs"] = jobs_out
    done = sum(1 for x in jobs_out if x["status"] == "done")
    failed = sum(1 for x in jobs_out if x["status"] == "failed")
    queued = sum(1 for x in jobs_out if x["status"] == "queued")
    processing = sum(1 for x in jobs_out if x["status"] == "processing")
    payload["counts"] = {
        "total": len(jobs_out),
        "queued": queued,
        "processing": processing,
        "done": done,
        "failed": failed,
    }
    return jsonify({"ok": True, "session": payload})


@app.get("/api/quickrun/history")
def api_quickrun_history():
    """List recent QuickRun sessions from the database (newest first)."""
    raw_limit = request.args.get("limit", "50")
    try:
        limit = int(raw_limit)
    except ValueError:
        limit = 50
    rows = _quickrun_list_history(limit)
    return jsonify({"ok": True, "sessions": rows})


@app.delete("/api/quickrun/session/<sid>")
def api_quickrun_delete_session(sid: str):
    """Remove a session record (and jobs) from the DB and delete its /tmp artifact folder."""
    if not re.fullmatch(r"[0-9a-f]{32}", sid):
        return jsonify({"error": "invalid session id"}), 400
    _quickrun_ensure_db()
    with _QUICKRUN_SESSION_LOCK:
        conn = sqlite3.connect(str(QUICKRUN_DB_PATH), timeout=60.0)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            cur = conn.execute("DELETE FROM quickrun_sessions WHERE id = ?", (sid,))
            conn.commit()
            deleted = cur.rowcount
        finally:
            conn.close()
    if deleted == 0:
        return jsonify({"error": "session not found"}), 404
    _quickrun_delete_session_artifacts(sid)
    return jsonify({"ok": True, "deleted": sid})


@app.get("/api/quickrun/results_for_video")
def api_quickrun_results_for_video():
    """Deduped output files for one video (indexed table survives session deletion)."""
    raw_vp = str(request.args.get("video_path", "")).strip()
    if not raw_vp:
        return jsonify({"error": "video_path is required"}), 400
    try:
        canon = _resolve_stored_video_path(raw_vp)
    except (ValueError, OSError):
        try:
            canon = str(Path(raw_vp).expanduser().resolve())
        except OSError:
            return jsonify({"error": "invalid video_path"}), 400
    project_q = str(request.args.get("project", "")).strip()
    proj_filter = project_q if _is_valid_project_name(project_q) else ""
    proj_arg = proj_filter if proj_filter else None
    _quickrun_sync_artifacts_from_jobs_for_video(canon)
    clip_raw = str(request.args.get("clip_id", "")).strip()
    clip_scope = -1
    if clip_raw:
        try:
            clip_scope = int(clip_raw)
        except ValueError:
            return jsonify({"error": "clip_id must be an integer"}), 400
        if clip_scope < 1:
            return jsonify({"error": "clip_id must be a positive integer"}), 400
    artifacts = _quickrun_fetch_video_artifacts(canon, proj_arg, clip_scope)
    failed_runs = _quickrun_failed_jobs_for_video(canon, proj_arg, clip_scope)
    return jsonify({
        "ok": True,
        "video_path": canon,
        "project_filter": proj_filter,
        "clip_id": clip_scope if clip_scope != -1 else None,
        "artifacts": artifacts,
        "failed_runs": failed_runs,
    })


@app.post("/api/project/quick_run_fastview")
def api_quick_run_fastview():
    """
    Write a unique /tmp TSV from project videos and start FastView/fastview_pipeline.py in the background.
    Default args match FastView/fastview_pipeline.py plus QuickRun output dir QuickTestForAging.
    Optional JSON keys: workers, window_overlap, speed_window, output_dir, weights, frame_skip,
    conf_thres, iou_thres, imgsz, device, limit, skip_detect, skip_track, skip_visualize, rerun.
    """
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    filter_paths = data.get("video_paths")
    if filter_paths is not None and not isinstance(filter_paths, list):
        return jsonify({"error": "video_paths must be a list or omitted"}), 400
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    projects = _read_projects_db()
    project = next((p for p in projects if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404

    fps_list = [str(x).strip() for x in (filter_paths or []) if str(x).strip()]
    entries = _quickrun_video_entries(project, fps_list if fps_list else None)
    if not entries:
        return jsonify({"error": "no videos to process"}), 400

    uid = uuid.uuid4().hex
    tmp_tsv = Path(f"/tmp/yolofly_quickrun_{uid}.tsv")
    tmp_log = Path(f"/tmp/yolofly_quickrun_{uid}.log")
    try:
        _write_fastview_pipeline_tsv(entries, tmp_tsv)
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500

    script = ROOT / "FastView" / "fastview_pipeline.py"
    if not script.is_file():
        return jsonify({"error": "FastView/fastview_pipeline.py not found"}), 404

    params, err = _parse_quick_run_pipeline_params(data)
    if err:
        return jsonify({"error": err}), 400
    assert params is not None
    cmd = _build_quickrun_cmd(script, tmp_tsv, params)

    try:
        log_f = tmp_log.open("w", encoding="utf-8")
    except OSError as exc:
        return jsonify({"error": f"cannot write log: {exc}"}), 500

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(FASTVIEW_WORKDIR.resolve()),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        log_f.close()
        return jsonify({"error": str(exc)}), 500
    log_f.close()
    return jsonify({
        "ok": True,
        "pid": proc.pid,
        "tsv_path": str(tmp_tsv.resolve()),
        "log_path": str(tmp_log.resolve()),
        "video_count": len(entries),
        "workdir": str(FASTVIEW_WORKDIR.resolve()),
        "command": cmd,
    })


@app.put("/api/project/video_meta")
def api_put_project_video_meta():
    data = request.get_json(force=True)
    name = str(data.get("name", "")).strip()
    video_path = str(data.get("video_path", "")).strip()
    meta = data.get("meta")
    refresh_total_frames = bool(data.get("refresh_total_frames", False))
    if not _is_valid_project_name(name):
        return jsonify({"error": "invalid project name"}), 400
    if not video_path:
        return jsonify({"error": "video_path is required"}), 400
    if meta is not None and not isinstance(meta, dict):
        return jsonify({"error": "meta must be an object"}), 400
    projects = _read_projects_db()
    project = next((p for p in projects if p["name"] == name), None)
    if not project:
        return jsonify({"error": "project not found"}), 404
    vp = _resolve_stored_video_path(video_path)
    target = next(
        (e for e in project["videos"] if e["path"] == vp or e.get("absolute_path") == vp),
        None,
    )
    if not target:
        return jsonify({"error": "video not found in project"}), 404
    if meta:
        if "disk_pixel" in meta:
            target["disk_pixel"] = _optional_float(meta.get("disk_pixel"))
        if "disk_radius_mm" in meta:
            target["disk_radius_mm"] = _optional_float(meta.get("disk_radius_mm"))
        if "frame_start" in meta:
            target["frame_start"] = _optional_int(meta.get("frame_start"))
        if "frame_end" in meta:
            target["frame_end"] = _optional_int(meta.get("frame_end"))
        if "fly_count" in meta:
            target["fly_count"] = _optional_int(meta.get("fly_count"))
        if "detailed_location" in meta:
            target["detailed_location"] = str(meta.get("detailed_location", "") or "").strip()
        if "split_x" in meta:
            target["split_x"] = _optional_int(meta.get("split_x"))
        if "split_y" in meta:
            target["split_y"] = _optional_int(meta.get("split_y"))
        if "total_frames" in meta:
            target["total_frames"] = _optional_int(meta.get("total_frames"))
        if "video_width" in meta:
            target["video_width"] = _optional_int(meta.get("video_width"))
        if "video_height" in meta:
            target["video_height"] = _optional_int(meta.get("video_height"))
    if refresh_total_frames:
        info = _probe_video_stream_info(target["path"])
        if info:
            if info.get("frame_count"):
                target["total_frames"] = info["frame_count"]
            if info.get("video_width"):
                target["video_width"] = info["video_width"]
            if info.get("video_height"):
                target["video_height"] = info["video_height"]
    _write_projects_db(projects)
    return jsonify({"ok": True, "project": _project_detail_payload(project)})


def _snapshot_output_manifest(save_dir: Path) -> dict:
    """
    detect_2 --snapshot-frame output: <stem>_frameN_snapshot_raw.png and
    labels/<stem>_frameN_snapshot.txt
    """
    save_dir = save_dir.resolve()
    if not save_dir.is_dir():
        return {"ok": False, "error": "not a directory"}
    raw_list = sorted(save_dir.glob("*_snapshot_raw.png"))
    raw_p = raw_list[0] if raw_list else None
    label_abs: Optional[str] = None
    label_exists = False
    if raw_p is not None:
        stem = raw_p.stem
        if stem.endswith("_snapshot_raw"):
            label_stem = stem[:-4]
            lab = save_dir / "labels" / f"{label_stem}.txt"
            label_abs = str(lab.resolve())
            label_exists = lab.is_file()
    return {
        "ok": True,
        "save_dir": str(save_dir),
        "raw_image_abs": str(raw_p.resolve()) if raw_p else None,
        "label_abs_path": label_abs,
        "label_exists": label_exists,
    }


def _tracking_output_manifest(save_dir: Path, video_path: Optional[str] = None) -> dict:
    """
    detect_2 tracking output folder: contains <video_name>.csv and one or more
    <video_name>_<frame_start>_.json files. If video_path is provided, prefer matching basenames.
    """
    save_dir = save_dir.resolve()
    if not save_dir.is_dir():
        return {"ok": False, "error": "not a directory"}
    csv_list = sorted(save_dir.glob("*.csv"))
    json_list = sorted(save_dir.glob("*.json"))
    if video_path:
        base = Path(str(video_path)).name
        csv_match = [p for p in csv_list if p.name.startswith(base)]
        json_match = [p for p in json_list if p.name.startswith(base + "_") or p.name.startswith(base)]
        if csv_match:
            csv_list = csv_match
        if json_match:
            json_list = json_match
    csv_p = csv_list[0] if csv_list else None
    json_p = json_list[0] if json_list else None
    frame_min = None
    frame_max = None
    if json_p and json_p.is_file():
        try:
            idx = _build_tracking_index(json_p)
            frame_min = idx.get("frame_min")
            frame_max = idx.get("frame_max")
        except Exception:
            frame_min = None
            frame_max = None
    return {
        "ok": True,
        "save_dir": str(save_dir),
        "csv_abs_path": str(csv_p.resolve()) if csv_p else None,
        "json_abs_path": str(json_p.resolve()) if json_p else None,
        "video_path": video_path or "",
        "frame_min": frame_min,
        "frame_max": frame_max,
    }


@app.get("/api/snapshot_explore/manifest")
def api_snapshot_explore_manifest():
    """Resolve raw PNG + label paths for deep-linking /detect_explore?snapshot_dir=…"""
    raw = str(request.args.get("dir", "")).strip()
    if not raw:
        return jsonify({"error": "dir is required"}), 400
    try:
        save_dir = _safe_path_any(raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    man = _snapshot_output_manifest(save_dir)
    if not man.get("ok"):
        return jsonify({"error": man.get("error", "invalid directory")}), 400
    return jsonify(man)


@app.get("/api/tracking_explore/manifest")
def api_tracking_explore_manifest():
    """Resolve video/csv/json paths for deep-linking /detect_explore?tracking_dir=…"""
    raw = str(request.args.get("dir", "")).strip()
    if not raw:
        return jsonify({"error": "dir is required"}), 400
    raw_video = str(request.args.get("video_path", "")).strip()
    video_path: Optional[str] = None
    if raw_video:
        try:
            video_path = _resolve_stored_video_path(raw_video)
        except (ValueError, OSError):
            return jsonify({"error": "invalid video_path"}), 400
    try:
        save_dir = _safe_path_any(raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    man = _tracking_output_manifest(save_dir, video_path=video_path)
    if not man.get("ok"):
        return jsonify({"error": man.get("error", "invalid directory")}), 400
    if not man.get("video_path"):
        csv_abs = str(man.get("csv_abs_path") or "")
        if csv_abs:
            guess_name = Path(csv_abs).name
            if guess_name.lower().endswith(".csv"):
                guess_name = guess_name[:-4]
            for p in _read_projects_db():
                for e in p.get("videos") or []:
                    vp = _video_entry_canon_key(e)
                    if vp and Path(vp).name == guess_name:
                        man["video_path"] = vp
                        break
                if man.get("video_path"):
                    break
    return jsonify(man)


@app.get("/api/label_abs")
def api_label_abs_get():
    """Load YOLO label text from an absolute path under ALLOWED_PATH_ROOTS."""
    path_str = str(request.args.get("path", "")).strip()
    if not path_str:
        return jsonify({"error": "path is required"}), 400
    try:
        f = _safe_path_any(path_str)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if f.suffix.lower() != ".txt":
        return jsonify({"error": "only .txt label files"}), 400
    if not f.exists():
        return jsonify({"exists": False, "content": ""})
    return jsonify({"exists": True, "content": f.read_text(encoding="utf-8")})


@app.post("/api/label_abs")
def api_label_abs_post():
    """Save YOLO label text to an absolute path (creates parent dirs)."""
    data = request.get_json(force=True)
    path_str = str(data.get("path", "")).strip()
    content = str(data.get("content", ""))
    if not path_str:
        return jsonify({"error": "path is required"}), 400
    try:
        f = _safe_path_any(path_str)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if f.suffix.lower() != ".txt":
        return jsonify({"error": "only .txt label files"}), 400
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return jsonify({"ok": True, "saved": str(f.resolve())})


@app.get("/api/run_assets")
def api_run_assets():
    run = request.args.get("run", "")
    run_dir = _safe_join(RUNS_ROOT, run)
    images = _list_files(run_dir, IMAGE_EXTS)
    videos = _list_files(run_dir, VIDEO_EXTS)
    labels = [f"labels/{p}" for p in _list_files(run_dir / "labels", {".txt"})]
    return jsonify({"images": images, "videos": videos, "labels": labels})


@app.get("/api/media")
def api_media():
    run = request.args.get("run", "")
    rel = request.args.get("path", "")
    run_dir = _safe_join(RUNS_ROOT, run)
    f = _safe_join(run_dir, rel)
    if not f.exists():
        return jsonify({"error": "file not found"}), 404
    return send_file(f)


@app.get("/api/video_frame")
def api_video_frame():
    run = request.args.get("run", "")
    rel = request.args.get("path", "")
    frame_idx = int(request.args.get("frame", "1"))
    run_dir = _safe_join(RUNS_ROOT, run)
    vid = _safe_join(run_dir, rel)
    if not vid.exists():
        return jsonify({"error": "video not found"}), 404

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        return jsonify({"error": "cannot open video"}), 400
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = max(1, frame_idx)
    if frame_count > 0:
        frame_idx = min(frame_idx, frame_count)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return jsonify({"error": "cannot read frame"}), 400

    ok, enc = cv2.imencode(".jpg", frame)
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return send_file(io.BytesIO(enc.tobytes()), mimetype="image/jpeg")


@app.get("/api/label")
def api_label():
    run = request.args.get("run", "")
    rel = request.args.get("path", "")
    run_dir = _safe_join(RUNS_ROOT, run)
    f = _resolve_label_path(run_dir, rel)
    if not f.exists():
        return jsonify({"exists": False, "content": ""})
    return jsonify({"exists": True, "content": f.read_text(encoding="utf-8")})


@app.get("/api/csv_files")
def api_csv_files():
    files = _list_files(CSV_ROOT, {".csv"})
    return jsonify({"csv_files": files})


@app.get("/api/json_files")
def api_json_files():
    files = _list_files(CSV_ROOT, {".json"})
    return jsonify({"json_files": files})


@app.get("/api/csv_preview")
def api_csv_preview():
    rel = request.args.get("path", "")
    limit = int(request.args.get("limit", "200"))
    try:
        f = _safe_csv_or_any(rel)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not f.exists():
        return jsonify({"error": "csv not found"}), 404
    rows = []
    # Support both comma-separated and whitespace-separated result files.
    with f.open("r", encoding="utf-8", newline="") as fh:
        for i, line in enumerate(fh):
            t = line.strip()
            if not t:
                continue
            row = _parse_result_row(t)
            rows.append(row)
            if len(rows) >= max(1, min(limit, 2000)):
                break
    return jsonify({"rows": rows, "path": rel})


@app.get("/api/csv_detections")
def api_csv_detections():
    """Load full detection rows from csv file for frame-wise plotting."""
    rel = request.args.get("path", "")
    try:
        f = _safe_csv_or_any(rel)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not f.exists():
        return jsonify({"error": "csv not found"}), 404
    rows = []
    with f.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            t = line.strip()
            if not t:
                continue
            if "," in t:
                row = [c.strip() for c in t.split(",") if c.strip() != ""]
            else:
                row = [c for c in re.split(r"\s+", t) if c != ""]
            rows.append(row)
    return jsonify({"rows": rows, "path": rel, "count": len(rows)})


@app.get("/api/csv_index")
def api_csv_index():
    rel = request.args.get("path", "")
    try:
        f = _safe_csv_or_any(rel)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not f.exists():
        return jsonify({"error": "csv not found"}), 404
    idx = _build_csv_index(f)
    return jsonify({
        "path": rel,
        "count": idx["count"],
        "frame_min": idx["frame_min"],
        "frame_max": idx["frame_max"],
        "classes": idx["classes"],
        "class_counts": idx["class_counts"],
    })


@app.get("/api/csv_frame_boxes")
def api_csv_frame_boxes():
    rel = request.args.get("path", "")
    frame = int(request.args.get("frame", "1"))
    try:
        f = _safe_csv_or_any(rel)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not f.exists():
        return jsonify({"error": "csv not found"}), 404
    idx = _build_csv_index(f)
    return jsonify({"frame": frame, "boxes": idx["by_frame"].get(frame, [])})


@app.get("/api/tracking_index")
def api_tracking_index():
    rel = request.args.get("path", "")
    try:
        f = _safe_csv_or_any(rel)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not f.exists():
        return jsonify({"error": "json not found"}), 404
    idx = _build_tracking_index(f)
    return jsonify({
        "path": rel,
        "count": idx["count"],
        "frame_min": idx["frame_min"],
        "frame_max": idx["frame_max"],
    })


@app.get("/api/tracking_frame")
def api_tracking_frame():
    rel = request.args.get("path", "")
    frame = int(request.args.get("frame", "1"))
    history = max(0, min(int(request.args.get("history", "0")), 500))
    try:
        f = _safe_csv_or_any(rel)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not f.exists():
        return jsonify({"error": "json not found"}), 404
    idx = _build_tracking_index(f)
    frames = []
    if history > 0:
        for fr in range(max(idx["frame_min"], frame - history), frame):
            flies = idx["by_frame"].get(fr)
            if flies:
                frames.append({"frame": fr, "flies": flies})
    return jsonify({
        "frame": frame,
        "flies": idx["by_frame"].get(frame, {}),
        "history": frames,
    })


@app.get("/api/video_frame_by_path")
def api_video_frame_by_path():
    video_path = request.args.get("video_path", "")
    frame_idx = int(request.args.get("frame", "1"))
    vid = _safe_path_any(video_path)
    if not vid.exists():
        return jsonify({"error": "video not found"}), 404
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        return jsonify({"error": "cannot open video"}), 400
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = max(1, frame_idx)
    if frame_count > 0:
        frame_idx = min(frame_idx, frame_count)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return jsonify({"error": "cannot read frame"}), 400
    ok, enc = cv2.imencode(".jpg", frame)
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return send_file(io.BytesIO(enc.tobytes()), mimetype="image/jpeg")


@app.get("/api/video_file_by_path")
def api_video_file_by_path():
    video_path = request.args.get("video_path", "")
    vid = _safe_path_any(video_path)
    if not vid.exists():
        return jsonify({"error": "video not found"}), 404
    return send_file(vid, conditional=True)


@app.get("/api/local_file")
def api_local_file():
    """Serve a single file from allowed roots (for browsing QuickRun CSV / plots / JSON)."""
    path_str = str(request.args.get("path", "")).strip()
    if not path_str:
        return jsonify({"error": "path is required"}), 400
    try:
        f = _safe_path_any(path_str)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not f.exists():
        return jsonify({"error": "path not found"}), 404
    if not f.is_file():
        return jsonify({"error": "not a file"}), 400
    ext = f.suffix.lower()
    mt = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".csv": "text/csv; charset=utf-8",
        ".json": "application/json",
        ".txt": "text/plain; charset=utf-8",
        ".log": "text/plain; charset=utf-8",
    }.get(ext, "application/octet-stream")
    return send_file(f, mimetype=mt, conditional=True, max_age=120)


def _guess_csv_delimiter(sample_line: str) -> str:
    tab_n = sample_line.count("\t")
    comma_n = sample_line.count(",")
    semi_n = sample_line.count(";")
    if tab_n >= comma_n and tab_n >= semi_n and tab_n > 0:
        return "\t"
    if semi_n > comma_n:
        return ";"
    return ","


@app.get("/api/csv_table")
def api_csv_table():
    """Return CSV/TSV rows as JSON for the table viewer (bounded rows)."""
    path_str = str(request.args.get("path", "")).strip()
    raw_max = request.args.get("max_rows", "25000")
    try:
        max_body_rows = max(1, min(int(raw_max), 100_000))
    except ValueError:
        max_body_rows = 25_000

    if not path_str:
        return jsonify({"error": "path is required"}), 400
    try:
        f = _safe_path_any(path_str)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not f.exists():
        return jsonify({"error": "path not found"}), 404
    if not f.is_file():
        return jsonify({"error": "not a file"}), 400
    ext = f.suffix.lower()
    if ext not in {".csv", ".tsv", ".txt"}:
        return jsonify({"error": "unsupported type; use .csv or .tsv"}), 400

    try:
        with f.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            pos0 = fh.tell()
            first_line = fh.readline()
            if not first_line.strip():
                return jsonify({
                    "ok": True,
                    "path": str(f.resolve()),
                    "delimiter": ",",
                    "headers": [],
                    "rows": [],
                    "truncated": False,
                    "row_count_returned": 0,
                })
            delim = _guess_csv_delimiter(first_line)
            fh.seek(pos0)
            reader = csv.reader(fh, delimiter=delim)
            reader_iter = iter(reader)
            try:
                headers = [str(c) for c in next(reader_iter)]
            except StopIteration:
                headers = []
            rows_out: List[List[str]] = []
            truncated = False
            for row in reader_iter:
                rows_out.append([str(c) for c in row])
                if len(rows_out) > max_body_rows:
                    rows_out.pop()
                    truncated = True
                    break
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "ok": True,
        "path": str(f.resolve()),
        "delimiter": delim,
        "headers": headers,
        "rows": rows_out,
        "truncated": truncated,
        "row_count_returned": len(rows_out),
        "max_rows": max_body_rows,
    })


@app.get("/api/total_speed_clips")
def api_total_speed_clips_list():
    """Persisted clip bands for total_speed_plot (keyed by resolved CSV path)."""
    path_str = str(request.args.get("path", "")).strip()
    if not path_str:
        return jsonify({"error": "path is required"}), 400
    try:
        canon = _total_speed_clips_csv_canonical(path_str)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    clips = _total_speed_clips_fetch_all(canon)
    return jsonify({"ok": True, "csv_path": canon, "clips": clips})


@app.post("/api/total_speed_clips")
def api_total_speed_clips_create():
    data = request.get_json(force=True)
    path_str = str(data.get("path", "")).strip()
    if not path_str:
        return jsonify({"error": "path is required"}), 400
    try:
        canon = _total_speed_clips_csv_canonical(path_str)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        start = float(data["start"])
        end = float(data["end"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "start and end must be numbers"}), 400
    name = str(data.get("name", "") or "").strip()[:200]
    if not name:
        name = "Clip"
    color_idx = int(data.get("color_idx", 0))
    cid = _total_speed_clips_insert(canon, name, start, end, color_idx)
    return jsonify({"ok": True, "id": cid})


@app.put("/api/total_speed_clips/<int:cid>")
def api_total_speed_clips_update(cid: int):
    data = request.get_json(force=True)
    path_str = str(data.get("path", "")).strip()
    if not path_str:
        return jsonify({"error": "path is required"}), 400
    try:
        canon = _total_speed_clips_csv_canonical(path_str)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        start = float(data["start"])
        end = float(data["end"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "start and end must be numbers"}), 400
    name = str(data.get("name", "") or "").strip()[:200]
    if not name:
        name = "Clip"
    if not _total_speed_clips_update(canon, cid, name, start, end):
        return jsonify({"error": "clip not found for this file"}), 404
    return jsonify({"ok": True, "id": cid})


@app.delete("/api/total_speed_clips/<int:cid>")
def api_total_speed_clips_delete(cid: int):
    path_str = str(request.args.get("path", "")).strip()
    if not path_str:
        return jsonify({"error": "path is required"}), 400
    try:
        canon = _total_speed_clips_csv_canonical(path_str)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not _total_speed_clips_delete(canon, cid):
        return jsonify({"error": "clip not found for this file"}), 404
    return jsonify({"ok": True, "deleted": cid})


@app.get("/api/video_info_by_path")
def api_video_info_by_path():
    video_path = request.args.get("video_path", "")
    vid = _safe_path_any(video_path)
    if not vid.exists():
        return jsonify({"error": "video not found"}), 404
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        return jsonify({"error": "cannot open video"}), 400
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)))
    height = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)))
    cap.release()
    return jsonify({
        "frame_count": frame_count,
        "fps": fps,
        "width": width if width > 0 else None,
        "height": height if height > 0 else None,
    })


@app.post("/api/label")
def api_save_label():
    data = request.get_json(force=True)
    run = data.get("run", "")
    rel = data.get("path", "")
    content = data.get("content", "")
    run_dir = _safe_join(RUNS_ROOT, run)
    # Save into labels/ by default when only filename is provided.
    rel_save = rel if ("/" in rel or "\\" in rel) else f"labels/{rel}"
    f = _safe_join(run_dir, rel_save)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return jsonify({"ok": True, "saved": str(f.relative_to(run_dir)).replace("\\", "/")})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

