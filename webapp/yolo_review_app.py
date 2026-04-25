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

import io
import json
import re
from pathlib import Path
from typing import Dict, List

import cv2
from flask import Flask, jsonify, redirect, request, send_file, send_from_directory


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs" / "detect"
CSV_ROOT = ROOT / "csv"
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


@app.get("/")
def index():
    return redirect("/detect_explore")


@app.get("/detect_explore")
def detect_explore():
    return send_from_directory(app.template_folder, "index.html")


@app.get("/api/runs")
def api_runs():
    runs = []
    if RUNS_ROOT.exists():
        runs = sorted([p.name for p in RUNS_ROOT.iterdir() if p.is_dir()])
    return jsonify({"runs": runs})


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
    f = _safe_join(CSV_ROOT, rel)
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
    f = _safe_join(CSV_ROOT, rel)
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
    f = _safe_join(CSV_ROOT, rel)
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
    f = _safe_join(CSV_ROOT, rel)
    if not f.exists():
        return jsonify({"error": "csv not found"}), 404
    idx = _build_csv_index(f)
    return jsonify({"frame": frame, "boxes": idx["by_frame"].get(frame, [])})


@app.get("/api/tracking_index")
def api_tracking_index():
    rel = request.args.get("path", "")
    f = _safe_join(CSV_ROOT, rel)
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
    f = _safe_join(CSV_ROOT, rel)
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
    cap.release()
    return jsonify({"frame_count": frame_count, "fps": fps})


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

