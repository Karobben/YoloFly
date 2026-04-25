#!/usr/bin/env python3
"""
Quick quality-check pipeline for long videos.

Runs a frame-skipped detection pass, then runs the windowed high-quality
post-tracker on the resulting CSV.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKDIR = ROOT.parent


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n" + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def csv_name_for(video: Path, frame_skip: int) -> str:
    suffix = f"_Fskip_{frame_skip}" if frame_skip > 1 else ""
    return f"{video.name}{suffix}.csv"


def first_frame_from_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            return int(round(float(line.replace(",", " ").split()[0])))
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch FastView QC from ../Video_list.csv."
    )
    parser.add_argument("--video-list", default=str(WORKDIR / "Video_list.csv"), help="Video list CSV/TSV path.")
    parser.add_argument(
        "-w",
        "--weights",
        default="YoloFly/runs/train/2022_05_11_p633_1280_5l_e700_b128/weights/best.pt",
        help="YOLO weights for detect_2.py.",
    )
    parser.add_argument("--frame-skip", type=int, default=30, help="Process one frame every N frames.")
    parser.add_argument("--workers", type=int, default=64, help="Number of tracking windows/workers.")
    parser.add_argument("--window-overlap", type=int, default=200, help="Overlap in processed frames.")
    parser.add_argument("--conf-thres", "--conf", type=float, default=0.3, help="YOLO confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="", help="CUDA device, for example 0 or cpu.")
    parser.add_argument("-o", "--output-dir", default="QuickView", help="Plot output directory under ../ by default.")
    parser.add_argument("--speed-window", type=int, default=300, help="Speed smoothing window for visualization.")
    parser.add_argument("--skip-detect", action="store_true", help="Reuse an existing CSV in csv/.")
    parser.add_argument("--skip-track", action="store_true", help="Only run detection and stop.")
    parser.add_argument("--skip-visualize", action="store_true", help="Do not create FastView plots.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows, for testing.")
    return parser.parse_args()


def parse_optional_split(value: str) -> str:
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "na"}:
        return ""
    return value


def read_video_list(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in re.split(r"[\t,]+", line)]
        if len(parts) < 7:
            raise ValueError(f"{path}:{line_no} needs at least 7 columns.")
        try:
            total_flies = int(float(parts[5]))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} column 6 must be total fly number: {parts[5]}") from exc
        rows.append({
            "name": parts[0],
            "total_flies": total_flies,
            "video": parts[6],
            "split_x": parse_optional_split(parts[7]) if len(parts) >= 8 else "",
            "split_y": parse_optional_split(parts[8]) if len(parts) >= 9 else "",
        })
    return rows


def resolve_workdir_path(path_text: str) -> Path:
    p = Path(path_text).expanduser()
    return p if p.is_absolute() else WORKDIR / p


def resolve_existing_path(path_text: str, base: Path = WORKDIR) -> Path:
    p = Path(path_text).expanduser()
    if p.is_absolute():
        resolved = p
    else:
        candidates = [
            (base / p).resolve(),
            (ROOT / p).resolve(),
            (Path.cwd() / p).resolve(),
        ]
        resolved = next((c for c in candidates if c.exists()), candidates[0])
    if not resolved.exists():
        raise FileNotFoundError(f"Path not found: {path_text} (resolved to {resolved})")
    return resolved


def region_count(split_x: str, split_y: str) -> int:
    nx = len([x for x in split_x.split(",") if x.strip()]) + 1 if split_x else 1
    ny = len([y for y in split_y.split(",") if y.strip()]) + 1 if split_y else 1
    return nx * ny


def flies_per_region(total_flies: int, split_x: str, split_y: str) -> int:
    n_regions = region_count(split_x, split_y)
    if total_flies % n_regions != 0:
        raise ValueError(
            f"Total flies ({total_flies}) is not divisible by region count ({n_regions}). "
            "Check split-x/split-y or Video_list.csv."
        )
    return total_flies // n_regions


def run_one(row: dict, args: argparse.Namespace) -> None:
    video = resolve_workdir_path(row["video"])
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    weights = resolve_existing_path(args.weights)

    csv_path = WORKDIR / "csv" / csv_name_for(video, args.frame_skip)
    track_output = csv_path.with_suffix("")
    split_x = row["split_x"]
    split_y = row["split_y"]
    per_region_flies = flies_per_region(row["total_flies"], split_x, split_y)

    if not args.skip_detect:
        detect_cmd = [
            sys.executable,
            str(ROOT / "detect_2.py"),
            "--weights",
            str(weights),
            "--source",
            str(video),
            "--frame-skip",
            str(args.frame_skip),
            "--conf-thres",
            str(args.conf_thres),
            "--iou-thres",
            str(args.iou_thres),
            "--imgsz",
            str(args.imgsz),
            "--device",
            str(args.device),
            "--quiet",
        ]
        run_cmd(detect_cmd, WORKDIR)

    needs_csv = (not args.skip_track) or (not args.skip_visualize)
    if needs_csv and not csv_path.exists():
        raise FileNotFoundError(f"Detection CSV not found: {csv_path}")

    if not args.skip_track:
        track_cmd = [
            sys.executable,
            str(ROOT / "utils" / "Post_track.py"),
            "-i",
            str(csv_path),
            "-o",
            str(track_output),
            "-v",
            str(video),
            "-n",
            str(per_region_flies),
            "--workers",
            str(args.workers),
            "--window-overlap",
            str(args.window_overlap),
        ]
        if split_x:
            track_cmd += ["--split-x", split_x]
        if split_y:
            track_cmd += ["--split-y", split_y]
        run_cmd(track_cmd, WORKDIR)

    tracked_json = Path(str(track_output) + "_" + str(first_frame_from_csv(csv_path)) + "_.json") if csv_path.exists() else None
    if not args.skip_visualize:
        if tracked_json is None:
            raise FileNotFoundError(f"Cannot determine tracking JSON because CSV is missing: {csv_path}")
        out_dir = resolve_workdir_path(args.output_dir)
        view_cmd = [
            sys.executable,
            str(ROOT / "FastView" / "visualize_tracks.py"),
            "-j",
            str(tracked_json),
            "--speed-window",
            str(args.speed_window),
            "-o",
            str(out_dir),
        ]
        run_cmd(view_cmd, WORKDIR)

    print("\nFastView outputs:")
    print(f"Detection CSV: {csv_path}")
    print(f"Tracked CSV:   {track_output}")
    print(f"Tracked JSON:  {tracked_json if tracked_json is not None else 'not available yet'}")


def main() -> None:
    args = parse_args()
    video_list = Path(args.video_list).expanduser()
    if not video_list.is_absolute():
        video_list = (Path.cwd() / video_list).resolve()
    rows = read_video_list(video_list)
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError(f"No videos found in {video_list}")
    for idx, row in enumerate(rows, start=1):
        print(f"\n=== FastView {idx}/{len(rows)}: {row['video']} ===", flush=True)
        run_one(row, args)


if __name__ == "__main__":
    main()
