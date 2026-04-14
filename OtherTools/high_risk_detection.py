#!/usr/bin/env python3
"""
High-risk event detection for fly-tracking data.

Detects three types of risk events that often indicate tracking errors:

  LOSS  – Frames where the effective body-detection count drops below the
          expected fly count (a fly disappeared from detection).

  GAIN  – Frames where the effective body-detection count exceeds the
          expected fly count (duplicate or false detection).

  LEAP  – Frames where one or more flies move more than a threshold multiple
          of their own body size within one second.  Rapid position jumps
          are a strong signal of ID switches in the tracker.

All three event types can optionally produce annotated ±1-second preview clips
via video_annotate_from_tracking.py.

Usage
-----
python OtherTools/high_risk_detection.py -id <video_id> [options]

Key options
  -n / --expected    Override expected fly count (default: from Video_list.csv)
  --leap-mult        Leap threshold in body-lengths per second (default: 1.0)
  --min-gap          Min consecutive abnormal frames for LOSS/GAIN (default: 10)
  --make-previews    Generate annotated preview clips for every event
  --outdir           Output directory for preview clips
  --preview-test     Limit to first 3 previews per event type (quick test)
  --no-leap          Skip leap detection (faster if JSON is large)
"""

import argparse
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FlyDic = Dict[str, Dict[str, Dict[str, Any]]]

_PIXEL_W = 1920
_PIXEL_H = 1080


# ---------------------------------------------------------------------------
# Shared helpers (also used in loss_detection.py)
# ---------------------------------------------------------------------------

def read_video_list(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video_list.csv not found at: {path}")
    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] < 7:
        raise ValueError(
            f"Video_list.csv at {path} must have at least 7 tab-separated columns "
            f"(video_id, petri_pixel, petri_mm, frame_start, frame_end, num_flies, abs_path). "
            f"Got {df.shape[1]} column(s)."
        )
    ncol = df.shape[1]
    base_cols = ["video_id", "petri_pixel", "petri_mm", "frame_start", "frame_end", "num_flies"]
    if ncol > 6:
        extra = [f"col_{i}" for i in range(7, ncol + 1)]
        cols = base_cols + extra
    else:
        cols = base_cols
    df.columns = cols
    df["video_path"] = df.iloc[:, -1]
    return df


def find_video_row(df: pd.DataFrame, video_id: str) -> pd.Series:
    match = df[df["video_id"] == video_id]
    if match.empty:
        raise ValueError(
            f"No row in Video_list.csv has video_id == '{video_id}'. "
            f"Available ids: {df['video_id'].unique()[:10]!r}"
        )
    return match.iloc[0]


def load_detection_csv(video_id: str) -> pd.DataFrame:
    csv_dir = os.path.join(PROJECT_ROOT, "csv")
    candidates = [f for f in os.listdir(csv_dir) if video_id in f and f.endswith(".csv")]
    if not candidates:
        raise FileNotFoundError(f"No detection CSV in {csv_dir} containing '{video_id}'.")
    csv_path = os.path.join(csv_dir, sorted(candidates, key=len)[0])
    df = pd.read_csv(csv_path, sep=" ", header=None)
    df = df.iloc[:, :6]
    df.columns = ["Frame", "class", "x", "y", "w", "h"]
    return df


def load_tracking_json(video_id: str, frame_start: int, frame_end: int) -> FlyDic:
    csv_dir = os.path.join(PROJECT_ROOT, "csv")
    candidates = [f for f in os.listdir(csv_dir) if video_id in f and f.endswith(".json")]
    if not candidates:
        print(f"[info] No tracking JSON found for '{video_id}'; skipping leap detection.")
        return {}
    json_path = os.path.join(csv_dir, sorted(candidates, key=len)[0])
    with open(json_path, "r") as f:
        raw = f.read().strip()
    fly_dic: FlyDic = {}
    if ";" in raw:
        for part in raw.split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                obj = json.loads(part)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not obj:
                continue
            key = list(obj.keys())[0]
            try:
                fi = int(key)
            except (TypeError, ValueError):
                continue
            if frame_start <= fi <= frame_end:
                fly_dic[key] = obj[key]
    else:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                for key, val in obj.items():
                    try:
                        fi = int(key)
                    except (TypeError, ValueError):
                        continue
                    if frame_start <= fi <= frame_end:
                        fly_dic[key] = val
        except json.JSONDecodeError:
            pass
    return fly_dic


def find_merged_cls0(
    det_cls0: pd.DataFrame,
    cover_thr: float = 0.9,
    min_covered: int = 2,
) -> set:
    """Return 0-based positions in det_cls0 that are merged (2+ flies as one)."""
    n = len(det_cls0)
    if n < min_covered + 1:
        return set()
    vals = det_cls0[["x", "y", "w", "h"]].to_numpy(dtype=float)
    x1 = vals[:, 0] - vals[:, 2] / 2
    y1 = vals[:, 1] - vals[:, 3] / 2
    x2 = vals[:, 0] + vals[:, 2] / 2
    y2 = vals[:, 1] + vals[:, 3] / 2
    areas = vals[:, 2] * vals[:, 3]
    merged: set = set()
    for i in range(n):
        covered = 0
        for j in range(n):
            if i == j:
                continue
            ix1 = max(x1[i], x1[j])
            iy1 = max(y1[i], y1[j])
            ix2 = min(x2[i], x2[j])
            iy2 = min(y2[i], y2[j])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / (areas[j] + 1e-9) >= cover_thr:
                covered += 1
        if covered >= min_covered:
            merged.add(i)
    return merged


# ---------------------------------------------------------------------------
# LOSS / GAIN detection (CSV-based)
# ---------------------------------------------------------------------------

def _intervals_from_flags(
    frames: np.ndarray,
    flags: np.ndarray,
    values: np.ndarray,
    min_gap: int,
    use_min: bool,
) -> List[Tuple[int, int, int]]:
    intervals: List[Tuple[int, int, int]] = []
    start_idx = None
    for i in range(len(flags)):
        if flags[i] and start_idx is None:
            start_idx = i
        elif not flags[i] and start_idx is not None:
            if i - start_idx >= min_gap:
                seg = values[start_idx:i]
                intervals.append(
                    (int(frames[start_idx]), int(frames[i - 1]),
                     int(seg.min() if use_min else seg.max()))
                )
            start_idx = None
    if start_idx is not None and len(frames) - start_idx >= min_gap:
        seg = values[start_idx:]
        intervals.append(
            (int(frames[start_idx]), int(frames[-1]),
             int(seg.min() if use_min else seg.max()))
        )
    return intervals


def find_loss_intervals(counts: pd.Series, expected: int, min_gap: int) -> List[Tuple[int, int, int]]:
    frames = counts.index.to_numpy()
    flags = (counts < expected).to_numpy().astype(int)
    return _intervals_from_flags(frames, flags, counts.to_numpy(), min_gap, use_min=True)


def find_excess_intervals(counts: pd.Series, expected: int, min_gap: int) -> List[Tuple[int, int, int]]:
    frames = counts.index.to_numpy()
    flags = (counts > expected).to_numpy().astype(int)
    return _intervals_from_flags(frames, flags, counts.to_numpy(), min_gap, use_min=False)


# ---------------------------------------------------------------------------
# LEAP detection (JSON-based)
# ---------------------------------------------------------------------------

def calc_mean_body_length(fly_dic: FlyDic) -> float:
    """
    Estimate mean body length (px) from the first frame that has both
    body and head data for at least one fly.

    Body length = 2 × Euclidean distance between head center and body center.
    This approximates the full head-to-tail length of the fly.
    """
    sorted_keys = sorted(fly_dic.keys(), key=lambda k: int(k))
    for fk in sorted_keys:
        lengths = []
        for d in fly_dic[fk].values():
            body = d.get("body")
            head = d.get("head")
            if not body or not head or len(body) < 2 or len(head) < 2:
                continue
            dx = (float(head[0]) - float(body[0])) * _PIXEL_W
            dy = (float(head[1]) - float(body[1])) * _PIXEL_H
            lengths.append(math.sqrt(dx ** 2 + dy ** 2) * 2)
        if lengths:
            return sum(lengths) / len(lengths)
    # fallback: typical body length for this setup (~72px observed)
    return 72.0


def detect_huge_leaps(
    fly_dic: FlyDic,
    fps: int,
    leap_multiplier: float = 1.0,
    max_frame_gap: int = 3,
) -> Tuple[Dict[str, List[Tuple[int, float, float]]], float]:
    """
    For each fly, find frames where the body-center displacement between
    adjacent frames exceeds:

        leap_multiplier × mean_body_length_px

    Body length is estimated as 2 × dist(head_center, body_center) from
    the first available frame, averaged across all flies.

    Normal per-frame movement is ~1–15 px; an ID switch typically causes
    a jump of 100–1000+ px, so one body-length (~72 px) is a robust threshold.

    Parameters
    ----------
    fly_dic         : FLY_matrix {frame_str: {fly_id: {body, head}}}
    fps             : video fps (used only for grouping, not for threshold)
    leap_multiplier : threshold multiplier (default 1.0 = one body length)
    max_frame_gap   : skip frame pairs with a gap larger than this

    Returns
    -------
    (events_dict, mean_body_length_px)
    events_dict: {fly_id: [(frame_int, dist_px, body_length_px), ...]}
    """
    if not fly_dic:
        return {}, 0.0

    mean_bl = calc_mean_body_length(fly_dic)
    threshold = leap_multiplier * mean_bl

    sorted_keys = sorted(fly_dic.keys(), key=lambda k: int(k))
    events: Dict[str, List[Tuple[int, float, float]]] = {}

    for idx in range(1, len(sorted_keys)):
        fk_prev = sorted_keys[idx - 1]
        fk_curr = sorted_keys[idx]
        fi_prev = int(fk_prev)
        fi_curr = int(fk_curr)
        gap = fi_curr - fi_prev
        if gap > max_frame_gap:
            continue

        for fly_id, data_curr in fly_dic[fk_curr].items():
            if fly_id not in fly_dic[fk_prev]:
                continue
            body_curr = data_curr.get("body")
            body_prev = fly_dic[fk_prev][fly_id].get("body")
            if not body_curr or not body_prev or len(body_curr) < 2 or len(body_prev) < 2:
                continue

            dx = (float(body_curr[0]) - float(body_prev[0])) * _PIXEL_W
            dy = (float(body_curr[1]) - float(body_prev[1])) * _PIXEL_H
            dist_px = math.sqrt(dx ** 2 + dy ** 2)

            if dist_px > threshold:
                events.setdefault(fly_id, []).append(
                    (fi_curr, round(dist_px, 1), round(mean_bl, 1))
                )

    return events, mean_bl


def group_leap_events(
    events: Dict[str, List[Tuple[int, float, float]]],
    fps: int,
    min_gap: int = 1,
) -> List[Tuple[int, int, str, float, float]]:
    """
    Merge per-fly leap frames that are within fps frames of each other into
    intervals.

    Returns a list of:
        (start_frame, end_frame, fly_id, max_dist_per_second, body_size_px)
    sorted by start_frame.
    """
    result: List[Tuple[int, int, str, float, float]] = []
    for fly_id, fly_events in events.items():
        fly_events_sorted = sorted(fly_events, key=lambda x: x[0])
        if not fly_events_sorted:
            continue
        seg_start = fly_events_sorted[0][0]
        seg_end = fly_events_sorted[0][0]
        seg_max_dist = fly_events_sorted[0][1]
        body_sz = fly_events_sorted[0][2]

        for fi, dist, bsz in fly_events_sorted[1:]:
            if fi - seg_end <= fps:  # within 1 second → extend interval
                seg_end = fi
                seg_max_dist = max(seg_max_dist, dist)
            else:
                result.append((seg_start, seg_end, fly_id, seg_max_dist, body_sz))
                seg_start, seg_end, seg_max_dist, body_sz = fi, fi, dist, bsz

        result.append((seg_start, seg_end, fly_id, seg_max_dist, body_sz))

    return sorted(result, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Preview clip generation
# ---------------------------------------------------------------------------

# Top-level function (must be picklable for multiprocessing)
def _run_preview_job(job: dict) -> str:
    """Execute one annotate command and return a status string."""
    cmd = job["cmd"]
    label = job["label"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    status = "OK" if result.returncode == 0 else f"exit {result.returncode}"
    return f"  [{status}] {label}"


def merge_event_groups(
    events: List[Tuple[str, int]],
    fps: int,
    frame_start: int,
    frame_end: int,
) -> List[Tuple[str, int, int, int]]:
    """
    Merge events whose ±1-second preview windows would overlap.

    Two events at frames A and B overlap when |B - A| < 2*fps
    (their windows [A-fps, A+fps] and [B-fps, B+fps] share frames).

    Parameters
    ----------
    events     : list of (tag, mark_frame), any order
    fps        : frames per second

    Returns
    -------
    list of (combined_tag, first_mark_frame, clip_fs, clip_fe) — one per group
    """
    if not events:
        return []

    sorted_events = sorted(events, key=lambda x: x[1])

    groups: List[List[Tuple[str, int]]] = []
    current: List[Tuple[str, int]] = [sorted_events[0]]

    for tag, frame in sorted_events[1:]:
        if frame - current[-1][1] < 2 * fps:   # windows overlap → extend group
            current.append((tag, frame))
        else:
            groups.append(current)
            current = [(tag, frame)]
    groups.append(current)

    result: List[Tuple[str, int, int, int]] = []
    for group in groups:
        tags = list(dict.fromkeys(t for t, _ in group))  # deduplicate, keep order
        combined_tag = "+".join(tags)
        frames = [f for _, f in group]
        first_mark = min(frames)
        fs = max(frame_start, min(frames) - fps)
        fe = min(frame_end,   max(frames) + fps)
        result.append((combined_tag, first_mark, fs, fe))

    return result


def build_preview_cmd(
    tag: str,
    mark_frame: int,
    fs: int,
    fe: int,
    video_id: str,
    outdir: str,
    vlist_path: str,
    annotate_script: str,
) -> dict:
    """Return a job dict (picklable) describing one preview clip."""
    out_name = os.path.join(outdir, f"{tag}Preview_{video_id}_{fs}_{fe}.mp4")
    cmd = [
        "python", annotate_script,
        "-id", video_id,
        "-vlist", vlist_path,
        "-o", out_name,
        "--fs", str(fs),
        "--fe", str(fe),
        "--mark-frame", str(mark_frame),
        "--mark-label", tag,
        "--no-behavior",
    ]
    return {
        "cmd": cmd,
        "label": f"{tag}  mark={mark_frame}  clip=[{fs},{fe}]  → {os.path.basename(out_name)}",
    }


def run_previews_parallel(jobs: List[dict], n_workers: int) -> None:
    """Run all preview jobs in parallel using a process pool."""
    if not jobs:
        return
    n = min(n_workers, len(jobs))
    print(f"  Dispatching {len(jobs)} clip(s) with {n} worker(s) ...")

    with ProcessPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(_run_preview_job, job): job for job in jobs}
        for fut in as_completed(futures):
            try:
                print(fut.result())
            except Exception as exc:
                print(f"  [error] {futures[fut]['label']}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-risk event detection: fly LOSS, GAIN, and LEAP."
    )
    parser.add_argument("-id", "--video_id", required=True,
                        help="Video id from Video_list.csv.")
    parser.add_argument("-n", "--expected", type=int, default=None,
                        help="Override expected fly count (default: from Video_list).")
    parser.add_argument("-vlist", "--video_list",
                        default=os.path.join(PROJECT_ROOT, "Video_list.csv"),
                        help="Path to Video_list.csv.")
    parser.add_argument("--min-gap", type=int, default=10,
                        help="Min consecutive abnormal frames to report LOSS/GAIN (default: 10).")
    parser.add_argument("--leap-mult", type=float, default=1.0,
                        help="Leap threshold: body-lengths per second (default: 1.0).")
    parser.add_argument("--no-leap", action="store_true",
                        help="Skip leap detection.")
    parser.add_argument("--make-previews", action="store_true",
                        help="Generate annotated preview clips for every event.")
    parser.add_argument("--outdir",
                        default=os.path.join(PROJECT_ROOT, "Video_post", "HighRiskCheck"),
                        help="Directory for preview clips (default: Video_post/HighRiskCheck).")
    parser.add_argument("--preview-test", action="store_true",
                        help="Limit to first 3 previews per event type.")
    parser.add_argument("-p", "--processes", type=int,
                        default=max(1, cpu_count() - 1),
                        help="Number of parallel workers for preview rendering "
                             "(default: CPU count − 1).")

    args = parser.parse_args()

    df_vlist = read_video_list(args.video_list)
    row = find_video_row(df_vlist, args.video_id)
    video_id = str(row["video_id"])
    frame_start = int(row["frame_start"])
    frame_end = int(row["frame_end"])
    expected = args.expected if args.expected is not None else int(row["num_flies"])

    # --- FPS from video ---
    video_path = str(row["video_path"])
    if not os.path.isabs(video_path):
        cand = os.path.join(PROJECT_ROOT, video_path)
        if os.path.exists(cand):
            video_path = cand
    fps = 30
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            v = cap.get(cv2.CAP_PROP_FPS)
            if v > 0:
                fps = int(round(v))
        cap.release()
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"Video        : {video_id}")
    print(f"Frame range  : [{frame_start}, {frame_end})")
    print(f"Expected flies: {expected}")
    print(f"FPS          : {fps}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # LOSS / GAIN detection
    # -----------------------------------------------------------------------
    det = load_detection_csv(video_id)
    det_bodies = det[(det["class"] == 0) &
                     (det["Frame"].between(frame_start, frame_end - 1))]

    if det_bodies.empty:
        print("[warn] No cls0 body detections found in configured frame range.")
    else:
        adj: dict = {}
        total_merged = 0
        for fid, grp in det_bodies.groupby("Frame"):
            merged = find_merged_cls0(grp)
            total_merged += len(merged)
            adj[fid] = len(grp) - len(merged)
        counts = pd.Series(adj, dtype=int).sort_index()
        counts = counts.reindex(np.arange(frame_start, frame_end), fill_value=0)

        print(f"[LOSS/GAIN] Merged detections excluded: {total_merged}")
        print(f"[LOSS/GAIN] Frames below expected: {(counts < expected).sum()}")
        print(f"[LOSS/GAIN] Frames above expected: {(counts > expected).sum()}")

        loss_intervals = find_loss_intervals(counts, expected, args.min_gap)
        excess_intervals = find_excess_intervals(counts, expected, args.min_gap)

        if loss_intervals:
            print("\n  LOSS intervals (start, end, min_count):")
            for s, e, m in loss_intervals:
                print(f"    {s} – {e}  (min observed: {m})")
        else:
            print("\n  No LOSS intervals found.")

        if excess_intervals:
            print("\n  GAIN intervals (start, end, max_count):")
            for s, e, m in excess_intervals:
                print(f"    {s} – {e}  (max observed: {m})")
        else:
            print("\n  No GAIN intervals found.")

    # -----------------------------------------------------------------------
    # LEAP detection
    # -----------------------------------------------------------------------
    leap_grouped: List[Tuple[int, int, str, float, float]] = []

    if not args.no_leap:
        print(f"\n[LEAP] Loading tracking JSON (threshold: {args.leap_mult}× body/s) ...")
        fly_dic = load_tracking_json(video_id, frame_start, frame_end)

        if fly_dic:
            raw_events, mean_bl = detect_huge_leaps(
                fly_dic, fps,
                leap_multiplier=args.leap_mult,
            )
            print(f"[LEAP] Mean body length: {mean_bl:.1f}px  "
                  f"threshold: {args.leap_mult}× = {args.leap_mult * mean_bl:.1f}px/frame")
            total_leap_frames = sum(len(v) for v in raw_events.values())
            print(f"[LEAP] Leap frames detected: {total_leap_frames} "
                  f"across {len(raw_events)} fly/flies")

            leap_grouped = group_leap_events(raw_events, fps)
            if leap_grouped:
                print(f"\n  LEAP intervals ({len(leap_grouped)} event(s)):")
                for s, e, fid, dist, bsz in leap_grouped:
                    ratio = dist / bsz if bsz > 0 else 0
                    print(f"    frames {s}–{e}  fly={fid}  "
                          f"jump={dist:.0f}px/frame  body_len={bsz:.0f}px  "
                          f"({ratio:.1f}× body length)")
            else:
                print("\n  No LEAP events found.")
        else:
            print("[LEAP] Skipped (no tracking JSON available).")

    # -----------------------------------------------------------------------
    # Preview clips (parallel)
    # -----------------------------------------------------------------------
    if args.make_previews:
        annotate_script = os.path.join(
            PROJECT_ROOT, "OtherTools", "video_annotate_from_tracking.py"
        )
        if not os.path.exists(annotate_script):
            print(f"[warn] {annotate_script} not found; skipping previews.")
            return

        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)
        vlist_path = os.path.join(PROJECT_ROOT, "Video_list.csv")
        max_previews = 3 if args.preview_test else None

        # ---- Step 1: gather all (tag, mark_frame) pairs ----
        raw_events: List[Tuple[str, int]] = []
        for s, e, m in (loss_intervals if not det_bodies.empty else []):
            raw_events.append(("LOSS", s))
        for s, e, m in (excess_intervals if not det_bodies.empty else []):
            raw_events.append(("GAIN", s))
        for s, e, fid, dist, bsz in leap_grouped:
            raw_events.append((f"LEAP_{fid}", s))

        if not raw_events:
            print("\nNo preview clips to generate.")
            return

        # ---- Step 2: merge nearby events ----
        merged = merge_event_groups(raw_events, fps, frame_start, frame_end)
        n_before = len(raw_events)
        n_after  = len(merged)
        if n_before > n_after:
            print(f"\n[merge] {n_before} events → {n_after} clip(s) after merging nearby windows")
        else:
            print(f"\n{n_after} event(s) (no merging needed)")

        if max_previews is not None:
            merged = merged[:max_previews]
            print(f"  (preview-test: limited to first {max_previews})")

        # ---- Step 3: build and dispatch in parallel ----
        jobs: List[dict] = [
            build_preview_cmd(
                tag, mark_frame, fs, fe,
                video_id, outdir, vlist_path, annotate_script,
            )
            for tag, mark_frame, fs, fe in merged
        ]

        n_workers = max(1, args.processes)
        print(f"Generating {len(jobs)} preview clip(s) → {outdir}")
        run_previews_parallel(jobs, n_workers)
        print(f"  Done. {len(jobs)} clip(s) created.")


if __name__ == "__main__":
    main()
