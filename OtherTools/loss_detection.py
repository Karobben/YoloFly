#!/usr/bin/env python3
"""
DEPRECATED – use high_risk_detection.py instead.

This script is kept for backward compatibility.
It now forwards all arguments to high_risk_detection.py.
"""
import os, sys, subprocess
_new = os.path.join(os.path.dirname(__file__), "high_risk_detection.py")
print("[loss_detection] Forwarding to high_risk_detection.py ...")
sys.exit(subprocess.run(["python", _new] + sys.argv[1:]).returncode)

"""
Original docstring (kept for reference):
Detect potential fly-loss events from detection CSVs.

This script mirrors the input conventions of `video_annotate_from_tracking.py`:
- It reads `Video_list.csv` (tab-separated, no header).
- You select a video by its id (first column).
- It then finds the corresponding detection CSV under `csv/`.

For each frame in the configured [frame_start, frame_end) range, it counts how
many flies are present and reports when the number of flies is lower than
an expected value.

Usage
-----

python OtherTools/loss_detection.py -id <video_id> -n <expected_flies> \
    [-vlist PATH] [--min-gap 10]

-id / --video_id : identifier from the first column of Video_list.csv
-n  / --expected : expected number of flies in the arena
--min-gap        : minimum consecutive frames below expected to report as an event
"""

import argparse
import os
import subprocess
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    base_cols = [
        "video_id",
        "petri_pixel",
        "petri_mm",
        "frame_start",
        "frame_end",
        "num_flies",
    ]
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
            f"Available ids include: {df['video_id'].unique()[:10]!r}"
        )
    return match.iloc[0]


def load_detection_csv(video_id: str) -> pd.DataFrame:
    """
    Load YOLO-style detection CSV: Frame, class, x, y, w, h (normalized).
    """
    csv_dir = os.path.join(PROJECT_ROOT, "csv")
    candidates = [
        f for f in os.listdir(csv_dir)
        if video_id in f and f.endswith(".csv")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No detection CSV found in {csv_dir} containing '{video_id}'."
        )
    csv_name = sorted(candidates, key=len)[0]
    csv_path = os.path.join(csv_dir, csv_name)

    df = pd.read_csv(csv_path, sep=" ", header=None)
    if df.shape[1] < 6:
        raise ValueError(
            f"Detection CSV {csv_path} must have at least 6 columns "
            f"(Frame, class, x, y, width, height). Got {df.shape[1]}."
        )
    df = df.iloc[:, :6]
    df.columns = ["Frame", "class", "x", "y", "w", "h"]
    return df


def find_merged_cls0(
    det_cls0: pd.DataFrame,
    cover_thr: float = 0.9,
    min_covered: int = 2,
) -> set:
    """
    Return the 0-based integer positions within det_cls0 whose boxes are
    likely 'merged' detections (the model saw 2+ flies as one large body).

    A box is considered merged when it covers >= cover_thr of the area of
    at least min_covered other bodies in the same frame.

    Coverage(A, B) = intersection(A, B) / area(B)
    """
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


def find_loss_intervals(
    counts: pd.Series,
    expected: int,
    min_gap: int,
) -> List[Tuple[int, int, int]]:
    """
    Given a per-frame count series, return intervals where the count is below
    expected for at least `min_gap` consecutive frames.

    Returns a list of (start_frame, end_frame, min_count).
    """
    below = counts < expected
    if not below.any():
        return []

    frames = counts.index.to_numpy()
    flags = below.to_numpy().astype(int)

    intervals: List[Tuple[int, int, int]] = []
    start_idx = None
    for i in range(len(flags)):
        if flags[i] and start_idx is None:
            start_idx = i
        elif not flags[i] and start_idx is not None:
            length = i - start_idx
            if length >= min_gap:
                seg_frames = frames[start_idx:i]
                seg_counts = counts.iloc[start_idx:i]
                intervals.append(
                    (int(seg_frames[0]), int(seg_frames[-1]), int(seg_counts.min()))
                )
            start_idx = None

    # tail
    if start_idx is not None:
        length = len(flags) - start_idx
        if length >= min_gap:
            seg_frames = frames[start_idx:]
            seg_counts = counts.iloc[start_idx:]
            intervals.append(
                (int(seg_frames[0]), int(seg_frames[-1]), int(seg_counts.min()))
            )

    return intervals


def find_excess_intervals(
    counts: pd.Series,
    expected: int,
    min_gap: int,
) -> List[Tuple[int, int, int]]:
    """
    Given a per-frame count series, return intervals where the count is above
    expected for at least `min_gap` consecutive frames.

    Returns a list of (start_frame, end_frame, max_count).
    """
    above = counts > expected
    if not above.any():
        return []

    frames = counts.index.to_numpy()
    flags = above.to_numpy().astype(int)

    intervals: List[Tuple[int, int, int]] = []
    start_idx = None
    for i in range(len(flags)):
        if flags[i] and start_idx is None:
            start_idx = i
        elif not flags[i] and start_idx is not None:
            length = i - start_idx
            if length >= min_gap:
                seg_frames = frames[start_idx:i]
                seg_counts = counts.iloc[start_idx:i]
                intervals.append(
                    (int(seg_frames[0]), int(seg_frames[-1]), int(seg_counts.max()))
                )
            start_idx = None

    # tail
    if start_idx is not None:
        length = len(flags) - start_idx
        if length >= min_gap:
            seg_frames = frames[start_idx:]
            seg_counts = counts.iloc[start_idx:]
            intervals.append(
                (int(seg_frames[0]), int(seg_frames[-1]), int(seg_counts.max()))
            )

    return intervals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect potential fly-loss events from detection CSVs."
    )
    parser.add_argument(
        "-id",
        "--video_id",
        required=True,
        help="Video id from the first column of Video_list.csv.",
    )
    parser.add_argument(
        "-n",
        "--expected",
        type=int,
        default=None,
        help="Expected number of flies in the arena (overrides num_flies from Video_list.csv if set).",
    )
    parser.add_argument(
        "-vlist",
        "--video_list",
        default=os.path.join(PROJECT_ROOT, "Video_list.csv"),
        help="Path to Video_list.csv (default: project root Video_list.csv).",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=10,
        help="Minimum consecutive frames below expected to report (default: 10).",
    )
    parser.add_argument(
        "--make-previews",
        action="store_true",
        help="If set, automatically call video_annotate_from_tracking.py to create preview videos around loss intervals.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Video_post", "LossCheck"),
        help="Directory to store generated preview videos (will be created if missing).",
    )
    parser.add_argument(
        "--preview-test",
        action="store_true",
        help="Only generate previews for the first 3 intervals (useful for quick testing).",
    )

    args = parser.parse_args()

    df_vlist = read_video_list(args.video_list)
    row = find_video_row(df_vlist, args.video_id)
    video_id = str(row["video_id"])
    frame_start = int(row["frame_start"])
    frame_end = int(row["frame_end"])
    # Use CLI expected if provided; otherwise fall back to num_flies from Video_list
    expected = args.expected if args.expected is not None else int(row["num_flies"])

    det = load_detection_csv(video_id)

    # Determine FPS for 1-second windows
    video_path = str(row["video_path"])
    if not os.path.isabs(video_path):
        candidate = os.path.join(PROJECT_ROOT, video_path)
        if os.path.exists(candidate):
            video_path = candidate
    fps = 30
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            if fps_val > 0:
                fps = int(round(fps_val))
        cap.release()
    except Exception:
        pass

    # Consider only body detections (cls 0) as flies
    det_bodies = det[det["class"] == 0]
    # Restrict to configured frame window
    det_bodies = det_bodies[
        det_bodies["Frame"].between(frame_start, frame_end - 1)
    ]

    if det_bodies.empty:
        print(
            f"[warn] No body detections (cls 0) for video_id={video_id} "
            f"in frames [{frame_start}, {frame_end})."
        )
        return

    # Compute per-frame effective count (excluding merged detections)
    adj: dict = {}
    total_merged = 0
    for frame_id, group in det_bodies.groupby("Frame"):
        merged_idxs = find_merged_cls0(group)
        total_merged += len(merged_idxs)
        adj[frame_id] = len(group) - len(merged_idxs)
    counts = pd.Series(adj, dtype=int).sort_index()

    # Ensure we have an entry for every frame in the window (fill missing with 0)
    full_index = np.arange(frame_start, frame_end)
    counts = counts.reindex(full_index, fill_value=0)

    print(f"Video: {video_id}")
    print(f"Frame range: [{frame_start}, {frame_end})")
    print(f"Expected flies: {expected}")
    print(f"Merged (2+ flies detected as one) detections excluded: {total_merged}")
    print(f"Frames with fewer than expected flies: {(counts < expected).sum()}")
    print(f"Frames with more than expected flies: {(counts > expected).sum()}")

    loss_intervals = find_loss_intervals(counts, expected, args.min_gap)
    excess_intervals = find_excess_intervals(counts, expected, args.min_gap)

    if loss_intervals:
        print("\nPotential fly-loss intervals (start_frame, end_frame, min_count):")
        for (s, e, m) in loss_intervals:
            print(f"  {s} – {e} (min flies observed: {m})")
    else:
        print("\nNo loss intervals meeting the criteria were found.")

    if excess_intervals:
        print("\nPotential over-count intervals (start_frame, end_frame, max_count):")
        for (s, e, m) in excess_intervals:
            print(f"  {s} – {e} (max flies observed: {m})")
    else:
        print("\nNo over-count intervals meeting the criteria were found.")

    # Optionally generate preview videos around loss intervals
    if args.make_previews and (loss_intervals or excess_intervals):
        annotate_script = os.path.join(
            PROJECT_ROOT, "OtherTools", "video_annotate_from_tracking.py"
        )
        if not os.path.exists(annotate_script):
            print(
                f"[warn] Cannot create previews: {annotate_script} not found."
            )
            return

        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)

        print(f"\nCreating preview videos around loss/over-count intervals (±1s) into: {outdir}")

        made = 0

        def make_preview(tag: str, mark_frame: int) -> None:
            nonlocal made
            if args.preview_test and made >= 3:
                return
            fs = max(frame_start, mark_frame - fps)
            fe = min(frame_end, mark_frame + fps)
            out_name = os.path.join(outdir, f"{tag}Preview_{video_id}_{fs}_{fe}.mp4")
            cmd = [
                "python",
                annotate_script,
                "-id",
                video_id,
                "-vlist",
                os.path.join(PROJECT_ROOT, "Video_list.csv"),
                "-o",
                out_name,
                "--fs",
                str(fs),
                "--fe",
                str(fe),
                "--mark-frame",
                str(mark_frame),
                "--mark-label",
                tag,
                "--no-behavior",
            ]
            print("  Running:", " ".join(cmd))
            subprocess.run(cmd, check=False)
            made += 1

        for (s, e, m) in loss_intervals:
            make_preview("LOSS", s)
        for (s, e, m) in excess_intervals:
            make_preview("GAIN", s)


if __name__ == "__main__":
    main()

