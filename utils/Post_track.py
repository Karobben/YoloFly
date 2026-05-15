#!/usr/bin/env python3

import numpy as np
import pandas as pd
import cv2, json, warnings, os
import math
import subprocess
import sys
import tempfile
import time
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from shapely.geometry import Polygon
from skimage.metrics import structural_similarity as ssim
from scipy.stats import median_abs_deviation as mad
from scipy.stats import zscore

from scipy.stats import norm
from scipy.spatial.distance import cdist
# create a polygon by following order:
# mute warning messages from pandas
from Head_bind import head_match
head_bind = head_match()

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    Progress = None
    SpinnerColumn = None
    BarColumn = None
    TextColumn = None
    TimeElapsedColumn = None
    TimeRemainingColumn = None
    RICH_AVAILABLE = False


import argparse

LOGGER = logging.getLogger("Post_track")


def setup_logging(level_name):
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER.setLevel(level)


@contextmanager
def progress_manager(enabled=True):
    if enabled and RICH_AVAILABLE:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        progress.start()
        try:
            yield progress
        finally:
            progress.stop()
    else:
        yield None


def progress_step(progress, task_id, *, advance=1, description=None):
    if progress is None or task_id is None:
        if description:
            LOGGER.info("STEP | %s", description)
        return
    kwargs = {"advance": advance}
    if description is not None:
        kwargs["description"] = description
    progress.update(task_id, **kwargs)

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input')     #输入文件
parser.add_argument('-o','-U','--output')     #输入文件
parser.add_argument('-v','-V','--video')     #输入文件
parser.add_argument('-n','-N','--num-fly', type=int, default=12, help='expected number of flies to track')
parser.add_argument('--initial-frame', type=int, default=None, help='start tracking from this frame index (must exist in detection table)')
parser.add_argument('--initial-results', default='', help='optional manually corrected detection file used to replace detections at --initial-frame')
parser.add_argument('--test', '--test-frames', type=int, default=None, help='process only the first N frames (default: all frames)')
parser.add_argument('--workers', type=int, default=1, help='number of parallel tracking windows')
parser.add_argument('--window-overlap', type=int, default=200, help='overlap size in processed frames')
parser.add_argument('--split-x', default='', help='comma-separated x split boundaries in pixels, e.g. 960 or 640,1280')
parser.add_argument('--split-y', default='', help='comma-separated y split boundaries in pixels, e.g. 540 or 360,720')
parser.add_argument('--frame-width', type=int, default=1920, help='video width in pixels for split boundaries')
parser.add_argument('--frame-height', type=int, default=1080, help='video height in pixels for split boundaries')
parser.add_argument('--head-reacquire-after', type=int, default=8, help='after this many consecutive fallback head frames, try nearest-head reacquire')
parser.add_argument('--head-reacquire-max-dist', type=float, default=0.03, help='max normalized XY distance for nearest-head reacquire')
parser.add_argument('--qa-report', default='', help='optional QA report JSON path (default: <output>.qa.json)')
parser.add_argument('--log-level', default='INFO', help='log level: DEBUG, INFO, WARNING, ERROR')
parser.add_argument('--log-every', type=int, default=1, help='print per-frame summary every N frames (default: every frame)')
parser.add_argument('--no-progress', action='store_true', help='disable rich progress bars')
parser.add_argument('--internal-window', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--internal-region', action='store_true', help=argparse.SUPPRESS)

##获取参数
args = parser.parse_args()
INPUT = args.input
OUTPUT = args.output
Video = args.video
NUM_FLY = args.num_fly
INITIAL_FRAME = args.initial_frame
INITIAL_RESULTS = str(args.initial_results or "").strip()
TEST_FRAMES = args.test
WORKERS = max(1, args.workers)
WINDOW_OVERLAP = max(0, args.window_overlap)
SPLIT_X = args.split_x
SPLIT_Y = args.split_y
FRAME_WIDTH = args.frame_width
FRAME_HEIGHT = args.frame_height
HEAD_REACQUIRE_AFTER = args.head_reacquire_after
HEAD_REACQUIRE_MAX_DIST = args.head_reacquire_max_dist
QA_REPORT_PATH = str(args.qa_report or "").strip()
LOG_LEVEL = str(args.log_level or "INFO")
LOG_EVERY = int(args.log_every)
ENABLE_PROGRESS = not args.no_progress

if LOG_EVERY < 1:
    raise ValueError("--log-every must be >= 1")
setup_logging(LOG_LEVEL)

if not INPUT:
    raise ValueError("--input is required")
if not OUTPUT:
    raise ValueError("--output is required")
if not Video:
    raise ValueError("--video is required")
if not Path(INPUT).exists():
    raise FileNotFoundError(f"Input file not found: {INPUT}")
if not Path(Video).exists():
    raise FileNotFoundError(f"Video file not found: {Video}")
if FRAME_WIDTH <= 0 or FRAME_HEIGHT <= 0:
    raise ValueError("--frame-width and --frame-height must be positive")
if TEST_FRAMES is not None and TEST_FRAMES <= 0:
    raise ValueError("--test must be >= 1")
if INITIAL_FRAME is not None and INITIAL_FRAME < 1:
    raise ValueError("--initial-frame must be >= 1")
if INITIAL_RESULTS and not Path(INITIAL_RESULTS).exists():
    raise FileNotFoundError(f"Initial results file not found: {INITIAL_RESULTS}")
if HEAD_REACQUIRE_AFTER < 1:
    raise ValueError("--head-reacquire-after must be >= 1")
if HEAD_REACQUIRE_MAX_DIST <= 0:
    raise ValueError("--head-reacquire-max-dist must be > 0")

# Runtime frame dimensions used by geometry/crop helpers. Defaults can be overridden from the video.
TRACK_FRAME_WIDTH = int(FRAME_WIDTH)
TRACK_FRAME_HEIGHT = int(FRAME_HEIGHT)


try:
    warnings.filterwarnings("ignore")
except:
    pd.options.mode.chained_assignment = None

#from Fly_Tra import fly_align 

## Functions

def Number_adjust(data, N = 13):
    Z_abs = abs(zscore(data))
    sorted_indices = np.argsort(Z_abs)#[::-1]
    tops = sorted_indices[:N]
    return tops

def Dots_Sort(points1, points2):
    # Generate two lists of 2D points
    #points1 = np.random.rand(10, 2)
    #points2 = np.random.rand(10, 2)
    # Calculate the pairwise distances between the points
    if len(points1) == 0 or len(points2) == 0:
        return pd.DataFrame(columns=[0, 1, 2])
    distances = cdist(points1, points2)
    # Sort the distances and get the indices of the sorted elements
    sorted_indices = np.argsort(distances, axis=None)
    # Keep track of which points have already been paired
    paired_points1 = set()
    paired_points2 = set()
    # Loop through the sorted distances and pair the closest points
    pairs = []
    for index in sorted_indices:
        i1, i2 = np.unravel_index(index, distances.shape)
        if i1 not in paired_points1 and i2 not in paired_points2:
            paired_points1.add(i1)
            paired_points2.add(i2)
            pairs.append((i1, i2, float(distances[i1, i2])))
    # Print the pairs
    Dots = pd.DataFrame(pairs)
    return(Dots)

def load_detection_table(input_path):
    path = Path(input_path)
    if path.suffix.lower() == ".npy":
        tb_np = np.load(input_path)
        tb = pd.DataFrame(tb_np)
    else:
        tb = pd.read_csv(input_path, sep=r"[,\s]+", header=None, engine="python")
    if tb.shape[1] < 6:
        raise ValueError("Input detection table must have at least 6 columns: frame class x y w h [conf].")
    if tb.shape[1] >= 7:
        tb = tb.iloc[:, :7].copy()
    else:
        tb = tb.iloc[:, :6].copy()
        tb[6] = 1.0
    tb.columns = list(range(7))
    tb[0] = tb[0].astype(float).round().astype(int)
    tb[1] = tb[1].astype(float).round().astype(int)
    for col in [2, 3, 4, 5, 6]:
        tb[col] = tb[col].astype(float)
    return tb.sort_values(0).reset_index(drop=True)


def load_manual_detection_rows(path, start_frame):
    """
    Load manually corrected detection rows for one frame.
    Supported formats:
    1) frame-first: frame class x y w h [conf]
    2) class-first snapshot labels: class x y w h [conf]  (frame will be set to start_frame)
    """
    raw = pd.read_csv(path, sep=r"[,\s]+", header=None, engine="python")
    if raw.empty:
        raise ValueError(f"Initial results file is empty: {path}")
    if raw.shape[1] < 5:
        raise ValueError(
            "Initial results file must have at least 5 columns: class x y w h (or frame class x y w h)."
        )

    c0 = pd.to_numeric(raw[0], errors="coerce")
    c1 = pd.to_numeric(raw[1], errors="coerce") if raw.shape[1] > 1 else pd.Series([], dtype=float)
    c2 = pd.to_numeric(raw[2], errors="coerce") if raw.shape[1] > 2 else pd.Series([], dtype=float)

    # Heuristic for class-first label rows:
    # - col0 looks like class IDs (small non-negative integers, including optional extra classes like 2/3)
    # - col1/col2 look like normalized coordinates in [0,1]
    # This avoids requiring only {0,1}, since some datasets include extra classes.
    c0_notna = c0.notna().all()
    c0_small_int_like = c0_notna and bool((c0 >= 0).all()) and bool((c0 <= 9).all()) and bool((np.abs(c0 - np.round(c0)) < 1e-6).all())
    c0_low_unique = c0_notna and int(c0.nunique()) <= 10
    c1_norm_like = c1.notna().all() and bool((c1 >= 0).all()) and bool((c1 <= 1).all()) if len(c1) else False
    c2_norm_like = c2.notna().all() and bool((c2 >= 0).all()) and bool((c2 <= 1).all()) if len(c2) else False
    looks_like_class_first = c0_small_int_like and c0_low_unique and c1_norm_like and c2_norm_like

    if looks_like_class_first:
        out = pd.DataFrame()
        out[1] = pd.to_numeric(raw[0], errors="coerce").round().astype(int)
        out[2] = pd.to_numeric(raw[1], errors="coerce")
        out[3] = pd.to_numeric(raw[2], errors="coerce")
        out[4] = pd.to_numeric(raw[3], errors="coerce")
        out[5] = pd.to_numeric(raw[4], errors="coerce")
        if raw.shape[1] >= 6:
            out[6] = pd.to_numeric(raw[5], errors="coerce")
        else:
            out[6] = 1.0
        out[0] = int(start_frame)
        out = out.dropna(subset=[0, 1, 2, 3, 4, 5]).copy()
        out = out[[0, 1, 2, 3, 4, 5, 6]].copy()
        out = out.sort_values(1).reset_index(drop=True)
        LOGGER.info(
            "Detected class-first manual labels (classes=%s); assigned frame=%s for %s rows from %s",
            sorted(pd.unique(out[1]).tolist()),
            int(start_frame),
            int(len(out)),
            path,
        )
        return out

    # Otherwise parse as frame-first detection table.
    tb = load_detection_table(path)
    frame_rows = tb[tb[0] == int(start_frame)].copy()
    if frame_rows.empty:
        raise ValueError(
            f"Initial results file has no rows for frame {start_frame}: {path}. "
            "If this is a snapshot label file (class x y w h ...), keep --initial-frame set and pass that file directly."
        )
    return frame_rows


def replace_frame_detections(base_tb, manual_path, start_frame):
    """
    Replace all detections at start_frame in base table with manual detections.
    """
    manual_rows = load_manual_detection_rows(manual_path, start_frame)
    replaced = pd.concat(
        [base_tb[base_tb[0] != int(start_frame)].copy(), manual_rows],
        ignore_index=True,
    )
    replaced = replaced.sort_values(0).reset_index(drop=True)
    LOGGER.info(
        "Applied initial-results override at frame=%s: replaced_with_rows=%s source=%s",
        int(start_frame),
        int(len(manual_rows)),
        manual_path,
    )
    return replaced

def select_top_confidence_bodies(body_df, num_fly):
    if len(body_df) <= num_fly:
        return body_df.copy()
    return body_df.sort_values(6, ascending=False).head(num_fly).sort_index().copy()

def parse_tracking_json(path):
    frames_out = {}
    p = Path(path)
    if not p.exists():
        return frames_out
    for part in p.read_text(encoding="utf-8", errors="replace").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            obj = json.loads(part)
        except json.JSONDecodeError:
            continue
        for frame, flies in obj.items():
            frames_out[int(frame)] = flies
    return frames_out

def map_json_ids(frame_dict, id_map):
    mapped = {}
    for frame, flies in frame_dict.items():
        mapped[frame] = {}
        for fly_id, data in flies.items():
            mapped[frame][id_map.get(fly_id, fly_id)] = data
    return mapped

def parse_split_boundaries(value, max_value):
    if not value:
        return []
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        v = int(round(float(item)))
        if 0 < v < max_value:
            out.append(v)
    return sorted(set(out))

def region_edges(split_x, split_y, width, height):
    x_edges = [0] + parse_split_boundaries(split_x, width) + [width]
    y_edges = [0] + parse_split_boundaries(split_y, height) + [height]
    regions = []
    rid = 0
    for yi in range(len(y_edges) - 1):
        for xi in range(len(x_edges) - 1):
            regions.append({
                "region_id": rid,
                "x0": x_edges[xi],
                "x1": x_edges[xi + 1],
                "y0": y_edges[yi],
                "y1": y_edges[yi + 1],
            })
            rid += 1
    return regions

def filter_region_table(tb, region, width, height):
    x = tb[2] * width
    y = tb[3] * height
    is_last_x = region["x1"] == width
    is_last_y = region["y1"] == height
    x_mask = (x >= region["x0"]) & ((x < region["x1"]) | (is_last_x & (x <= region["x1"])))
    y_mask = (y >= region["y0"]) & ((y < region["y1"]) | (is_last_y & (y <= region["y1"])))
    return tb[x_mask & y_mask].copy()

def remap_tracked_csv(csv_path, id_offset):
    df = pd.read_csv(csv_path, header=None)
    if df.empty:
        return df, id_offset
    id_col = df.columns[-2]
    ids = sorted(df[id_col].dropna().unique(), key=fly_sort_key_for_id)
    id_map = {old: f"fly_{id_offset + i}" for i, old in enumerate(ids)}
    df[id_col] = df[id_col].map(lambda x: id_map.get(x, x))
    return df, id_offset + len(ids)

def fly_sort_key_for_id(fly_id):
    s = str(fly_id)
    return int(s.split("_")[-1]) if "_" in s and s.split("_")[-1].isdigit() else s

def remap_tracking_json_file(json_path, id_offset):
    frames_dict = parse_tracking_json(json_path)
    ids = sorted({fly for flies in frames_dict.values() for fly in flies}, key=fly_sort_key_for_id)
    id_map = {old: f"fly_{id_offset + i}" for i, old in enumerate(ids)}
    return map_json_ids(frames_dict, id_map), id_offset + len(ids)

def run_split_region_tracking(tb, frames, output, video, num_fly, workers, overlap, split_x, split_y, width, height):
    regions = region_edges(split_x, split_y, width, height)
    if len(regions) <= 1:
        return False
    LOGGER.info(
        "Region tracking enabled: %s regions from split-x=%s split-y=%s",
        len(regions),
        split_x or "none",
        split_y or "none",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    merged_rows = []
    merged_json_by_frame = {}
    id_offset = 0

    with tempfile.TemporaryDirectory(prefix="post_track_regions_") as tmpdir:
        tmpdir = Path(tmpdir)
        with progress_manager(ENABLE_PROGRESS and len(regions) > 0) as progress:
            region_task = progress.add_task("Regions", total=len(regions)) if progress is not None else None
            for region in regions:
                rid = region["region_id"]
                progress_step(progress, region_task, advance=0, description=f"region-{rid}: prepare")
                region_tb = filter_region_table(tb, region, width, height)
                if region_tb.empty:
                    LOGGER.info("Region %s empty, skipped.", rid)
                    progress_step(progress, region_task, description=f"region-{rid}: skipped")
                    continue
                region_input = tmpdir / f"region_{rid:03d}.csv"
                region_output = tmpdir / f"region_{rid:03d}_tracked.csv"
                region_tb.to_csv(region_input, sep=" ", header=False, index=False)
                cmd = [
                    sys.executable, str(script_path),
                    "-i", str(region_input),
                    "-o", str(region_output),
                    "-v", str(video),
                    "-n", str(num_fly),
                    "--workers", str(workers),
                    "--window-overlap", str(overlap),
                    "--internal-region",
                ]
                progress_step(progress, region_task, advance=0, description=f"region-{rid}: run")
                subprocess.run(cmd, check=True)
                region_start = int(region_tb[0].min())
                region_json = Path(str(region_output) + "_" + str(region_start) + "_.json")

                region_id_offset = id_offset
                remapped_csv, id_offset = remap_tracked_csv(region_output, region_id_offset)
                if not remapped_csv.empty:
                    merged_rows.append(remapped_csv)
                remapped_json, id_offset_json = remap_tracking_json_file(region_json, region_id_offset)
                id_offset = max(id_offset, id_offset_json)
                for fr in sorted(remapped_json):
                    merged_json_by_frame.setdefault(fr, {}).update(remapped_json[fr])
                progress_step(progress, region_task, description=f"region-{rid}: merged")

    if merged_rows:
        merged = pd.concat(merged_rows)
        merged.sort_values([merged.columns[0], merged.columns[-2]]).to_csv(output, header=False, index=False)
    else:
        output.write_text("", encoding="utf-8")
    json_output = Path(str(output) + "_" + str(int(frames[0])) + "_.json")
    json_parts = [json.dumps({fr: merged_json_by_frame[fr]}) + ";" for fr in sorted(merged_json_by_frame)]
    json_output.write_text("".join(json_parts), encoding="utf-8")
    LOGGER.info("Saved split-region CSV: %s", output)
    LOGGER.info("Saved split-region JSON: %s", json_output)
    return True

def stitch_id_map(prev_frames, curr_frames):
    shared = sorted(set(prev_frames).intersection(curr_frames))
    prev_ids = sorted({fly for fr in shared for fly in prev_frames.get(fr, {})})
    curr_ids = sorted({fly for fr in shared for fly in curr_frames.get(fr, {})})
    if not shared or not prev_ids or not curr_ids:
        return {fly: fly for fly in curr_ids}

    costs = np.full((len(curr_ids), len(prev_ids)), 1e9, dtype=float)
    for i, curr_id in enumerate(curr_ids):
        for j, prev_id in enumerate(prev_ids):
            distances = []
            for fr in shared:
                curr_body = curr_frames.get(fr, {}).get(curr_id, {}).get("body")
                prev_body = prev_frames.get(fr, {}).get(prev_id, {}).get("body")
                if curr_body is None or prev_body is None:
                    continue
                distances.append(np.linalg.norm(np.array(curr_body[:2]) - np.array(prev_body[:2])))
            if distances:
                costs[i, j] = float(np.mean(distances))

    pairs = Dots_Sort(costs.argmin(axis=1).reshape(-1, 1), np.arange(len(prev_ids)).reshape(-1, 1))
    id_map = {}
    used_curr = set()
    used_prev = set()
    for flat_idx in np.argsort(costs, axis=None):
        i, j = np.unravel_index(flat_idx, costs.shape)
        if costs[i, j] >= 1e8:
            break
        if i in used_curr or j in used_prev:
            continue
        id_map[curr_ids[i]] = prev_ids[j]
        used_curr.add(i)
        used_prev.add(j)
    for curr_id in curr_ids:
        id_map.setdefault(curr_id, curr_id)
    return id_map

def build_windows(frames, workers, overlap):
    workers = max(1, min(workers, len(frames)))
    if overlap > 0 and workers > 1:
        max_workers_by_overlap = max(1, len(frames) // overlap)
        if workers > max_workers_by_overlap:
            LOGGER.warning(
                "Reducing workers from %s to %s: %s processed frames with overlap %s would create mostly-overlap windows.",
                workers,
                max_workers_by_overlap,
                len(frames),
                overlap,
            )
            workers = max_workers_by_overlap
    core_size = int(math.ceil(len(frames) / workers))
    windows = []
    for worker_id in range(workers):
        core_start = worker_id * core_size
        core_end = min(len(frames), (worker_id + 1) * core_size)
        if core_start >= core_end:
            break
        actual_start = max(0, core_start - overlap)
        actual_end = min(len(frames), core_end + overlap)
        windows.append({
            "worker_id": worker_id,
            "core_frames": [int(x) for x in frames[core_start:core_end]],
            "actual_frames": [int(x) for x in frames[actual_start:actual_end]],
        })
    return windows

def window_start_counts(tb, frame, num_fly):
    frame_rows = tb[tb[0] == int(frame)]
    body_count = len(select_top_confidence_bodies(frame_rows[frame_rows[1] == 0], num_fly))
    head_count = int((frame_rows[1] == 1).sum())
    return body_count, head_count

def window_start_is_valid(tb, frame, num_fly):
    body_count, head_count = window_start_counts(tb, frame, num_fly)
    return body_count == int(num_fly) and head_count >= int(num_fly)

def ensure_valid_window_start(tb, frames, win, num_fly):
    """
    Worker initialization needs a reliable first frame. If the planned window
    start does not have enough bodies/heads, prepend earlier frames until it does.
    """
    if not win["actual_frames"]:
        return win

    frame_to_pos = {int(frame): idx for idx, frame in enumerate(frames)}
    actual_start_frame = int(win["actual_frames"][0])
    actual_start_pos = frame_to_pos[actual_start_frame]

    for pos in range(actual_start_pos, -1, -1):
        candidate_frame = int(frames[pos])
        if window_start_is_valid(tb, candidate_frame, num_fly):
            if pos == actual_start_pos:
                return win
            adjusted = dict(win)
            adjusted["actual_frames"] = [int(x) for x in frames[pos:frame_to_pos[int(win["actual_frames"][-1])] + 1]]
            LOGGER.info(
                "Adjusted worker %s start frame from %s to %s for valid initialization.",
                win["worker_id"],
                actual_start_frame,
                candidate_frame,
            )
            return adjusted

    body_count, head_count = window_start_counts(tb, actual_start_frame, num_fly)
    raise ValueError(
        f"Worker {win['worker_id']} cannot find a valid initialization frame at or before "
        f"{actual_start_frame}. Need {num_fly} bodies after filtering and at least {num_fly} heads; "
        f"planned start has bodies={body_count}, heads={head_count}."
    )

def run_window_worker(script_path, window_input, window_output, video, num_fly):
    cmd = [
        sys.executable, str(script_path),
        "-i", str(window_input),
        "-o", str(window_output),
        "-v", str(video),
        "-n", str(num_fly),
        "--internal-window",
    ]
    subprocess.run(cmd, check=True)
    start_frame = int(pd.read_csv(window_input, sep=r"[,\s]+", header=None, engine="python", nrows=1).iloc[0, 0])
    return {
        "output": Path(window_output),
        "json": Path(str(window_output) + "_" + str(start_frame) + "_.json"),
    }

def run_windowed_tracking(tb, frames, output, video, num_fly, workers, overlap):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    windows = build_windows(frames, workers, overlap)
    LOGGER.info(
        "Window tracking enabled: %s processed frames, %s workers, overlap=%s",
        len(frames),
        len(windows),
        overlap,
    )
    script_path = Path(__file__).resolve()

    with tempfile.TemporaryDirectory(prefix="post_track_windows_") as tmpdir:
        tmpdir = Path(tmpdir)
        jobs = []
        with progress_manager(ENABLE_PROGRESS and len(windows) > 0) as progress:
            prep_task = progress.add_task("Window prepare", total=len(windows)) if progress is not None else None
            run_task = progress.add_task("Window run", total=len(windows)) if progress is not None else None
            merge_task = progress.add_task("Window merge", total=len(windows)) if progress is not None else None
            for win in windows:
                win = ensure_valid_window_start(tb, frames, win, num_fly)
                wid = win["worker_id"]
                progress_step(progress, prep_task, advance=0, description=f"window-{wid}: prepare")
                win_input = tmpdir / f"window_{wid:03d}.csv"
                win_output = tmpdir / f"window_{wid:03d}_tracked.csv"
                tb[tb[0].isin(win["actual_frames"])].to_csv(win_input, sep=" ", header=False, index=False)
                jobs.append((win, win_input, win_output))
                progress_step(progress, prep_task, description=f"window-{wid}: prepared")

            if not jobs:
                output.write_text("", encoding="utf-8")
                json_output = Path(str(output) + "_" + str(int(frames[0])) + "_.json")
                json_output.write_text("", encoding="utf-8")
                LOGGER.warning("No windows were generated; wrote empty outputs.")
                return

            results = {}
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                future_map = {
                    executor.submit(run_window_worker, script_path, inp, out, video, num_fly): win
                    for win, inp, out in jobs
                }
                for future in as_completed(future_map):
                    win = future_map[future]
                    wid = win["worker_id"]
                    progress_step(progress, run_task, description=f"window-{wid}: done")
                    results[wid] = future.result()

            merged_rows = []
            merged_json_parts = []
            prev_full_json = None
            for win in windows:
                wid = win["worker_id"]
                progress_step(progress, merge_task, advance=0, description=f"window-{wid}: merge")
                result = results[wid]
                core_set = set(win["core_frames"])
                tracked = pd.read_csv(result["output"], header=None)
                if tracked.empty:
                    progress_step(progress, merge_task, description=f"window-{wid}: merge-empty")
                    continue
                id_col = tracked.columns[-2]
                curr_full_json = parse_tracking_json(result["json"])
                if prev_full_json is None:
                    id_map = {fly: fly for flies in curr_full_json.values() for fly in flies}
                else:
                    id_map = stitch_id_map(prev_full_json, curr_full_json)

                tracked[id_col] = tracked[id_col].map(lambda x: id_map.get(x, x))
                core_rows = tracked[tracked[0].isin(core_set)]
                merged_rows.append(core_rows)

                mapped_core_json = map_json_ids(
                    {fr: flies for fr, flies in curr_full_json.items() if fr in core_set},
                    id_map,
                )
                for fr in sorted(mapped_core_json):
                    merged_json_parts.append(json.dumps({fr: mapped_core_json[fr]}) + ";")
                prev_full_json = map_json_ids(curr_full_json, id_map)
                progress_step(progress, merge_task, description=f"window-{wid}: merged")

            if merged_rows:
                pd.concat(merged_rows).sort_values(0).to_csv(output, header=False, index=False)
            else:
                output.write_text("", encoding="utf-8")
            json_output = Path(str(output) + "_" + str(int(frames[0])) + "_.json")
            json_output.write_text("".join(merged_json_parts), encoding="utf-8")
            LOGGER.info("Saved stitched CSV: %s", output)
            LOGGER.info("Saved stitched JSON: %s", json_output)

def box_center(img_f, Type = 'R'):
    img_f[img_f>50] = 0
    nonzero_indices = np.nonzero(img_f)
    if len(nonzero_indices[0]) == 0:
        return (0, 0)
    # Create a DataFrame with the non-zero indices and their corresponding values
    df = pd.DataFrame({
        'col': nonzero_indices[0],
        'row': nonzero_indices[1]
    })
    if Type == "R":
        # Normalize by the actual image/crop shape instead of fixed 1920x1080.
        h, w = img_f.shape[:2]
        if w <= 0 or h <= 0:
            return (0, 0)
        return ((df.row.mean() - (w / 2)) / w, (df.col.mean() - (h / 2)) / h)
    else:
        return (df.row.mean(), df.col.mean())

def crop_box(img, ob_tb):
    if img is None or len(ob_tb) == 0:
        return None
    if isinstance(ob_tb, pd.DataFrame):
        if ob_tb.empty:
            return None
        row = ob_tb.iloc[0]
    elif isinstance(ob_tb, pd.Series):
        row = ob_tb
    else:
        row = ob_tb
    try:
        xc = float(row[2])
        yc = float(row[3])
        bw = float(row[4])
        bh = float(row[5])
    except Exception:
        return None
    x1 = max(0, int(xc - bw / 2))
    x2 = min(img.shape[1], int(xc + bw / 2))
    y1 = max(0, int(yc - bh / 2))
    y2 = min(img.shape[0], int(yc + bh / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def denormalize_box_df(df):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    out[2] *= TRACK_FRAME_WIDTH
    out[4] *= TRACK_FRAME_WIDTH
    out[3] *= TRACK_FRAME_HEIGHT
    out[5] *= TRACK_FRAME_HEIGHT
    return out


def _safe_head_row(tb_head, idx):
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return None
    if i < 0 or i >= len(tb_head):
        return None
    return list(tb_head.iloc[i, 1:])


def _nearest_head_idx(tb_head, target_xy, used_idx, max_dist):
    if tb_head is None or len(tb_head) == 0:
        return None
    tx, ty = target_xy
    best_idx = None
    best_dist = None
    for i in range(len(tb_head)):
        if i in used_idx:
            continue
        try:
            hx = float(tb_head.iloc[i, 1])
            hy = float(tb_head.iloc[i, 2])
        except Exception:
            continue
        d = float(np.hypot(hx - tx, hy - ty))
        if best_dist is None or d < best_dist:
            best_dist = d
            best_idx = i
    if best_idx is None:
        return None
    if best_dist is None or best_dist > max_dist:
        return None
    return best_idx

def _box_polygon_center_xywh(arry):
    try:
        xc = float(arry[0])
        yc = float(arry[1])
        bw = float(arry[2])
        bh = float(arry[3])
    except Exception:
        return None
    if bw <= 0 or bh <= 0:
        return None
    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0
    if x2 <= x1 or y2 <= y1:
        return None
    return Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

def _overlap_ratio(poly_a, poly_b):
    if poly_a is None or poly_b is None:
        return 0.0
    try:
        if poly_a.area <= 0 or poly_b.area <= 0:
            return 0.0
        inter = float(poly_a.intersection(poly_b).area)
        if inter <= 0:
            return 0.0
        return inter / float(poly_a.area)
    except Exception:
        return 0.0

def _body_has_c4_protection(body_xywh, c4_rows):
    if c4_rows is None or len(c4_rows) == 0:
        return False
    body_poly = _box_polygon_center_xywh(body_xywh)
    if body_poly is None:
        return False
    for _, c4 in c4_rows.iterrows():
        c4_poly = _box_polygon_center_xywh(c4.iloc[2:6].to_numpy())
        if _overlap_ratio(body_poly, c4_poly) >= C4_BODY_SAFE_OVERLAP:
            return True
    return False

def _mount_has_opposite_orientations(body_rows, head_rows):
    if body_rows is None or len(body_rows) < 2 or head_rows is None or len(head_rows) == 0:
        return False
    dirs = []
    for _, b in body_rows.iterrows():
        bx = float(b[2])
        by = float(b[3])
        best = None
        best_d = None
        for _, h in head_rows.iterrows():
            hx = float(h[2])
            hy = float(h[3])
            d = float(np.hypot(hx - bx, hy - by))
            if best_d is None or d < best_d:
                best_d = d
                best = (hx, hy)
        if best is None:
            continue
        vx = best[0] - bx
        vy = best[1] - by
        norm = float(np.hypot(vx, vy))
        if norm <= 1e-9:
            continue
        dirs.append((vx / norm, vy / norm))
    if len(dirs) < 2:
        return False
    # Two opposite body->head directions suggest a plausible mount pair.
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            dot = float(dirs[i][0] * dirs[j][0] + dirs[i][1] * dirs[j][1])
            if dot <= MOUNT_DIRECTION_DOT_MAX:
                return True
    return False

def _mount_protected_indices(tmp_b, tmp_full, body_area_mean):
    out = set()
    if tmp_b is None or len(tmp_b) == 0 or tmp_full is None or len(tmp_full) == 0:
        return out
    c5_rows = tmp_full[tmp_full[1] == 5]
    if len(c5_rows) == 0:
        return out
    head_rows = tmp_full[tmp_full[1] == 1]
    body_areas = (tmp_b[4].to_numpy() * tmp_b[5].to_numpy()) if len(tmp_b) else np.array([])
    for _, c5 in c5_rows.iterrows():
        c5_poly = _box_polygon_center_xywh(c5.iloc[2:6].to_numpy())
        c5_area = float(c5[4]) * float(c5[5]) if float(c5[4]) > 0 and float(c5[5]) > 0 else 0.0
        if c5_poly is None or c5_area <= 0:
            continue
        in_mount_idx = []
        for i, b in enumerate(tmp_b.itertuples(index=False)):
            b_xywh = np.array([b[2], b[3], b[4], b[5]], dtype=float)
            b_poly = _box_polygon_center_xywh(b_xywh)
            if _overlap_ratio(b_poly, c5_poly) >= MOUNT_BODY_OVERLAP_MIN:
                in_mount_idx.append(i)
        if not in_mount_idx:
            continue
        if len(in_mount_idx) >= 2:
            safe_flags = []
            for i in in_mount_idx:
                b_area = float(body_areas[i]) if i < len(body_areas) else 0.0
                if body_area_mean > 0:
                    safe_flags.append((b_area / body_area_mean) <= Box_size_check_d)
                else:
                    safe_flags.append(False)
            has_safe = any(safe_flags)
            body_rows = tmp_b.iloc[in_mount_idx]
            if has_safe or _mount_has_opposite_orientations(body_rows, head_rows):
                out.update(in_mount_idx)
            continue
        # Only one body recognized inside c5.
        i = in_mount_idx[0]
        b_area = float(body_areas[i]) if i < len(body_areas) else 0.0
        ratio_to_c5 = b_area / c5_area if c5_area > 0 else 0.0
        if (
            MOUNT_BODY_C5_SIMILAR_MIN <= ratio_to_c5 <= MOUNT_BODY_C5_SIMILAR_MAX
            or ratio_to_c5 < MOUNT_BODY_C5_SMALL_MAX
        ):
            out.add(i)
    return out

def _clearlist_protected_indices(tmp_b, tmp_full):
    out = set()
    if tmp_b is None or len(tmp_b) == 0:
        return out
    c4_rows = tmp_full[tmp_full[1] == 4] if tmp_full is not None and len(tmp_full) > 0 else pd.DataFrame()
    body_area_mean = float((tmp_b[4] * tmp_b[5]).mean()) if len(tmp_b) > 0 else 0.0
    for i in range(len(tmp_b)):
        body_xywh = tmp_b.iloc[i, 2:6].to_numpy()
        if _body_has_c4_protection(body_xywh, c4_rows):
            out.add(i)
    out.update(_mount_protected_indices(tmp_b, tmp_full, body_area_mean))
    return out


def _iter_mount_pairs_prev_frame(s_tmp_b, tmp_full):
    """
    Detect (fly_id_a, fly_id_b, center_a, center_b) for each class-5 mount region
    on the previous frame, using the same safety rules as _mount_protected_indices.
    Centers are normalized (x, y) arrays.
    """
    pairs = []
    if s_tmp_b is None or len(s_tmp_b) < 2 or tmp_full is None or len(tmp_full) == 0:
        return pairs
    c5_rows = tmp_full[tmp_full[1] == 5]
    if len(c5_rows) == 0:
        return pairs
    head_rows = tmp_full[tmp_full[1] == 1]
    tmp_b = s_tmp_b
    body_area_mean = float((tmp_b[4] * tmp_b[5]).mean()) if len(tmp_b) > 0 else 0.0
    seen = set()

    for _, c5 in c5_rows.iterrows():
        c5_poly = _box_polygon_center_xywh(c5.iloc[2:6].to_numpy())
        if c5_poly is None:
            continue
        in_mount_idx = []
        overlap_scores = []
        for i, b in enumerate(tmp_b.itertuples(index=False)):
            b_xywh = np.array([b[2], b[3], b[4], b[5]], dtype=float)
            b_poly = _box_polygon_center_xywh(b_xywh)
            r = _overlap_ratio(b_poly, c5_poly)
            if r >= MOUNT_BODY_OVERLAP_MIN:
                in_mount_idx.append(i)
                overlap_scores.append(r)
        if len(in_mount_idx) < 2:
            continue
        if len(in_mount_idx) > 2:
            order = np.argsort(-np.array(overlap_scores, dtype=float))[:2]
            in_mount_idx = [in_mount_idx[int(j)] for j in order]
            body_rows = tmp_b.iloc[in_mount_idx]
        else:
            body_rows = tmp_b.iloc[in_mount_idx]
        safe_flags = []
        for i in in_mount_idx:
            b_area = float(tmp_b.iloc[i, 4] * tmp_b.iloc[i, 5])
            if body_area_mean > 0:
                safe_flags.append((b_area / body_area_mean) <= Box_size_check_d)
            else:
                safe_flags.append(False)
        has_safe = any(safe_flags)
        if not (has_safe or _mount_has_opposite_orientations(body_rows, head_rows)):
            continue
        id_a = tmp_b.iloc[in_mount_idx[0]].ID
        id_b = tmp_b.iloc[in_mount_idx[1]].ID
        key = (str(id_a), str(id_b))
        if key in seen or (str(id_b), str(id_a)) in seen:
            continue
        seen.add(key)
        ca = tmp_b.iloc[in_mount_idx[0]][2:4].to_numpy(dtype=float)
        cb = tmp_b.iloc[in_mount_idx[1]][2:4].to_numpy(dtype=float)
        pairs.append((str(id_a), str(id_b), ca, cb))
    return pairs


def _mount_pair_for_fly(mount_pairs, fly_id):
    fid = str(fly_id)
    for id_a, id_b, ca, cb in mount_pairs:
        if fid == id_a or fid == id_b:
            return (id_a, id_b, ca, cb)
    return None


def _mount_partner_id(pair_tuple, fly_id):
    id_a, id_b, _, _ = pair_tuple
    fid = str(fly_id)
    if fid == id_a:
        return id_b
    if fid == id_b:
        return id_a
    return None


def _maybe_swap_mount_pair_ids(tmp_b, s_tmp_b, tmp_prev_full, qa):
    """
    If the previous frame had a mount pair inside class-5, greedy nearest matching can
    swap the two IDs. Pick the assignment that minimizes total center displacement from
    the previous frame (equivalent to preferring the previous inter-fly offset vector).
    """
    if tmp_b is None or tmp_b.empty or s_tmp_b is None or len(s_tmp_b) == 0:
        return tmp_b
    if tmp_prev_full is None or len(tmp_prev_full) == 0:
        return tmp_b
    pairs = _iter_mount_pairs_prev_frame(s_tmp_b, tmp_prev_full)
    if not pairs:
        return tmp_b
    out = tmp_b
    for id_a, id_b, ca, cb in pairs:
        if not (out["ID"] == id_a).any() or not (out["ID"] == id_b).any():
            continue
        rows_a = out.index[out["ID"] == id_a]
        rows_b = out.index[out["ID"] == id_b]
        if len(rows_a) != 1 or len(rows_b) != 1:
            continue
        ia, ib = int(rows_a[0]), int(rows_b[0])
        if ia == ib:
            continue
        pa = out.loc[ia, [2, 3]].to_numpy(dtype=float)
        pb = out.loc[ib, [2, 3]].to_numpy(dtype=float)
        cost_ok = float(np.linalg.norm(pa - ca) + np.linalg.norm(pb - cb))
        cost_sw = float(np.linalg.norm(pa - cb) + np.linalg.norm(pb - ca))
        if cost_sw + 1e-6 < cost_ok:
            if out is tmp_b:
                out = tmp_b.copy()
            out.loc[ia, "ID"] = id_b
            out.loc[ib, "ID"] = id_a
            qa["mount_id_swap_fixes"] += 1
    return out


def _mount_relative_recovered_row(ob_ls, partner_id, frame, s_tmp_b, tmp_b_current):
    """
    Place the lost fly at partner_current_center + (lost_prev - partner_prev),
    preserving box size from the last tracked frame.
    """
    lost_prev = s_tmp_b[s_tmp_b["ID"] == ob_ls]
    partner_prev = s_tmp_b[s_tmp_b["ID"] == partner_id]
    partner_now = tmp_b_current[tmp_b_current["ID"] == partner_id]
    if lost_prev.empty or partner_prev.empty or partner_now.empty:
        return None
    lx = float(lost_prev.iloc[0][2])
    ly = float(lost_prev.iloc[0][3])
    px0 = float(partner_prev.iloc[0][2])
    py0 = float(partner_prev.iloc[0][3])
    px1 = float(partner_now.iloc[0][2])
    py1 = float(partner_now.iloc[0][3])
    rel_x = lx - px0
    rel_y = ly - py0
    pred_x = min(max(px1 + rel_x, 0.0), 1.0)
    pred_y = min(max(py1 + rel_y, 0.0), 1.0)
    obl = lost_prev.copy()
    obl.iloc[0, 0] = frame
    obl.iloc[0, 2] = pred_x
    obl.iloc[0, 3] = pred_y
    obl["find"] = False
    return obl


def creat_polygon(arry):
    # arry = S_TMP_B[S_TMP_B.ID==ob_ls].to_numpy()[0][2:6]
    X1 = arry[0]
    Y1 = arry[1]
    X2 = arry[0] + arry[2]  
    Y2 = arry[1] + arry[3] 
    x = [X1, X2, X2, X1]
    y = [Y1, Y1, Y2, Y2]
    return Polygon([[i,j]for i,j in zip(x,y)])

def Overlap_test(ob_ls, TMP_B):
    if TMP_B is None or TMP_B.empty:
        return False
    if TMP_B[TMP_B.ID == ob_ls].empty:
        return False
    rct_los = creat_polygon(TMP_B[TMP_B.ID==ob_ls].to_numpy()[0][2:6])
    if rct_los.area <= 0:
        return False
    Inter_dict1 = {}
    Inter_dict2 = {}
    for line in range(len(TMP_B)):
        if TMP_B.ID.iloc[line] != ob_ls:
            rct_tag = creat_polygon(TMP_B.iloc[line,2:6].to_numpy())
            if rct_tag.area <= 0:
                continue
            inter_area = rct_los.intersection(rct_tag).area
            Inter_dict1.update({TMP_B.ID.iloc[line]: inter_area / rct_los.area})
            Inter_dict2.update({TMP_B.ID.iloc[line]: inter_area / rct_tag.area})
    if not Inter_dict1 or not Inter_dict2:
        return False
    if max(Inter_dict1.values()) < max(Inter_dict2.values()):
        Inter_dict1 =  Inter_dict2
    if max(Inter_dict1.values()) >= Overlap_thres:
        ob_ov = max(Inter_dict1, key= Inter_dict1.get )
        TMP_cache = TB_cache[TB_cache.ID == ob_ov]
        TMP_cache['Area'] = TMP_cache[4] * TMP_cache[5]
        Ar_change = (TMP_B[TMP_B.ID == ob_ov][4] * TMP_B[TMP_B.ID == ob_ov][5]).to_list()[0] / TMP_cache.Area.mean()
        if Ar_change >= Box_size_check:
            bst_frame = TMP_cache[0][np.abs(TMP_cache.Area - TMP_cache.Area.mean())== min(np.abs(TMP_cache.Area - TMP_cache.Area.mean()))]
            # then, read images and drift by the center
            cap.set(1,frame)
            ret,img_t=cap.read()
            if not ret or img_t is None:
                return False
            OB_TB = denormalize_box_df(S_TMP_B[S_TMP_B.ID == ob_ov])
            img_t = crop_box(img_t, OB_TB)
            if img_t is None:
                return False
            img_t = cv2.cvtColor(img_t,cv2.COLOR_RGB2GRAY)
            img_t = cv2.GaussianBlur(img_t, (5, 5), 10)
            return (bst_frame.to_list()[0], ob_ov, box_center(img_t))
    return False

def Obj_los_test(frame, ob_ls, cap, prev_frame=None):
    Result = None
    OB_TB = denormalize_box_df(S_TMP_B[S_TMP_B.ID == ob_ls])
    if OB_TB is None or len(OB_TB) == 0:
        return {'Type' : "CroLst", "drift" :(0,0)}
    #OB_TB = S_TMP_B[S_TMP_B.ID == 'fly_1']

    cap.set(1,frame)
    ret,img_t=cap.read()
    if not ret or img_t is None:
        LOGGER.warning("Cannot read frame for lost-object recovery: frame=%s", frame)
        return {'Type' : "CroLst", "drift" :(0,0)}
    img_t = crop_box(img_t, OB_TB)
    if img_t is None:
        LOGGER.warning("Invalid crop for lost-object recovery: frame=%s id=%s", frame, ob_ls)
        return {'Type' : "CroLst", "drift" :(0,0)}
    img_t = cv2.cvtColor(img_t,cv2.COLOR_RGB2GRAY)
    img_t = cv2.GaussianBlur(img_t, (5, 5), 10)

    # ratio of fly-body to the blank background. A normal single fly ~= 14, >20 means over corroded.
    CoverR = len(img_t.ravel()[img_t.ravel()<50])/len(img_t.ravel())
    LOGGER.debug("Lost-object recovery mask ratio: frame=%s id=%s ratio=%.4f", frame, ob_ls, CoverR)
    if CoverR>=.19:
        LOGGER.debug("Over-corroded crop in lost-object recovery: frame=%s id=%s", frame, ob_ls)
        return {'Type' : "CroLst", "drift" : box_center(img_t)}
    if CoverR==0:
        LOGGER.debug("Lost-object recovery empty mask: frame=%s id=%s", frame, ob_ls)
        return {'Type' : "CroLst", "drift" : (0,0)}

    cap.set(1, prev_frame if prev_frame is not None else frame-1)
    ret,img_f=cap.read()
    if not ret or img_f is None:
        LOGGER.warning(
            "Cannot read previous frame for lost-object recovery: frame=%s prev_frame=%s",
            frame,
            prev_frame if prev_frame is not None else frame - 1,
        )
        return {'Type' : "CroLst", "drift" :(0,0)}
    img_f = crop_box(img_f, OB_TB)
    if img_f is None:
        return {'Type' : "CroLst", "drift" :(0,0)}
    img_f = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)
    img_f = cv2.GaussianBlur(img_f, (5, 5), 10)
    # similarity clean vs single fly: 68.8%
    try:
        ssim_V = ssim(img_f, img_t)
        LOGGER.debug("Lost-object recovery SSIM: frame=%s id=%s ssim=%.4f", frame, ob_ls, float(ssim_V))
        if ssim_V>=.85:
            LOGGER.debug("Single-fly loss branch selected: frame=%s id=%s", frame, ob_ls)
            
            # ove lap check
            rct_los = creat_polygon(S_TMP_B[S_TMP_B.ID==ob_ls].to_numpy()[0][2:6])
            Inter_dict = {}
            for line in range(len(TMP_B)):
                Inter_dict.update({ TMP_B.ID.iloc[line] : rct_los.intersection(creat_polygon(TMP_B.iloc[line,2:6].to_numpy())).area/ rct_los.area})
            Over_p = list(np.where(np.array(list(Inter_dict.values())) > .2)[0])
            if len(Over_p) > 0:
                LOGGER.debug("Lost-object overlap branch selected: frame=%s id=%s", frame, ob_ls)
                #max_value = max(Inter_dict, key=Inter_dict.get)
                cap.set(1,frame)
                ret,img_n=cap.read()
                if not ret or img_n is None:
                    return {'Type' : "CroLst", "drift" : box_center(img_t)}

                Scores = {}
                fly_cache = TB_cache[TB_cache.ID == ob_ls] 
                for i in TB_cache[0].unique():
                    cap.set(1, i)
                    ret,img_p=cap.read()
                    if not ret or img_p is None:
                        continue
                    Loc = TB_cache[TB_cache[0] == i]
                    OB_TB = denormalize_box_df(Loc[Loc.ID== ob_ls])
                    img_old = crop_box(img_p, OB_TB)
                    if img_old is None:
                        continue
                    img_old = cv2.cvtColor(img_old, cv2.COLOR_RGB2GRAY)
                    img_old[img_old>50] =0
                    img_now = crop_box(img_n, OB_TB)
                    if img_now is None:
                        continue
                    img_now = cv2.cvtColor(img_now, cv2.COLOR_RGB2GRAY)
                    img_now[img_now>50] =0
                    Scores.update({ i: ssim(img_now, img_old)})
                if not Scores:
                    return {'Type' : "CroLst", "drift" : box_center(img_t)}
                best_frame = max(Scores, key=Scores.get)
                return {'Type' : "Overlap", "frame": best_frame, "drift" : box_center(img_t)}
            else:
                LOGGER.debug("No overlap candidate found: frame=%s id=%s", frame, ob_ls)
                return {'Type' : "CroLst", "drift" : box_center(img_t)}
    
    except:
        LOGGER.debug("Small box size in lost-object recovery: frame=%s id=%s", frame, ob_ls)
    LOGGER.debug("Fast-moving/low-similarity fallback used: frame=%s id=%s", frame, ob_ls)
    return {'Type' : "CroLst", "drift" :(0,0)}

def Leap_Check(TB_Leap, frame, cap):
    for leap_i in range(TB_Leap.shape[0]):
        leap_id_from  = TB_Leap.iloc[leap_i, 0]
        leap_id_to     = TB_Leap.iloc[leap_i, 1]
        leap_fly = S_TMP_B.iloc[leap_id_from].ID
        leap_body_from = S_TMP_B[S_TMP_B.ID == leap_fly].to_numpy()[0][2:6]
        leap_body_to = TMP_B.iloc[leap_id_to].to_numpy()[2:6]
        # maxmize the window
        leap_w = int(np.max([leap_body_from[2], leap_body_to[2]]) * 1920)
        leap_h = int(np.max([leap_body_from[3], leap_body_to[3]]) * 1080)
        # window define
        w_BeLeap = leap_body_from.copy()
        w_BeLeap[2] = leap_w
        w_BeLeap[3] = leap_h
        w_BeLeap[0] = int(leap_body_from[0] * 1920 - leap_w/2)
        w_BeLeap[1] = int(leap_body_from[1] * 1080 - leap_h/2)
        w_AfLeap = leap_body_to.copy()
        w_AfLeap[2] = leap_w
        w_AfLeap[3] = leap_h
        w_AfLeap[0] = int(leap_body_to[0] * 1920 - leap_w/2)
        w_AfLeap[1] = int(leap_body_to[1] * 1080 - leap_h/2)
        # retrieve the window to check the leap similarity
        cap.set(1, frame)
        ret,img_f=cap.read()
        img_f = cv2.cvtColor(img_f,cv2.COLOR_RGB2GRAY)
        img_f = cv2.GaussianBlur(img_f, (5, 5), 10)
        # check the similarity
        img_BeLeap = img_f[w_BeLeap[1]:w_BeLeap[1]+w_BeLeap[3], w_BeLeap[0]:w_BeLeap[0]+w_BeLeap[2]]
        img_AfLeap = img_f[w_AfLeap[1]:w_AfLeap[1]+w_AfLeap[3], w_AfLeap[0]:w_AfLeap[0]+w_AfLeap[2]]
        img_BeLeap = cv2.GaussianBlur(img_BeLeap, (5, 5), 10)
        img_AfLeap = cv2.GaussianBlur(img_AfLeap, (5, 5), 10)
        # check the similarity
        ssim_V = ssim(img_BeLeap, img_AfLeap)
        # if no leap
        cap.set(1, frame-1)
        ret,img_f=cap.read()
        img_f = cv2.cvtColor(img_f,cv2.COLOR_RGB2GRAY)
        img_f = cv2.GaussianBlur(img_f, (5, 5), 10)
        # check the similarity
        img_NoLeap = img_f[w_BeLeap[1]:w_BeLeap[1]+w_BeLeap[3], w_BeLeap[0]:w_BeLeap[0]+w_BeLeap[2]]
        img_NoLeap = cv2.GaussianBlur(img_NoLeap, (5, 5), 10)
        # check the similarity
        ssim_Vn = ssim(img_NoLeap, img_BeLeap)

        if ssim_Vn > 0.8 and ssim_Vn > ssim_V:
            # fake leap, it is a false positive
            TB_Leap.Check.iloc[leap_i] = 'jitter'

    return TB_Leap 

## Functions down

# argumetns


CSV_f = INPUT#"Mix7.MP4.csv"
#Video = "/home/ken/Videos/Mix7.MP4"
Num = NUM_FLY
#OUTPUT = 'test.csv'

Box_size_check_d = 1.6
Box_size_check = 1.3
Overlap_thres  = .45
C4_BODY_SAFE_OVERLAP = 0.85
MOUNT_BODY_OVERLAP_MIN = 0.50
MOUNT_BODY_C5_SIMILAR_MIN = 0.65
MOUNT_BODY_C5_SIMILAR_MAX = 1.35
MOUNT_BODY_C5_SMALL_MAX = 0.65
MOUNT_DIRECTION_DOT_MAX = 0.15

TB = load_detection_table(INPUT)
frames_all = sorted(TB[0].unique())
if len(frames_all) == 0:
    raise ValueError("No detection frames found in input.")
if INITIAL_FRAME is not None:
    if int(INITIAL_FRAME) not in set(frames_all):
        raise ValueError(f"--initial-frame {INITIAL_FRAME} is not present in input detections")
    frames = [f for f in frames_all if int(f) >= int(INITIAL_FRAME)]
else:
    frames = list(frames_all)
if len(frames) == 0:
    raise ValueError("No frames remain after applying --initial-frame filter.")
if TEST_FRAMES is not None and len(frames) > TEST_FRAMES:
    frames = frames[:TEST_FRAMES]
TB = TB[TB[0].isin(frames)].reset_index(drop=True)
Start = int(frames[0])
if INITIAL_RESULTS:
    TB = replace_frame_detections(TB, INITIAL_RESULTS, Start)
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
LOGGER.info(
    "Tracking start: input=%s video=%s output=%s frames=%s (from %s to %s) num_fly=%s",
    INPUT,
    Video,
    OUTPUT,
    len(frames),
    int(frames[0]),
    int(frames[-1]),
    Num,
)

if (SPLIT_X or SPLIT_Y) and not args.internal_region and not args.internal_window:
    did_split = run_split_region_tracking(
        TB, frames, OUTPUT, Video, Num, WORKERS, WINDOW_OVERLAP,
        SPLIT_X, SPLIT_Y, FRAME_WIDTH, FRAME_HEIGHT,
    )
    if did_split:
        sys.exit(0)

if WORKERS > 1 and not args.internal_window:
    run_windowed_tracking(TB, frames, OUTPUT, Video, Num, WORKERS, WINDOW_OVERLAP)
    sys.exit(0)

cap=cv2.VideoCapture(Video)
if not cap.isOpened():
    raise ValueError(f"Cannot open video: {Video}")
video_w = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)))
video_h = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)))
if video_w > 0 and video_h > 0:
    TRACK_FRAME_WIDTH = video_w
    TRACK_FRAME_HEIGHT = video_h
LOGGER.info(
    "Geometry setup: frame_width=%s frame_height=%s workers=%s split=(%s,%s) progress=%s",
    TRACK_FRAME_WIDTH,
    TRACK_FRAME_HEIGHT,
    WORKERS,
    SPLIT_X or "none",
    SPLIT_Y or "none",
    ENABLE_PROGRESS and RICH_AVAILABLE,
)

# Define the Start 
S_TMP = TB[TB[0]==Start]
S_TMP_B = S_TMP[S_TMP[1]==0]
if len(S_TMP_B) < Num:
    raise ValueError(f"Frame {Start} has only {len(S_TMP_B)} body detections, fewer than --num-fly {Num}.")
S_TMP_B = select_top_confidence_bodies(S_TMP_B, Num)
S_TMP_B['ID'] = ['fly_' +str(i) for i in range(len(S_TMP_B))]
S_TMP_B = S_TMP_B.copy()
Dots_from = S_TMP_B.iloc[:,2:4].to_numpy()
S_TMP_B['find'] = True
# Save the first frame
S_TMP_B.to_csv(OUTPUT, header=False, index=False)

TB_cache = S_TMP_B

FLY_matrix = {}
TMP_Dict = {}
for fly in S_TMP_B.ID:
    TMP_Dict.update({fly : {'body': S_TMP_B[S_TMP_B.ID==fly].to_numpy().tolist()[0][2:6]}})
FLY_matrix.update({Start: TMP_Dict})

# head bind
TB_head = S_TMP[S_TMP[1] == 1].iloc[:, 1:6]
head_bind.main(FLY_matrix, Start, TB_head)
used_head_start = set()
for fly in FLY_matrix[Start].keys():
    match_idx = None
    try:
        match_idx = int(head_bind.MATCH_result[fly])
    except Exception:
        match_idx = None
    head_row = _safe_head_row(TB_head, match_idx) if match_idx is not None else None
    if head_row is not None and int(match_idx) not in used_head_start:
        FLY_matrix[Start][fly].update({"head": head_row})
        used_head_start.add(int(match_idx))
    else:
        FLY_matrix[Start][fly].update({"head": FLY_matrix[Start][fly]["body"]})

# get the mean values for fly's body-length
fly_len = np.mean([cdist([i['body'][:2]], [i['head'][:2]])[0] for i in FLY_matrix[Start].values()])

# write results
dic_ID = list(FLY_matrix.keys())[-1]
tmp = {dic_ID:FLY_matrix[dic_ID]}
FLY_matrix_tmp_str = json.dumps(tmp) +";"
tracking_json_path = OUTPUT+"_"+str(Start)+"_.json"
Path(tracking_json_path).unlink(missing_ok=True)
Trac_out = open(tracking_json_path, "a", encoding="utf-8")
Trac_out.write(FLY_matrix_tmp_str)

qa = {
    "input": str(INPUT),
    "video": str(Video),
    "output_csv": str(OUTPUT),
    "output_json": str(tracking_json_path),
    "num_fly": int(Num),
    "workers": int(WORKERS),
    "window_overlap": int(WINDOW_OVERLAP),
    "split_x": str(SPLIT_X or ""),
    "split_y": str(SPLIT_Y or ""),
    "track_frame_width": int(TRACK_FRAME_WIDTH),
    "track_frame_height": int(TRACK_FRAME_HEIGHT),
    "head_reacquire_after": int(HEAD_REACQUIRE_AFTER),
    "head_reacquire_max_dist": float(HEAD_REACQUIRE_MAX_DIST),
    "initial_frame": int(Start),
    "initial_results": str(INITIAL_RESULTS or ""),
    "frames_total": int(len(frames)),
    "frames_tracked": 1,
    "match_pairs_step1": 0,
    "match_pairs_step2": 0,
    "missing_events": 0,
    "missing_objects_total": 0,
    "lost_recovery_crolst": 0,
    "lost_recovery_overlap": 0,
    "overlap_adjustments": 0,
    "head_bind_success": 0,
    "head_fallback": 0,
    "head_reacquire_attempts": 0,
    "head_reacquire_success": 0,
    "mount_id_swap_fixes": 0,
    "mount_relative_recovery": 0,
}
run_started_ts = time.time()
head_fallback_streak = {str(fly): 0 for fly in S_TMP_B.ID}

# head QA on start frame
used_head_idx = set()
for fly in FLY_matrix[Start].keys():
    match_idx = None
    try:
        match_idx = int(head_bind.MATCH_result[fly])
    except Exception:
        match_idx = None
    head_row = _safe_head_row(TB_head, match_idx) if match_idx is not None else None
    if head_row is not None and int(match_idx) not in used_head_idx:
        qa["head_bind_success"] += 1
        used_head_idx.add(int(match_idx))
        head_fallback_streak[str(fly)] = 0
    else:
        qa["head_fallback"] += 1
        head_fallback_streak[str(fly)] = 1


prev_frame = Start
pipeline_steps = (
    "select-body",
    "id-match",
    "box-adjust",
    "missing-recover",
    "save-csv",
    "head-bind",
    "flush-cache",
)
total_tracked_frames = max(0, len(frames) - 1)
with progress_manager(ENABLE_PROGRESS and total_tracked_frames > 0) as progress:
    frame_task = None
    step_task = None
    if progress is not None:
        frame_task = progress.add_task("Frames", total=total_tracked_frames)
        step_task = progress.add_task("Pipeline steps", total=total_tracked_frames * len(pipeline_steps))
    for frame_idx, frame in enumerate(frames[1:], start=1):
        frame = int(frame)
        qa["frames_tracked"] += 1
        qa_snapshot = {
            "match_pairs_step1": int(qa["match_pairs_step1"]),
            "match_pairs_step2": int(qa["match_pairs_step2"]),
            "missing_objects_total": int(qa["missing_objects_total"]),
            "overlap_adjustments": int(qa["overlap_adjustments"]),
            "head_bind_success": int(qa["head_bind_success"]),
            "head_fallback": int(qa["head_fallback"]),
            "head_reacquire_attempts": int(qa["head_reacquire_attempts"]),
            "head_reacquire_success": int(qa["head_reacquire_success"]),
        }
        # Define the Start
        TMP = TB[TB[0] == frame]
        TMP_B = select_top_confidence_bodies(TMP[TMP[1] == 0], Num)
        TMP_B['ID'] = None
        TMP_B['find'] = True
        # Check the over-sized box and update it if it caused by two/more flies
        TMP_B['Areas'] = (TMP_B[4] * TMP_B[5] * TRACK_FRAME_WIDTH * TRACK_FRAME_HEIGHT).to_numpy()
        protected_idx = _clearlist_protected_indices(TMP_B, TMP)

        Clear_lst = []
        for line in np.where(TMP_B.Areas / TMP_B.Areas.mean() > Box_size_check_d)[0]:
            if int(line) in protected_idx:
                continue
            TMP_BL = TMP_B.iloc[line, :]
            # Check the overlap
            rct_los = creat_polygon(TMP_BL.to_numpy()[2:6])
            Inter_dict1 = [rct_los.intersection(creat_polygon(TMP_B.iloc[line, 2:6].to_numpy())).area / rct_los.area for line in range(len(TMP_B))]
            Inter_dict2 = [rct_los.intersection(creat_polygon(TMP_B.iloc[line, 2:6].to_numpy())).area / creat_polygon(TMP_B.iloc[line, 2:6].to_numpy()).area for line in range(len(TMP_B))]
            Over_p = list(np.where(np.array(Inter_dict1) > Overlap_thres)[0]) + list(np.where(np.array(Inter_dict2) > Overlap_thres)[0])
            while line in Over_p:
                Over_p.remove(line)
            if len(Over_p) > 0:
                Clear_lst += [line]
        TMP_B = TMP_B.drop(TMP_B.index[Clear_lst])

        # remove the Areas column
        TMP_B = TMP_B.drop('Areas', axis=1)
        Dots_to = TMP_B.iloc[:, 2:4].to_numpy()
        progress_step(progress, step_task, description=f"[{frame}] {pipeline_steps[0]}")

        # two steps sort
        ## Step 1
        Dots = Dots_Sort(Dots_from, Dots_to)
        qa["match_pairs_step1"] += int(len(Dots))
        # inherit IDs from previous based on sorting distance
        for i in range(len(Dots)):
            TMP_B.ID.iloc[Dots[1][i]] = S_TMP_B.ID.iloc[Dots[0][i]]

        ## Step 2 Leap Check
        S_TMP_B = S_TMP_B.reset_index(drop=True)

        TB_Leap = Dots[np.array(Dots[2] > fly_len*4).astype(int) + np.array(~S_TMP_B.iloc[Dots[0].tolist()].find) == 2]
        if len(TB_Leap) > 0:
            TB_Leap['Check'] = np.nan 
            # Leap Check
            TB_Leap = Leap_Check(TB_Leap, frame, cap)
            for leap_i in range(TB_Leap.shape[0]):
                if TB_Leap.Check.iloc[leap_i] ==  'jitter':
                    TMP_B.ID.iloc[TB_Leap.iloc[leap_i, 1]] = np.nan
                    
        progress_step(progress, step_task, description=f"[{frame}] {pipeline_steps[1]}")
        TMP_prev_det = TB[TB[0] == prev_frame]
        TMP_B = _maybe_swap_mount_pair_ids(TMP_B, S_TMP_B, TMP_prev_det, qa)

        # Adjust the size of boxs
        TMP_B['Areas'] = (TMP_B[4] * TMP_B[5] * TRACK_FRAME_WIDTH * TRACK_FRAME_HEIGHT).to_numpy()
        for ob_ov in TMP_B.ID[~TMP_B.ID.isna()]:
            if TMP_B.Areas[TMP_B.ID == ob_ov].iloc[0] / TMP_B.Areas.mean() > Box_size_check_d:
                Over_adjust = Overlap_test(ob_ov, TMP_B[~TMP_B.ID.isna()])
                if Over_adjust != False:
                    qa["overlap_adjustments"] += 1
                    ob_ov = Over_adjust[1]
                    TMP_chage = TB_cache[TB_cache[0] == Over_adjust[0]]
                    TMP_chage = TMP_chage[TMP_chage.ID == Over_adjust[1]]
                    TMP_B.iloc[np.where(TMP_B.ID == ob_ov)[0], 4] = TMP_chage[4] * .9
                    TMP_B.iloc[np.where(TMP_B.ID == ob_ov)[0], 5] = TMP_chage[5] * .9
        # remove the Areas column
        TMP_B = TMP_B.drop('Areas', axis=1)
        Dots_to = TMP_B.iloc[:, 2:4].to_numpy()
        progress_step(progress, step_task, description=f"[{frame}] {pipeline_steps[2]}")

        # Missing Check
        matched_count = TMP_B.ID.notna().sum()
        if matched_count < len(S_TMP_B):
            qa["missing_events"] += 1
            qa["missing_objects_total"] += int(len(S_TMP_B) - matched_count)
            mount_pairs_prev = _iter_mount_pairs_prev_frame(S_TMP_B, TMP_prev_det)
            for losN in range(len(S_TMP_B) - matched_count):
                ob_ls = S_TMP_B.ID[S_TMP_B.ID.isin(TMP_B.ID) == False].iloc[0]
                used_mount_relative = False
                mpair = _mount_pair_for_fly(mount_pairs_prev, ob_ls)
                if mpair is not None:
                    partner = _mount_partner_id(mpair, ob_ls)
                    if partner is not None and (TMP_B["ID"] == partner).any():
                        obl_mr = _mount_relative_recovered_row(
                                ob_ls, partner, frame, S_TMP_B, TMP_B
                                )
                        if obl_mr is not None:
                            TMP_B = pd.concat([TMP_B, obl_mr])
                            qa["mount_relative_recovery"] += 1
                            used_mount_relative = True
                if not used_mount_relative:
                    Lost = Obj_los_test(frame, ob_ls, cap, prev_frame)
                    if Lost['Type'] == "CroLst":
                        qa["lost_recovery_crolst"] += 1
                        # update the frame from the object lost
                        Obl_TB = S_TMP_B[S_TMP_B.ID == ob_ls]
                        Obl_TB[0] = frame
                        # Keep last known location when object is lost.
                        Obl_TB.find = False
                        TMP_B = pd.concat([TMP_B, Obl_TB])
                    elif Lost['Type'] == "Overlap":
                        qa["lost_recovery_overlap"] += 1
                        Obl_TB = TB_cache[TB_cache[0] == Lost['frame']]
                        Obl_TB = Obl_TB[Obl_TB.ID == ob_ls]
                        Obl_TB[0] = frame
                        # Keep last known location when object is lost.
                        Obl_TB.find = False
                        TMP_B = pd.concat([TMP_B, Obl_TB])
                    else:
                        LOGGER.error("Unexpected lost-object type: frame=%s id=%s type=%s", frame, ob_ls, Lost.get("Type"))
                Over_adjust = Overlap_test(ob_ls, TMP_B)
                if Over_adjust != False:
                    qa["overlap_adjustments"] += 1
                    ob_ov = Over_adjust[1]
                    TMP_chage = TB_cache[TB_cache[0] == Over_adjust[0]]
                    TMP_chage = TMP_chage[TMP_chage.ID == Over_adjust[1]]
                    TMP_B.iloc[np.where(TMP_B.ID == ob_ov)[0], 4] = TMP_chage[4] * .9
                    TMP_B.iloc[np.where(TMP_B.ID == ob_ov)[0], 5] = TMP_chage[5] * .9
        progress_step(progress, step_task, description=f"[{frame}] {pipeline_steps[3]}")

        # remove false positive results
        TMP_B = TMP_B[TMP_B.ID.isna() == False]
        Dots_to = TMP_B.iloc[:, 2:4].to_numpy()
        # save the results
        TMP_B.to_csv(OUTPUT, header=False, index=False, mode='a')
        progress_step(progress, step_task, description=f"[{frame}] {pipeline_steps[4]}")

        # Update FLY_matrix
        TMP_Dict = {}
        for fly in TMP_B.ID:
            TMP_Dict.update({fly: {'body': TMP_B[TMP_B.ID == fly].to_numpy().tolist()[0][2:6]}})
        FLY_matrix.update({frame: TMP_Dict})
        if len(FLY_matrix) > 10:
            FLY_matrix.pop(list(FLY_matrix.keys())[0])

        # head bind
        TB_head = TMP[TMP[1] == 1].iloc[:, 1:6]
        head_bind.main(FLY_matrix, frame, TB_head)
        used_head_idx = set()
        for fly in FLY_matrix[frame].keys():
            fly_key = str(fly)
            match_idx = None
            try:
                match_idx = int(head_bind.MATCH_result[fly])
            except Exception:
                match_idx = None
            head_row = _safe_head_row(TB_head, match_idx) if match_idx is not None else None
            if head_row is not None and int(match_idx) not in used_head_idx:
                FLY_matrix[frame][fly].update({"head": head_row})
                used_head_idx.add(int(match_idx))
                head_fallback_streak[fly_key] = 0
                qa["head_bind_success"] += 1
                continue

            qa["head_fallback"] += 1
            head_fallback_streak[fly_key] = head_fallback_streak.get(fly_key, 0) + 1
            try:
                # Inherate the head from previous frame based on relative position
                last_body = FLY_matrix[prev_frame][fly]['body']
                last_head = FLY_matrix[prev_frame][fly]['head']
                new_body = FLY_matrix[frame][fly]['body']
                rel_pos = [last_head[0] - last_body[0], last_head[1] - last_body[1]]
                rel_pos_new = [rel_pos[0] + new_body[0], rel_pos[1] + new_body[1]]
                fallback_head = rel_pos_new + last_head[2:4]
                FLY_matrix[frame][fly].update({"head": fallback_head})

            except Exception:
                # Keep prior fallback value if anything unexpected occurs.
                pass
        progress_step(progress, step_task, description=f"[{frame}] {pipeline_steps[5]}")

        # write results
        dic_ID = int(list(FLY_matrix.keys())[-1])
        tmp = {dic_ID: FLY_matrix[dic_ID]}
        FLY_matrix_tmp_str = json.dumps(tmp) + ";"
        Trac_out.write(FLY_matrix_tmp_str)

        # update catch table
        TB_cache = pd.concat([TB_cache, TMP_B])
        TB_cache = TB_cache[TB_cache[0].isin(TB_cache[0].unique()[-10:])]

        S_TMP_B = TMP_B.copy()
        Dots_from = Dots_to.copy()
        prev_frame = frame
        progress_step(progress, step_task, description=f"[{frame}] {pipeline_steps[6]}")
        progress_step(progress, frame_task, description=f"Frames ({frame_idx}/{total_tracked_frames})")

        if frame_idx == 1 or frame_idx == total_tracked_frames or frame_idx % LOG_EVERY == 0:
            LOGGER.info(
                    "Frame %s/%s (%s): step1=%s step2=%s missing=%s overlap_adj=%s head_bind=%s fallback=%s reacquire=%s/%s",
                    frame_idx,
                    total_tracked_frames,
                    frame,
                    int(qa["match_pairs_step1"] - qa_snapshot["match_pairs_step1"]),
                    int(qa["match_pairs_step2"] - qa_snapshot["match_pairs_step2"]),
                    int(qa["missing_objects_total"] - qa_snapshot["missing_objects_total"]),
                    int(qa["overlap_adjustments"] - qa_snapshot["overlap_adjustments"]),
                    int(qa["head_bind_success"] - qa_snapshot["head_bind_success"]),
                    int(qa["head_fallback"] - qa_snapshot["head_fallback"]),
                    int(qa["head_reacquire_success"] - qa_snapshot["head_reacquire_success"]),
                    int(qa["head_reacquire_attempts"] - qa_snapshot["head_reacquire_attempts"]),
                    )

Trac_out.close()
cap.release()

qa["head_fallback_streak_max"] = int(max(head_fallback_streak.values()) if head_fallback_streak else 0)
qa["runtime_seconds"] = float(max(0.0, time.time() - run_started_ts))
qa["lost_recovery_total"] = int(qa["lost_recovery_crolst"] + qa["lost_recovery_overlap"])
qa["head_reacquire_fail"] = int(qa["head_reacquire_attempts"] - qa["head_reacquire_success"])
qa_path = QA_REPORT_PATH if QA_REPORT_PATH else f"{OUTPUT}.qa.json"
qa_path_obj = Path(qa_path)
qa_path_obj.parent.mkdir(parents=True, exist_ok=True)
qa_path_obj.write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
LOGGER.info("Saved QA report: %s", qa_path_obj)
LOGGER.info(
    "Tracking complete: frames_tracked=%s lost_recovery=%s head_fallback=%s runtime=%.2fs",
    qa["frames_tracked"],
    qa["lost_recovery_total"],
    qa["head_fallback"],
    qa["runtime_seconds"],
)



'''
head_bind.main(FLY_matrix, Num_frame, TB_head)
print("Head Match:", head_bind.MATCH_result)
for fly in FLY_matrix[Num_frame].keys():
    FLY_matrix[Num_frame][fly].update({"head":list(TB_head.iloc[int(head_bind.MATCH_result[fly]),1:])})                        #print(FLY_matrix)
dic_ID = list(FLY_matrix.keys())[-1]
tmp = {dic_ID:FLY_matrix[dic_ID]}
FLY_matrix_tmp_str = json.dumps(tmp) +";"
print(FLY_matrix_tmp_str)
# update the tar_tr_start
os.system("rm  csv/" + Video+"_"+str(tar_tr_start)+"_.json")
Trac_out = open("csv/" + Video+"_"+str(tar_tr_start)+"_.json", "a")
Trac_out.write(FLY_matrix_tmp_str)
'''


'''
ptLeftTop = (int(OB_TB[2] - OB_TB[4]/2 ), int(OB_TB[3]- OB_TB[5]/2))
ptRightBottom = (int(OB_TB[2] + OB_TB[4]/2), int(OB_TB[3] + OB_TB[5]/2))
point_color = (0, 0, 255) # BGR
thickness = 2
lineType = 8
cap.set(1,frame-1)
ret,img_f=cap.read()
img_f = img_f[int(OB_TB[3]- OB_TB[5]/2):int(OB_TB[3] + OB_TB[5]/2), int(OB_TB[2] - OB_TB[4]/2 ):int(OB_TB[2] + OB_TB[4]/2)]
img_f = cv2.cvtColor(img_f,cv2.COLOR_RGB2GRAY)
img_f = cv2.GaussianBlur(img_f, (5, 5), 100)
#img_f[img_f >= 50]=0
'''


'''    
# codes visualize the result and output as video

TBR = pd.read_csv(OUTPUT, header= None)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (1920,1080))

cap=cv2.VideoCapture(Video)
Num = Start -1
cap.set(1, Num+1)

while Num <= End:
    Num += 1 
    TMP = TBR[TBR[0]==Num]
    ret,img = cap.read()
    cv2.putText(img, str(Num) ,(100, 100), cv2.FONT_HERSHEY_COMPLEX, .5, (100, 200, 200), 2)

    for fly in range(len(TMP)):
        ptLeftTop = (int( 1920 * (TMP.iloc[fly, 2]- TMP.iloc[fly, 4]/2)), int( 1080 * (TMP.iloc[fly, 3]- TMP.iloc[fly, 5]/2)))
        ptRightBottom = (int( 1920 * (TMP.iloc[fly, 2]+ TMP.iloc[fly, 4]/2)), int( 1080 * (TMP.iloc[fly, 3] + TMP.iloc[fly, 5]/2)))
        point_color = (0, 0, 255) # BGR
        thickness = 1
        lineType = 8
        cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        cv2.putText(img, TMP.iloc[fly, 6] ,(int( 1920 * TMP.iloc[fly, 2]), int( 1080 * TMP.iloc[fly, 3])), cv2.FONT_HERSHEY_COMPLEX, .5, (100, 200, 200), 2)
    cv2.imshow("video", img)
    out.write(img)
    if cv2.waitKey(30)&0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
out.release()
cv2.destroyAllWindows()



TBR = pd.read_csv('Mix7.MP4.csv', header= None, sep = ' ')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Standard.avi',fourcc, 20.0, (1920,1080))

cap=cv2.VideoCapture(Video)
Num = Start -1
cap.set(1, Num+1)

while Num <= End:
    Num += 1 
    TMP = TBR[TBR[0]==Num]
    TMP = TMP[TMP[1]==0]
    ret,img = cap.read()
    cv2.putText(img, str(Num) ,(100, 100), cv2.FONT_HERSHEY_COMPLEX, .5, (100, 200, 200), 2)

    for fly in range(len(TMP)):
        ptLeftTop = (int( 1920 * (TMP.iloc[fly, 2]- TMP.iloc[fly, 4]/2)), int( 1080 * (TMP.iloc[fly, 3]- TMP.iloc[fly, 5]/2)))
        ptRightBottom = (int( 1920 * (TMP.iloc[fly, 2]+ TMP.iloc[fly, 4]/2)), int( 1080 * (TMP.iloc[fly, 3] + TMP.iloc[fly, 5]/2)))
        point_color = (0, 0, 255) # BGR
        thickness = 1
        lineType = 8
        cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
    cv2.imshow("video", img)
    out.write(img)
    if cv2.waitKey(30)&0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
out.release()
cv2.destroyAllWindows()




'''

'''
# code for show specific fly in a frame

frame = 661 
fly = 'fly_3'

TBR = pd.read_csv(OUTPUT, header= None)
TMP = TBR[TBR[0]==frame]
fly_TB = TMP[TMP[6] == fly]
fly_TB[2] *= 1920
fly_TB[4] *= 1920
fly_TB[3] *= 1080
fly_TB[5] *= 1080

cap=cv2.VideoCapture(Video)
cap.set(1,frame)
ret,img_f=cap.read()
img_f = img_f[int(fly_TB[3]- fly_TB[5]/2):int(fly_TB[3] + fly_TB[5]/2), int(fly_TB[2] - fly_TB[4]/2 ):int(fly_TB[2] + fly_TB[4]/2)]
img_f = cv2.cvtColor(img_f,cv2.COLOR_RGB2GRAY)
img_f = cv2.GaussianBlur(img_f, (5, 5), 100)
img_f[img_f >= 50]=0

nonzero_indices = np.nonzero(img_f)

# Create a DataFrame with the non-zero indices and their corresponding values
df = pd.DataFrame({
    'row': nonzero_indices[0],
    'col': nonzero_indices[1],
    'value': img_f[nonzero_indices]
})

# weight the points
df.value = df.value/df.value.max()
df['srow'] = df.row * df.value
df['scol'] = df.col * df.value



plt.imshow(img_f)
plt.show()
plt.scatter( df.col.median(), df.row.median(), c= 'r')
plt.scatter(df.col.mean(), df.row.mean(),  c= 'black')


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB))
ax1.plot(int(box_center(img_f, '')[0]), int(box_center(img_f, '')[1]), 'ro')
ax2.imshow(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))
ax2.plot(int(box_center(img_t, '')[0]), int(box_center(img_t, '')[1]), 'ro')
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(cv2.cvtColor(img_old, cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(img_now, cv2.COLOR_BGR2RGB))
plt.show()


cv2.destroyAllWindows()
cv2.imshow("video",img_f)
if cv2.waitKey(0)&0xFF==ord('q'):
    cv2.destroyAllWindows()
'''
