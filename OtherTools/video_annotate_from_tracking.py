#!/usr/bin/env python3
"""
Create an annotated video segment using detection/tracking results.

Reads `Video_list.csv` to get full information for a given video id
(first column of Video_list), overlays detections on the original
video, and saves an annotated video.

Inputs
------
- Video_list.csv (tab-separated, 5 columns):
  1) video_id   : path or name of the video file (as used in detection)
  2) petri_pixel
  3) petri_mm
  4) frame_start
  5) frame_end

CLI
---
python OtherTools/video_annotate_from_tracking.py -id <video_id> [-o OUTPUT] [--show]

-id / --video_id : string from the first column of Video_list.csv
-o / --output    : optional output path for annotated video
--show           : optionally display frames while processing (slow)
"""

import argparse
import json
import math
import os
import sys
from typing import Tuple, Optional, Dict, Any, List

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Color palette for fly IDs (hex → BGR), based on ColorBrewer Set3-like colors
FLY_PALETTE = [
    (0xE3, 0xCE, 0xA6),  # Light Blue:  #A6CEE3
    (0xB4, 0x78, 0x1F),  # Dark Blue:   #1F78B4
    (0x8A, 0xDF, 0xB2),  # Light Green: #B2DF8A
    (0x2C, 0xA0, 0x33),  # Dark Green:  #33A02C
    (0x99, 0x9A, 0xFB),  # Light Red:   #FB9A99
    (0x1C, 0x1A, 0xE3),  # Dark Red:    #E31A1C
    (0x6F, 0xBF, 0xFD),  # Light Orange:#FDBF6F
    (0x00, 0x7F, 0xFF),  # Dark Orange: #FF7F00
    (0xD6, 0xB2, 0xCA),  # Light Purple:#CAB2D6
    (0x9A, 0x3D, 0x6A),  # Dark Purple: #6A3D9A
    (0x99, 0xFF, 0xFF),  # Light Yellow:#FFFF99
    (0x28, 0x59, 0xB1),  # Dark Yellow/Brown: #B15928
]

# Global mapping from fly_id → color index, initialized lazily per video
FLY_ID_COLOR_MAP: Dict[str, int] = {}


def find_merged_cls0(
    det_cls0: pd.DataFrame,
    cover_thr: float = 0.9,
    min_covered: int = 2,
) -> set:
    """
    Return 0-based positions within det_cls0 that are 'merged' detections:
    one large body box that covers >= cover_thr of the area of at least
    min_covered other bodies in the same frame.

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


def init_fly_color_map(fly_dic: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """
    Initialize FLY_ID_COLOR_MAP so that colors follow fly indices strictly:
    fly_0 → first palette color, fly_1 → second, etc., cycling when needed.

    If names are not in 'fly_<n>' format, they are ordered lexicographically.
    """
    global FLY_ID_COLOR_MAP
    if FLY_ID_COLOR_MAP:
        return

    fly_ids = set()
    for frame_dict in fly_dic.values():
        fly_ids.update(frame_dict.keys())

    def sort_key(fid: str):
        if fid.startswith("fly_"):
            try:
                return int(fid.split("_", 1)[1])
            except ValueError:
                return fid
        return fid

    ordered = sorted(fly_ids, key=sort_key)
    for idx, fid in enumerate(ordered):
        FLY_ID_COLOR_MAP[fid] = idx % len(FLY_PALETTE)


def fly_color(fly_id: str, fly_dic: Dict[str, Dict[str, Dict[str, Any]]]) -> Tuple[int, int, int]:
    """
    Map a fly_id deterministically to a color in FLY_PALETTE, using a strict
    ordering from fly_0, fly_1, ... across this video.
    """
    if not FLY_ID_COLOR_MAP:
        init_fly_color_map(fly_dic)
    idx = FLY_ID_COLOR_MAP.get(fly_id)
    if idx is None:
        # Fallback: append new ids at the end, cycling palette
        idx = len(FLY_ID_COLOR_MAP) % len(FLY_PALETTE)
        FLY_ID_COLOR_MAP[fly_id] = idx
    return FLY_PALETTE[idx]


def read_video_list(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video_list.csv not found at: {path}")
    # Video_list is tab-separated, no header.
    # At minimum we expect 5 columns; if there are more, the LAST column is
    # treated as the absolute path to the video.
    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] < 5:
        raise ValueError(
            f"Video_list.csv at {path} must have at least 5 tab-separated columns "
            f"(video_id, petri_pixel, petri_mm, frame_start, frame_end, [abs_path]). "
            f"Got {df.shape[1]} column(s)."
        )
    ncol = df.shape[1]
    df = df.iloc[:, :ncol]
    # Name the first five columns explicitly; any middle columns (if present)
    # get generic names, and the last column is exposed as video_path.
    base_cols = ["video_id", "petri_pixel", "petri_mm", "frame_start", "frame_end"]
    if ncol > 5:
        extra = [f"col_{i}" for i in range(6, ncol + 1)]
        cols = base_cols + extra
    else:
        cols = base_cols
    df.columns = cols
    df["video_path"] = df.iloc[:, -1]
    return df


def find_video_row(df: pd.DataFrame, video_id: str) -> pd.Series:
    """
    Find the row whose first column (video_id) matches the given id.
    """
    match = df[df["video_id"] == video_id]
    if match.empty:
        raise ValueError(
            f"No row in Video_list.csv has video_id == '{video_id}'. "
            f"Available ids include: {df['video_id'].unique()[:10]!r}"
        )
    # If duplicated, just take the first
    return match.iloc[0]


def open_video(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    return cap


def get_video_meta(cap: cv2.VideoCapture) -> Tuple[int, int, int, float]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    return width, height, total_frames, fps


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
    # If multiple, pick the shortest name as the base detection file
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


def load_tracking_json(
    video_id: str,
    frame_start: int,
    frame_end: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load FLY_matrix tracking JSON for the given video and frame range.

    The file format follows the rest of the Post_data pipeline:
    - Preferred: semicolon-separated JSON objects, each like {"123": {...}}
    - Fallback : one big JSON object mapping frame -> fly_id -> {body, head}
    """
    csv_dir = os.path.join(PROJECT_ROOT, "csv")
    candidates = [
        f for f in os.listdir(csv_dir)
        if video_id in f and f.endswith(".json")
    ]
    if not candidates:
        # Tracking JSON is optional for this visualizer; just return empty.
        print(f"[info] No tracking JSON found in {csv_dir} containing '{video_id}'.")
        return {}

    json_name = sorted(candidates, key=len)[0]
    json_path = os.path.join(csv_dir, json_name)

    with open(json_path, "r") as f:
        raw = f.read().strip()

    fly_dic: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Try semicolon-separated stream first
    if ";" in raw:
        parts = [p for p in raw.split(";") if p.strip()]
        for part in parts:
            try:
                obj = json.loads(part)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not obj:
                continue
            frame_key = list(obj.keys())[0]
            try:
                frame_int = int(frame_key)
            except (TypeError, ValueError):
                continue
            if frame_start <= frame_int <= frame_end:
                fly_dic[frame_key] = obj[frame_key]
        if fly_dic:
            return fly_dic

    # Fallback: assume one big JSON object
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[warn] Could not parse tracking JSON at {json_path}; skipping fly IDs.")
        return {}

    if isinstance(obj, dict):
        for frame_key, val in obj.items():
            try:
                frame_int = int(frame_key)
            except (TypeError, ValueError):
                continue
            if frame_start <= frame_int <= frame_end:
                fly_dic[frame_key] = val

    return fly_dic


def draw_detections(
    frame_bgr: np.ndarray,
    det_frame: pd.DataFrame,
    width: int,
    height: int,
    show_behavior: bool = True,
) -> np.ndarray:
    """
    Overlay behavior-class detection boxes and labels on a frame.

    The CSV uses normalized (x_center, y_center, w, h) in [0,1].
    If show_behavior is False, the input frame is returned unchanged.
    """
    if not show_behavior:
        return frame_bgr

    img = frame_bgr.copy()
    for _, row in det_frame.iterrows():
        cls = int(row["class"])
        if cls in (0, 1):
            continue  # skip body/head boxes (cls 0 and 1)
        x_c = float(row["x"]) * width
        y_c = float(row["y"]) * height
        w = float(row["w"]) * width
        h = float(row["h"]) * height
        x1 = int(max(0, x_c - w / 2))
        y1 = int(max(0, y_c - h / 2))
        x2 = int(min(width - 1, x_c + w / 2))
        y2 = int(min(height - 1, y_c + h / 2))

        # Behavior-class specific colors (using provided palette)
        # ed6a5a, 26547c, ffd166
        if cls == 2:       # grooming
            color = (0x66, 0xD1, 0xFF)     # ffd166
            label = "grooming"
        elif cls == 3:     # chasing
            color = (0x7C, 0x54, 0x26)     # 26547c
            label = "chasing"
        elif cls == 4:     # flapping
            color = (0x6F, 0x47, 0xEF)     # ef476f
            label = "flapping"
        else:
            color = (0, 255, 0)       # default green
            label = f"cls {cls}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
    return img


def draw_head_body_arrows(
    frame_bgr: np.ndarray,
    fly_dic: Dict[str, Dict[str, Dict[str, Any]]],
    frame_idx: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Draw an arrow from each fly's body center into its head center, using the
    tracking JSON (so color can follow fly id).
    """
    key = str(frame_idx)
    if not fly_dic or key not in fly_dic:
        return frame_bgr

    img = frame_bgr.copy()
    flies = fly_dic[key]
    for fly_id, parts in flies.items():
        body = parts.get("body")
        head = parts.get("head")
        if not body or not head or len(body) < 2 or len(head) < 2:
            continue
        bx, by = float(body[0]) * width, float(body[1]) * height
        hx, hy = float(head[0]) * width, float(head[1]) * height
        pt1 = (int(bx), int(by))
        pt2 = (int(hx), int(hy))
        color = fly_color(fly_id, fly_dic)
        cv2.arrowedLine(img, pt1, pt2, color, 3, tipLength=0.35)
    return img


def draw_fly_ids(
    frame_bgr: np.ndarray,
    fly_dic: Dict[str, Dict[str, Dict[str, Any]]],
    frame_idx: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Overlay fly IDs using the tracking JSON (FLY_matrix).

    For each fly at this frame, we draw a small circle at the body center
    and label it with the fly's id string.
    """
    img = frame_bgr.copy()
    key = str(frame_idx)
    if key not in fly_dic:
        return img

    flies = fly_dic[key]
    for fly_id, parts in flies.items():
        body = parts.get("body")
        head = parts.get("head")
        if not body or len(body) < 2:
            continue

        # Body center (for circle)
        bx_n, by_n = float(body[0]), float(body[1])
        bx = int(bx_n * width)
        by = int(by_n * height)
        color = fly_color(fly_id, fly_dic)
        cv2.circle(img, (bx, by), 5, color, -1)

        # Head box (cls1) for placing the ID at the right-top corner
        if head and len(head) >= 4:
            hx_n, hy_n, hw_n, hh_n = map(float, head[:4])
            # (x_center, y_center, w, h) normalized
            hx_rt = (hx_n + hw_n / 2.0) * width
            hy_top = (hy_n - hh_n / 2.0) * height
            text_x = int(hx_rt + 3)
            text_y = int(max(0, hy_top - 3))
        else:
            # Fallback: place near body if head box missing
            text_x = bx + 5
            text_y = by - 5

        cv2.putText(
            img,
            str(fly_id),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return img


def highlight_detection_mismatch(
    frame_bgr: np.ndarray,
    fly_dic: Dict[str, Dict[str, Dict[str, Any]]],
    det_frame: pd.DataFrame,
    frame_idx: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Compare JSON tracking (FLY_matrix) with CSV cls0 body detections for this frame.

    JSON is the standard (tracked fly identities).
    - JSON body with no close CSV cls0 match → LOSS (red box, fly in tracking but not detected).
    - CSV cls0 body with no close JSON match → GAIN (blue box, extra detection not in tracking).

    Matching uses center-point distance (not IoU) since JSON and CSV share the same
    coordinate origin – centers that belong to the same fly will be very close or identical.

    Also draws a small count overlay (JSON:N | CSV:M) so mismatches are always visible.
    """
    key = str(frame_idx)

    img = frame_bgr.copy()

    # --- Build JSON body centers + rects (normalized → pixel) ---
    json_centers: list[tuple[float, float]] = []
    json_rects_px: list[tuple[int, int, int, int]] = []

    if key in fly_dic:
        for parts in fly_dic[key].values():
            body = parts.get("body")
            if body and len(body) >= 2:
                cx_n, cy_n = float(body[0]), float(body[1])
                cx, cy = cx_n * width, cy_n * height
                json_centers.append((cx, cy))
                if len(body) >= 4:
                    w_n, h_n = float(body[2]), float(body[3])
                    bw, bh = w_n * width, h_n * height
                    x1 = int(max(0, cx - bw / 2))
                    y1 = int(max(0, cy - bh / 2))
                    x2 = int(min(width - 1, cx + bw / 2))
                    y2 = int(min(height - 1, cy + bh / 2))
                    json_rects_px.append((x1, y1, x2, y2))
                else:
                    r = int(0.03 * min(width, height))
                    json_rects_px.append((int(cx) - r, int(cy) - r,
                                          int(cx) + r, int(cy) + r))

    # --- Build CSV cls0 body centers + rects (normalized → pixel) ---
    csv_centers: list[tuple[float, float]] = []
    csv_rects_px: list[tuple[int, int, int, int]] = []

    for _, row in det_frame[det_frame["class"] == 0].iterrows():
        cx_n, cy_n = float(row["x"]), float(row["y"])
        cx, cy = cx_n * width, cy_n * height
        csv_centers.append((cx, cy))
        bw = float(row["w"]) * width
        bh = float(row["h"]) * height
        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(width - 1, cx + bw / 2))
        y2 = int(min(height - 1, cy + bh / 2))
        csv_rects_px.append((x1, y1, x2, y2))

    # --- Filter out merged cls0 detections before matching/counting ---
    csv_cls0_df = det_frame[det_frame["class"] == 0].reset_index(drop=True)
    merged_csv_pos = find_merged_cls0(csv_cls0_df)

    # Rebuild csv lists excluding merged indices
    eff_csv_centers: list = []
    eff_csv_rects: list = []
    merged_csv_centers: list = []
    merged_csv_rects: list = []
    for pos, (ctr, rect) in enumerate(zip(csv_centers, csv_rects_px)):
        if pos in merged_csv_pos:
            merged_csv_centers.append(ctr)
            merged_csv_rects.append(rect)
        else:
            eff_csv_centers.append(ctr)
            eff_csv_rects.append(rect)

    n_json = len(json_centers)
    n_csv_raw = len(csv_centers)
    n_csv_eff = len(eff_csv_centers)
    n_merged = len(merged_csv_pos)

    # --- Count overlay (always visible) ---
    if n_json != n_csv_eff:
        count_color = (0, 0, 255)   # red when effective counts differ
    else:
        count_color = (0, 0, 0)     # black when equal
    count_text = (
        f"JSON:{n_json} | CSV:{n_csv_eff}"
        + (f" (+{n_merged} merged)" if n_merged else "")
    )
    (cw, ch), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(img, count_text,
                (max(5, width - cw - 10), 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, count_color, 2, cv2.LINE_AA)

    if not json_centers and not eff_csv_centers:
        return img

    json_pts = np.array(json_centers)   if json_centers   else np.zeros((0, 2))
    csv_pts  = np.array(eff_csv_centers) if eff_csv_centers else np.zeros((0, 2))
    csv_rects_px = eff_csv_rects  # use filtered list for GAIN boxes below

    def dist_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not len(a) or not len(b):
            return np.full((len(a), len(b)), np.inf)
        return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))

    dists = dist_matrix(json_pts, csv_pts)  # shape (n_json, n_csv)

    # -----------------------------------------------------------------
    # One-to-one greedy matching: each JSON fly claims its closest
    # unmatched CSV detection.  Any leftover CSV detection = GAIN
    # (catches duplicates even when they're very close to a tracked fly).
    # Any unmatched JSON fly = LOSS.
    # -----------------------------------------------------------------
    matched_csv = set()
    matched_json = set()

    if len(json_pts) and len(csv_pts):
        # Build sorted list of (dist, json_idx, csv_idx)
        pairs = sorted(
            [(dists[j, i], j, i) for j in range(len(json_pts)) for i in range(len(csv_pts))],
            key=lambda x: x[0],
        )
        for d, j, i in pairs:
            if j not in matched_json and i not in matched_csv:
                matched_json.add(j)
                matched_csv.add(i)

    # JSON flies with no CSV match → LOSS (red)
    for j, rect in enumerate(json_rects_px):
        if j not in matched_json:
            x1, y1, x2, y2 = rect
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, "LOSS", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    # CSV detections with no JSON match → GAIN (blue)
    for i, rect in enumerate(csv_rects_px):
        if i not in matched_csv:
            x1, y1, x2, y2 = rect
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(img, "GAIN", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Leap detection and arrow overlay
# ---------------------------------------------------------------------------

_LEAP_COLOR = (0, 100, 255)  # orange-red (BGR)


def build_leap_map(
    fly_dic: Dict[str, Any],
    leap_multiplier: float = 1.0,
    max_frame_gap: int = 3,
) -> Dict[int, Dict[str, tuple]]:
    """
    Pre-compute all leap frames from fly_dic.

    A leap is when a fly's body-center moves more than
    (leap_multiplier × mean_body_length) pixels between adjacent frames.

    Returns
    -------
    { frame_int: { fly_id: (prev_xy_norm, curr_xy_norm, dist_px) } }
      prev_xy_norm / curr_xy_norm : [x_norm, y_norm] in [0,1]
    """
    if not fly_dic:
        return {}

    # Mean body length: 2 × dist(head_center, body_center) from first frame
    sorted_keys = sorted(fly_dic.keys(), key=lambda k: int(k))
    mean_bl = 72.0  # fallback
    for fk in sorted_keys:
        lengths = []
        for d in fly_dic[fk].values():
            body, head = d.get("body"), d.get("head")
            if body and head and len(body) >= 2 and len(head) >= 2:
                dx = (float(head[0]) - float(body[0])) * 1920
                dy = (float(head[1]) - float(body[1])) * 1080
                lengths.append(math.sqrt(dx ** 2 + dy ** 2) * 2)
        if lengths:
            mean_bl = sum(lengths) / len(lengths)
            break

    threshold = leap_multiplier * mean_bl
    leap_map: Dict[int, Dict[str, tuple]] = {}

    for idx in range(1, len(sorted_keys)):
        fk_prev = sorted_keys[idx - 1]
        fk_curr = sorted_keys[idx]
        gap = int(fk_curr) - int(fk_prev)
        if gap > max_frame_gap:
            continue
        for fly_id, data_curr in fly_dic[fk_curr].items():
            if fly_id not in fly_dic[fk_prev]:
                continue
            body_curr = data_curr.get("body")
            body_prev = fly_dic[fk_prev][fly_id].get("body")
            if not body_curr or not body_prev or len(body_curr) < 2 or len(body_prev) < 2:
                continue
            dx = (float(body_curr[0]) - float(body_prev[0])) * 1920
            dy = (float(body_curr[1]) - float(body_prev[1])) * 1080
            dist_px = math.sqrt(dx ** 2 + dy ** 2)
            if dist_px > threshold:
                leap_map.setdefault(int(fk_curr), {})[fly_id] = (
                    [float(body_prev[0]), float(body_prev[1])],
                    [float(body_curr[0]), float(body_curr[1])],
                    round(dist_px, 1),
                )

    return leap_map


def draw_leap_arrows(
    frame_bgr: np.ndarray,
    leap_at_frame: Dict[str, tuple],
    width: int,
    height: int,
) -> np.ndarray:
    """
    For each leaping fly:
    - Draw a red warning border around the entire frame.
    - Draw an arrow from previous position (BEGIN) to current position (END).
    - Label BEGIN and END with fly id and distance.
    - Show a LEAP summary at the top-left.
    """
    img = frame_bgr.copy()

    # --- Full-frame warning border (same style as LOSS/GAIN) ---
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), _LEAP_COLOR, 4)

    # --- Top-left summary label ---
    fly_ids = list(leap_at_frame.keys())
    summary = "LEAP: " + ", ".join(fly_ids)
    cv2.putText(img, summary,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, _LEAP_COLOR, 2, cv2.LINE_AA)

    # --- Per-fly arrow and markers ---
    for fly_id, (prev_norm, curr_norm, dist_px) in leap_at_frame.items():
        x1 = int(float(prev_norm[0]) * width)
        y1 = int(float(prev_norm[1]) * height)
        x2 = int(float(curr_norm[0]) * width)
        y2 = int(float(curr_norm[1]) * height)

        # Arrow: BEGIN → END
        cv2.arrowedLine(img, (x1, y1), (x2, y2), _LEAP_COLOR, 3,
                        cv2.LINE_AA, tipLength=0.25)

        # BEGIN marker (solid circle + label)
        cv2.circle(img, (x1, y1), 8, _LEAP_COLOR, -1)
        cv2.putText(img,
                    f"BEGIN {fly_id}",
                    (x1 + 10, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, _LEAP_COLOR, 2, cv2.LINE_AA)

        # END marker (hollow circle + label with distance)
        cv2.circle(img, (x2, y2), 8, _LEAP_COLOR, 2)
        cv2.putText(img,
                    f"END {fly_id}  {dist_px:.0f}px",
                    (x2 + 10, max(15, y2 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, _LEAP_COLOR, 2, cv2.LINE_AA)

    return img


def build_output_path(video_id: str, frame_start: int, frame_end: int, output_arg: Optional[str]) -> str:
    if output_arg:
        # If user gave a full path, respect it; otherwise put under project root
        if os.path.isabs(output_arg):
            return output_arg
        return os.path.join(PROJECT_ROOT, output_arg)
    out_dir = os.path.join(PROJECT_ROOT, "Video_post")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"Preview_{video_id}_{frame_start}_{frame_end}.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create annotated video using tracking/detection results."
    )
    parser.add_argument(
        "-id",
        "--video_id",
        required=True,
        help="Video id from the first column of Video_list.csv.",
    )
    parser.add_argument(
        "-vlist",
        "--video_list",
        default=os.path.join(PROJECT_ROOT, "Video_list.csv"),
        help="Path to Video_list.csv (default: project root Video_list.csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output video path (default: Video_post/Preview_<id>_<fs>_<fe>.mp4).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display frames while processing (slower).",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=None,
        help="Override frame_start from Video_list.csv (inclusive).",
    )
    parser.add_argument(
        "--fe",
        type=int,
        default=None,
        help="Override frame_end from Video_list.csv (exclusive).",
    )
    parser.add_argument(
        "--mark-frame",
        type=int,
        default=None,
        help="Highlight a specific frame index inside the preview window.",
    )
    parser.add_argument(
        "--mark-label",
        type=str,
        default=None,
        help="Text label to show when mark-frame is reached (e.g. LOSS or GAIN).",
    )
    parser.add_argument(
        "--no-behavior",
        action="store_true",
        help="Do not draw behavior-class boxes/labels (only flies, arrows, and frame index).",
    )
    parser.add_argument(
        "--no-leap",
        action="store_true",
        help="Do not draw leap arrows.",
    )
    parser.add_argument(
        "--leap-mult",
        type=float,
        default=1.0,
        help="Leap threshold multiplier (body-lengths per frame, default: 1.0).",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Process only the first second of the frame range (for quick testing).",
    )

    args = parser.parse_args()

    # 1) Read Video_list and select row
    df_vlist = read_video_list(args.video_list)
    row = find_video_row(df_vlist, args.video_id)
    video_id = str(row["video_id"])
    frame_start = int(row["frame_start"])
    frame_end = int(row["frame_end"])

    # 2) Open raw video
    # Use absolute path from the LAST column of Video_list (video_path).
    video_path = str(row["video_path"])
    if not os.path.isabs(video_path):
        # If for some reason it's not absolute, still allow relative to project root
        candidate = os.path.join(PROJECT_ROOT, video_path)
        if os.path.exists(candidate):
            video_path = candidate
    cap = open_video(video_path)
    width, height, total_frames, fps = get_video_meta(cap)

    # Optional override of frame range
    if args.fs is not None:
        frame_start = int(args.fs)
    if args.fe is not None:
        frame_end = int(args.fe)

    if frame_start < 0 or frame_end <= frame_start:
        raise ValueError(
            f"Invalid frame range in Video_list for '{video_id}': "
            f"frame_start={frame_start}, frame_end={frame_end}"
        )
    if frame_start >= total_frames:
        raise ValueError(
            f"frame_start ({frame_start}) >= total_frames ({total_frames}) in video {video_path}"
        )

    if args.test:
        frames_one_sec = int(round(fps))
        frame_end = min(frame_end, frame_start + frames_one_sec)
        print(f"[test] Limiting to first second: frames {frame_start}–{frame_end} ({frames_one_sec} frames)")

    # 3) Load detection CSV
    det = load_detection_csv(video_id)
    # 3b) Load tracking JSON (optional; used for fly IDs and arrows)
    fly_dic = load_tracking_json(video_id, frame_start, frame_end)

    # 4) Prepare writer
    out_path = build_output_path(video_id, frame_start, frame_end, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # 5) Iterate frames, overlay detections for requested range
    current_frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    current_frame_idx = frame_start

    mark_frame = args.mark_frame
    mark_label = args.mark_label or "MARK"
    show_behavior = not args.no_behavior

    # Pre-compute leap map (once) over the loaded frame range
    leap_map: Dict[int, Dict[str, tuple]] = {}
    if fly_dic and not args.no_leap:
        leap_map = build_leap_map(fly_dic, leap_multiplier=args.leap_mult)
        if leap_map:
            print(f"[leap] {sum(len(v) for v in leap_map.values())} leap event(s) "
                  f"across {len(leap_map)} frame(s) will be marked.")

    try:
        while current_frame_idx < frame_end:
            ret, frame = cap.read()
            if not ret:
                break

            det_frame = det[det["Frame"] == current_frame_idx]
            annotated = draw_detections(frame, det_frame, width, height, show_behavior)
            if fly_dic:
                annotated = draw_head_body_arrows(
                    annotated, fly_dic, current_frame_idx, width, height
                )
                annotated = draw_fly_ids(
                    annotated, fly_dic, current_frame_idx, width, height
                )

            # Always highlight JSON/CSV head-box mismatch when JSON is available
            if fly_dic:
                annotated = highlight_detection_mismatch(
                    annotated,
                    fly_dic,
                    det_frame,
                    current_frame_idx,
                    width,
                    height,
                )

            # Draw leap arrows if any fly leaped on this frame
            if leap_map.get(current_frame_idx):
                annotated = draw_leap_arrows(
                    annotated, leap_map[current_frame_idx], width, height
                )

            # Frame index display (top-right); highlight if this is the marked frame
            base_text = f"Frame {current_frame_idx}"
            if mark_frame is not None and current_frame_idx == mark_frame:
                frame_text = f"{base_text} [{mark_label}]"
                frame_color = (0, 0, 255)  # red for gain/loss mark
                # optional border highlight
                cv2.rectangle(
                    annotated,
                    (0, 0),
                    (width - 1, height - 1),
                    frame_color,
                    3,
                )
            else:
                frame_text = base_text
                frame_color = (0, 0, 0)  # black by default

            (tw, th), _ = cv2.getTextSize(
                frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            text_x = max(5, width - tw - 10)
            text_y = 30
            cv2.putText(
                annotated,
                frame_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                frame_color,
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)

            if args.show:
                cv2.imshow("Annotated", annotated)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
                    break

            current_frame_idx += 1
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Annotated video saved to: {out_path}")


if __name__ == "__main__":
    main()

