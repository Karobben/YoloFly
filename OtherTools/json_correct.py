#!/usr/bin/env python3
"""
Manually correct a FLY_matrix tracking JSON file.

Two correction types are supported:

  swap           – Swap two fly IDs over a frame range.
                   Use this when the tracker confused two flies' identities
                   from a certain frame onwards.

  delete_inherit – Remove a fly ID from a frame range and replace it with the
                   data from the frame immediately before the range starts.
                   Use this when tracking loses a fly and assigns a wild
                   position for a few frames.

Correction table (CSV / TSV / XLSX, with a header row):

  action         | fly_id_1  | fly_id_2  | frame_start | frame_end
  ---------------|-----------|-----------|-------------|----------
  swap           | fly_0     | fly_3     | 500         |          <- from 500 to end
  swap           | fly_1     | fly_5     | 820         | 960      <- only 820-960
  delete_inherit | fly_2     |           | 830         | 850      <- replace 830-850

Column rules
  - action      : "swap" or "delete_inherit" (case-insensitive)
  - fly_id_1    : required for both actions
  - fly_id_2    : required for "swap", ignored for "delete_inherit"
  - frame_start : first frame to apply (inclusive)
  - frame_end   : last frame to apply (inclusive).
                  For "swap", if blank the swap is applied until the last frame.
                  For "delete_inherit", this field is required.

CLI
---
python OtherTools/json_correct.py \\
    -id <video_id> \\
    -c corrections.csv \\
    [-vlist Video_list.csv] [--json-path PATH] [--out PATH] [--dry-run]

-id / --video_id    : video id from Video_list.csv (used to find the JSON)
-c  / --corrections : path to the correction table (CSV / TSV / XLSX)
-vlist              : path to Video_list.csv (default: project root)
--json-path         : override the JSON file path directly
--out               : output JSON path (default: overwrite in place)
--dry-run           : print what would change but do not write
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FlyDic = Dict[str, Dict[str, Dict[str, Any]]]


# ---------------------------------------------------------------------------
# Helpers: loading
# ---------------------------------------------------------------------------

def load_json(path: str) -> FlyDic:
    """Load a semicolon-separated or single-object FLY_matrix JSON."""
    with open(path, "r") as f:
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
            if isinstance(obj, dict) and obj:
                key = list(obj.keys())[0]
                fly_dic[key] = obj[key]
    else:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            fly_dic = obj

    return fly_dic


def save_json(fly_dic: FlyDic, path: str) -> None:
    """Write fly_dic back as a semicolon-separated stream of JSON objects."""
    parts = []
    for frame_key in sorted(fly_dic.keys(), key=lambda k: int(k)):
        parts.append(json.dumps({frame_key: fly_dic[frame_key]}))
    with open(path, "w") as f:
        f.write(";".join(parts) + ";")
    print(f"[saved] {path}")


def find_json_path(video_id: str) -> str:
    """Locate the JSON file under csv/ that contains video_id."""
    csv_dir = os.path.join(PROJECT_ROOT, "csv")
    candidates = [
        f for f in os.listdir(csv_dir)
        if video_id in f and f.endswith(".json")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No JSON found in {csv_dir} containing '{video_id}'."
        )
    return os.path.join(csv_dir, sorted(candidates, key=len)[0])


def load_corrections(path: str) -> pd.DataFrame:
    """Load correction table from CSV / TSV / XLSX."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip().lower() for c in df.columns]
    required = {"action", "fly_id_1", "frame_start"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Correction table is missing columns: {missing}. "
            f"Expected at least: action, fly_id_1, fly_id_2, frame_start, frame_end."
        )
    return df


# ---------------------------------------------------------------------------
# Core correction functions
# ---------------------------------------------------------------------------

def apply_swap(
    fly_dic: FlyDic,
    fly_id_1: str,
    fly_id_2: str,
    frame_start: int,
    frame_end: Optional[int],
    dry_run: bool,
) -> int:
    """
    Swap fly_id_1 and fly_id_2 data over [frame_start, frame_end].
    Returns the number of frames modified.
    """
    all_frames = sorted(fly_dic.keys(), key=lambda k: int(k))
    changed = 0

    for fk in all_frames:
        fi = int(fk)
        if fi < frame_start:
            continue
        if frame_end is not None and fi > frame_end:
            break

        frame_data = fly_dic[fk]
        has_1 = fly_id_1 in frame_data
        has_2 = fly_id_2 in frame_data

        if not has_1 and not has_2:
            continue

        if dry_run:
            print(
                f"  [dry-run] swap frame {fk}: "
                f"{fly_id_1}={'present' if has_1 else 'absent'}  "
                f"{fly_id_2}={'present' if has_2 else 'absent'}"
            )
        else:
            d1 = deepcopy(frame_data.get(fly_id_1))
            d2 = deepcopy(frame_data.get(fly_id_2))

            if d1 is not None:
                frame_data[fly_id_2] = d1
            elif fly_id_2 in frame_data:
                del frame_data[fly_id_2]

            if d2 is not None:
                frame_data[fly_id_1] = d2
            elif fly_id_1 in frame_data:
                del frame_data[fly_id_1]

        changed += 1

    return changed


def apply_delete_inherit(
    fly_dic: FlyDic,
    fly_id: str,
    frame_start: int,
    frame_end: int,
    dry_run: bool,
) -> int:
    """
    In [frame_start, frame_end] replace fly_id's data with a copy of the
    data from the frame immediately before frame_start.
    Returns the number of frames modified.
    """
    all_frames = sorted(fly_dic.keys(), key=lambda k: int(k))

    # Find the last frame before frame_start that has fly_id
    inherit_data = None
    inherit_frame = None
    for fk in all_frames:
        fi = int(fk)
        if fi >= frame_start:
            break
        if fly_id in fly_dic[fk]:
            inherit_data = deepcopy(fly_dic[fk][fly_id])
            inherit_frame = fi

    if inherit_data is None:
        print(
            f"  [warn] delete_inherit: no data for {fly_id} before frame "
            f"{frame_start}. Skipping this rule."
        )
        return 0

    print(
        f"  Inheriting {fly_id} from frame {inherit_frame} "
        f"into frames {frame_start}–{frame_end}"
    )

    changed = 0
    for fk in all_frames:
        fi = int(fk)
        if fi < frame_start:
            continue
        if fi > frame_end:
            break

        if dry_run:
            current = fly_dic[fk].get(fly_id, "<absent>")
            print(f"  [dry-run] delete_inherit frame {fk}: {fly_id} -> inherited position")
        else:
            fly_dic[fk][fly_id] = deepcopy(inherit_data)

        changed += 1

    return changed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually correct a FLY_matrix tracking JSON."
    )
    parser.add_argument(
        "-id", "--video_id",
        default=None,
        help="Video id (used to locate JSON under csv/).",
    )
    parser.add_argument(
        "-c", "--corrections",
        required=True,
        help="Path to correction table (CSV / TSV / XLSX).",
    )
    parser.add_argument(
        "-vlist", "--video_list",
        default=os.path.join(PROJECT_ROOT, "Video_list.csv"),
        help="Path to Video_list.csv (only used if --json-path not given).",
    )
    parser.add_argument(
        "--json-path",
        default=None,
        help="Direct path to the JSON file to correct (overrides -id lookup).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: overwrite in place).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing.",
    )

    args = parser.parse_args()

    # --- Resolve JSON path ---
    if args.json_path:
        json_path = args.json_path
    elif args.video_id:
        json_path = find_json_path(args.video_id)
    else:
        parser.error("Provide either -id <video_id> or --json-path <path>.")

    if not os.path.exists(json_path):
        print(f"[error] JSON file not found: {json_path}")
        sys.exit(1)

    print(f"JSON file : {json_path}")

    # --- Load JSON ---
    fly_dic = load_json(json_path)
    frames_total = len(fly_dic)
    all_frame_ints = sorted(int(k) for k in fly_dic.keys())
    last_frame = all_frame_ints[-1] if all_frame_ints else 0
    print(f"Frames loaded: {frames_total}  (range {all_frame_ints[0]}–{last_frame})")

    # --- Load correction table ---
    corrections = load_corrections(args.corrections)
    print(f"Corrections loaded: {len(corrections)} row(s)")

    total_changed = 0

    for idx, row in corrections.iterrows():
        action = str(row.get("action", "")).strip().lower()
        fly_id_1 = str(row.get("fly_id_1", "")).strip()
        fly_id_2 = str(row.get("fly_id_2", "")).strip() if pd.notna(row.get("fly_id_2")) else ""

        try:
            fs = int(row["frame_start"])
        except (ValueError, KeyError):
            print(f"[warn] Row {idx}: invalid frame_start, skipping.")
            continue

        fe_raw = row.get("frame_end")
        fe: Optional[int] = None
        if pd.notna(fe_raw) and str(fe_raw).strip() not in ("", "nan"):
            try:
                fe = int(fe_raw)
            except ValueError:
                pass

        print(f"\nRow {idx}: action={action}  fly_id_1={fly_id_1}  "
              f"fly_id_2={fly_id_2}  frames={fs}–{fe if fe is not None else 'end'}")

        if action == "swap":
            if not fly_id_2:
                print(f"  [warn] swap requires fly_id_2. Skipping row {idx}.")
                continue
            fe_used = fe if fe is not None else last_frame
            n = apply_swap(fly_dic, fly_id_1, fly_id_2, fs, fe_used, args.dry_run)
            print(f"  Swapped {fly_id_1} ↔ {fly_id_2} in {n} frame(s).")
            total_changed += n

        elif action == "delete_inherit":
            if fe is None:
                print(f"  [warn] delete_inherit requires frame_end. Skipping row {idx}.")
                continue
            n = apply_delete_inherit(fly_dic, fly_id_1, fs, fe, args.dry_run)
            print(f"  delete_inherit applied to {fly_id_1} in {n} frame(s).")
            total_changed += n

        else:
            print(f"  [warn] Unknown action '{action}'. Skipping row {idx}.")

    print(f"\nTotal frames affected: {total_changed}")

    if args.dry_run:
        print("[dry-run] No file written.")
        return

    out_path = args.out or json_path
    save_json(fly_dic, out_path)


if __name__ == "__main__":
    main()
