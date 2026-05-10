#!/usr/bin/env python3

import numpy as np
import pandas as pd
import cv2, json, warnings, os
import math
import subprocess
import sys
import tempfile
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
    from rich.progress import track
except ImportError:
    def track(sequence, *args, **kwargs):
        return sequence


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','-I','--input')     #输入文件
parser.add_argument('-o','-U','--output')     #输入文件
parser.add_argument('-v','-V','--video')     #输入文件
parser.add_argument('-n','-N','--num-fly', type=int, default=12, help='expected number of flies to track')
parser.add_argument('--workers', type=int, default=1, help='number of parallel tracking windows')
parser.add_argument('--window-overlap', type=int, default=200, help='overlap size in processed frames')
parser.add_argument('--split-x', default='', help='comma-separated x split boundaries in pixels, e.g. 960 or 640,1280')
parser.add_argument('--split-y', default='', help='comma-separated y split boundaries in pixels, e.g. 540 or 360,720')
parser.add_argument('--frame-width', type=int, default=1920, help='video width in pixels for split boundaries')
parser.add_argument('--frame-height', type=int, default=1080, help='video height in pixels for split boundaries')
parser.add_argument('--internal-window', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--internal-region', action='store_true', help=argparse.SUPPRESS)

##获取参数
args = parser.parse_args()
INPUT = args.input
OUTPUT = args.output
Video = args.video
NUM_FLY = args.num_fly
WORKERS = max(1, args.workers)
WINDOW_OVERLAP = max(0, args.window_overlap)
SPLIT_X = args.split_x
SPLIT_Y = args.split_y
FRAME_WIDTH = args.frame_width
FRAME_HEIGHT = args.frame_height


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
        return pd.DataFrame(columns=[0, 1])
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
            pairs.append((i1, i2))
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
    print(f"Region tracking: {len(regions)} regions from split-x={split_x or 'none'} split-y={split_y or 'none'}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    merged_rows = []
    merged_json_by_frame = {}
    id_offset = 0

    with tempfile.TemporaryDirectory(prefix="post_track_regions_") as tmpdir:
        tmpdir = Path(tmpdir)
        for region in regions:
            region_tb = filter_region_table(tb, region, width, height)
            if region_tb.empty:
                print(f"Region {region['region_id']} empty, skipped.")
                continue
            region_input = tmpdir / f"region_{region['region_id']:03d}.csv"
            region_output = tmpdir / f"region_{region['region_id']:03d}_tracked.csv"
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

    if merged_rows:
        merged = pd.concat(merged_rows)
        merged.sort_values([merged.columns[0], merged.columns[-2]]).to_csv(output, header=False, index=False)
    else:
        output.write_text("", encoding="utf-8")
    json_output = Path(str(output) + "_" + str(int(frames[0])) + "_.json")
    json_parts = [json.dumps({fr: merged_json_by_frame[fr]}) + ";" for fr in sorted(merged_json_by_frame)]
    json_output.write_text("".join(json_parts), encoding="utf-8")
    print(f"Saved split-region CSV: {output}")
    print(f"Saved split-region JSON: {json_output}")
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
            print(
                f"Reducing workers from {workers} to {max_workers_by_overlap}: "
                f"{len(frames)} processed frames with overlap {overlap} would create mostly-overlap windows."
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
    print(f"Window tracking: {len(frames)} processed frames, {len(windows)} workers, {overlap} processed-frame overlap")
    script_path = Path(__file__).resolve()

    with tempfile.TemporaryDirectory(prefix="post_track_windows_") as tmpdir:
        tmpdir = Path(tmpdir)
        jobs = []
        for win in windows:
            win_input = tmpdir / f"window_{win['worker_id']:03d}.csv"
            win_output = tmpdir / f"window_{win['worker_id']:03d}_tracked.csv"
            tb[tb[0].isin(win["actual_frames"])].to_csv(win_input, sep=" ", header=False, index=False)
            jobs.append((win, win_input, win_output))

        results = {}
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            future_map = {
                executor.submit(run_window_worker, script_path, inp, out, video, num_fly): win
                for win, inp, out in jobs
            }
            for future in as_completed(future_map):
                win = future_map[future]
                results[win["worker_id"]] = future.result()

        merged_rows = []
        merged_json_parts = []
        prev_full_json = None
        for win in windows:
            result = results[win["worker_id"]]
            core_set = set(win["core_frames"])
            tracked = pd.read_csv(result["output"], header=None)
            if tracked.empty:
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

        if merged_rows:
            pd.concat(merged_rows).sort_values(0).to_csv(output, header=False, index=False)
        else:
            output.write_text("", encoding="utf-8")
        json_output = Path(str(output) + "_" + str(int(frames[0])) + "_.json")
        json_output.write_text("".join(merged_json_parts), encoding="utf-8")
        print(f"Saved stitched CSV: {output}")
        print(f"Saved stitched JSON: {json_output}")

def box_center(img_f, Type = 'R'):
    img_f[img_f>50] = 0
    nonzero_indices = np.nonzero(img_f)
    # Create a DataFrame with the non-zero indices and their corresponding values
    df = pd.DataFrame({
        'col': nonzero_indices[0],
        'row': nonzero_indices[1]
    })
    if Type == "R":
        return  ((df.row.mean()-(len(img_f[0])/2))/1920, (df.col.mean()-(len(img_f)/2))/1080 )
    else:
        return (df.row.mean(), df.col.mean())

def crop_box(img, ob_tb):
    if img is None or len(ob_tb) == 0:
        return None
    x1 = max(0, int(ob_tb[2] - ob_tb[4] / 2))
    x2 = min(img.shape[1], int(ob_tb[2] + ob_tb[4] / 2))
    y1 = max(0, int(ob_tb[3] - ob_tb[5] / 2))
    y2 = min(img.shape[0], int(ob_tb[3] + ob_tb[5] / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

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
    rct_los = creat_polygon(TMP_B[TMP_B.ID==ob_ls].to_numpy()[0][2:6])
    Inter_dict1 = {}
    Inter_dict2 = {}
    for line in range(len(TMP_B)):
        if TMP_B.ID.iloc[line] != ob_ls:
            rct_tag = creat_polygon(TMP_B.iloc[line,2:6].to_numpy())
            Inter_dict1.update({ TMP_B.ID.iloc[line] : rct_los.intersection(rct_tag).area/ rct_los.area})
            Inter_dict2.update({ TMP_B.ID.iloc[line] : rct_los.intersection(rct_tag).area/ rct_tag.area})
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
            OB_TB = S_TMP_B[S_TMP_B.ID == ob_ov]
            OB_TB[2] *= 1920
            OB_TB[4] *= 1920
            OB_TB[3] *= 1080
            OB_TB[5] *= 1080
            img_t = crop_box(img_t, OB_TB)
            if img_t is None:
                return False
            img_t = cv2.cvtColor(img_t,cv2.COLOR_RGB2GRAY)
            img_t = cv2.GaussianBlur(img_t, (5, 5), 10)
            return (bst_frame.to_list()[0], ob_ov, box_center(img_t))
    return False

def Obj_los_test(frame, ob_ls, cap, prev_frame=None):
    Result = None
    OB_TB = S_TMP_B[S_TMP_B.ID == ob_ls]
    #OB_TB = S_TMP_B[S_TMP_B.ID == 'fly_1']
    OB_TB[2] *= 1920
    OB_TB[4] *= 1920
    OB_TB[3] *= 1080
    OB_TB[5] *= 1080

    cap.set(1,frame)
    ret,img_t=cap.read()
    if not ret or img_t is None:
        print("Cannot read frame for lost-object recovery:", frame)
        return {'Type' : "CroLst", "drift" :(0,0)}
    img_t = crop_box(img_t, OB_TB)
    if img_t is None:
        print("Invalid crop for lost-object recovery:", frame, ob_ls)
        return {'Type' : "CroLst", "drift" :(0,0)}
    img_t = cv2.cvtColor(img_t,cv2.COLOR_RGB2GRAY)
    img_t = cv2.GaussianBlur(img_t, (5, 5), 10)

    # ratio of fly-body to the blank background. A normal single fly ~= 14, >20 means over corroded.
    CoverR = len(img_t.ravel()[img_t.ravel()<50])/len(img_t.ravel())
    print("Mask ratio:", CoverR )
    if CoverR>=.19:
        print("Over Corroded caused object lost, test the overlap")
        return {'Type' : "CroLst", "drift" : box_center(img_t)}
    if CoverR==0:
        print("Object lost")
        return {'Type' : "CroLst", "drift" : (0,0)}

    cap.set(1, prev_frame if prev_frame is not None else frame-1)
    ret,img_f=cap.read()
    if not ret or img_f is None:
        print("Cannot read previous frame for lost-object recovery:", prev_frame if prev_frame is not None else frame-1)
        return {'Type' : "CroLst", "drift" :(0,0)}
    img_f = crop_box(img_f, OB_TB)
    if img_f is None:
        return {'Type' : "CroLst", "drift" :(0,0)}
    img_f = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)
    img_f = cv2.GaussianBlur(img_f, (5, 5), 10)
    # similarity clean vs single fly: 68.8%
    try:
        ssim_V = ssim(img_f, img_t)
        print(ssim_V )
        if ssim_V>=.85:
            print("Single fly lost")
            
            # ove lap check
            rct_los = creat_polygon(S_TMP_B[S_TMP_B.ID==ob_ls].to_numpy()[0][2:6])
            Inter_dict = {}
            for line in range(len(TMP_B)):
                Inter_dict.update({ TMP_B.ID.iloc[line] : rct_los.intersection(creat_polygon(TMP_B.iloc[line,2:6].to_numpy())).area/ rct_los.area})
            Over_p = list(np.where(np.array(list(Inter_dict.values())) > .2)[0])
            if len(Over_p) > 0:
                print("Overlap caused in frame:", frame)
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
                    OB_TB = Loc[Loc.ID== ob_ls]
                    OB_TB[2] *= 1920
                    OB_TB[4] *= 1920
                    OB_TB[3] *= 1080
                    OB_TB[5] *= 1080
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
                print(' No overlap')
                return {'Type' : "CroLst", "drift" : box_center(img_t)}
    
    except:
        print('Small box size')
    print('Less similarities, fastmoving')
    return {'Type' : "CroLst", "drift" :(0,0)}

## Functions down

# argumetns


CSV_f = INPUT#"Mix7.MP4.csv"
#Video = "/home/ken/Videos/Mix7.MP4"
Num = NUM_FLY
#OUTPUT = 'test.csv'

Box_size_check_d = 1.6
Box_size_check = 1.3
Overlap_thres  = .45

TB = load_detection_table(INPUT)
frames = sorted(TB[0].unique())
if len(frames) == 0:
    raise ValueError("No detection frames found in input.")
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

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

Start = int(frames[0])
# Define the Start 
S_TMP = TB[TB[0]==Start]
S_TMP_B = S_TMP[S_TMP[1]==0]
if len(S_TMP_B) < Num:
    raise ValueError(f"Frame {Start} has only {len(S_TMP_B)} body detections, fewer than --num-fly {Num}.")
S_TMP_B = select_top_confidence_bodies(S_TMP_B, Num)
S_TMP_B['ID'] = ['fly_' +str(i) for i in range(len(S_TMP_B))]
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
TB_head = S_TMP[S_TMP[1]==1].iloc[:,1:6]
head_bind.main(FLY_matrix, Start, TB_head)
#print("Head Match:", head_bind.MATCH_result)
for fly in FLY_matrix[Start].keys():
    try:
        FLY_matrix[Start][fly].update({"head":list(TB_head.iloc[int(head_bind.MATCH_result[fly]),1:])})
    except:
        FLY_matrix[Start][fly].update({"head":FLY_matrix[Start][fly]['body']})


# write results
dic_ID = list(FLY_matrix.keys())[-1]
tmp = {dic_ID:FLY_matrix[dic_ID]}
FLY_matrix_tmp_str = json.dumps(tmp) +";"
Path(OUTPUT+"_"+str(Start)+"_.json").unlink(missing_ok=True)
Trac_out = open(OUTPUT+"_"+str(Start)+"_.json", "a")
Trac_out.write(FLY_matrix_tmp_str)


prev_frame = Start
for frame in track(frames[1:], description="Tracking frames"):
    frame = int(frame)
    # Define the Start 
    TMP = TB[TB[0]==frame]
    TMP_B = select_top_confidence_bodies(TMP[TMP[1]==0], Num)
    TMP_B['ID'] = None
    TMP_B['find'] = True
    # Check the over-sized box and update it if it caused by two/more flies
    TMP_B['Areas'] = (TMP_B[4]*TMP_B[5]*1920*1080).to_numpy()

    Clear_lst = []
    for line in np.where( TMP_B.Areas/ TMP_B.Areas.mean() > Box_size_check_d)[0]:
        TMP_BL = TMP_B.iloc[line, :]
        # Check the overlap
        rct_los = creat_polygon(TMP_BL.to_numpy()[2:6])
        Inter_dict1 = [ rct_los.intersection(creat_polygon(TMP_B.iloc[line,2:6].to_numpy())).area / rct_los.area  for line in range(len(TMP_B)) ] 
        Inter_dict2 = [  rct_los.intersection(creat_polygon(TMP_B.iloc[line,2:6].to_numpy())).area / creat_polygon(TMP_B.iloc[line,2:6].to_numpy()).area for line in range(len(TMP_B))]
        Over_p = list(np.where(np.array(Inter_dict1) > Overlap_thres)[0]) + list(np.where(np.array(Inter_dict2) > Overlap_thres)[0])
        while line in Over_p:
            Over_p.remove(line)
        if len(Over_p) > 0:
            Clear_lst += [line]
    TMP_B = TMP_B.drop(TMP_B.index[Clear_lst])


    # remove the Areas column
    TMP_B = TMP_B.drop('Areas', axis = 1)
    Dots_to = TMP_B.iloc[:,2:4].to_numpy()

    # two steps sort
    ## Step 1
    Dots = Dots_Sort(Dots_from[S_TMP_B.find], Dots_to)
    # inherit IDs from previous based on sorting distance 
    for i in range(len(Dots)):
        TMP_B.ID.iloc[Dots[1][i]] = S_TMP_B.ID[S_TMP_B.find].iloc[Dots[0][i]]
    ## Step 2
    Dots = Dots_Sort(Dots_from[~S_TMP_B.find],Dots_to[TMP_B.ID.isna()])
    mask = TMP_B[TMP_B.ID.isna()]
    for i in range(len(Dots)):
        TMP_B.ID.iloc[ TMP_B.index == mask.iloc[Dots[1][i]].name] = S_TMP_B.ID[~S_TMP_B.find].iloc[Dots[0][i]]

    # Adjust the size of boxs
    TMP_B['Areas'] = (TMP_B[4]*TMP_B[5]*1920*1080).to_numpy()
    for ob_ov in TMP_B.ID[~TMP_B.ID.isna()]:
        if TMP_B.Areas[TMP_B.ID==ob_ov].iloc[0]/ TMP_B.Areas.mean() > Box_size_check_d:
            Over_adjust = Overlap_test(ob_ov, TMP_B[~TMP_B.ID.isna()])
            if Over_adjust != False:
                ob_ov = Over_adjust[1]
                #print("Adjust the size of ", frame, ob_ov)
                TMP_chage = TB_cache[ TB_cache[0]== Over_adjust[0]]
                TMP_chage = TMP_chage[ TMP_chage.ID == Over_adjust[1]]
                TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],4] = TMP_chage[4] * .9
                TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],5] = TMP_chage[5] * .9
    # remove the Areas column
    TMP_B = TMP_B.drop('Areas', axis = 1)
    Dots_to = TMP_B.iloc[:,2:4].to_numpy()


    # Missing Check
    matched_count = TMP_B.ID.notna().sum()
    if matched_count < len(S_TMP_B):
        #print("object lost, Number:", len(S_TMP_B) - len(TMP_B[TMP_B.ID != None]))
        for losN in range(len(S_TMP_B) - matched_count):
            ob_ls = S_TMP_B.ID[S_TMP_B.ID.isin(TMP_B.ID)==False].iloc[0]
            #print("lost object:", ob_ls, "in frame:", frame)
            Lost = Obj_los_test(frame, ob_ls, cap, prev_frame)
            if Lost['Type'] == "CroLst":
                # update the frame from the object lost
                Obl_TB = S_TMP_B[S_TMP_B.ID==ob_ls]
                Obl_TB[0] = frame
                Obl_TB[2] += Lost['drift'][0]
                Obl_TB[3] += Lost['drift'][1]
                Obl_TB.find = False
                TMP_B = pd.concat([TMP_B, Obl_TB])
            elif Lost['Type'] == "Overlap":
                Obl_TB = TB_cache[TB_cache[0]== Lost['frame']]
                Obl_TB = Obl_TB[Obl_TB.ID==ob_ls]
                Obl_TB[0] = frame
                Obl_TB[2] += Lost['drift'][0]
                Obl_TB[3] += Lost['drift'][1]
                Obl_TB.find = False
                TMP_B = pd.concat([TMP_B, Obl_TB])
            else:
                print("\n\nERROR!!!:", frame, ob_ls, "\n\n")
            Over_adjust = Overlap_test(ob_ls, TMP_B)
            if Over_adjust != False:
                ob_ov = Over_adjust[1]
                #print("Adjust the size of ", frame, ob_ov)
                #print(Over_adjust)
                TMP_chage = TB_cache[ TB_cache[0]== Over_adjust[0]]
                TMP_chage = TMP_chage[ TMP_chage.ID == Over_adjust[1]]
                #TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],2] += Over_adjust[2][0]
                #TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],3] += Over_adjust[2][1]
                TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],4] = TMP_chage[4] * .9
                TMP_B.iloc[np.where(TMP_B.ID==ob_ov)[0],5] = TMP_chage[5] * .9

    # remove false positive results
    TMP_B = TMP_B[TMP_B.ID.isna()==False]
    Dots_to = TMP_B.iloc[:,2:4].to_numpy()
    # give the ID to new frame
    # save the results
    TMP_B.to_csv(OUTPUT, header=False, index=False, mode='a')

    # Update FLY_matrix
    TMP_Dict = {}
    for fly in TMP_B.ID:
        TMP_Dict.update({fly : {'body': TMP_B[TMP_B.ID==fly].to_numpy().tolist()[0][2:6]}})
    FLY_matrix.update({frame: TMP_Dict})
    if len(FLY_matrix) >10:
        FLY_matrix.pop(list(FLY_matrix.keys())[0])
    #print("FLY_matrix:", len(FLY_matrix))
   
    # head bind
    TB_head = TMP[TMP[1]==1].iloc[:,1:6]
    head_bind.main(FLY_matrix, frame, TB_head)
    #print("Head Match:", head_bind.MATCH_result)
    for fly in FLY_matrix[frame].keys():
        try:
            FLY_matrix[frame][fly].update({"head":list(TB_head.iloc[int(head_bind.MATCH_result[fly]),1:])})                        #print(FLY_matrix)
        except:
            # Inherate the head from previous frame based on relative position
            last_body = FLY_matrix[prev_frame][fly]['body']
            last_head = FLY_matrix[prev_frame][fly]['head']
            new_body  = FLY_matrix[frame][fly]['body']
            rel_pos = [last_head[0] - last_body[0], last_head[1] - last_body[1]]
            rel_pos_new = [rel_pos[0]+ new_body[0], rel_pos[1]+ new_body[1]]
            FLY_matrix[frame][fly].update({"head": rel_pos_new + last_head[2:4]})

    # write results
    dic_ID = int(list(FLY_matrix.keys())[-1])
    tmp = {dic_ID:FLY_matrix[dic_ID]}
    FLY_matrix_tmp_str = json.dumps(tmp) +";"
    Trac_out = open(OUTPUT+"_"+str(Start)+"_.json", "a")
    Trac_out.write(FLY_matrix_tmp_str)

    # update catch table
    TB_cache = pd.concat([TB_cache, TMP_B])
    TB_cache = TB_cache[TB_cache[0].isin(TB_cache[0].unique()[-10:])]

    #S_TMP_B = None
    S_TMP_B = TMP_B
    Dots_from = Dots_to
    prev_frame = frame

Trac_out.close()



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
