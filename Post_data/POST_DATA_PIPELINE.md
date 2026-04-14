# Post_data pipeline: input–output flow

This directory contains **post-processing scripts** for fly-tracking data: they turn detection + tracking outputs (CSV + JSON) into **per-fly behavior metrics**, **chase/interaction events**, and **corrected behavior-class tables**, then **plots** and **statistics**.

---

## 1. Overview: where data comes from and goes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BEFORE Post_data (run detection from project root)                          │
│  e.g. python detect_2.py --source video.mp4 --bh-count --tar-track          │
│       --head-bind --num-fly 13 ...                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  csv/                                                                       │
│    • {Video}.csv           ← raw detections (frame, class, x, y, w, h)     │
│    • {Video}_{frame}_.json ← FLY_matrix (frame → fly_id → body/head)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │  + Video_list.csv (video_id, plate, frame range)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Post_data pipeline (run.py or step-by-step)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌─────────────────┐    ┌─────────────────────────┐    ┌─────────────────────┐
│ 1_single_fly_   │    │ 2_Chas_behavior_arg.py   │    │ 3_single_and_        │
│ run_arg.py      │    │ (chase / interaction     │    │ Chascls_arg.py       │
│ (per-fly        │───▶│  events)                 │───▶│ (merge + behavior    │
│  metrics)       │    │                          │    │  class)              │
└────────┬────────┘    └────────────┬────────────┘    └──────────┬────────────┘
         │                          │                            │
         ▼                          ▼                            ▼
   Video_post/                Video_post/                  Video_post/
   {id}_{fs}_{fe}.csv        Interection_{id}_             Correct_{id}_{fs}_{fe}.csv
                             {fs}_{fe}.csv
                                                                    │
                                                                    ▼
                                                         ┌─────────────────────┐
                                                         │ Plot_1.py           │
                                                         │ (plots + R stats)  │
                                                         └─────────────────────┘
```

---

## 2. Prerequisites (inputs that must already exist)

These are produced by the **detection scripts** (e.g. `detect_2.py`, `detect_220101.py`) when run with `--bh-count`, `--tar-track`, `--head-bind`, etc. Paths are relative to the **project root** unless you run from another working directory.

| Location | Content |
|----------|--------|
| **`csv/{Video}.csv`** | Raw detections: space-separated; columns include **frame**, **class** (0=body, 1=head, 2+=other), **x, y, width, height** (normalized). One row per detection per frame. |
| **`csv/{Video}_{frame_start}_.json`** | Tracking output: concatenation of `"{frame_id: { fly_id: {'body': [...], 'head': [...]}, ... }};"` chunks. Used as FLY_matrix (frame → fly → body/head). |

Example: after running detection on `adf6254_Movie_S3.mp4` you might have:

- `csv/adf6254_Movie_S3.mp4.csv`
- `csv/adf6254_Movie_S3.mp4_1_.json`

---

## 3. Video_list.csv (driver for the pipeline)

**Path:** Must be in the **current working directory** when you run `run.py` (e.g. project root or `Post_data/`).

**Format:** One line per video, **tab-separated** (`\t`). No header—the script reads raw lines and splits on tabs. Exactly **5 columns**:

| Column | Name          | Meaning |
|--------|---------------|--------|
| 1      | **video_id**  | Exact video name as in `csv/` (e.g. `adf6254_Movie_S3.mp4`). Used to find `csv/{video_id}.csv` and `csv/{video_id}_*_.json`. |
| 2      | **petri_pixel** | Plate diameter (or width) in **pixels** (e.g. 1920). |
| 3      | **petri_mm**  | Same dimension in **mm** (e.g. 50). Defines mm-per-pixel scale. |
| 4      | **frame_start** | First frame to analyze (inclusive). |
| 5      | **frame_end** | Last frame to analyze (inclusive). |

**Example file** (tabs between columns):

```
adf6254_Movie_S3.mp4	1920	50	500	5000
another_video.mp4	1920	50	1	3000
```

**Rules:**

- Use **tabs** between columns, not spaces or commas.
- One line per video; add more lines for more videos.
- `video_id` must match the detection outputs in `csv/` (e.g. `csv/adf6254_Movie_S3.mp4.csv`).
- Use **QC.py** (see below) to choose a good frame range before filling `Video_list.csv`.

---

## 4. Script-by-script input → output

### 4.1 QC.py (quality control and frame-range selection)

**Purpose:** Plot each fly’s path and “energy” (motion over time), fit a GAM, find peaks. Use outputs to decide **frame_start** and **frame_end** for `Video_list.csv`.

| Input | Output |
|-------|--------|
| `csv/{Video}.csv` | **`QC/{Video}_{fly_id}_path.png`** – path in arena (if `-p Y`). |
| `csv/{Video}_*_.json` (one JSON per video) | **`QC/{Video}_{fly_id}_Energy.png`** – energy over time. |
| (same) | **`QC/{Video}_Energy.png`** – combined GAM + peaks. |
| (same) | **`QC/{Video}_E_peak.csv`** – peak positions (for choosing active segments). |

**Usage:**

```bash
python QC.py -p Y    # Y = save per-fly path plots; N = skip
```

**Working directory:** Script looks for `csv/` in the current directory; creates **`QC/`** if missing.

---

### 4.2 1_single_fly_run_arg.py (single-fly metrics)

**Purpose:** For one video and one frame range: load **raw detection CSV + tracking JSON**, compute per-fly per-frame metrics (nearest-neighbor distance, body/head angle, speed, moving angle, grooming, singing, etc.), and write one table per video×range.

| Input | Output |
|-------|--------|
| `csv/{video_id}.csv` | Raw detections per frame (Frame, class, x, y, width, height). |
| `csv/{video_id}_*_.json` | FLY_matrix tracking JSON: for each frame, each fly has `body` and `head` boxes. The file may be **a semicolon-separated stream of JSON objects** or **one big JSON object**; the script supports both. |
| (same) | **`Video_post/{video_id}_{frame_start}_{frame_end}.csv`** – per-frame × fly table (Frame, Fly_s, Nst_dist, Nst_num, body/head angles, speed, Grooming, Sing, Chasing, Hold, etc.). |

**Usage:**

```bash
python 1_single_fly_run_arg.py -i <video_id> -pp <petri_pixel> -pm <petri_mm> -fs <frame_start> -fe <frame_end>
```

**Working directory:** Expects **`csv/`** and creates **`Video_post/`** in the current directory.

---

### 4.3 2_Chas_behavior_arg.py (chase / interaction events)

**Purpose:** Detect chasing events (who chases whom, head-to/toward, angles), compute chaser/target responses, assign Chase_ID, fill gaps, classify interaction class (Touch, Grooming, Sing, Hold). **Requires both the raw detection/tracking data and the single-fly table from step 1.**

| Input | Output |
|-------|--------|
| `csv/{video_id}.csv` | Raw detections per frame (same file used in 1_). |
| `csv/{video_id}_*_.json` | FLY_matrix tracking JSON (same file used in 1_); again, can be semicolon-separated JSON objects or one big JSON. |
| **`Video_post/{video_id}_{frame_start}_{frame_end}.csv`** (from 1_) | Single-fly per-frame metrics used to filter candidate chase frames and compute responses. |
| (all above) | **`Video_post/Interection_{video_id}_{frame_start}_{frame_end}.csv`** – chase / interaction events and target/chaser metrics (Sfly_act, Tfly_act, Head_to, Direction, etc.). |

**Usage:**

```bash
python 2_Chas_behavior_arg.py -i <video_id> -pp <petri_pixel> -pm <petri_mm> -fs <frame_start> -fe <frame_end>
```

**Working directory:** Same as 1_: **`csv/`**, **`Video_post/`** in current directory.

---

### 4.4 3_single_and_Chascls_arg.py (merge + behavior classification)

**Purpose:** Merge the single-fly table with the interaction table, add chase/chain info, and produce the final **Correct_** table with unified behavior labels. **This step does not read any JSON; it only uses the CSV outputs from 1_ and 2_.**

| Input | Output |
|-------|--------|
| **`Video_post/{video_id}_{frame_start}_{frame_end}.csv`** (from 1_) | **`Video_post/Correct_{video_id}_{frame_start}_{frame_end}.csv`** |
| **`Video_post/Interection_{video_id}_{frame_start}_{frame_end}.csv`** (from 2_) | Single table: per-frame, per-fly, with behavior classes and chase info. |

**Usage:**

```bash
python 3_single_and_Chascls_arg.py -i <video_id> -fs <frame_start> -fe <frame_end>
```

**Working directory:** Reads/writes **`Video_post/`** in current directory.

---

### 4.5 run.py (orchestrator)

**Purpose:** Run the pipeline for every line in **`Video_list.csv`**: step 1 (single-fly), step 2 (chase), step 3 (merge), then **Plot_1.py**. Each line must be tab-separated with exactly 5 columns (see [§3. Video_list.csv](#3-video_listcsv-driver-for-the-pipeline)).

| Input | Output |
|-------|--------|
| **`Video_list.csv`** (in current directory) | For each line: calls 1_, 2_, then 3_ with the same video and frame range. |
| **`csv/*.csv`**, **`csv/*.json`** | Then runs **Plot_1.py** once. |

**Usage:**

```bash
python run.py -p 10
```

- **`-p 10`** = number of parallel processes for the multiprocessing pools.

**Flow inside run.py:**

1. **multicore()** – for each line in `Video_list.csv`:  
   `1_single_fly_run_arg.py -i Line[0] -pp Line[1] -pm Line[2] -fs Line[3] -fe Line[4]`
2. **multicore2()** – for each line:  
   `2_Chas_behavior_arg.py -i Line[0] -pp Line[1] -pm Line[2] -fs Line[3] -fe Line[4]`
3. **multicore3()** – for each line:  
   `3_single_and_Chascls_arg.py -i Line[0] -fs Line[3] -fe Line[4]`
4. **Plot_1(Process)** – run **Plot_1.py -p &lt;Process&gt;**.

**Working directory:** Must be the directory that contains **`Video_list.csv`** and where **`csv/`** and **`Video_post/`** are visible (often project root or `Post_data/`).

---

### 4.6 Plot_1.py (plots and R statistics)

**Purpose:** For every **`Video_post/Correct_*.csv`**, generate boxplots (e.g. nearest distance, behavior counts), write R scripts, and run R for statistical tests.

| Input | Output |
|-------|--------|
| **`Video_post/Correct_*.csv`** | **`Video_post/Nearst_dist/*.png`**, **`*.R`**, R outputs. |
| (same) | **`Video_post/Nearst_num/*.png`**, **`*.R`**, R outputs. |
| (same) | **`Video_post/Behav_count/`** – behavior count plots and R results. |

**Usage:**

```bash
python Plot_1.py -p 10
```

**Working directory:** Expects **`Video_post/`** in current directory; uses **`Post_data/R/`** (e.g. `Nst_dist.R`, `Behav_count.R`) for R script templates.

---

### 4.7 plot_run.py (plate and chain plots)

**Purpose:** For each row in **`Video_list.csv`**, run **Plate_plot.py** and **Plot_chain.py** for that video’s frame range.

| Input | Output |
|-------|--------|
| **`Video_list.csv`** | For each line: calls Plate_plot.py and Plot_chain.py. |
| **`csv/{video}.csv`** (Plate_plot) | Plate position plots. |
| **`Video_post/Correct_{video}_*.csv`** (Plot_chain) | Chain/chase plots. |

**Usage:**

```bash
python plot_run.py -p 5
```

---

### 4.8 Other plotting scripts

All assume **`Video_post/Correct_*.csv`** (and sometimes **`Video_list.csv`** or **`config.json`**) exist.

| Script | Main input | Purpose |
|--------|------------|--------|
| **Plate_plot.py** | `csv/{video}.csv` | Position/class plots in arena. |
| **Plot_chain.py** | `Video_post/Correct_{video}*.csv` | Chase chain visualization. |
| **Plot_arror.py** | `Video_post/Correct_{video}.csv` | Arrow/heading plot for given flies and frame range. |
| **Plot_arror_multy.py** | (same) | Multi-panel arrow plots. |
| **Plot_nw_fly.py** | (same) | Network/position plots. |

**config.json** in this directory defines **PALETTE** (Fly, Groom, Chase, Sing, Hold) and **Palette_sequential** used by these plotters.

---

### 4.9 Batch / multi-video variants (no Video_list)

| Script | Input | Output |
|--------|--------|--------|
| **1_single_fly_run.py** | Discovers videos from **`csv/`** and JSON filenames. | **`Video_post/{video_id}.csv`** (full video, no frame range in name). |
| **3_single_and_Chascls.py** | All single-fly CSVs in **`Video_post/`** and matching **Interection_*.csv**. | Merged/corrected tables (depends on `Singl_list` / `Inter_list` in script). |
| **4_Sing_cor.py** | **`Video_post/Interection_*.csv`** and corresponding single-fly tables. | Correction/summary over interactions (script-specific list of files). |

---

### 4.10 R scripts (called by Plot_1.py or manually)

| Script | Typical input | Purpose |
|--------|----------------|--------|
| **R/Nst_dist.R** | Path to a **Correct_*.csv** (injected by Plot_1). | Nearest-distance stats. |
| **R/Nst_dist_all.R** | Multiple **Correct_*.csv** (Stat_Correct_*, etc.). | Group comparisons, T-tests, ANOVA, Tukey/Dunnett. |
| **R/Behav_count.R**, **R/Behav_count_all.R** | **Correct_*.csv**. | Behavior count stats and group comparisons. |
| **NetD3_FLY.R** | **Correct_*.csv**, **Video_mate.csv**. | Network visualization (e.g. D3). |

---

## 5. Recommended workflow (summary)

1. **Run detection** (from project root) so **`csv/{Video}.csv`** and **`csv/{Video}_*_.json`** exist.
2. **QC:**  
   `python QC.py -p Y`  
   Inspect **`QC/*_Energy.png`** and **`QC/*_E_peak.csv`** to choose **frame_start** and **frame_end**.
3. **Create Video_list.csv** (tab-separated: video_id, petri_pixel, petri_mm, frame_start, frame_end).
4. **Run pipeline:**  
   `python run.py -p 10`
5. **Optional plots:**  
   `python plot_run.py -p 5`  
   Or run individual Plot_* scripts with **-i**, **-f**, **-e**, etc., as needed.

---

## 6. Directory layout (after a full run)

```
Project root (or Post_data/)
├── Video_list.csv
├── csv/
│   ├── {Video}.csv
│   └── {Video}_{frame}_.json
├── Video_post/
│   ├── {video_id}_{fs}_{fe}.csv
│   ├── Interection_{video_id}_{fs}_{fe}.csv
│   ├── Correct_{video_id}_{fs}_{fe}.csv
│   ├── Nearst_dist/
│   ├── Nearst_num/
│   └── Behav_count/
├── QC/
│   ├── {Video}_Energy.png
│   ├── {Video}_E_peak.csv
│   └── {Video}_{fly_id}_path.png, _Energy.png
└── Post_data/
    ├── POST_DATA_PIPELINE.md   (this file)
    ├── run.py
    ├── 1_single_fly_run_arg.py
    ├── 2_Chas_behavior_arg.py
    ├── 3_single_and_Chascls_arg.py
    ├── Plot_1.py
    ├── plot_run.py
    ├── QC.py
    ├── config.json
    ├── R/
    └── ... (other Plot_*, multy, etc.)
```

This document describes the **input and output flow** of the Post_data pipeline; for script-specific options, run each script with `--help` or read the source.
