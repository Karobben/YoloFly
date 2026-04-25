# FastView

FastView is a quick quality-check pipeline for long videos. It is designed for screening, not replacing final review:

1. Run `detect_2.py` with `--frame-skip` to create a smaller detection CSV.
2. Run `utils/Post_track.py` on that CSV with windowed parallel tracking.
3. Load the CSV, raw video, and tracking JSON in the web app `CSV Explore` section.

## Example

Run from the parent directory (`/home/wenkanl2/Jie`) in the `yolo` environment. The pipeline reads `Video_list.csv` automatically:

```bash
conda run -n yolo python YoloFly/FastView/fastview_pipeline.py \
  -w YoloFly/runs/train/2022_05_11_p633_1280_51_e700_b128/weights/best.pt \
  --frame-skip 30 \
  --workers 64 \
  --speed-window 300 \
  -o QuickView
```

`Video_list.csv` columns used by FastView:

```text
column 6: total number of flies
column 7: video path
column 8: split-x pixel boundary, optional
column 9: split-y pixel boundary, optional
```

If a split is present, the total fly number is divided by the number of regions. For example, total `24` with `split-x 1000` becomes `12` flies per region.

The detection output is:

```text
csv/C0157_12male.MP4_Fskip_30.csv
```

The tracking outputs are:

```text
csv/C0157_12male.MP4_Fskip_30
csv/C0157_12male.MP4_Fskip_30_1_.json
```

## Reuse Existing CSV

If the frame-skipped detection CSVs already exist:

```bash
conda run -n yolo python YoloFly/FastView/fastview_pipeline.py \
  -w YoloFly/runs/train/2022_05_11_p633_1280_51_e700_b128/weights/best.pt \
  --frame-skip 30 \
  --workers 64 \
  --skip-detect
```

## Window Tracking

`--workers` controls the number of tracking windows. Window size is calculated automatically from the number of processed frames.

`--window-overlap` defaults to `200` processed frames. For `--frame-skip 30`, this means about 6000 raw video frames of overlap, which gives enough shared frames for ID stitching.

## Web Review

Start the app:

```bash
conda run -n yolo python webapp/yolo_review_app.py
```

Open `http://localhost:8000/detect_explore`, then in `CSV Explore`:

- Load the frame-skipped CSV.
- Set the raw video path.
- Load the tracking JSON.
- Use the class filters, fly ID arrows, and history dots for quality checking.

## Static Track Plot

Create a quick faceted track image from a tracking JSON:

```bash
conda run -n yolo python FastView/visualize_tracks.py \
  -j csv/C0157_12male.MP4_Fskip_30_tracked.csv_1_.json \
  -o csv/FastView_plots
```

This makes one subplot per `fly_*` ID. Points are body centers, colored by frame.

By default, the script creates the track plot, a faceted moving-speed line plot, and a total moving-speed line plot in the output directory:

```text
csv/FastView_plots/C0157_12male.MP4_Fskip_30_tracked.csv_1_.json.tracks_by_fly.png
csv/FastView_plots/C0157_12male.MP4_Fskip_30_tracked.csv_1_.json.speed_by_fly.png
csv/FastView_plots/C0157_12male.MP4_Fskip_30_tracked.csv_1_.json.total_speed.png
```

Speed is normalized body-center displacement per raw frame:

```text
sqrt(dx^2 + dy^2) / frame_difference
```

The speed line is smoothed by a centered rolling window. Default is `--speed-window 30`:

```bash
conda run -n yolo python FastView/visualize_tracks.py \
  -j csv/C0157_12male.MP4_Fskip_30_tracked.csv_1_.json \
  -o csv/FastView_plots \
  --speed-window 11
```

To make only the speed plot:

```bash
conda run -n yolo python FastView/visualize_tracks.py \
  -j csv/C0157_12male.MP4_Fskip_30_tracked.csv_1_.json \
  --plot speed \
  -o csv/FastView_plots
```

To make only the total speed plot:

```bash
conda run -n yolo python FastView/visualize_tracks.py \
  -j csv/C0157_12male.MP4_Fskip_30_tracked.csv_1_.json \
  --plot total-speed \
  -o csv/FastView_plots
```
