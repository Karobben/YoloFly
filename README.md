<p>
About YOLOv5:<br>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>

<!--
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->
</div>

<video width="320" height="240" controls>
  <source src="movie.mp4" type="Example/6_dishes.Mp4">
</video>

## Install

```bash
# this repository only support for older version if pytorch
conda create -y --name yolo python=3.7 Tensorflow=2.0.0 pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

git clone https://github.com/Dengflylab/YoloFly.git  # clone
cd YoloFly
pip install -r requirements.txt
```


```bash
# assign the loc of the model, video, etc
Model=/mnt/Ken_lap/Github/yolov5/runs/train/2022_07_01_p677_1280_5l_e700_b128_withBW/weights/best.pt
Video=/mnt/Ken_lap/Vlog/flies/20210412-promE-V105005_29C6d.mp4
NUM=13
python3 detect_220101.py --weight $Model  --source $Video --view-img  --conf-thres 0.4 --bh-count --tar-track --head-bind --img-size 1280 --num-fly $NUM

```

## <div align="center">Document &#129712;蝇  </div>

==Before doing every thing:==
```bash
mkdir ../png_DB csv
mkdir ../png_DB/png
```
<pre>
├── png_DB
│   └── png
└── yolov5
    ├── csv
    ├── data
    ├── mask
    ├── models
    ├── other_tools
    ├── runs
    └── utils

</pre>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

---

Thanks for Yolov5, the words below is the torturous of this repository

some scripts:

Detect_2.py
A customized script for specific output.

Quick start:


:fly:
:mosquito:
:microbe:

```python
mkdir csv ../png_DB
mkdir ../png_DB/png

python3 detect_2.py --weight runs/train/exp/weights/best.pt  --source test.mp4 --view-img  --conf-thres 0.4 --chain-det
python3 path_ink.py -i test.mp4
```

To do work


Training:
  - [x] 5 k training set
  - [ ] 10k training set
  - [ ] 20k training set
  - [ ] 50k training set
  - [ ] 100k training set

Features:
  - [X] path ink
  - [X] save images and annotations
  - [X] Flies detection
  - [X] Flies head detection
  - [X] Chasing Behaviors
    - [ ] Chasing duration
  - [ ] Chains
    - [X] Chain by radium
    - [ ] Chasing correction
    - [ ] Chasing duration
  - [ ] Mating
  - [ ] Tracking
    - [ ] Tracking by latest dots
    - [ ] Tracking correction
    - [ ] Tracking Tracking
    - [ ] Movement statistics


### Generate new image data from video for labeling


```bash
# Path for your model
Model=runs/train/2022_03_01_p529_1280_5l_e700_b128/weights/best.pt
Video=/mnt/8A26661926660713/Vlog/upload/cacer/C0022_Trim.mp4

python3 detect_220101.py  --weight $Model   --source $Video --conf 0.4 --head-bind --img-size 1280 --num-fly 11 --img-save
```

### Sorting the flies

```bash
python /mnt/Data/PopOS/Github/Yolo/YoloFly_ken/utils/Post_track.py
```

If you have more than 1 Platte in a video, we can plot the position for each fly to check if there are any switch between different petri dish.
```bash
# plot the position of the fly from each video to check switch betw
python Post_data/QC.py
```

### Detection class labels

The YOLO model outputs the following classes. Each row in a detection CSV (`csv/*.csv`) is:
`Frame  class  x_center  y_center  width  height` (all space-separated, coordinates normalized 0–1).

| Class | Label       | Description                                                                 | Typical box size (normalized) |
|-------|-------------|-----------------------------------------------------------------------------|-------------------------------|
| 0     | **body**    | Entire fly body — the main detection used for tracking and counting.        | w ≈ 0.066, h ≈ 0.079         |
| 1     | **head**    | Fly head only — small box used for head-binding and orientation estimation.  | w ≈ 0.017, h ≈ 0.019         |
| 2     | **grooming**| Fly performing grooming behaviour.                                          | —                             |
| 3     | **chasing** | Chasing interaction between two flies.                                      | —                             |
| 4     | **flapping**| Wing-flapping / singing behaviour.                                          | —                             |
| 5     | **holding** | Holding / mounting interaction.                                             | —                             |

> **cls 0 is ~17× larger in area than cls 1.** cls 0 covers the whole body; cls 1 covers only the head.

---

### Video_list.csv format (post-processing pipeline)

`Video_list.csv` is a **tab-separated** table with **no header**. Each line describes one video and one frame range:

| Column | Name            | Description                                                                                 |
|--------|-----------------|---------------------------------------------------------------------------------------------|
| 1      | `video_id`      | Identifier used in detection results and `Video_post` filenames; also the `-id` argument.  |
| 2      | `petri_pixel`   | Plate diameter (or relevant length) in **pixels**.                                         |
| 3      | `petri_mm`      | Same plate length in **millimeters**.                                                      |
| 4      | `frame_start`   | First frame index (**inclusive**) for this segment.                                        |
| 5      | `frame_end`     | Last frame index (**exclusive**) for this segment.                                         |
| 6      | `num_flies`     | Expected number of flies in the arena (used by analysis/QA tools).                         |
| 7      | `abs_path`      | Absolute path to the corresponding video file.                                             |

Example line (single tab between columns, no header row):

```text
adf6254_Movie_S3.mp4	1080	95	3000	17900	13	/mnt/Data/Videos/adf6254_Movie_S3.mp4
```
