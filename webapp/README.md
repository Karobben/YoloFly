# YOLO Review Web App

Server web app for reviewing/editing YOLO labels against images/video frames from `runs/detect`.

## Run

```bash
cd /home/wenkanl2/Jie/YoloFly
python webapp/yolo_review_app.py
```

Open: `http://127.0.0.1:8787`

## Usage

- Select a run folder from `runs/detect`.
- Open an image, or open a frame from a video.
- Set label path (relative to run, usually under `labels/`), then load.
- Edit boxes:
  - Click to select.
  - Hold **Shift + drag** to add a new box.
  - Press **Delete** to remove selected.
- Save label to run folder.

## Notes

- Label format: `class xc yc w h [conf]` normalized.
- App only reads/writes inside `runs/detect` for safety.
