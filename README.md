# Plane Detection Experiment

An experimental Python script that downloads a video with `yt-dlp`, processes it frame by frame with YOLOv8, and displays possible airplane detections with OpenCV.

## Entry point

```text
plane_detect/plane_detect/plane_tracker.py
```

The script:

1. Prompts for a YouTube video URL.
2. Downloads the video as `video.mp4` when a local copy is not present.
3. Loads the `yolov8n.pt` model through Ultralytics.
4. Starts processing from a user-selected timestamp.
5. Shows annotated frames in an OpenCV window; press `q` to stop.

## Requirements

- Python
- `yt-dlp` available as a command-line program
- Ultralytics YOLO
- OpenCV
- NumPy
- A desktop environment capable of displaying an OpenCV window

No dependency versions are recorded in this repository. The model weights may be downloaded automatically by Ultralytics, and downloading or processing third-party videos may be subject to the source platform's terms and the video's rights.

## Run

After preparing a compatible environment:

```bash
python plane_detect/plane_detect/plane_tracker.py
```

## Limitations

This is a learning experiment, not a validated tracking or aviation-safety system. The current filter accepts numeric class IDs `4` and `5` in addition to the class name `airplane`; class mappings can vary, so non-airplane objects may be labeled as planes. No accuracy benchmark or test dataset is included.
