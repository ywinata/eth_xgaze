# EyeTrax Prototype

Local Windows webcam gaze-pointer prototype. It uses OpenCV for webcam capture,
MediaPipe Face Mesh/Iris landmarks for eye features, and a small user calibration
model to map those features to screen coordinates.

## Why this approach

- MediaPipe 0.10 Face Mesh with `refine_landmarks=True` is open source, free, and fast
  enough for low-end laptops at 640x480.
- Calibration is required because ordinary laptop webcams do not directly know
  where on the screen you are looking.
- Heavier open-source options exist, including OpenFace and deep gaze models, but
  they are harder to install and often less pleasant on a low-end Windows laptop.

## Setup

Recommended local venv setup:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_model.py
.venv\Scripts\python.exe app.py --calibrate
```

Conda setup, if `eyetrax_prototype` uses Python 3.11:

```bat
conda env create -f environment.yml
conda activate eyetrax_prototype
python download_model.py
python app.py --calibrate
```

Later runs can use:

```bat
run.bat
```

## Controls

- Click `Calibrate` and look at each red dot until it advances.
- During calibration, the bottom-left readout shows target screen `x,y`, captured
  eye-feature `x,y`, and sample count for the current dot.
- Calibration uses a 5x5 grid and rejects blink/closed-eye frames before saving
  samples.
- Keep your laptop and head position similar after calibration.
- Close the camera preview or click `Quit` to exit.

## Low-end laptop tips

```bat
python app.py --width 424 --height 240
python app.py --no-preview
```

## Accuracy tips

- Recalibrate after changing display resolution, scaling, monitor layout, laptop
  angle, webcam position, seat height, or lighting.
- Calibration is full screen because the targets are real screen coordinates.
  The control window does not need to be full screen after calibration.
- Keep your head still while each red calibration dot is shown. The app now uses
  all detected frames per dot, so stable samples matter more than speed.
- If good samples are rejected too often, lower the threshold slightly:
  `run.bat --calibrate --min-eye-quality 0.12`
- The camera preview is mirrored for comfort. That is okay because calibration
  and prediction use the same mirrored frame pipeline.
- For best accuracy, use `--width 640 --height 480` first. Use lower resolution
  only if the laptop is too slow.

## Notes

This is a webcam prototype, not medical-grade or research-grade eye tracking.
Accuracy depends heavily on lighting, webcam position, glasses glare, and keeping
your head near the calibrated position.
