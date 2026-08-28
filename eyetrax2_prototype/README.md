# EyeTrax2 Prototype

Second local Windows gaze-pointer prototype using
[`ck-zhang/eyetrax`](https://github.com/ck-zhang/eyetrax).

It keeps the same flow as the first prototype:

- full-screen calibration dots
- live calibration capture readout
- blink rejection during calibration and prediction
- trained model saved to `eyetrax_gaze_model.pkl`
- red transparent pointer overlay on screen
- press `Esc` during calibration to cancel; press `Esc` during runtime to quit
  cleanly; prediction stays disabled until a trained gaze model exists

## Setup

```bat
conda env create -f environment.yml
conda activate eyetrax2_prototype
python download_model.py
python app.py --calibrate
```

Later runs:

```bat
run.bat
```

Recalibrate:

```bat
run.bat --calibrate
```

After this update, recalibration is required because the trained model format
changed and old models are ignored.

## Useful Options

```bat
run.bat --calibrate --grid-rows 7 --grid-cols 7 --grid-margin 0.06 --edge-margin 0.03 --min-samples-per-target 24
run.bat --filter kalman_ema --ema-alpha 0.70
run.bat --model svr --calibrate
run.bat --max-gaze-jump 420
run.bat --calibrate --no-edge-targets
run.bat --no-preview
```

## Notes

This prototype now adds app-level improvements around EyeTrax:

- default `7x7` calibration grid
- bezel-near edge calibration targets
- more samples per dot
- robust outlier trimming before training
- median output filtering before Kalman/EMA
- large jump rejection

EyeTrax extracts head-normalized eye landmarks, rejects blinks, supports multiple
models, and provides Kalman/EMA smoothing. This is still webcam gaze tracking, so
precision is limited by webcam placement, lighting, glasses glare, and head
movement.
