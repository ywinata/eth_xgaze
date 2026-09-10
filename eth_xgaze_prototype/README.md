# ETH-XGaze Prototype

Local webcam gaze-pointer prototype using the official
[`xucong-zhang/ETH-XGaze`](https://github.com/xucong-zhang/ETH-XGaze) code as
the core model and data-normalization path.

The cloned ETH-XGaze repo is expected here:

```bat
C:\Users\ywinata_kadence\Documents\CV Code\eth-xgaze
```

## Setup

```bat
conda env create -f environment.yml
conda activate eth_xgaze_prototype
python download_checkpoint.py
python download_emotion_model.py
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

Default calibration is laptop-friendly: `5x5` grid and `18` valid samples per
dot.

## What It Does

- captures webcam frames with OpenCV
- detects a face and 68 landmarks with dlib
- runs ETH-XGaze face normalization and ResNet-50 gaze estimation
- maps ETH-XGaze pitch/yaw plus lightly weighted head pose to screen coordinates
- shows a transparent red pointer overlay

## Required ETH-XGaze Files

The official repo includes the dlib landmark model and face model. The pretrained
checkpoint is not in git; run `python download_checkpoint.py` or manually place:

```bat
C:\Users\ywinata_kadence\Documents\CV Code\eth-xgaze\ckpt\epoch_24_ckpt.pth.tar
```

## Useful Options

```bat
run.bat --calibrate
run.bat --calibrate --grid-rows 7 --grid-cols 7 --min-samples-per-target 18
run.bat --calibrate --feature-mode gaze --grid-rows 7 --grid-cols 7
run.bat --calibrate --feature-mode gaze_head
run.bat --calibrate --head-weight 0.15
run.bat --calibrate --min-eye-open 0.14
run.bat --device cuda
run.bat --width 424 --height 240
run.bat --calibrate --face-upsample 2
run.bat --calibrate --calibration-bg white
run.bat --calibrate --calibration-camera-overlay
run.bat --preview
run.bat --emotion
run.bat --preview --emotion
run.bat --preview --emotion --emotion-debug
run.bat --emotion --emotion-interval-ms 100
```

## Calibration Notes

Default calibration now uses `--feature-mode hybrid`, which maps ETH-XGaze
pitch/yaw plus lightly weighted head pose to the screen. Keep your head mostly
still and move your eyes to each dot; small natural head motion is okay.

If head movement still dominates, lower it:

```bat
run.bat --calibrate --head-weight 0.15
```

If the pointer cannot reach the screen edges, raise it:

```bat
run.bat --calibrate --head-weight 0.35
```

`--feature-mode gaze` maps only pitch/yaw. It usually allows freer eye movement
but may cover a smaller screen area.

`--feature-mode gaze_head` restores the previous gaze-plus-head-pose mapping for
comparison. It can feel stable, but it often makes the pointer depend too much on
head direction.

The calibration screen defaults to a black background with dim status text. This
keeps 49-dot calibration less distracting, so your attention can stay on the red
target.

Calibration samples are collected only when the face is detected, ETH-XGaze runs,
and both eyes are open enough. If normal open eyes are rejected too often, lower
the threshold slightly with `--min-eye-open 0.14`.

Optional emotion detection is off by default to keep gaze estimation smooth.
Enable the lightweight FER+ ONNX mode with `--emotion`; it updates at most every
`150 ms` by default. Use `--emotion-interval-ms 100` for stronger PCs or
`--emotion-interval-ms 300` if the laptop feels heavy. Labels are `neutral`,
`happy`, `surprise`, `sad`, `angry`, `disgust`, `fear`, and `contempt`.
By default the UI shows the latest winning label without majority smoothing; add
`--emotion-debug` to show all label probabilities in the preview/status for
tuning.

If calibration says `face not detected`, make sure the physical camera switch is
on, try brighter front lighting, keep your face centered in the webcam, or run
with `--face-upsample 2`. For debugging only, add
`--calibration-camera-overlay`; a green box labelled `dlib` or `cv2` means face
detection is working before ETH-XGaze inference.

This is a prototype. Accuracy depends heavily on webcam angle, lighting, glasses
glare, display geometry, and keeping your head position similar after calibration.
