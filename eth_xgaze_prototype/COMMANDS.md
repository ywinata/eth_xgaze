# ETH-XGaze Prototype Commands

Run commands from this folder:

```bat
C:\Users\ywinata_kadence\Documents\ChatGPT\Eye Track\eth_xgaze_prototype
```

## Common Commands

```bat
run.bat
run.bat --calibrate
run.bat --emotion
run.bat --calibrate --emotion
run.bat --preview
run.bat --preview --emotion
run.bat --preview --emotion --emotion-debug
run.bat --calibrate --calibration-camera-overlay
```

Recommended laptop calibration:

```bat
run.bat --calibrate
```

Higher quality calibration for stronger PC:

```bat
run.bat --calibrate --grid-rows 7 --grid-cols 7 --min-samples-per-target 18
```

Emotion/facial-expression cue:

```bat
python download_emotion_model.py
run.bat --emotion
run.bat --preview --emotion
run.bat --preview --emotion --emotion-debug
run.bat --emotion --emotion-interval-ms 100
run.bat --emotion --emotion-interval-ms 300
```

Debug face detection:

```bat
run.bat --calibrate --calibration-camera-overlay
run.bat --calibrate --face-upsample 2
```

Try gaze-only mode:

```bat
run.bat --calibrate --feature-mode gaze
```

Try different head-pose influence:

```bat
run.bat --calibrate --head-weight 0.15
run.bat --calibrate --head-weight 0.35
```

## Changeable Settings

| Setting | Default | Values / Example | Notes |
| --- | ---: | --- | --- |
| `--eth-xgaze-dir` | `C:\Users\ywinata_kadence\Documents\CV Code\eth-xgaze` | path | Official ETH-XGaze repo folder. |
| `--checkpoint` | `...\ckpt\epoch_24_ckpt.pth.tar` | path | ETH-XGaze pretrained checkpoint. |
| `--calibration` | `eth_xgaze_screen_calibration.json` | path | Calibration output/input file. |
| `--device` | `cpu` | `cpu`, `cuda` | Use `cuda` only when GPU/Torch supports it. |
| `--camera` | `0` | `0`, `1`, etc. | Webcam index. |
| `--width` | `640` | `424`, `640`, `1280` | Camera capture width. Lower can be lighter. |
| `--height` | `480` | `240`, `480`, `720` | Camera capture height. Lower can be lighter. |
| `--face-upsample` | `1` | `0`, `1`, `2` | Higher may detect face better but costs CPU. |
| `--calibration-bg` | `black` | `black`, `white` | Calibration screen background. |
| `--calibration-camera-overlay` | off | flag | Shows camera thumbnail during calibration for debugging. |
| `--no-calibration-camera-overlay` | on | flag | Keeps calibration camera overlay hidden. |
| `--pointer-size` | `34` | pixels | Red gaze pointer size. |
| `--grid-rows` | `5` | `5`, `7`, etc. | Calibration grid rows. |
| `--grid-cols` | `5` | `5`, `7`, etc. | Calibration grid columns. |
| `--grid-margin` | `0.08` | `0.05`, `0.10` | Screen edge margin for calibration dots. |
| `--min-samples-per-target` | `18` | `12`, `18`, `24` | Valid samples needed per dot. Higher is slower but can improve fit. |
| `--capture-delay` | `0.65` | seconds | Wait before collecting samples after dot appears. |
| `--min-target-seconds` | `1.8` | seconds | Minimum time spent on each calibration dot. |
| `--max-target-seconds` | `4.0` | seconds | Maximum time before moving on. |
| `--median-window` | `5` | `3`, `5`, `7` | Pointer median smoothing window. |
| `--ema-alpha` | `0.60` | `0.4` to `0.8` | Pointer smoothing. Higher follows movement faster. |
| `--ridge` | `0.01` | `0.001`, `0.01`, `0.1` | Calibration regression regularization. |
| `--robust-percentile` | `80` | `70`, `80`, `90` | Drops noisier calibration samples above this percentile. |
| `--feature-mode` | `hybrid` | `gaze`, `hybrid`, `gaze_head` | Gaze-only or gaze plus head-pose features. |
| `--head-weight` | `0.25` | `0.15`, `0.25`, `0.35` | Head-pose influence in `hybrid` mode. |
| `--min-eye-open` | `0.16` | `0.14`, `0.16`, `0.20` | Minimum eye-open score for calibration/runtime. |
| `--emotion` | off | flag | Enables FER+ ONNX emotion detection. |
| `--no-emotion` | on | flag | Disables emotion detection. |
| `--emotion-model` | `models\emotion-ferplus-12-int8.onnx` | path | FER+ ONNX model file. |
| `--emotion-interval-ms` | `150` | `100`, `150`, `300` | Lower is more responsive, higher is lighter. |
| `--emotion-window` | `1` | `1`, `3`, `5` | Majority-vote smoothing for emotion label. `1` follows the latest frame. |
| `--emotion-debug` | off | flag | Shows every emotion label probability in preview/status. |
| `--calibrate` | off | flag | Opens calibration mode. |
| `--preview` | off | flag | Opens webcam preview window. |
| `--no-preview` | on | flag | Keeps webcam preview hidden. |

## Presets

Balanced laptop:

```bat
run.bat --calibrate --grid-rows 5 --grid-cols 5 --min-samples-per-target 18
```

Fast calibration:

```bat
run.bat --calibrate --grid-rows 5 --grid-cols 5 --min-samples-per-target 12
```

More accurate but slower:

```bat
run.bat --calibrate --grid-rows 7 --grid-cols 7 --min-samples-per-target 18 --capture-delay 0.8
```

Lighter runtime:

```bat
run.bat --width 424 --height 240 --no-preview --no-emotion
```

Emotion detection, responsive default:

```bat
run.bat --emotion --emotion-interval-ms 150 --emotion-window 1
```

Emotion labels:

```text
neutral, happy, surprise, sad, angry, disgust, fear, contempt
```
