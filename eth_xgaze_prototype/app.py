from __future__ import annotations

import argparse
import csv
import ctypes
from ctypes import wintypes
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


enable_dpi_awareness()

import cv2
import dlib
import numpy as np
import tkinter as tk
import torch
import torch.nn as nn
from imutils import face_utils
from PIL import Image, ImageTk
from torchvision import transforms


ETH_XGAZE_DIR = Path(r"C:\Users\ywinata_kadence\Documents\CV Code\eth-xgaze")
CHECKPOINT = ETH_XGAZE_DIR / "ckpt" / "epoch_24_ckpt.pth.tar"
CALIBRATION_FILE = Path("eth_xgaze_screen_calibration.json")
EMOTION_MODEL = Path("models") / "emotion-ferplus-12-int8.onnx"
EMOTION_LABELS = ("neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt")
NEGATIVE_EMOTION_LABELS = ("sad", "angry", "disgust", "fear")
MODEL_VERSION = 4
PREVIEW_WINDOW = "ETH-XGaze camera preview"
PREVIEW_RECORDING_FILE = Path("preview_recording.mp4")
PREVIEW_RECORDING_FPS = 20.0
METRICS_FILE = Path("preview_metrics.csv")
METRICS_FIELDS = [
    "timestamp_s",
    "frame_index",
    "face_detected",
    "gaze_x",
    "gaze_y",
    "pitch",
    "yaw",
    "eye_open",
    "emotion_label",
    "emotion_intensity",
    "emotion_intensity_delta",
    "emotion_intensity_sigma",
    "valence",
    "valence_delta",
    "valence_sigma",
    "positive",
    "negative",
    "neutral_prob",
    "happy_prob",
    "surprise_prob",
    "sad_prob",
    "angry_prob",
    "disgust_prob",
    "fear_prob",
    "contempt_prob",
    "expression_overall",
    "expression_overall_sigma",
    "brow_sigma",
    "eye_sigma",
    "mouth_sigma",
]


def list_screens(root: tk.Tk) -> list[dict[str, int | bool]]:
    if sys.platform != "win32":
        return [{"x": 0, "y": 0, "w": root.winfo_screenwidth(), "h": root.winfo_screenheight(), "primary": True}]

    monitors: list[dict[str, int | bool]] = []

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    def callback(handle, _dc, _rect, _data):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        ctypes.windll.user32.GetMonitorInfoW(handle, ctypes.byref(info))
        rect = info.rcMonitor
        monitors.append(
            {
                "x": int(rect.left),
                "y": int(rect.top),
                "w": int(rect.right - rect.left),
                "h": int(rect.bottom - rect.top),
                "primary": bool(info.dwFlags & 1),
            }
        )
        return True

    monitor_enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    ctypes.windll.user32.EnumDisplayMonitors(0, 0, monitor_enum_proc(callback), 0)
    return monitors or [{"x": 0, "y": 0, "w": root.winfo_screenwidth(), "h": root.winfo_screenheight(), "primary": True}]


def select_screen(root: tk.Tk, value: str) -> dict[str, int | bool]:
    screens = list_screens(root)
    raw = str(value).strip().lower()
    if raw == "auto":
        return next((screen for screen in screens if screen["primary"]), screens[0])
    try:
        return screens[int(raw)]
    except (ValueError, IndexError):
        return screens[0]


def resolve_camera_index(value: str) -> int:
    raw = str(value).strip().lower()
    if raw == "auto":
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def find_haar_cascade() -> Optional[Path]:
    name = "haarcascade_frontalface_default.xml"
    roots = []
    haarcascades = getattr(getattr(cv2, "data", None), "haarcascades", "")
    if haarcascades:
        roots.append(Path(haarcascades))
    roots.extend(
        [
            Path(sys.prefix) / "Library" / "etc" / "haarcascades",
            Path(sys.prefix) / "share" / "opencv4" / "haarcascades",
            Path(getattr(cv2, "__file__", "")).parent / "data",
        ]
    )
    for root in roots:
        path = root / name
        if path.exists():
            return path
    return None


class PointerWindow:
    def __init__(self, root: tk.Tk, size: int) -> None:
        self.size = size
        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.configure(bg="black")
        try:
            self.window.attributes("-transparentcolor", "black")
        except tk.TclError:
            self.window.attributes("-alpha", 0.75)
        canvas = tk.Canvas(self.window, width=size, height=size, bg="black", highlightthickness=0)
        canvas.pack()
        canvas.create_oval(3, 3, size - 3, size - 3, fill="red", outline="white", width=2)
        canvas.create_oval(size / 2 - 3, size / 2 - 3, size / 2 + 3, size / 2 + 3, fill="white", outline="")
        self.hide()

    def move(self, x: int, y: int) -> None:
        self.window.geometry(f"{self.size}x{self.size}+{x - self.size // 2}+{y - self.size // 2}")
        self.window.deiconify()

    def hide(self) -> None:
        self.window.withdraw()


class EthXGazeModel(nn.Module):
    def __init__(self, eth_dir: Path) -> None:
        super().__init__()
        sys.path.insert(0, str(eth_dir))
        from modules import resnet50

        self.gaze_network = resnet50(pretrained=False)
        self.gaze_fc = nn.Sequential(nn.Linear(2048, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.gaze_network(x)
        return self.gaze_fc(features.view(features.size(0), -1))


class EthXGazeCore:
    def __init__(self, eth_dir: Path, checkpoint: Path, device: str, width: int, height: int, face_upsample: int) -> None:
        self.eth_dir = eth_dir
        self.requested_device = device
        self.device = self.resolve_device(device)
        self.device_note = f"{device} -> {self.device.type}" if device != self.device.type else self.device.type
        self.face_upsample = face_upsample
        self._verify_files(checkpoint)
        self.detector = dlib.get_frontal_face_detector()
        cascade_path = find_haar_cascade()
        self.cascade = cv2.CascadeClassifier(str(cascade_path)) if cascade_path else cv2.CascadeClassifier()
        self.predictor = dlib.shape_predictor(str(eth_dir / "modules" / "shape_predictor_68_face_landmarks.dat"))
        face_model_load = np.loadtxt(eth_dir / "face_model.txt")
        self.face_model = face_model_load[[20, 23, 26, 29, 15, 19], :].reshape(6, 1, 3)
        self.camera = self._camera_matrix(width, height)
        self.distortion = np.zeros((5, 1), dtype=np.float64)
        self.model = EthXGazeModel(eth_dir).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)

    def _verify_files(self, checkpoint: Path) -> None:
        missing = [
            path
            for path in [
                self.eth_dir / "modules" / "shape_predictor_68_face_landmarks.dat",
                self.eth_dir / "face_model.txt",
                self.eth_dir / "modules" / "resnet.py",
                checkpoint,
            ]
            if not path.exists()
        ]
        if missing:
            lines = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Missing ETH-XGaze files:\n{lines}\nRun python download_checkpoint.py if only the checkpoint is missing.")

    @staticmethod
    def _camera_matrix(width: int, height: int) -> np.ndarray:
        focal = float(width)
        return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def detect_face(self, frame: np.ndarray) -> Optional[tuple[dlib.rectangle, str]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb, self.face_upsample)
        if faces:
            return faces[0], "dlib"
        if not faces and not self.cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hits = self.cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(80, 80))
            if len(hits):
                x, y, w, h = hits[0]
                return dlib.rectangle(int(x), int(y), int(x + w), int(y + h)), "cv2"
        return None

    def estimate(self, frame: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, dlib.rectangle, str, float, np.ndarray]]:
        detection = self.detect_face(frame)
        if detection is None:
            return None
        face, detector_name = detection
        shape = face_utils.shape_to_np(self.predictor(frame, face))
        eye_quality = self.eye_open_quality(shape)
        landmarks = shape[[36, 39, 42, 45, 31, 35], :].astype(float).reshape(6, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(self.face_model, landmarks, self.camera, self.distortion, flags=cv2.SOLVEPNP_EPNP)
        if ok:
            ok, rvec, tvec = cv2.solvePnP(self.face_model, landmarks, self.camera, self.distortion, rvec, tvec, True)
        if not ok:
            return None
        patch = self._normalize_face(frame, landmarks, rvec, tvec)
        rgb_patch = patch[:, :, [2, 1, 0]]
        tensor = self.transform(rgb_patch).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            gaze = self.model(tensor)[0].cpu().numpy().astype(np.float32)
        self._draw_debug(frame, shape, gaze)
        pose = np.concatenate([rvec.reshape(-1).astype(np.float32), tvec.reshape(-1).astype(np.float32)])
        return np.concatenate([gaze, pose]), gaze, face, detector_name, eye_quality, shape

    @staticmethod
    def eye_open_quality(landmarks: np.ndarray) -> float:
        def ear(eye: np.ndarray) -> float:
            vertical = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
            horizontal = 2.0 * max(float(np.linalg.norm(eye[0] - eye[3])), 1e-6)
            return float(vertical / horizontal)

        return min(ear(landmarks[36:42]), ear(landmarks[42:48]))

    def _normalize_face(self, img: np.ndarray, landmarks: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        focal_norm = 960
        distance_norm = 600
        roi_size = (224, 224)
        h_r = cv2.Rodrigues(rvec)[0]
        fc = np.dot(h_r, self.face_model.reshape(6, 3).T) + tvec.reshape(3, 1)
        eye_center = np.mean(fc[:, 0:4], axis=1).reshape(3, 1)
        nose_center = np.mean(fc[:, 4:6], axis=1).reshape(3, 1)
        face_center = np.mean(np.concatenate((eye_center, nose_center), axis=1), axis=1).reshape(3, 1)
        distance = np.linalg.norm(face_center)
        cam_norm = np.array([[focal_norm, 0, 112], [0, focal_norm, 112], [0, 0, 1.0]])
        scale = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, distance_norm / distance]])
        h_rx = h_r[:, 0]
        forward = (face_center / distance).reshape(3)
        down = np.cross(forward, h_rx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        rotate = np.c_[right, down, forward].T
        warp = np.dot(np.dot(cam_norm, scale), np.dot(rotate, np.linalg.inv(self.camera)))
        return cv2.warpPerspective(img, warp, roi_size)

    @staticmethod
    def _draw_debug(frame: np.ndarray, landmarks: np.ndarray, gaze: np.ndarray) -> None:
        for x, y in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        length = min(w, h) / 4
        dx = -length * np.sin(gaze[1]) * np.cos(gaze[0])
        dy = -length * np.sin(gaze[0])
        cv2.arrowedLine(frame, center, (int(center[0] + dx), int(center[1] + dy)), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2)


class EmotionRecognizer:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Emotion model not found: {model_path}")
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame: np.ndarray, face: dlib.rectangle) -> tuple[str, float, dict[str, float]]:
        h, w = frame.shape[:2]
        pad = int(max(face.width(), face.height()) * 0.18)
        x1 = max(0, face.left() - pad)
        y1 = max(0, face.top() - pad)
        x2 = min(w, face.right() + pad)
        y2 = min(h, face.bottom() + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown", 0.0, {}
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(1, 1, 64, 64)
        scores = self.session.run(None, {self.input_name: image})[0][0]
        scores = scores - np.max(scores)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        index = int(np.argmax(probs))
        return EMOTION_LABELS[index], float(probs[index]), {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}


class LandmarkExpressionIntensity:
    ANCHORS = np.array([27, 28, 29, 30, 31, 35, 36, 39, 42, 45])
    REGIONS = {
        "brow": np.arange(17, 27),
        "eye": np.arange(36, 48),
        "mouth": np.arange(48, 68),
    }

    def __init__(self, baseline_frames: int) -> None:
        self.baseline_frames = max(1, baseline_frames)
        self.samples: deque[np.ndarray] = deque(maxlen=self.baseline_frames)
        self.baseline: Optional[np.ndarray] = None
        self.face_width = 1.0
        self.baseline_stats: dict[str, tuple[float, float]] = {}
        self.last_scores: dict[str, float] = {}

    def reset(self) -> None:
        self.samples.clear()
        self.baseline = None
        self.face_width = 1.0
        self.baseline_stats = {}
        self.last_scores = {}

    def update(self, landmarks: np.ndarray) -> dict[str, float]:
        points = landmarks.astype(np.float32)
        if self.baseline is None:
            self.samples.append(points)
            if len(self.samples) == self.samples.maxlen:
                self.baseline = np.median(np.stack(self.samples), axis=0).astype(np.float32)
                self.face_width = max(float(np.linalg.norm(self.baseline[0] - self.baseline[16])), 1.0)
                sample_scores = [self.score(sample) for sample in self.samples]
                self.baseline_stats = {
                    key: (
                        float(np.median([scores[key] for scores in sample_scores])),
                        max(float(np.std([scores[key] for scores in sample_scores])), 1e-3),
                    )
                    for key in ["overall", *self.REGIONS.keys()]
                }
            self.last_scores = {"ready": 0.0, "progress": len(self.samples) / float(self.samples.maxlen)}
            return self.last_scores

        scores = self.score(points)
        scores["ready"] = 1.0
        for key, (center, noise) in self.baseline_stats.items():
            scores[f"{key}_delta"] = scores[key] - center
            scores[f"{key}_sigma"] = (scores[key] - center) / noise
        self.last_scores = scores
        return scores

    def score(self, points: np.ndarray) -> dict[str, float]:
        affine, _ = cv2.estimateAffinePartial2D(points[self.ANCHORS], self.baseline[self.ANCHORS], method=cv2.LMEDS)
        aligned = cv2.transform(points.reshape(1, -1, 2), affine).reshape(-1, 2) if affine is not None else points
        residual = np.linalg.norm(aligned - self.baseline, axis=1) / self.face_width
        scores = {"overall": float(np.mean(residual[17:68]))}
        scores.update({name: float(np.mean(residual[indexes])) for name, indexes in self.REGIONS.items()})
        return scores


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("ETH-XGaze Prototype")
        self.root.geometry("560x240")
        self.root.minsize(560, 240)
        self.screen = select_screen(self.root, args.screen)
        self.screen_x, self.screen_y = int(self.screen["x"]), int(self.screen["y"])
        self.screen_w, self.screen_h = int(self.screen["w"]), int(self.screen["h"])
        self.camera_index = resolve_camera_index(args.camera)
        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.core = EthXGazeCore(
            Path(args.eth_xgaze_dir),
            Path(args.checkpoint),
            args.device,
            args.width,
            args.height,
            args.face_upsample,
        )
        self.pointer = PointerWindow(self.root, args.pointer_size)
        self.status = tk.StringVar(value=f"Calibrate first. Device: {self.core.device_note}.")
        self.emotion: Optional[EmotionRecognizer] = None
        self.emotion_error = ""
        if args.emotion:
            try:
                self.emotion = EmotionRecognizer(Path(args.emotion_model))
            except Exception as exc:
                self.emotion_error = f"Emotion unavailable: {exc}"
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = self.load_calibration()
        if self.weights is not None:
            self.status.set(f"ETH-XGaze calibration loaded. Device: {self.core.device_note}.")
        if self.emotion_error:
            self.status.set(self.emotion_error)
        self.preview_writer: Optional[cv2.VideoWriter] = None
        self.preview_recording_active = bool(args.record_preview)
        self.metrics_file = None
        self.metrics_writer: Optional[csv.DictWriter] = None
        self.metrics_recording_active = bool(args.record_metrics)
        self.last_metrics_at = 0.0
        self.frame_index = 0
        self.latest_face_detected = False
        self.latest_gaze_xy: tuple[Optional[int], Optional[int]] = (None, None)
        self.latest_pitch_yaw: tuple[Optional[float], Optional[float]] = (None, None)
        self.latest_eye_open: Optional[float] = None
        self.preview_window_size: Optional[tuple[int, int]] = None
        self.preview_window_maximized = False
        self.preview_window_created = False
        self.stopped = False
        tk.Label(self.root, text="ETH-XGaze Prototype", font=("Segoe UI", 13, "bold")).pack(pady=(12, 4))
        buttons = tk.Frame(self.root)
        buttons.pack(pady=(8, 10))
        tk.Button(buttons, text="Calibrate", command=self.start_calibration).pack(side=tk.LEFT, padx=6)
        self.record_start_button = tk.Button(buttons, text="Start Recording", command=self.start_preview_recording)
        self.record_start_button.pack(side=tk.LEFT, padx=6)
        self.record_stop_button = tk.Button(buttons, text="Stop & Save", command=self.stop_preview_recording)
        self.record_stop_button.pack(side=tk.LEFT, padx=6)
        tk.Button(buttons, text="Quit", command=self.stop).pack(side=tk.LEFT, padx=6)
        tk.Label(self.root, textvariable=self.status, wraplength=520, height=5, justify=tk.CENTER).pack(fill=tk.X, padx=16)
        self.calibration_screen: Optional[tk.Toplevel] = None
        self.calibration_phase = "gaze"
        self.canvas: Optional[tk.Canvas] = None
        self.targets: list[tuple[int, int]] = []
        self.target_index = -1
        self.target_started = 0.0
        self.target_samples: list[np.ndarray] = []
        self.features: list[np.ndarray] = []
        self.labels: list[list[int]] = []
        self.history: deque[tuple[int, int]] = deque(maxlen=args.median_window)
        self.smoothed: Optional[tuple[float, float]] = None
        self.last_emotion_at = 0.0
        self.emotion_labels: deque[str] = deque(maxlen=args.emotion_window)
        self.emotion_label = "off"
        self.emotion_confidence = 0.0
        self.emotion_probs: dict[str, float] = {}
        self.emotion_intensity = 0.0
        self.emotion_positive = 0.0
        self.emotion_negative = 0.0
        self.emotion_valence = 0.0
        self.emotion_intensity_delta = 0.0
        self.emotion_valence_delta = 0.0
        self.emotion_intensity_sigma = 0.0
        self.emotion_valence_sigma = 0.0
        self.emotion_baseline_samples: list[dict[str, float]] = []
        self.emotion_baseline_probs: dict[str, float] = {}
        self.emotion_baseline_stats: dict[str, tuple[float, float]] = {}
        self.expression_intensity: Optional[LandmarkExpressionIntensity] = (
            LandmarkExpressionIntensity(args.expression_baseline_frames) if args.expression_intensity else None
        )
        self.expression_scores: dict[str, float] = {}
        self.camera_image = None
        self.update_recording_buttons()
        if args.calibrate:
            self.root.after(500, self.start_calibration)
        self.root.after(10, self.tick)
        self.root.bind_all("<Escape>", self.handle_escape)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    def run(self) -> None:
        self.root.mainloop()

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        self.pointer.hide()
        self.close_preview_writer()
        self.close_metrics_writer()
        self.capture.release()
        if not self.args.no_preview:
            cv2.destroyAllWindows()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def handle_escape(self, _event=None) -> None:
        if self.calibration_screen:
            self.close_calibration_screen()
            self.status.set("Calibration cancelled.")
        else:
            self.stop()

    def close_calibration_screen(self) -> None:
        if self.calibration_screen:
            try:
                self.calibration_screen.grab_release()
            except tk.TclError:
                pass
            self.calibration_screen.destroy()
            self.calibration_screen = None
            self.canvas = None
        self.root.deiconify()

    def start_preview_recording(self) -> None:
        if self.args.no_preview:
            self.status.set("Enable --preview before recording preview.")
            return
        self.close_preview_writer()
        self.close_metrics_writer()
        self.preview_recording_active = True
        self.metrics_recording_active = True
        self.update_recording_buttons()
        self.status.set(f"Recording preview/metrics to {PREVIEW_RECORDING_FILE} and {self.args.metrics_file}.")

    def stop_preview_recording(self) -> None:
        was_recording = (
            self.preview_recording_active
            or self.preview_writer is not None
            or self.metrics_recording_active
            or self.metrics_writer is not None
        )
        self.preview_recording_active = False
        self.metrics_recording_active = False
        self.close_preview_writer()
        self.close_metrics_writer()
        self.update_recording_buttons()
        if was_recording:
            self.status.set(f"Saved recording: {PREVIEW_RECORDING_FILE}; metrics: {self.args.metrics_file}.")

    def update_recording_buttons(self) -> None:
        if self.args.no_preview:
            self.record_start_button.config(state=tk.DISABLED)
            self.record_stop_button.config(state=tk.DISABLED)
            return
        active = self.preview_recording_active or self.metrics_recording_active
        self.record_start_button.config(state=tk.DISABLED if active else tk.NORMAL)
        self.record_stop_button.config(state=tk.NORMAL if active else tk.DISABLED)

    def load_calibration(self) -> Optional[np.ndarray]:
        path = Path(self.args.calibration)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("version") != MODEL_VERSION
            or data.get("screen_size") != [self.screen_w, self.screen_h]
            or data.get("screen_origin", [0, 0]) != [self.screen_x, self.screen_y]
            or data.get("feature_mode") != self.args.feature_mode
            or float(data.get("head_weight", self.args.head_weight)) != float(self.args.head_weight)
        ):
            return None
        self.feature_mean = np.array(data["feature_mean"], dtype=np.float32)
        self.feature_std = np.array(data["feature_std"], dtype=np.float32)
        return np.array(data["weights"], dtype=np.float32)

    def has_calibration(self) -> bool:
        return self.weights is not None and self.feature_mean is not None and self.feature_std is not None

    def save_calibration(self) -> None:
        data = {
            "version": MODEL_VERSION,
            "screen_size": [self.screen_w, self.screen_h],
            "screen_origin": [self.screen_x, self.screen_y],
            "weights": self.weights.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "feature_mode": self.args.feature_mode,
            "head_weight": self.args.head_weight,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        Path(self.args.calibration).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def start_calibration(self) -> None:
        try:
            cv2.destroyWindow(PREVIEW_WINDOW)
        except cv2.error:
            pass
        self.targets = self.build_targets()
        self.features = []
        self.labels = []
        self.history.clear()
        self.smoothed = None
        self.pointer.hide()
        self.root.withdraw()
        self.calibration_screen = tk.Toplevel(self.root)
        self.calibration_screen.overrideredirect(True)
        self.calibration_screen.configure(bg=self.args.calibration_bg)
        self.calibration_screen.geometry(f"{self.screen_w}x{self.screen_h}+{self.screen_x}+{self.screen_y}")
        self.calibration_screen.attributes("-topmost", True)
        self.canvas = tk.Canvas(
            self.calibration_screen,
            width=self.screen_w,
            height=self.screen_h,
            bg=self.args.calibration_bg,
            bd=0,
            highlightthickness=0,
        )
        self.canvas.place(x=0, y=0, width=self.screen_w, height=self.screen_h)
        self.calibration_screen.grab_set()
        self.target_index = -1
        if self.needs_neutral_baseline():
            if self.expression_intensity is not None:
                self.expression_intensity.reset()
            self.expression_scores = {}
            self.reset_emotion_baseline()
            self.calibration_phase = "expression_baseline"
            self.draw_expression_baseline(False)
        else:
            self.calibration_phase = "gaze"
            self.next_target()
        self.calibration_screen.lift()
        self.calibration_screen.focus_force()
        self.calibration_screen.update_idletasks()
        self.calibration_screen.update()

    def draw_expression_baseline(self, valid: bool, eye_quality: float = 0.0, detector_name: str = "") -> None:
        if not self.canvas:
            return
        self.canvas.delete("all")
        text_color = "#cbd5e1" if self.args.calibration_bg == "black" else "#1f2937"
        status_color = "#6ee7ff" if valid and self.args.calibration_bg == "black" else ("#0369a1" if valid else "#dc2626")
        progress = self.neutral_baseline_progress()
        cx, cy = self.screen_w // 2, self.screen_h // 2
        self.canvas.create_oval(cx - 12, cy - 12, cx + 12, cy + 12, fill="#38bdf8", outline="white", width=2)
        self.canvas.create_text(cx, 72, text="Neutral baseline", fill=text_color, font=("Segoe UI", 18, "bold"))
        self.canvas.create_text(
            cx,
            118,
            text="Look at the center, keep your head still, relax your face.",
            fill=text_color,
            font=("Segoe UI", 13),
        )
        self.canvas.create_rectangle(cx - 180, cy + 42, cx + 180, cy + 62, outline=text_color, width=2)
        self.canvas.create_rectangle(cx - 178, cy + 44, cx - 178 + int(356 * progress), cy + 60, fill="#22c55e", outline="")
        self.canvas.create_text(
            26,
            self.screen_h - 82,
            text=(
                f"baseline: {progress:.0%}\n"
                f"eye-open: {eye_quality:.3f} / {self.args.min_eye_open:.3f}\n"
                f"{f'face/gaze ok ({detector_name})' if valid else 'face/gaze/eyes not ready'}"
            ),
            fill=status_color,
            font=("Consolas", 14, "bold"),
            anchor="w",
        )

    def build_targets(self) -> list[tuple[int, int]]:
        mx = int(self.screen_w * self.args.grid_margin)
        my = int(self.screen_h * self.args.grid_margin)
        xs = np.linspace(mx, self.screen_w - mx, self.args.grid_cols, dtype=int)
        ys = np.linspace(my, self.screen_h - my, self.args.grid_rows, dtype=int)
        center = (self.screen_w // 2, self.screen_h // 2)
        targets = sorted({(int(x), int(y)) for y in ys for x in xs}, key=lambda p: (p[1], p[0]))
        if center in targets:
            targets.remove(center)
        return [center, *targets]

    def next_target(self) -> None:
        self.target_index += 1
        self.target_samples = []
        self.target_started = time.time()
        if self.target_index >= len(self.targets):
            self.finish_calibration()
        else:
            self.draw_target(False)

    def draw_target(
        self,
        valid: bool,
        eye_quality: float = 0.0,
        frame: Optional[np.ndarray] = None,
        face: Optional[dlib.rectangle] = None,
        detector_name: str = "",
    ) -> None:
        if not self.canvas:
            return
        self.canvas.delete("all")
        x, y = self.targets[self.target_index]
        text_color = "#cbd5e1" if self.args.calibration_bg == "black" else "#1f2937"
        status_color = "#6ee7ff" if valid and self.args.calibration_bg == "black" else ("#0369a1" if valid else "#dc2626")
        self.canvas.create_oval(x - 18, y - 18, x + 18, y + 18, fill="red", outline="white", width=3)
        self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="white", outline="")
        self.canvas.create_text(
            self.screen_w // 2,
            70,
            text=f"Look at the red dot: {self.target_index + 1}/{len(self.targets)}",
            fill=text_color,
            font=("Segoe UI", 18, "bold"),
        )
        self.canvas.create_text(
            26,
            self.screen_h - 96,
            text=(
                f"samples: {len(self.target_samples)} / {self.args.min_samples_per_target}\n"
                f"eye-open: {eye_quality:.3f} / {self.args.min_eye_open:.3f}\n"
                f"{f'face/gaze ok ({detector_name})' if valid else 'face/gaze/eyes not ready'}"
            ),
            fill=status_color,
            font=("Consolas", 14, "bold"),
            anchor="w",
        )
        if self.args.calibration_camera_overlay and frame is not None:
            self.draw_camera_overlay(frame, face, detector_name)

    def draw_camera_overlay(self, frame: np.ndarray, face: Optional[dlib.rectangle], detector_name: str) -> None:
        if not self.canvas:
            return
        thumb = frame.copy()
        if face is not None:
            cv2.rectangle(thumb, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(thumb, detector_name, (face.left(), max(20, face.top() - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(thumb, "no face", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((320, 240))
        self.camera_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(self.screen_w - image.width - 24, self.screen_h - image.height - 24, image=self.camera_image, anchor="nw")

    def finish_calibration(self) -> None:
        self.close_calibration_screen()
        needed = len(self.targets) * self.args.min_samples_per_target
        if len(self.features) < needed:
            self.status.set(f"Calibration failed: {len(self.features)}/{needed} valid samples.")
            return
        selected = self.select_features(np.array(self.features, dtype=np.float32), self.args.feature_mode)
        self.feature_mean = selected.mean(axis=0)
        self.feature_std = selected.std(axis=0)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        x = self.design(self.normalize_features(selected))
        y = np.array(self.labels, dtype=np.float32)
        ridge = self.args.ridge * np.eye(x.shape[1], dtype=np.float32)
        self.weights = np.linalg.solve(x.T @ x + ridge, x.T @ y)
        self.save_calibration()
        self.status.set(f"Calibration saved to {self.args.calibration}.")

    @staticmethod
    def select_features(features: np.ndarray, feature_mode: str) -> np.ndarray:
        features = np.atleast_2d(features)
        if feature_mode == "gaze":
            return features[:, :2]
        return features

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Calibration normalization is not ready.")
        features = (features - self.feature_mean) / self.feature_std
        if self.args.feature_mode == "hybrid":
            features[:, 2:] *= self.args.head_weight
        return features

    @staticmethod
    def design(features: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(features)
        return np.hstack([np.ones((features.shape[0], 1), dtype=np.float32), features, features[:, :2] ** 2])

    def tick(self) -> None:
        if self.stopped:
            return
        ok, frame = self.capture.read()
        if ok:
            frame = cv2.flip(frame, 1)
            self.handle_frame(frame)
            self.write_metrics_record()
            if not self.args.no_preview and not self.calibration_screen:
                if self.preview_was_closed():
                    self.stop()
                    return
                preview = self.make_preview_frame(frame)
                self.draw_preview_overlay(preview)
                self.write_preview_recording(preview)
                self.resize_preview_window(preview)
                cv2.imshow(PREVIEW_WINDOW, preview)
                self.maximize_preview_window()
                cv2.waitKey(1)
                if self.preview_was_closed():
                    self.stop()
                    return
        if not self.stopped:
            self.root.after(16, self.tick)

    def preview_was_closed(self) -> bool:
        if not self.preview_window_created:
            return False
        try:
            return cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def maximize_preview_window(self) -> None:
        if self.preview_window_maximized or sys.platform != "win32":
            return
        hwnd = ctypes.windll.user32.FindWindowW(None, PREVIEW_WINDOW)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 3)
            self.preview_window_maximized = True

    def resize_preview_window(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        size = (width, height)
        if self.preview_window_size == size:
            return
        self.preview_window_size = size
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        self.preview_window_created = True
        cv2.resizeWindow(PREVIEW_WINDOW, width, height)
        x = self.screen_x + max(0, (self.screen_w - width) // 2)
        y = self.screen_y + max(0, (self.screen_h - height) // 2)
        cv2.moveWindow(PREVIEW_WINDOW, x, y)

    def write_preview_recording(self, frame: np.ndarray) -> None:
        if not self.preview_recording_active:
            return
        if self.preview_writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.preview_writer = cv2.VideoWriter(
                str(PREVIEW_RECORDING_FILE),
                fourcc,
                PREVIEW_RECORDING_FPS,
                (width, height),
            )
        if self.preview_writer.isOpened():
            self.preview_writer.write(frame)
        else:
            self.preview_recording_active = False
            self.update_recording_buttons()
            self.status.set(f"Could not write preview recording: {PREVIEW_RECORDING_FILE}.")

    def close_preview_writer(self) -> None:
        if self.preview_writer is not None:
            self.preview_writer.release()
            self.preview_writer = None

    def write_metrics_record(self) -> None:
        if not self.metrics_recording_active:
            return
        now = time.perf_counter()
        if (now - self.last_metrics_at) * 1000.0 < self.args.metrics_interval_ms:
            return
        self.last_metrics_at = now
        if self.metrics_writer is None:
            self.metrics_file = open(self.args.metrics_file, "w", newline="", encoding="utf-8")
            self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=METRICS_FIELDS)
            self.metrics_writer.writeheader()
        gaze_x, gaze_y = self.latest_gaze_xy
        pitch, yaw = self.latest_pitch_yaw
        row = {
            "timestamp_s": f"{time.time():.3f}",
            "frame_index": self.frame_index,
            "face_detected": int(self.latest_face_detected),
            "gaze_x": "" if gaze_x is None else gaze_x,
            "gaze_y": "" if gaze_y is None else gaze_y,
            "pitch": "" if pitch is None else f"{pitch:.6f}",
            "yaw": "" if yaw is None else f"{yaw:.6f}",
            "eye_open": "" if self.latest_eye_open is None else f"{self.latest_eye_open:.6f}",
            "emotion_label": self.emotion_label if self.emotion is not None else "",
            "emotion_intensity": f"{self.emotion_intensity:.6f}",
            "emotion_intensity_delta": f"{self.emotion_intensity_delta:.6f}",
            "emotion_intensity_sigma": f"{self.emotion_intensity_sigma:.6f}",
            "valence": f"{self.emotion_valence:.6f}",
            "valence_delta": f"{self.emotion_valence_delta:.6f}",
            "valence_sigma": f"{self.emotion_valence_sigma:.6f}",
            "positive": f"{self.emotion_positive:.6f}",
            "negative": f"{self.emotion_negative:.6f}",
            "expression_overall": self.format_metric("overall"),
            "expression_overall_sigma": self.format_metric("overall_sigma"),
            "brow_sigma": self.format_metric("brow_sigma"),
            "eye_sigma": self.format_metric("eye_sigma"),
            "mouth_sigma": self.format_metric("mouth_sigma"),
        }
        for label in EMOTION_LABELS:
            row[f"{label}_prob"] = f"{self.emotion_probs.get(label, 0.0):.6f}"
        self.metrics_writer.writerow(row)
        if self.metrics_file is not None:
            self.metrics_file.flush()

    def format_metric(self, key: str) -> str:
        if not self.expression_scores or key not in self.expression_scores:
            return ""
        return f"{self.expression_scores[key]:.6f}"

    def close_metrics_writer(self) -> None:
        if self.metrics_file is not None:
            self.metrics_file.close()
            self.metrics_file = None
            self.metrics_writer = None

    def preview_scale(self, frame: np.ndarray) -> float:
        raw = str(self.args.preview_scale).strip().lower()
        if raw == "auto":
            height, width = frame.shape[:2]
            available_w = max(320, self.screen_w - 80)
            available_h = max(240, self.screen_h - 180)
            return max(0.25, min(2.0, available_w / width, available_h / height))
        try:
            return max(0.25, min(4.0, float(raw)))
        except ValueError:
            return 1.0

    def make_preview_frame(self, frame: np.ndarray) -> np.ndarray:
        scale = self.preview_scale(frame)
        if abs(scale - 1.0) < 0.01:
            return frame.copy()
        height, width = frame.shape[:2]
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

    def draw_preview_overlay(self, frame: np.ndarray) -> None:
        if self.emotion is None and self.expression_intensity is None:
            return
        rows: list[list[str]] = []
        if self.emotion is not None:
            intensity = f"{self.emotion_intensity:.0%}"
            if self.emotion_baseline_probs:
                intensity = f"{intensity} ({self.emotion_intensity_delta:+.0%})"
            rows.append(["EMOTION SUMMARY"])
            rows.append(["emotion", self.emotion_label, "intensity", intensity])
        if self.emotion is not None and self.args.emotion_debug and self.emotion_probs:
            rows.extend(
                [
                    ["EMOTION PROBABILITY"],
                    ["valence", f"{self.emotion_valence:+.2f}", "delta", f"{self.emotion_valence_delta:+.2f}"],
                    ["positive", f"{self.emotion_positive:.0%}", "negative", f"{self.emotion_negative:.0%}"],
                    ["neutral", f"{self.emotion_probs['neutral']:.0%}", "happy", f"{self.emotion_probs['happy']:.0%}"],
                    ["surprise", f"{self.emotion_probs['surprise']:.0%}", "sad", f"{self.emotion_probs['sad']:.0%}"],
                    ["angry", f"{self.emotion_probs['angry']:.0%}", "disgust", f"{self.emotion_probs['disgust']:.0%}"],
                    ["fear", f"{self.emotion_probs['fear']:.0%}", "contempt", f"{self.emotion_probs['contempt']:.0%}"],
                ]
            )
        if self.expression_intensity is not None:
            rows.append(["LANDMARK INTENSITY"])
            rows.extend(self.format_expression_rows())
        user_rows = self.user_overlay_rows()
        if not rows and not user_rows:
            return
        if user_rows:
            self.draw_overlay_panel(frame, user_rows, "left")
        if rows:
            self.draw_overlay_panel(frame, rows, "right")

    def user_overlay_rows(self) -> list[list[str]]:
        rows: list[list[str]] = [["USER SUMMARY"]]
        if self.emotion is not None:
            rows.append(["emotion", self.emotion_label, "", ""])
            if self.emotion_baseline_stats:
                rows.append(["valence", self.valence_label(self.emotion_valence_sigma), "", ""])
                rows.append(["emotion intensity", self.change_label(self.emotion_intensity_sigma), "", ""])
            else:
                rows.append(["emotion baseline", "collecting", "", ""])
        if self.expression_intensity is not None:
            if not self.expression_scores:
                rows.append(["expression baseline", "waiting", "", ""])
            elif not self.expression_scores.get("ready", 0.0):
                rows.append(["expression baseline", f"{self.expression_scores.get('progress', 0.0):.0%}", "", ""])
            else:
                rows.extend(
                    [
                        ["expression", self.change_label(self.expression_scores.get("overall_sigma", 0.0)), "", ""],
                        ["brow", self.change_label(self.expression_scores.get("brow_sigma", 0.0)), "eye", self.change_label(self.expression_scores.get("eye_sigma", 0.0))],
                        ["mouth", self.change_label(self.expression_scores.get("mouth_sigma", 0.0)), "", ""],
                    ]
                )
        return rows if len(rows) > 1 else []

    def draw_overlay_panel(self, frame: np.ndarray, rows: list[list[str]], side: str) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 21
        col_gap = 26
        normalized_rows = [row + [""] * (4 - len(row)) for row in rows]
        col_widths = [
            max(cv2.getTextSize(row[col], font, font_scale, thickness)[0][0] for row in normalized_rows)
            for col in range(4)
        ]
        panel_w = min(frame.shape[1] - 20, max(300 if side == "left" else 400, sum(col_widths) + col_gap + 32))
        panel_h = 16 + line_height * len(rows)
        x0 = 10 if side == "left" else max(10, frame.shape[1] - panel_w - 10)
        y0 = 10
        col_x = [
            x0 + 12,
            x0 + 12 + col_widths[0] + 12,
            x0 + 12 + col_widths[0] + col_widths[1] + col_gap,
            x0 + 12 + col_widths[0] + col_widths[1] + col_widths[2] + col_gap + 12,
        ]
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        for i, row in enumerate(normalized_rows):
            y = y0 + 23 + i * line_height
            if len(rows[i]) == 1:
                cv2.putText(frame, row[0], (x0 + 12, y), font, font_scale, (160, 255, 255), thickness, cv2.LINE_AA)
                continue
            for text, x in zip(row, col_x):
                if text:
                    cv2.putText(frame, text, (x, y), font, font_scale, (80, 255, 255), thickness, cv2.LINE_AA)

    def handle_frame(self, frame: np.ndarray) -> None:
        self.frame_index += 1
        self.latest_face_detected = False
        self.latest_gaze_xy = (None, None)
        self.latest_pitch_yaw = (None, None)
        self.latest_eye_open = None
        result = self.core.estimate(frame)
        valid = result is not None
        if valid:
            self.latest_face_detected = True
            self.latest_pitch_yaw = (float(result[1][0]), float(result[1][1]))
            self.latest_eye_open = float(result[4])
        if self.calibration_screen:
            elapsed = time.time() - self.target_started
            eye_quality = result[4] if result is not None else 0.0
            valid_sample = valid and eye_quality >= self.args.min_eye_open
            if self.calibration_phase == "expression_baseline":
                if valid_sample:
                    if self.expression_intensity is not None:
                        self.update_expression_intensity(result[5])
                    self.update_emotion_baseline(frame, result[2])
                    self.draw_expression_baseline(True, eye_quality, result[3])
                    if self.neutral_baseline_ready():
                        self.finish_emotion_baseline()
                        self.calibration_phase = "gaze"
                        self.next_target()
                    return
                face = self.core.detect_face(frame)
                self.draw_expression_baseline(False, eye_quality, face[1] if face else "")
                return
            if not valid_sample:
                self.target_started = time.time()
                face = self.core.detect_face(frame)
                self.draw_target(False, eye_quality, frame, face[0] if face else None, face[1] if face else "")
                return
            if elapsed >= self.args.capture_delay:
                self.target_samples.append(result[0])
            self.draw_target(True, eye_quality, frame, result[2], result[3])
            if elapsed >= self.args.min_target_seconds and len(self.target_samples) >= self.args.min_samples_per_target:
                x, y = self.targets[self.target_index]
                samples = self.trim(self.target_samples)
                self.features.extend(samples)
                self.labels.extend([[x, y]] * len(samples))
                self.next_target()
            return
        if not valid:
            self.pointer.hide()
            self.status.set("Face not detected.")
            return
        if result[4] < self.args.min_eye_open:
            self.pointer.hide()
            self.status.set(f"Eyes not open enough: {result[4]:.3f} / {self.args.min_eye_open:.3f}")
            return
        expression = self.update_emotion(frame, result[2])
        self.update_expression_intensity(result[5])
        if not self.has_calibration():
            self.pointer.hide()
            suffix = self.format_expression_status()
            self.status.set(f"No screen calibration. Click Calibrate.{suffix}")
            return
        row = self.design(self.normalize_features(self.select_features(result[0], self.args.feature_mode)))[0]
        x, y = row @ self.weights
        x = int(min(max(x, 0), self.screen_w - 1))
        y = int(min(max(y, 0), self.screen_h - 1))
        self.history.append((x, y))
        if len(self.history) >= 3:
            x = int(np.median([p[0] for p in self.history]))
            y = int(np.median([p[1] for p in self.history]))
        x, y = self.smooth(x, y)
        self.latest_gaze_xy = (x, y)
        self.pointer.move(self.screen_x + x, self.screen_y + y)
        suffix = f" | emotion: {self.format_emotion_status(expression)}" if expression else ""
        suffix += self.format_expression_status()
        self.status.set(f"ETH-XGaze pitch/yaw: {result[1][0]:+.3f}, {result[1][1]:+.3f} ({result[3]}) -> x={x} y={y}{suffix}")

    def update_emotion(self, frame: np.ndarray, face: dlib.rectangle) -> str:
        if self.emotion is None:
            return ""
        now = time.time()
        if (now - self.last_emotion_at) * 1000.0 >= self.args.emotion_interval_ms:
            self.last_emotion_at = now
            label, confidence, probs = self.emotion.predict(frame, face)
            self.emotion_confidence = confidence
            self.emotion_probs = probs
            self.update_emotion_intensity()
            self.emotion_labels.append(label)
            self.emotion_label = Counter(self.emotion_labels).most_common(1)[0][0]
        return self.emotion_label

    def format_emotion_status(self, label: str) -> str:
        if not self.args.emotion_debug or not self.emotion_probs:
            return f"{label}, model intensity {self.format_percent_delta(self.emotion_intensity, self.emotion_intensity_delta)}"
        probs = ", ".join(f"{name} {self.emotion_probs[name]:.0%}" for name in EMOTION_LABELS)
        return (
            f"{label}, model intensity {self.format_percent_delta(self.emotion_intensity, self.emotion_intensity_delta)}, "
            f"valence {self.emotion_valence:+.2f} ({self.emotion_valence_delta:+.2f}), {probs}"
        )

    def update_emotion_intensity(self) -> None:
        neutral = self.emotion_probs.get("neutral", 0.0)
        self.emotion_intensity = max(0.0, min(1.0, 1.0 - neutral))
        self.emotion_positive = self.emotion_probs.get("happy", 0.0)
        self.emotion_negative = sum(self.emotion_probs.get(label, 0.0) for label in NEGATIVE_EMOTION_LABELS)
        self.emotion_valence = self.emotion_positive - self.emotion_negative
        if self.emotion_baseline_stats:
            self.emotion_intensity_delta = self.emotion_intensity - self.emotion_baseline_stats["intensity"][0]
            self.emotion_valence_delta = self.emotion_valence - self.emotion_baseline_stats["valence"][0]
            self.emotion_intensity_sigma = self.emotion_intensity_delta / self.emotion_baseline_stats["intensity"][1]
            self.emotion_valence_sigma = self.emotion_valence_delta / self.emotion_baseline_stats["valence"][1]
        else:
            self.emotion_intensity_delta = 0.0
            self.emotion_valence_delta = 0.0
            self.emotion_intensity_sigma = 0.0
            self.emotion_valence_sigma = 0.0

    def needs_neutral_baseline(self) -> bool:
        return self.expression_intensity is not None or self.emotion is not None

    def reset_emotion_baseline(self) -> None:
        self.emotion_baseline_samples = []
        self.emotion_baseline_probs = {}
        self.emotion_baseline_stats = {}
        self.emotion_intensity_delta = 0.0
        self.emotion_valence_delta = 0.0
        self.emotion_intensity_sigma = 0.0
        self.emotion_valence_sigma = 0.0

    def update_emotion_baseline(self, frame: np.ndarray, face: dlib.rectangle) -> None:
        if self.emotion is None or len(self.emotion_baseline_samples) >= self.args.expression_baseline_frames:
            return
        _label, _confidence, probs = self.emotion.predict(frame, face)
        self.emotion_baseline_samples.append(probs)

    def finish_emotion_baseline(self) -> None:
        if not self.emotion_baseline_samples:
            return
        self.emotion_baseline_probs = {
            label: float(np.median([sample.get(label, 0.0) for sample in self.emotion_baseline_samples]))
            for label in EMOTION_LABELS
        }
        baseline_metrics = [self.emotion_metrics(sample) for sample in self.emotion_baseline_samples]
        self.emotion_baseline_stats = {
            key: (
                float(np.median([metrics[key] for metrics in baseline_metrics])),
                max(float(np.std([metrics[key] for metrics in baseline_metrics])), 1e-3),
            )
            for key in ["intensity", "valence", "positive", "negative"]
        }

    @staticmethod
    def emotion_metrics(probs: dict[str, float]) -> dict[str, float]:
        positive = probs.get("happy", 0.0)
        negative = sum(probs.get(label, 0.0) for label in NEGATIVE_EMOTION_LABELS)
        return {
            "intensity": max(0.0, min(1.0, 1.0 - probs.get("neutral", 0.0))),
            "positive": positive,
            "negative": negative,
            "valence": positive - negative,
        }

    def neutral_baseline_progress(self) -> float:
        progresses = []
        if self.expression_intensity is not None:
            progresses.append(self.expression_scores.get("progress", 1.0 if self.expression_scores.get("ready", 0.0) else 0.0))
        if self.emotion is not None:
            progresses.append(min(1.0, len(self.emotion_baseline_samples) / float(max(1, self.args.expression_baseline_frames))))
        return min(progresses) if progresses else 1.0

    def neutral_baseline_ready(self) -> bool:
        expression_ready = self.expression_intensity is None or bool(self.expression_scores.get("ready", 0.0))
        emotion_ready = self.emotion is None or len(self.emotion_baseline_samples) >= self.args.expression_baseline_frames
        return expression_ready and emotion_ready

    @staticmethod
    def format_percent_delta(value: float, delta: float) -> str:
        return f"{value:.0%} ({delta:+.0%})"

    @staticmethod
    def change_label(sigma: float) -> str:
        value = abs(sigma)
        if value < 1.0:
            return "stable"
        if value < 2.0:
            return "slight"
        if value < 3.0:
            return "moderate"
        return "strong"

    @staticmethod
    def valence_label(sigma: float) -> str:
        if abs(sigma) < 1.0:
            return "stable"
        direction = "more positive" if sigma > 0 else "more negative"
        strength = App.change_label(sigma)
        return f"{strength} {direction}"

    def update_expression_intensity(self, landmarks: np.ndarray) -> None:
        if self.expression_intensity is not None:
            self.expression_scores = self.expression_intensity.update(landmarks)

    def format_expression_status(self) -> str:
        if self.expression_intensity is None or not self.expression_scores:
            return ""
        if not self.expression_scores.get("ready", 0.0):
            return f" | expression baseline {self.expression_scores.get('progress', 0.0):.0%}"
        return f" | intensity {self.expression_scores['overall']:.3f}"

    def format_expression_rows(self) -> list[list[str]]:
        if not self.expression_scores:
            return [["baseline", "waiting", "", ""]]
        if not self.expression_scores.get("ready", 0.0):
            return [["baseline", f"{self.expression_scores.get('progress', 0.0):.0%}", "", ""]]
        return [
            ["overall", self.change_label(self.expression_scores.get("overall_sigma", 0.0)), "sigma", f"{self.expression_scores.get('overall_sigma', 0.0):+.1f}"],
            ["brow", self.change_label(self.expression_scores.get("brow_sigma", 0.0)), "sigma", f"{self.expression_scores.get('brow_sigma', 0.0):+.1f}"],
            ["eye", self.change_label(self.expression_scores.get("eye_sigma", 0.0)), "sigma", f"{self.expression_scores.get('eye_sigma', 0.0):+.1f}"],
            ["mouth", self.change_label(self.expression_scores.get("mouth_sigma", 0.0)), "sigma", f"{self.expression_scores.get('mouth_sigma', 0.0):+.1f}"],
        ]

    def trim(self, samples: list[np.ndarray]) -> list[np.ndarray]:
        x = np.vstack(samples)
        center = np.median(x, axis=0)
        dist = np.linalg.norm(x - center, axis=1)
        cutoff = np.percentile(dist, self.args.robust_percentile)
        kept = x[dist <= cutoff]
        return [row for row in (kept if len(kept) >= self.args.min_samples_per_target else x)]

    def smooth(self, x: int, y: int) -> tuple[int, int]:
        if self.smoothed is None:
            self.smoothed = (float(x), float(y))
        else:
            a = self.args.ema_alpha
            self.smoothed = (a * x + (1 - a) * self.smoothed[0], a * y + (1 - a) * self.smoothed[1])
        return int(self.smoothed[0]), int(self.smoothed[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETH-XGaze webcam gaze pointer prototype.")
    parser.add_argument("--eth-xgaze-dir", default=str(ETH_XGAZE_DIR))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--calibration", default=str(CALIBRATION_FILE))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--screen", default="auto")
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--face-upsample", type=int, default=1)
    parser.add_argument("--calibration-bg", choices=["black", "white"], default="black")
    parser.set_defaults(calibration_camera_overlay=False)
    parser.add_argument("--calibration-camera-overlay", action="store_true")
    parser.add_argument("--no-calibration-camera-overlay", dest="calibration_camera_overlay", action="store_false")
    parser.add_argument("--pointer-size", type=int, default=34)
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--grid-margin", type=float, default=0.08)
    parser.add_argument("--min-samples-per-target", type=int, default=18)
    parser.add_argument("--capture-delay", type=float, default=0.65)
    parser.add_argument("--min-target-seconds", type=float, default=1.8)
    parser.add_argument("--max-target-seconds", type=float, default=4.0)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--ema-alpha", type=float, default=0.60)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--robust-percentile", type=float, default=80)
    parser.add_argument("--feature-mode", choices=["gaze", "hybrid", "gaze_head"], default="hybrid")
    parser.add_argument("--head-weight", type=float, default=0.25)
    parser.add_argument("--min-eye-open", type=float, default=0.16)
    parser.set_defaults(emotion=False)
    parser.add_argument("--emotion", action="store_true")
    parser.add_argument("--no-emotion", dest="emotion", action="store_false")
    parser.add_argument("--emotion-model", default=str(EMOTION_MODEL))
    parser.add_argument("--emotion-interval-ms", type=int, default=150)
    parser.add_argument("--emotion-window", type=int, default=1)
    parser.add_argument("--emotion-debug", action="store_true")
    parser.set_defaults(expression_intensity=False)
    parser.add_argument("--expression-intensity", action="store_true")
    parser.add_argument("--no-expression-intensity", dest="expression_intensity", action="store_false")
    parser.add_argument("--expression-baseline-frames", type=int, default=45)
    parser.add_argument("--expression-debug", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.set_defaults(no_preview=True)
    parser.add_argument("--preview", dest="no_preview", action="store_false")
    parser.add_argument("--no-preview", dest="no_preview", action="store_true")
    parser.add_argument("--preview-scale", default="auto")
    parser.set_defaults(record_preview=False)
    parser.add_argument("--record-preview", action="store_true")
    parser.add_argument("--no-record-preview", dest="record_preview", action="store_false")
    parser.set_defaults(record_metrics=False)
    parser.add_argument("--record-metrics", action="store_true")
    parser.add_argument("--no-record-metrics", dest="record_metrics", action="store_false")
    parser.add_argument("--metrics-file", default=str(METRICS_FILE))
    parser.add_argument("--metrics-interval-ms", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    App(parse_args()).run()
