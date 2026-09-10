from __future__ import annotations

import argparse
import ctypes
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
MODEL_VERSION = 4
PREVIEW_WINDOW = "ETH-XGaze camera preview"


def screen_size(root: tk.Tk) -> tuple[int, int]:
    if sys.platform == "win32":
        return ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    return root.winfo_screenwidth(), root.winfo_screenheight()


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
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
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

    @staticmethod
    def expression_label(landmarks: np.ndarray, eye_quality: float) -> str:
        face_width = max(float(np.linalg.norm(landmarks[0] - landmarks[16])), 1e-6)
        mouth_width = float(np.linalg.norm(landmarks[48] - landmarks[54])) / face_width
        mouth_open = float(np.linalg.norm(landmarks[62] - landmarks[66])) / max(float(np.linalg.norm(landmarks[48] - landmarks[54])), 1e-6)
        if mouth_open > 0.24:
            return "mouth open"
        if mouth_width > 0.36 and mouth_open < 0.14:
            return "smile"
        if eye_quality < 0.20:
            return "squint"
        return "neutral"

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


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("ETH-XGaze Prototype")
        self.root.geometry("420x180")
        self.screen_w, self.screen_h = screen_size(self.root)
        self.capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
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
        self.status = tk.StringVar(value="Calibrate first.")
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = self.load_calibration()
        if self.weights is not None:
            self.status.set("ETH-XGaze calibration loaded.")
        tk.Label(self.root, text="ETH-XGaze Prototype", font=("Segoe UI", 13, "bold")).pack(pady=(12, 4))
        tk.Label(self.root, textvariable=self.status, wraplength=380).pack()
        tk.Button(self.root, text="Calibrate", command=self.start_calibration).pack(side=tk.LEFT, padx=32, pady=18)
        tk.Button(self.root, text="Quit", command=self.stop).pack(side=tk.RIGHT, padx=32, pady=18)
        self.calibration_screen: Optional[tk.Toplevel] = None
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
        self.camera_image = None
        if args.calibrate:
            self.root.after(500, self.start_calibration)
        self.root.after(10, self.tick)
        self.root.bind_all("<Escape>", self.handle_escape)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    def run(self) -> None:
        self.root.mainloop()

    def stop(self) -> None:
        self.pointer.hide()
        self.capture.release()
        if not self.args.no_preview:
            cv2.destroyAllWindows()
        self.root.destroy()

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

    def load_calibration(self) -> Optional[np.ndarray]:
        path = Path(self.args.calibration)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("version") != MODEL_VERSION
            or data.get("screen_size") != [self.screen_w, self.screen_h]
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
        self.calibration_screen.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
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
        self.next_target()
        self.calibration_screen.lift()
        self.calibration_screen.focus_force()
        self.calibration_screen.update_idletasks()
        self.calibration_screen.update()

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
        ok, frame = self.capture.read()
        if ok:
            frame = cv2.flip(frame, 1)
            self.handle_frame(frame)
            if not self.args.no_preview and not self.calibration_screen:
                cv2.imshow(PREVIEW_WINDOW, frame)
                cv2.waitKey(1)
        self.root.after(16, self.tick)

    def handle_frame(self, frame: np.ndarray) -> None:
        result = self.core.estimate(frame)
        valid = result is not None
        if self.calibration_screen:
            elapsed = time.time() - self.target_started
            eye_quality = result[4] if result is not None else 0.0
            valid_sample = valid and eye_quality >= self.args.min_eye_open
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
        if not self.has_calibration():
            self.pointer.hide()
            self.status.set("No screen calibration. Click Calibrate.")
            return
        expression = self.update_emotion(result[5], result[4])
        row = self.design(self.normalize_features(self.select_features(result[0], self.args.feature_mode)))[0]
        x, y = row @ self.weights
        x = int(min(max(x, 0), self.screen_w - 1))
        y = int(min(max(y, 0), self.screen_h - 1))
        self.history.append((x, y))
        if len(self.history) >= 3:
            x = int(np.median([p[0] for p in self.history]))
            y = int(np.median([p[1] for p in self.history]))
        x, y = self.smooth(x, y)
        self.pointer.move(x, y)
        suffix = f" | expression: {expression}" if expression else ""
        self.status.set(f"ETH-XGaze pitch/yaw: {result[1][0]:+.3f}, {result[1][1]:+.3f} ({result[3]}) -> x={x} y={y}{suffix}")

    def update_emotion(self, landmarks: np.ndarray, eye_quality: float) -> str:
        if not self.args.emotion:
            return ""
        now = time.time()
        if (now - self.last_emotion_at) * 1000.0 >= self.args.emotion_interval_ms:
            self.last_emotion_at = now
            self.emotion_labels.append(self.core.expression_label(landmarks, eye_quality))
            self.emotion_label = Counter(self.emotion_labels).most_common(1)[0][0]
        return self.emotion_label

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
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--camera", type=int, default=0)
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
    parser.add_argument("--emotion-interval-ms", type=int, default=200)
    parser.add_argument("--emotion-window", type=int, default=5)
    parser.add_argument("--calibrate", action="store_true")
    parser.set_defaults(no_preview=True)
    parser.add_argument("--preview", dest="no_preview", action="store_false")
    parser.add_argument("--no-preview", dest="no_preview", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    App(parse_args()).run()
