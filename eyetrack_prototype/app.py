from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


CALIBRATION_FILE = Path("gaze_calibration.json")
MODEL_FILE = Path("models/face_landmarker.task")


@dataclass
class GazeSample:
    features: np.ndarray
    frame: np.ndarray
    quality: float


class GazeEstimator:
    def __init__(self, camera_index: int, width: int, height: int, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing {model_path}. Run download_model.py first or pass --model."
            )
        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.weights_x: Optional[np.ndarray] = None
        self.weights_y: Optional[np.ndarray] = None
        self.smoothing = 0.12
        self.smoothed: Optional[tuple[float, float]] = None
        self.prediction_history: deque[tuple[float, float]] = deque(maxlen=5)
        self.timestamp_ms = 0

    def close(self) -> None:
        self.capture.release()
        self.landmarker.close()
        cv2.destroyAllWindows()

    def load_calibration(self, path: Path, screen_w: int, screen_h: int) -> bool:
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("screen_size") != [screen_w, screen_h]:
            return False
        self.weights_x = np.array(data["weights_x"], dtype=np.float32)
        self.weights_y = np.array(data["weights_y"], dtype=np.float32)
        return True

    def save_calibration(self, path: Path, screen_w: int, screen_h: int) -> None:
        if self.weights_x is None or self.weights_y is None:
            return
        data = {
            "weights_x": self.weights_x.tolist(),
            "weights_y": self.weights_y.tolist(),
            "screen_size": [screen_w, screen_h],
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def calibrate(self, raw_features: list[np.ndarray], targets: list[tuple[int, int]]) -> None:
        x = self._design_matrix(raw_features)
        y = np.array(targets, dtype=np.float32)
        ridge = 1e-3 * np.eye(x.shape[1], dtype=np.float32)
        self.weights_x = np.linalg.solve(x.T @ x + ridge, x.T @ y[:, 0])
        self.weights_y = np.linalg.solve(x.T @ x + ridge, x.T @ y[:, 1])
        self.smoothed = None
        self.prediction_history.clear()

    def read(self) -> Optional[GazeSample]:
        ok, frame = self.capture.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp_ms += 33
        result = self.landmarker.detect_for_video(image, self.timestamp_ms)
        if not result.face_landmarks:
            return GazeSample(np.zeros(12, dtype=np.float32), frame, 0.0)
        landmarks = result.face_landmarks[0]
        features, quality = self._features(landmarks)
        self._draw_debug(frame, landmarks)
        return GazeSample(features, frame, quality)

    def predict(self, features: np.ndarray, screen_w: int, screen_h: int) -> Optional[tuple[int, int]]:
        if self.weights_x is None or self.weights_y is None or not features.any():
            return None
        row = self._design_matrix([features])[0]
        x = float(row @ self.weights_x)
        y = float(row @ self.weights_y)
        x = min(max(x, 0), screen_w - 1)
        y = min(max(y, 0), screen_h - 1)
        self.prediction_history.append((x, y))
        xs = [point[0] for point in self.prediction_history]
        ys = [point[1] for point in self.prediction_history]
        x = float(np.median(xs))
        y = float(np.median(ys))
        if self.smoothed is None:
            self.smoothed = (x, y)
        else:
            px, py = self.smoothed
            a = self.smoothing
            self.smoothed = (px + (x - px) * a, py + (y - py) * a)
        return int(self.smoothed[0]), int(self.smoothed[1])

    @staticmethod
    def _features(landmarks) -> tuple[np.ndarray, float]:
        left_iris = _mean_xy(landmarks, range(468, 473))
        right_iris = _mean_xy(landmarks, range(473, 478))
        iris = (left_iris + right_iris) / 2.0

        left_eye_center = (_xy(landmarks, 33) + _xy(landmarks, 133)) / 2.0
        right_eye_center = (_xy(landmarks, 362) + _xy(landmarks, 263)) / 2.0
        eye_center = (left_eye_center + right_eye_center) / 2.0
        face_eye_span = np.linalg.norm(_xy(landmarks, 33) - _xy(landmarks, 263))
        eye_span = max(float(face_eye_span), 0.001)
        left_eye_w = max(float(np.linalg.norm(_xy(landmarks, 33) - _xy(landmarks, 133))), 0.001)
        right_eye_w = max(float(np.linalg.norm(_xy(landmarks, 362) - _xy(landmarks, 263))), 0.001)
        left_open = float(np.linalg.norm(_xy(landmarks, 159) - _xy(landmarks, 145)) / left_eye_w)
        right_open = float(np.linalg.norm(_xy(landmarks, 386) - _xy(landmarks, 374)) / right_eye_w)
        quality = min(left_open, right_open)

        nose = _xy(landmarks, 1)
        chin = _xy(landmarks, 152)
        forehead = _xy(landmarks, 10)
        face_h = max(float(np.linalg.norm(chin - forehead)), 0.001)
        roll = math.atan2(
            float(_xy(landmarks, 263)[1] - _xy(landmarks, 33)[1]),
            float(_xy(landmarks, 263)[0] - _xy(landmarks, 33)[0]),
        )

        left_rel = (left_iris - left_eye_center) / eye_span
        right_rel = (right_iris - right_eye_center) / eye_span
        rel = (iris - eye_center) / eye_span
        features = np.array(
            [
                rel[0],
                rel[1],
                left_rel[0],
                left_rel[1],
                right_rel[0],
                right_rel[1],
                iris[0],
                iris[1],
                nose[0],
                nose[1],
                face_h,
                roll,
            ],
            dtype=np.float32,
        )
        return features, quality

    @staticmethod
    def _design_matrix(raw_features: list[np.ndarray]) -> np.ndarray:
        x = np.vstack(raw_features).astype(np.float32)
        core = x[:, :6]
        return np.hstack([np.ones((x.shape[0], 1), dtype=np.float32), x, core * core])

    @staticmethod
    def _draw_debug(frame: np.ndarray, landmarks) -> None:
        h, w = frame.shape[:2]
        for idx in list(range(468, 478)) + [33, 133, 362, 263, 1]:
            p = landmarks[idx]
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 255, 0), -1)


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
        pad = 3
        canvas.create_oval(pad, pad, size - pad, size - pad, fill="red", outline="white", width=2)
        canvas.create_oval(size / 2 - 3, size / 2 - 3, size / 2 + 3, size / 2 + 3, fill="white", outline="")
        self.hide()

    def move(self, x: int, y: int) -> None:
        self.window.geometry(f"{self.size}x{self.size}+{x - self.size // 2}+{y - self.size // 2}")
        self.window.deiconify()

    def hide(self) -> None:
        self.window.withdraw()


class EyeTrackApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = tk.Tk()
        self.root.title("EyeTrax Prototype")
        self.root.geometry("360x160")
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.estimator = GazeEstimator(args.camera, args.width, args.height, Path(args.model))
        self.pointer = PointerWindow(self.root, args.pointer_size)
        self.show_preview = not args.no_preview
        self.min_eye_quality = args.min_eye_quality
        self.calibration_path = Path(args.calibration)
        loaded = self.estimator.load_calibration(self.calibration_path, self.screen_w, self.screen_h)

        self.status = tk.StringVar(value="Calibration loaded." if loaded else "Calibrate first.")
        tk.Label(self.root, text="EyeTrax Prototype", font=("Segoe UI", 13, "bold")).pack(pady=(12, 4))
        tk.Label(self.root, textvariable=self.status, wraplength=330).pack()
        tk.Button(self.root, text="Calibrate", command=self.start_calibration).pack(side=tk.LEFT, padx=24, pady=18)
        tk.Button(self.root, text="Quit", command=self.stop).pack(side=tk.RIGHT, padx=24, pady=18)

        self.calibration_targets: list[tuple[int, int]] = []
        self.calibration_features: list[np.ndarray] = []
        self.calibration_sample_targets: list[tuple[int, int]] = []
        self.calibration_screen: Optional[tk.Toplevel] = None
        self.target_canvas: Optional[tk.Canvas] = None
        self.target_index = -1
        self.target_started = 0.0
        self.target_samples: list[np.ndarray] = []
        self.last_calibration_feature: Optional[np.ndarray] = None
        self.min_samples_per_target = 6

        if args.calibrate:
            self.root.after(500, self.start_calibration)
        self.root.after(10, self.tick)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    def run(self) -> None:
        self.root.mainloop()

    def stop(self) -> None:
        self.pointer.hide()
        self.estimator.close()
        self.root.destroy()

    def start_calibration(self) -> None:
        margin_x = int(self.screen_w * 0.07)
        margin_y = int(self.screen_h * 0.07)
        xs = np.linspace(margin_x, self.screen_w - margin_x, 5, dtype=int)
        ys = np.linspace(margin_y, self.screen_h - margin_y, 5, dtype=int)
        self.calibration_targets = [(int(x), int(y)) for y in ys for x in xs]
        center = (self.screen_w // 2, self.screen_h // 2)
        if center in self.calibration_targets:
            self.calibration_targets.remove(center)
        self.calibration_targets.insert(0, center)
        self.calibration_features = []
        self.calibration_sample_targets = []
        self.target_index = -1
        self.pointer.hide()
        self.calibration_screen = tk.Toplevel(self.root)
        self.calibration_screen.attributes("-fullscreen", True)
        self.calibration_screen.attributes("-topmost", True)
        self.calibration_screen.configure(bg="black")
        self.target_canvas = tk.Canvas(self.calibration_screen, bg="black", highlightthickness=0)
        self.target_canvas.pack(fill=tk.BOTH, expand=True)
        self.next_target()

    def next_target(self) -> None:
        self.target_index += 1
        self.target_samples = []
        self.last_calibration_feature = None
        self.target_started = time.time()
        if self.target_index >= len(self.calibration_targets):
            self.finish_calibration()
            return
        self.draw_target()

    def draw_target(self, detected: bool = False, elapsed: float = 0.0, quality: float = 0.0) -> None:
        if not self.target_canvas:
            return
        self.target_canvas.delete("all")
        x, y = self.calibration_targets[self.target_index]
        r = 18
        self.target_canvas.create_oval(x - r, y - r, x + r, y + r, fill="red", outline="white", width=3)
        self.target_canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="white", outline="")
        self.target_canvas.create_text(
            self.screen_w // 2,
            70,
            text=f"Look at the red dot: {self.target_index + 1}/{len(self.calibration_targets)}",
            fill="white",
            font=("Segoe UI", 22, "bold"),
        )
        captured = "none"
        if self.last_calibration_feature is not None:
            captured = f"{self.last_calibration_feature[0]:+.5f}, {self.last_calibration_feature[1]:+.5f}"
        collect_state = "COLLECTING" if elapsed > 0.65 and detected else "WAITING"
        detect_state = "eyes open" if detected else "no eyes / blink rejected"
        self.target_canvas.create_text(
            26,
            self.screen_h - 150,
            text=(
                f"target screen x,y: {x}, {y}\n"
                f"captured feature x,y: {captured}\n"
                f"eye-open quality: {quality:.3f} / min {self.min_eye_quality:.3f}\n"
                f"samples for this dot: {len(self.target_samples)}\n"
                f"{collect_state} - {detect_state}"
            ),
            fill="#6ee7ff" if detected else "#ff7777",
            font=("Consolas", 18, "bold"),
            anchor="w",
        )

    def finish_calibration(self) -> None:
        if self.calibration_screen:
            self.calibration_screen.destroy()
            self.calibration_screen = None
        if len(self.calibration_features) >= len(self.calibration_targets) * self.min_samples_per_target:
            self.estimator.calibrate(self.calibration_features, self.calibration_sample_targets)
            self.estimator.save_calibration(self.calibration_path, self.screen_w, self.screen_h)
            self.status.set(f"Calibration saved to {self.calibration_path}.")
        else:
            self.status.set("Calibration failed: face/eyes were not detected enough.")

    def tick(self) -> None:
        sample = self.estimator.read()
        if sample is not None:
            self.handle_sample(sample)
            if self.show_preview:
                cv2.imshow("EyeTrax camera preview", sample.frame)
                cv2.waitKey(1)
        self.root.after(16, self.tick)

    def handle_sample(self, sample: GazeSample) -> None:
        if self.calibration_screen:
            elapsed = time.time() - self.target_started
            has_face = bool(sample.features.any())
            detected = has_face and sample.quality >= self.min_eye_quality
            if has_face:
                self.last_calibration_feature = sample.features
            if elapsed > 0.65 and detected:
                self.target_samples.append(sample.features)
            self.draw_target(detected, elapsed, sample.quality)
            enough_samples = len(self.target_samples) >= self.min_samples_per_target
            timed_out = elapsed > 3.0
            if elapsed > 1.65 and (enough_samples or timed_out):
                if self.target_samples:
                    x, y = self.calibration_targets[self.target_index]
                    self.calibration_features.extend(self.target_samples)
                    self.calibration_sample_targets.extend([(x, y)] * len(self.target_samples))
                self.next_target()
            return

        if sample.quality < self.min_eye_quality:
            self.status.set(f"Blink/eyes not open enough: quality={sample.quality:.3f}")
            return

        point = self.estimator.predict(sample.features, self.screen_w, self.screen_h)
        if point:
            self.pointer.move(*point)
            self.status.set(f"Gaze: x={point[0]} y={point[1]}")
        else:
            self.pointer.hide()
            self.status.set("No calibration or no face detected.")


def _xy(landmarks, idx: int) -> np.ndarray:
    p = landmarks[idx]
    return np.array([p.x, p.y], dtype=np.float32)


def _mean_xy(landmarks, indexes) -> np.ndarray:
    return np.mean([_xy(landmarks, idx) for idx in indexes], axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local webcam gaze pointer prototype.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--pointer-size", type=int, default=34)
    parser.add_argument("--calibration", default=str(CALIBRATION_FILE))
    parser.add_argument("--model", default=str(MODEL_FILE))
    parser.add_argument("--min-eye-quality", type=float, default=0.16)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    EyeTrackApp(parse_args()).run()
