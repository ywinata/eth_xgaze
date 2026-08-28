from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from eyetrax.filters import KalmanEMASmoother, KalmanSmoother, NoSmoother, make_kalman
from eyetrax.gaze import GazeEstimator


MODEL_FILE = Path("models/face_landmarker.task")
GAZE_MODEL_FILE = Path("eyetrax_gaze_model.pkl")
MODEL_VERSION = 3


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


class EyeTraxPrototype:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("EyeTrax2 Prototype")
        self.root.geometry("390x170")
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        self.estimator = GazeEstimator(
            model_name=args.model,
            face_landmarker_model=args.face_model,
            blink_threshold_ratio=args.blink_threshold_ratio,
        )
        self.model_meta_file = Path(args.gaze_model).with_suffix(".json")
        loaded = self._can_load_model(args.gaze_model)
        if loaded:
            self.estimator.load_model(args.gaze_model)
        self.model_ready = loaded

        self.smoother = self._make_smoother(args.filter, args.ema_alpha)
        self.prediction_history: deque[tuple[int, int]] = deque(maxlen=args.median_window)
        self.last_pointer: Optional[tuple[int, int]] = None
        self.capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        self.pointer = PointerWindow(self.root, args.pointer_size)
        self.status = tk.StringVar(value="EyeTrax model loaded." if loaded else "Calibrate first.")
        tk.Label(self.root, text="EyeTrax2 Prototype", font=("Segoe UI", 13, "bold")).pack(pady=(12, 4))
        tk.Label(self.root, textvariable=self.status, wraplength=360).pack()
        tk.Button(self.root, text="Calibrate", command=self.start_calibration).pack(side=tk.LEFT, padx=28, pady=18)
        tk.Button(self.root, text="Quit", command=self.stop).pack(side=tk.RIGHT, padx=28, pady=18)

        self.calibration_screen: Optional[tk.Toplevel] = None
        self.target_canvas: Optional[tk.Canvas] = None
        self.targets: list[tuple[int, int]] = []
        self.target_index = -1
        self.target_started = 0.0
        self.target_samples: list[np.ndarray] = []
        self.features: list[np.ndarray] = []
        self.labels: list[list[int]] = []
        self.last_feature: Optional[np.ndarray] = None
        self.min_samples_per_target = args.min_samples_per_target

        if args.calibrate:
            self.root.after(500, self.start_calibration)
        self.root.after(10, self.tick)
        self.root.bind_all("<Escape>", self.handle_escape)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    @staticmethod
    def _make_smoother(name: str, ema_alpha: float):
        if name == "kalman":
            return KalmanSmoother(make_kalman())
        if name == "kalman_ema":
            return KalmanEMASmoother(make_kalman(), ema_alpha=ema_alpha)
        return NoSmoother()

    def _can_load_model(self, gaze_model: str) -> bool:
        meta_path = Path(gaze_model).with_suffix(".json")
        if not Path(gaze_model).exists() or not meta_path.exists():
            return False
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data.get("version") == MODEL_VERSION and data.get("screen_size") == [self.screen_w, self.screen_h]

    def run(self) -> None:
        self.root.mainloop()

    def stop(self) -> None:
        self.pointer.hide()
        self.capture.release()
        self.estimator.close()
        cv2.destroyAllWindows()
        self.root.destroy()

    def handle_escape(self, _event=None) -> None:
        if self.calibration_screen:
            self.cancel_calibration()
        else:
            self.stop()

    def start_calibration(self) -> None:
        self.targets = self.build_targets()
        center = (self.screen_w // 2, self.screen_h // 2)
        if center in self.targets:
            self.targets.remove(center)
        self.targets.insert(0, center)

        self.features = []
        self.labels = []
        self.prediction_history.clear()
        self.last_pointer = None
        self.pointer.hide()
        self.calibration_screen = tk.Toplevel(self.root)
        self.calibration_screen.attributes("-fullscreen", True)
        self.calibration_screen.attributes("-topmost", True)
        self.calibration_screen.configure(bg="black")
        self.calibration_screen.focus_force()
        self.target_canvas = tk.Canvas(self.calibration_screen, bg="black", highlightthickness=0)
        self.target_canvas.pack(fill=tk.BOTH, expand=True)
        self.target_index = -1
        self.next_target()

    def cancel_calibration(self, _event=None) -> None:
        if not self.calibration_screen:
            return
        self.calibration_screen.destroy()
        self.calibration_screen = None
        self.target_canvas = None
        self.targets = []
        self.target_samples = []
        self.features = []
        self.labels = []
        self.last_feature = None
        self.status.set("Calibration cancelled with Esc.")

    def build_targets(self) -> list[tuple[int, int]]:
        mx = int(self.screen_w * self.args.grid_margin)
        my = int(self.screen_h * self.args.grid_margin)
        xs = np.linspace(mx, self.screen_w - mx, self.args.grid_cols, dtype=int)
        ys = np.linspace(my, self.screen_h - my, self.args.grid_rows, dtype=int)
        targets = {(int(x), int(y)) for y in ys for x in xs}
        if not self.args.no_edge_targets:
            ex = int(self.screen_w * self.args.edge_margin)
            ey = int(self.screen_h * self.args.edge_margin)
            edge_xs = np.linspace(ex, self.screen_w - ex, self.args.grid_cols, dtype=int)
            edge_ys = np.linspace(ey, self.screen_h - ey, self.args.grid_rows, dtype=int)
            for x in edge_xs:
                targets.add((int(x), ey))
                targets.add((int(x), self.screen_h - ey))
            for y in edge_ys:
                targets.add((ex, int(y)))
                targets.add((self.screen_w - ex, int(y)))
        return sorted(targets, key=lambda p: (p[1], p[0]))

    def next_target(self) -> None:
        self.target_index += 1
        self.target_samples = []
        self.last_feature = None
        self.target_started = time.time()
        if self.target_index >= len(self.targets):
            self.finish_calibration()
            return
        self.draw_target(False, 0.0)

    def draw_target(self, valid: bool, elapsed: float, blink: bool = False) -> None:
        if not self.target_canvas:
            return
        self.target_canvas.delete("all")
        x, y = self.targets[self.target_index]
        r = 18
        self.target_canvas.create_oval(x - r, y - r, x + r, y + r, fill="red", outline="white", width=3)
        self.target_canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="white", outline="")
        self.target_canvas.create_text(
            self.screen_w // 2,
            70,
            text=f"Look at the red dot: {self.target_index + 1}/{len(self.targets)}",
            fill="white",
            font=("Segoe UI", 22, "bold"),
        )
        captured = "none"
        if self.last_feature is not None:
            captured = f"{self.last_feature[0]:+.5f}, {self.last_feature[1]:+.5f}"
        state = "COLLECTING" if elapsed > self.args.capture_delay and valid else "WAITING"
        reason = "ok" if valid else ("blink rejected" if blink else "face not detected")
        self.target_canvas.create_text(
            26,
            self.screen_h - 150,
            text=(
                f"target screen x,y: {x}, {y}\n"
                f"EyeTrax feature[0:2]: {captured}\n"
                f"samples for this dot: {len(self.target_samples)} / {self.min_samples_per_target}\n"
                f"{state} - {reason}"
            ),
            fill="#6ee7ff" if valid else "#ff7777",
            font=("Consolas", 18, "bold"),
            anchor="w",
        )

    def finish_calibration(self) -> None:
        if self.calibration_screen:
            self.calibration_screen.destroy()
            self.calibration_screen = None
        needed = len(self.targets) * self.min_samples_per_target
        if len(self.features) < needed:
            self.status.set(f"Calibration failed: {len(self.features)}/{needed} valid samples.")
            return
        self.estimator.train(np.array(self.features), np.array(self.labels))
        self.estimator.save_model(self.args.gaze_model)
        self.model_meta_file.write_text(
            json.dumps(
                {
                    "version": MODEL_VERSION,
                    "screen_size": [self.screen_w, self.screen_h],
                    "model": self.args.model,
                    "feature_mode": "eyetrax_raw",
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.model_ready = True
        self.smoother = self._make_smoother(self.args.filter, self.args.ema_alpha)
        self.status.set(f"EyeTrax calibration saved to {self.args.gaze_model}.")

    def robust_target_samples(self, samples: list[np.ndarray]) -> list[np.ndarray]:
        if not samples:
            return []
        x = np.vstack(samples)
        median = np.median(x, axis=0)
        dist = np.linalg.norm(x - median, axis=1)
        cutoff = np.percentile(dist, self.args.robust_percentile)
        kept = x[dist <= cutoff]
        if len(kept) < self.min_samples_per_target:
            kept = x
        return [row for row in kept]

    def read_frame(self):
        ok, frame = self.capture.read()
        if not ok:
            return None
        return cv2.flip(frame, 1)

    def tick(self) -> None:
        frame = self.read_frame()
        if frame is not None:
            self.handle_frame(frame)
            if not self.args.no_preview:
                cv2.imshow("EyeTrax2 camera preview", frame)
                cv2.waitKey(1)
        self.root.after(16, self.tick)

    def handle_frame(self, frame) -> None:
        features, blink = self.estimator.extract_features(frame)
        valid = features is not None and not blink

        if self.calibration_screen:
            elapsed = time.time() - self.target_started
            if features is not None:
                self.last_feature = features
            if elapsed > self.args.capture_delay and valid:
                self.target_samples.append(features)
            self.draw_target(valid, elapsed, blink)
            enough_samples = len(self.target_samples) >= self.min_samples_per_target
            timed_out = elapsed > self.args.max_target_seconds
            if elapsed > self.args.min_target_seconds and (enough_samples or timed_out):
                if self.target_samples:
                    x, y = self.targets[self.target_index]
                    samples = self.robust_target_samples(self.target_samples)
                    self.features.extend(samples)
                    self.labels.extend([[x, y]] * len(samples))
                self.next_target()
            return

        if not valid:
            self.pointer.hide()
            self.status.set("Blink/face not detected.")
            return
        if not self.model_ready:
            self.pointer.hide()
            self.status.set("No trained gaze model yet. Calibrate first.")
            return

        x, y = self.estimator.predict(np.array([features]))[0]
        x = min(max(int(x), 0), self.screen_w - 1)
        y = min(max(int(y), 0), self.screen_h - 1)
        if self.last_pointer:
            dx = x - self.last_pointer[0]
            dy = y - self.last_pointer[1]
            if float(np.hypot(dx, dy)) > self.args.max_gaze_jump:
                self.status.set(f"Rejected noisy jump: x={x} y={y}")
                return
        self.prediction_history.append((x, y))
        if len(self.prediction_history) >= 3:
            x = int(np.median([p[0] for p in self.prediction_history]))
            y = int(np.median([p[1] for p in self.prediction_history]))
        x, y = self.smoother.step(x, y)
        self.last_pointer = (x, y)
        self.pointer.move(x, y)
        self.status.set(f"EyeTrax gaze: x={x} y={y}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EyeTrax-based local gaze pointer prototype.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--pointer-size", type=int, default=34)
    parser.add_argument("--face-model", default=str(MODEL_FILE))
    parser.add_argument("--gaze-model", default=str(GAZE_MODEL_FILE))
    parser.add_argument("--model", choices=["ridge", "elastic_net", "svr", "tiny_mlp"], default="ridge")
    parser.add_argument("--filter", choices=["kalman", "kalman_ema", "none"], default="kalman_ema")
    parser.add_argument("--ema-alpha", type=float, default=0.65)
    parser.add_argument("--grid-rows", type=int, default=7)
    parser.add_argument("--grid-cols", type=int, default=7)
    parser.add_argument("--grid-margin", type=float, default=0.06)
    parser.add_argument("--edge-margin", type=float, default=0.03)
    parser.add_argument("--no-edge-targets", action="store_true")
    parser.add_argument("--min-samples-per-target", type=int, default=24)
    parser.add_argument("--capture-delay", type=float, default=0.65)
    parser.add_argument("--min-target-seconds", type=float, default=2.2)
    parser.add_argument("--max-target-seconds", type=float, default=4.0)
    parser.add_argument("--blink-threshold-ratio", type=float, default=0.8)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--max-gaze-jump", type=float, default=520)
    parser.add_argument("--robust-percentile", type=float, default=80)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    EyeTraxPrototype(parse_args()).run()
