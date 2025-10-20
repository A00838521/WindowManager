#!/usr/bin/env python3
"""
Control de ventanas en macOS con gestos de mano (MediaPipe Tasks)

Versión macOS: usa pywinctl y un overlay Cocoa (overlay_helper.py) para resaltar.
"""
import os
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
import subprocess
import json

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    import pywinctl as pwc
except Exception as e:
    raise RuntimeError(
        "pywinctl no está disponible. Instala las dependencias con 'pip install -r requirements.txt'."
    ) from e

INDEX_TIP = 8
THUMB_TIP = 4
PALM_REF_A = 5
PALM_REF_B = 17

def _dist2d(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def is_pinch(hand_landmarks, pinch_ratio_thresh: float = 0.40) -> bool:
    try:
        idx_tip = hand_landmarks[INDEX_TIP]
        th_tip = hand_landmarks[THUMB_TIP]
        palm_a = hand_landmarks[PALM_REF_A]
        palm_b = hand_landmarks[PALM_REF_B]
    except Exception:
        return False
    palm_w = _dist2d(palm_a, palm_b) + 1e-6
    pinch_d = _dist2d(idx_tip, th_tip)
    return (pinch_d / palm_w) < pinch_ratio_thresh

def landmark_xy_norm(hand_landmarks, landmark_id=INDEX_TIP):
    lmk = hand_landmarks[landmark_id]
    return float(lmk.x), float(lmk.y)

@dataclass
class DragState:
    active: bool = False
    window: Optional[pwc.Window] = None
    window_size: Optional[Tuple[int, int]] = None
    offset: Optional[Tuple[int, int]] = None
    last_pos: Optional[Tuple[int, int]] = None

def get_screen_size() -> Tuple[int, int]:
    try:
        return pwc.getScreenSize()
    except Exception:
        import AppKit
        vf = AppKit.NSScreen.mainScreen().visibleFrame()
        return int(vf.size.width), int(vf.size.height)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def snap_window(win: pwc.Window, screen_w: int, screen_h: int, x: int, y: int):
    margin = int(0.1 * screen_h)
    side_margin = int(0.12 * screen_w)
    if y <= margin:
        try:
            win.resizeTo(screen_w, screen_h)
            win.moveTo(0, 0)
        except Exception:
            pass
        return
    if x <= side_margin:
        try:
            win.resizeTo(screen_w // 2, screen_h)
            win.moveTo(0, 0)
        except Exception:
            pass
        return
    if x >= (screen_w - side_margin):
        try:
            win.resizeTo(screen_w // 2, screen_h)
            win.moveTo(screen_w // 2, 0)
        except Exception:
            pass

def window_contains(win: pwc.Window, x: int, y: int) -> bool:
    try:
        (wx, wy) = win.topleft
        (w, h) = win.size
        return wx <= x < wx + w and wy <= y < wy + h
    except Exception:
        return False

def find_window_at_point(x: int, y: int) -> Optional[pwc.Window]:
    try:
        if hasattr(pwc, "getWindowAt"):
            win = pwc.getWindowAt(x, y)  # type: ignore[attr-defined]
            if win:
                return win
    except Exception:
        pass
    try:
        active = pwc.getActiveWindow()
    except Exception:
        active = None
    if active and window_contains(active, x, y):
        return active
    try:
        wins = pwc.getAllWindows()
        candidates = [w for w in wins if getattr(w, 'isVisible', True) and window_contains(w, x, y)]
        if not candidates:
            return None
        for w in candidates:
            if getattr(w, 'isActive', False):
                return w
        return candidates[0]
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Control de ventanas por gestos de mano (macOS)")
    parser.add_argument("--model", default="hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=-1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--inference-scale", type=float, default=0.5)
    parser.add_argument("--flip", type=int, default=1, choices=[0, 1])
    parser.add_argument("--invert-y", action="store_true")
    parser.add_argument("--backend", default="auto", choices=["auto", "avf", "any"]) 
    parser.add_argument("--pinch-thresh", type=float, default=0.40)
    parser.add_argument("--drag-fps", type=float, default=18.0)
    parser.add_argument("--preview", type=int, default=0, choices=[0,1])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--scan-hz", type=float, default=5.0)
    parser.add_argument("--move-threshold", type=int, default=3)
    parser.add_argument("--overlay", type=int, default=1, choices=[0,1])
    args = parser.parse_args()

    try:
        _ = pwc.getActiveWindow()
    except Exception:
        print("ADVERTENCIA: Habilita Accesibilidad para tu terminal/VS Code en macOS.")

    def try_open_camera(index: int, backend_flag: int):
        cap_local = cv2.VideoCapture(index, backend_flag)
        if cap_local.isOpened():
            cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap_local.set(cv2.CAP_PROP_FPS, 24)
        return cap_local

    backend_order = (
        [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if args.backend == "auto"
        else [cv2.CAP_AVFOUNDATION] if args.backend == "avf" else [cv2.CAP_ANY]
    )
    def try_indices(indices, backends):
        for b in backends:
            for idx in indices:
                cap_t = try_open_camera(idx, b)
                if cap_t.isOpened():
                    return cap_t, idx, b
        return None, None, None

    if args.camera >= 0:
        cap, used_idx, used_backend = try_indices([args.camera], backend_order)
    else:
        cap, used_idx, used_backend = try_indices([0], [cv2.CAP_AVFOUNDATION])
        if not cap:
            cap, used_idx, used_backend = try_indices([0,1,2,3], backend_order)
    if not cap or not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")
    else:
        print(f"Cámara abierta en índice {used_idx} con backend {used_backend}.")

    model_path = args.model
    if not os.path.exists(model_path):
        from urllib.request import urlretrieve
        print("Descargando modelo de hand_landmarker…")
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            model_path,
        )
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    screen_w, screen_h = get_screen_size()

    ds = DragState()
    paused = False
    last_update = 0.0
    min_update_dt = 1.0 / max(1.0, args.drag_fps)
    last_scan = 0.0
    min_scan_dt = 1.0 / max(1.0, args.scan_hz)
    last_cursor = None
    highlighted_win = None
    filt_x = None
    filt_y = None
    alpha = 0.35

    overlay_proc = None
    last_overlay = (None, None, None, None, None)
    if args.overlay:
        try:
            overlay_proc = subprocess.Popen(
                ["python3", os.path.join(os.path.dirname(__file__), "overlay_helper.py")],
                stdin=subprocess.PIPE,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            print(f"Overlay iniciado (pid={overlay_proc.pid}).")
        except Exception as e:
            print(f"Overlay no disponible: {e}. Continúo sin sombreado.")

    def overlay_send(win: Optional[pwc.Window], visible: bool):
        if overlay_proc is None or overlay_proc.stdin is None:
            return
        if not args.overlay:
            return
        try:
            if win and visible:
                (wx, wy) = win.topleft
                (w, h) = win.size
                cocoa_y = int(screen_h - (wy + h))
                msg = {"x": int(wx), "y": cocoa_y, "w": int(w), "h": int(h), "visible": True}
            else:
                msg = {"x": 0, "y": 0, "w": 10, "h": 10, "visible": False}
            key = (msg["x"], msg["y"], msg["w"], msg["h"], msg["visible"])
            nonlocal last_overlay
            if key != last_overlay:
                overlay_proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
                overlay_proc.stdin.flush()
                last_overlay = key
        except Exception:
            pass

    try:
        while True:
            if paused:
                time.sleep(0.01)
                continue

            ret, frame_bgr = cap.read()
            if not ret:
                break
            if args.flip:
                frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            infer_img = frame_rgb
            if 0.3 <= args.inference_scale < 1.0:
                new_w = max(64, int(frame_rgb.shape[1] * args.inference_scale))
                new_h = max(64, int(frame_rgb.shape[0] * args.inference_scale))
                infer_img = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_img)
            timestamp_ms = time.monotonic_ns() // 1_000_000
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                pinching = is_pinch(hand, args.pinch_thresh)
                x_norm, y_norm = landmark_xy_norm(hand, INDEX_TIP)
                cur_x = int(max(0.0, min(1.0, x_norm)) * screen_w)
                cur_y = int(max(0.0, min(1.0, y_norm)) * screen_h)
                if filt_x is None:
                    filt_x, filt_y = cur_x, cur_y
                else:
                    filt_x = int(alpha * filt_x + (1 - alpha) * cur_x)
                    filt_y = int(alpha * filt_y + (1 - alpha) * cur_y)
                x_screen, y_screen = filt_x, filt_y
                if args.invert_y:
                    y_screen = screen_h - y_screen

                now = time.monotonic()
                moved = (last_cursor is None) or (abs(x_screen - last_cursor[0]) > args.move_threshold or abs(y_screen - last_cursor[1]) > args.move_threshold)
                if not pinching and not ds.active and moved and (now - last_scan) >= min_scan_dt:
                    last_scan = now
                    last_cursor = (x_screen, y_screen)
                    try:
                        highlighted_win = find_window_at_point(x_screen, y_screen)
                    except Exception:
                        highlighted_win = None
                    overlay_send(highlighted_win, True if highlighted_win else False)

                if now - last_update >= min_update_dt:
                    last_update = now
                    try:
                        if pinching and not ds.active:
                            win = highlighted_win or find_window_at_point(x_screen, y_screen)
                            if win and getattr(win, 'isVisible', True):
                                ds.active = True
                                ds.window = win
                                w, h = win.size
                                ds.window_size = (w, h)
                                wx, wy = win.topleft
                                ds.offset = (x_screen - wx, y_screen - wy)
                                ds.last_pos = (wx, wy)
                                try:
                                    win.activate()
                                except Exception:
                                    pass
                                overlay_send(win, False)
                        elif pinching and ds.active and ds.window is not None:
                            w, h = ds.window_size if ds.window_size is not None else ds.window.size
                            new_x = max(0, min(screen_w - w, x_screen - ds.offset[0]))
                            new_y = max(0, min(screen_h - h, y_screen - ds.offset[1]))
                            lp = ds.last_pos
                            if lp is None or abs(new_x - lp[0]) > args.move_threshold or abs(new_y - lp[1]) > args.move_threshold:
                                ds.window.moveTo(int(new_x), int(new_y))
                                ds.last_pos = (int(new_x), int(new_y))
                        elif (not pinching) and ds.active and ds.window is not None:
                            snap_window(ds.window, screen_w, screen_h, x_screen, y_screen)
                            overlay_send(ds.window, True)
                            ds = DragState()
                    except Exception:
                        ds = DragState()
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            if overlay_proc and overlay_proc.stdin:
                overlay_proc.stdin.write((json.dumps({"visible": False}) + "\n").encode("utf-8"))
                overlay_proc.stdin.flush()
                overlay_proc.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()
