#!/usr/bin/env python3
"""
macOS: Mover el cursor según la punta del índice (MediaPipe + Quartz)
"""
import os, time, argparse
from typing import Optional, Tuple
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import Quartz

INDEX_TIP = 8

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def get_screen_size() -> Tuple[int, int]:
    b = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return int(b.size.width), int(b.size.height)

def warp_mouse(x_cg: int, y_cg: int):
    Quartz.CGWarpMouseCursorPosition((x_cg, y_cg))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hand_landmarker.task")
    p.add_argument("--camera", type=int, default=-1)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--inference-scale", type=float, default=0.5)
    p.add_argument("--flip", type=int, default=1, choices=[0,1])
    p.add_argument("--invert-y", action="store_true")
    p.add_argument("--backend", default="auto", choices=["auto","avf","any"])
    p.add_argument("--cursor-fps", type=float, default=24.0)
    p.add_argument("--move-threshold", type=int, default=4)
    p.add_argument("--preview", type=int, default=0, choices=[0,1])
    args = p.parse_args()

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
            cap, used_idx, used_backend = try_indices([0,1,2], backend_order)
    if not cap or not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Prueba --backend avf o --camera 1.")
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
    filt_x: Optional[int] = None
    filt_y: Optional[int] = None
    alpha = 0.35
    last_move = 0.0
    min_move_dt = 1.0 / max(1.0, args.cursor_fps)
    last_cursor: Optional[Tuple[int,int]] = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
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
                tip = result.hand_landmarks[0][INDEX_TIP]
                x_norm, y_norm = float(tip.x), float(tip.y)
                cur_x = int(max(0.0, min(1.0, x_norm)) * screen_w)
                cur_y = int(max(0.0, min(1.0, y_norm)) * screen_h)
                if filt_x is None:
                    filt_x, filt_y = cur_x, cur_y
                else:
                    filt_x = int(alpha * filt_x + (1 - alpha) * cur_x)
                    filt_y = int(alpha * filt_y + (1 - alpha) * cur_y)
                x_scr, y_scr = filt_x, filt_y
                y_cg = y_scr if args.invert_y else (screen_h - y_scr)
                x_cg = x_scr

                now = time.monotonic()
                moved = (last_cursor is None) or (abs(x_scr - last_cursor[0]) > args.move_threshold or abs(y_scr - last_cursor[1]) > args.move_threshold)
                if moved and (now - last_move) >= min_move_dt:
                    Quartz.CGWarpMouseCursorPosition((int(x_cg), int(y_cg)))
                    last_move = now
                    last_cursor = (x_scr, y_scr)

    finally:
        try:
            cap.release()
        except Exception:
            pass

if __name__ == "__main__":
    main()
