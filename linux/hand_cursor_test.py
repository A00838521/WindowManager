#!/usr/bin/env python3
"""
Linux (Wayland/Hyprland): Mover el cursor según la punta del índice (MediaPipe)

- Intenta primero WLR virtual pointer vía hyprctl (Hyprland) usando hyprland-ipc si está disponible.
- Fallback: pyautogui (X11/Wayland con portal), o python-xlib (X11).

Requisitos sugeridos en Arch:
  - Hyprland + hyprland (hyprctl)
  - python-pyautogui (AUR) o pip install pyautogui
  - python-evdev (opcional para precisión), python-xlib (para X11)
  - pip install -r requirements.txt (mediapipe, opencv-python)

Permisos Wayland: para mover el cursor de forma fiable en Hyprland, lo más estable es usar hyprctl keyword o hyprctl dispatch (require IPC).
"""
import os, time, argparse, shutil, json, subprocess
from typing import Optional, Tuple

import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

INDEX_TIP = 8
_MOVE_ANNOUNCED = False


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def try_hyprland_move_abs(x: int, y: int) -> bool:
    """Intentar mover el cursor con hyprctl dispatch si existe un dispatcher válido.
    En varias versiones, 'cursorpos' NO existe y hyprctl imprime 'Invalid dispatcher' con exit 0.
    Silenciamos salida y consideramos fallo si vemos 'Invalid dispatcher'."""
    hyprctl = shutil.which("hyprctl")
    if not hyprctl:
        return False
    try:
        proc = subprocess.run(
            [hyprctl, "dispatch", "cursorpos", str(x), str(y)],
            capture_output=True,
            text=True,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if "Invalid dispatcher" in out or proc.returncode != 0:
            return False
        return True
    except Exception:
        return False


def try_ydotool_move_abs(x: int, y: int) -> bool:
    """Mover cursor absoluto con ydotool (requiere ydotoold activo).
    ydotool mousemove -a -x X -y Y"""
    ydotool = shutil.which("ydotool")
    if not ydotool:
        return False
    try:
        proc = subprocess.run(
            [ydotool, "mousemove", "-a", "-x", str(x), "-y", str(y)],
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0
    except Exception:
        return False


def try_pyautogui_move(x: int, y: int) -> bool:
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.moveTo(x, y)
        return True
    except Exception:
        return False


def get_screen_size() -> Tuple[int, int]:
    """Detecta tamaño de pantalla:
    1) hyprctl -j monitors
    2) pyautogui.size()
    3) fallback 1920x1080
    """
    hyprctl = shutil.which("hyprctl")
    if hyprctl:
        try:
            proc = subprocess.run([hyprctl, "-j", "monitors"], capture_output=True, text=True)
            if proc.returncode == 0 and proc.stdout:
                mons = json.loads(proc.stdout)
                # Elegir monitor activo con focus o el primero
                mon = None
                for m in mons:
                    if m.get("focused"):
                        mon = m
                        break
                if mon is None and mons:
                    mon = mons[0]
                if mon:
                    # Hyprland expone size: { x, y }
                    if "width" in mon and "height" in mon:
                        return int(mon["width"]), int(mon["height"])
                    if isinstance(mon.get("size"), dict):
                        return int(mon["size"].get("x", 1920)), int(mon["size"].get("y", 1080))
        except Exception:
            pass

    try:
        import pyautogui
        w, h = pyautogui.size()
        return int(w), int(h)
    except Exception:
        return 1920, 1080


def move_cursor_abs(x: int, y: int) -> bool:
    global _MOVE_ANNOUNCED
    # Preferir ydotool en Wayland/Hyprland
    if try_ydotool_move_abs(x, y):
        if not _MOVE_ANNOUNCED:
            print("[input] Usando ydotool para mover el cursor (Wayland)")
            _MOVE_ANNOUNCED = True
        return True
    # Intentar Hyprland dispatch (si existe)
    if try_hyprland_move_abs(x, y):
        if not _MOVE_ANNOUNCED:
            print("[input] Usando hyprctl dispatch para mover el cursor")
            _MOVE_ANNOUNCED = True
        return True
    # Luego pyautogui (X11/Wayland portal)
    if try_pyautogui_move(x, y):
        if not _MOVE_ANNOUNCED:
            print("[input] Usando pyautogui para mover el cursor")
            _MOVE_ANNOUNCED = True
        return True
    return False


def main():
    p = argparse.ArgumentParser(description="Mover cursor (Linux/Hyprland)")
    p.add_argument("--model", default="hand_landmarker.task")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--inference-scale", type=float, default=0.5)
    p.add_argument("--flip", type=int, default=1, choices=[0,1])
    p.add_argument("--invert-y", action="store_true")
    p.add_argument("--cursor-fps", type=float, default=24.0)
    p.add_argument("--move-threshold", type=int, default=4)
    p.add_argument("--preview", type=int, default=0, choices=[0,1])
    args = p.parse_args()

    # Cámara (en Linux suele ser índice 0 con V4L2)
    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 24)
    if not cap or not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara en Linux. Prueba --camera 1 o revisa V4L2.")

    # Modelo
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

    # Suavizado y rate limit
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
                cur_x = int(clamp(x_norm, 0.0, 1.0) * screen_w)
                cur_y = int(clamp(y_norm, 0.0, 1.0) * screen_h)

                if filt_x is None:
                    filt_x, filt_y = cur_x, cur_y
                else:
                    filt_x = int(alpha * filt_x + (1 - alpha) * cur_x)
                    filt_y = int(alpha * filt_y + (1 - alpha) * cur_y)

                x_scr, y_scr = filt_x, filt_y
                # En Wayland, la coordenada Y suele tener origen arriba-izquierda; ajustar si fuese necesario.
                y_out = y_scr if args.invert_y else y_scr
                x_out = x_scr

                now = time.monotonic()
                moved = (last_cursor is None) or (abs(x_scr - last_cursor[0]) > args.move_threshold or abs(y_scr - last_cursor[1]) > args.move_threshold)
                if moved and (now - last_move) >= min_move_dt:
                    moved_ok = move_cursor_abs(int(x_out), int(y_out))
                    if not moved_ok:
                        # Si no se pudo mover, podemos informar una vez
                        pass
                    last_move = now
                    last_cursor = (x_scr, y_scr)

    finally:
        try:
            cap.release()
        except Exception:
            pass

if __name__ == "__main__":
    main()
