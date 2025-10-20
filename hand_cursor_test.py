#!/usr/bin/env python3
"""
Prueba: mover el cursor según la posición del dedo índice (MediaPipe + macOS Quartz)

- Usa MediaPipe HandLandmarker para obtener la punta del índice (landmark 8).
- Mapea coordenadas normalizadas [0..1] a pantalla y mueve el cursor.
- Incluye suavizado (EMA), umbral de movimiento y limitación de tasa.

Requisitos:
  - Python 3.9–3.11 (recomendado 3.10)
  - pip install -r requirements.txt (mediapipe, opencv-python, pyobjc)
  - macOS: dar permisos de Accesibilidad a tu Terminal/VS Code y a Python al primer intento de mover el cursor.

Controles:
  - "q": salir (si preview=1)
  - Espacio: pausar/reanudar
"""
import os
import time
import argparse
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# macOS Quartz para mover el cursor y obtener tamaño de pantalla
try:
    import Quartz
except Exception as e:
    raise RuntimeError(
        "Falta Quartz (pyobjc-framework-Quartz). Instala las dependencias con 'pip install -r requirements.txt'."
    ) from e

INDEX_TIP = 8


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def get_screen_size() -> Tuple[int, int]:
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return int(bounds.size.width), int(bounds.size.height)


def warp_mouse(x_cg: int, y_cg: int):
    # CGWarpMouseCursorPosition usa coordenadas con origen en esquina inferior-izquierda (Cocoa)
    Quartz.CGWarpMouseCursorPosition((x_cg, y_cg))


def main():
    parser = argparse.ArgumentParser(description="Mover cursor con la punta del índice (prueba)")
    parser.add_argument("--model", default="hand_landmarker.task", help="Ruta al archivo .task del modelo")
    parser.add_argument("--camera", type=int, default=-1, help="Índice de cámara; -1 intenta FaceTime HD")
    parser.add_argument("--width", type=int, default=960, help="Ancho de captura de cámara")
    parser.add_argument("--height", type=int, default=540, help="Alto de captura de cámara")
    parser.add_argument("--inference-scale", type=float, default=0.5, help="Escala de downsample para la inferencia (0.3–1.0)")
    parser.add_argument("--flip", type=int, default=1, choices=[0, 1], help="Vista espejo (1=Sí, 0=No)")
    parser.add_argument("--invert-y", action="store_true", help="No inviertas Y (útil si ves el cursor al revés)")
    parser.add_argument("--backend", default="auto", choices=["auto", "avf", "any"], help="Backend cámara")
    parser.add_argument("--cursor-fps", type=float, default=24.0, help="Frecuencia máxima de movimientos del cursor (Hz)")
    parser.add_argument("--move-threshold", type=int, default=4, help="Umbral (px) para mover el cursor")
    parser.add_argument("--preview", type=int, default=1, choices=[0,1], help="Mostrar preview de cámara con HUD (1=Sí, 0=No)")
    args = parser.parse_args()

    # Abrir cámara (preferir AVFoundation en macOS)
    def try_open_camera(index: int, backend_flag: int):
        cap_local = cv2.VideoCapture(index, backend_flag)
        if cap_local.isOpened():
            cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap_local.set(cv2.CAP_PROP_FPS, 24)
        return cap_local

    backend_order = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if args.backend == "auto" else \
                    [cv2.CAP_AVFOUNDATION] if args.backend == "avf" else [cv2.CAP_ANY]

    def try_indices(indices, backends):
        for b in backends:
            for idx in indices:
                cap_t = try_open_camera(idx, b)
                if cap_t.isOpened():
                    return cap_t, idx, b
        return None, None, None

    indices_default = [0, 1, 2]
    if args.camera >= 0:
        cap, used_idx, used_backend = try_indices([args.camera], backend_order)
    else:
        cap, used_idx, used_backend = try_indices([0], [cv2.CAP_AVFOUNDATION])
        if not cap:
            cap, used_idx, used_backend = try_indices(indices_default, backend_order)
    if not cap or not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Prueba --backend avf o --camera 1.")
    else:
        print(f"Cámara abierta en índice {used_idx} con backend {used_backend}.")

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
    print(f"Tamaño de pantalla: {screen_w}x{screen_h}")

    # Suavizado y rate limit
    filt_x: Optional[int] = None
    filt_y: Optional[int] = None
    alpha = 0.35
    last_move = 0.0
    min_move_dt = 1.0 / max(1.0, args.cursor_fps)
    last_cursor: Optional[Tuple[int,int]] = None

    paused = False
    t0 = time.time()
    frames = 0

    try:
        while True:
            if paused:
                if args.preview:
                    dummy = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    cv2.imshow("Hand Cursor Test", dummy)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        break
                    if key == 32:
                        paused = not paused
                time.sleep(0.01)
                continue

            ok, frame_bgr = cap.read()
            if not ok:
                print("No se pudo leer frame de cámara. Saliendo…")
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

            hud = []
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                tip = hand[INDEX_TIP]
                x_norm, y_norm = float(tip.x), float(tip.y)
                # Mapeo a pantalla (origen arriba-izquierda en normalizado)
                cur_x = int(clamp(x_norm, 0.0, 1.0) * screen_w)
                cur_y = int(clamp(y_norm, 0.0, 1.0) * screen_h)

                if filt_x is None:
                    filt_x, filt_y = cur_x, cur_y
                else:
                    filt_x = int(alpha * filt_x + (1 - alpha) * cur_x)
                    filt_y = int(alpha * filt_y + (1 - alpha) * cur_y)

                x_scr, y_scr = filt_x, filt_y
                # Convertir a coordenadas CG (origen abajo-izquierda) a menos que se pida no invertir
                y_cg = y_scr if args.invert_y else (screen_h - y_scr)
                x_cg = x_scr

                now = time.monotonic()
                moved = (last_cursor is None) or (abs(x_scr - last_cursor[0]) > args.move_threshold or abs(y_scr - last_cursor[1]) > args.move_threshold)
                if moved and (now - last_move) >= min_move_dt:
                    warp_mouse(int(x_cg), int(y_cg))
                    last_move = now
                    last_cursor = (x_scr, y_scr)

                hud.append(f"pos=({x_scr},{y_scr}) cg=({int(x_cg)},{int(y_cg)})")
            else:
                # Reiniciar filtro si perdemos mano para evitar saltos grandes al reenganchar
                filt_x = None
                filt_y = None

            if args.preview:
                frames += 1
                elapsed = time.time() - t0
                fps = frames / elapsed if elapsed > 0 else 0.0
                overlay = frame_bgr.copy()
                if result.hand_landmarks:
                    cx = int(clamp(float(result.hand_landmarks[0][INDEX_TIP].x), 0.0, 1.0) * overlay.shape[1])
                    cy = int(clamp(float(result.hand_landmarks[0][INDEX_TIP].y), 0.0, 1.0) * overlay.shape[0])
                    cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), 2)
                y0 = 20
                cv2.putText(overlay, f"FPS: {fps:.1f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
                y0 += 22
                for line in hud[-3:]:
                    cv2.putText(overlay, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2, cv2.LINE_AA)
                    y0 += 22
                cv2.imshow("Hand Cursor Test", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return
                if key == 32:
                    paused = not paused

    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
