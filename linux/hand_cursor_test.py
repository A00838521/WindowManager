#!/usr/bin/env python3
"""
Linux (Wayland/Hyprland): Control del cursor con la punta del índice (MediaPipe).

Qué hace
- Detecta la mano (MediaPipe) y mueve el cursor en Wayland/Hyprland.
- Orden de backends: ydotool (preferido, absoluto) → hyprctl (si compatible) → pyautogui.
- Auto-ajuste: detecta Hz del monitor y fps de la cámara para ajustar suavizado y tasa de emisión.

Uso rápido
    QT_QPA_PLATFORM=xcb python linux/hand_cursor_test.py --preview 1

Flags útiles
- --invert-x: invierte el eje X si tu cámara/posición lo requiere.
- --invert-y / --no-invert-y: por defecto invertido (suele sentirse natural en escritorio).
- --auto / --no-auto: auto-detecta y ajusta cursor-fps, tau y filtro. Por defecto activado.

Requisitos
- Hyprland (hyprctl) y ydotoold activo (para Wayland). En Arch: ydotool
- Python 3.11 con: mediapipe, opencv-python, numpy, pyautogui
"""
import os, time, argparse, shutil, json, subprocess, math, threading
from typing import Optional, Tuple

# Evitar fallo de Qt en Wayland cuando se usa preview/imshow
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

INDEX_TIP = 8
_MOVE_ANNOUNCED = False


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def try_hyprland_move_abs(x: int, y: int) -> bool:
    """Mueve el cursor con hyprctl si existe un dispatcher válido en esta versión.

    Returns:
        True si el dispatcher cursorpos funciona; False si no existe o falla.
    """
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

    Returns:
        True si el comando ydotool finaliza correctamente; False en caso contrario.
    """
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
    """Mover cursor con pyautogui (fallback X11/Wayland portal)."""
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.moveTo(x, y)
        return True
    except Exception:
        return False


def get_screen_size() -> Tuple[int, int]:
    """Tamaño de pantalla por Hyprland; fallback a pyautogui o 1920x1080.

    Returns:
        (ancho, alto) en píxeles.
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
    """Envía movimiento de cursor usando el mejor backend disponible.

    Returns:
        True si algún backend pudo mover el cursor; False si todos fallaron.
    """
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
    """Punto de entrada del control de cursor por mano.

    - Configura cámara y modelo de MediaPipe.
    - Lanza un hilo animador desacoplado del fps de la cámara.
    - Aplica filtro One Euro y mapea a coordenadas de pantalla.
    """
    p = argparse.ArgumentParser(description="Mover cursor (Linux/Hyprland)")
    p.add_argument("--model", default="hand_landmarker.task")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--inference-scale", type=float, default=0.5)
    # Por defecto sin espejo; usa --invert-x si lo necesitas
    p.add_argument("--flip", type=int, default=0, choices=[0,1])
    p.add_argument("--invert-x", dest="invert_x", action="store_true")
    p.add_argument("--no-invert-x", dest="invert_x", action="store_false")
    p.set_defaults(invert_x=True)
    # Invertir Y por defecto para alinear movimiento de la mano con el puntero
    ygrp = p.add_mutually_exclusive_group()
    ygrp.add_argument("--invert-y", dest="invert_y", action="store_true")
    ygrp.add_argument("--no-invert-y", dest="invert_y", action="store_false")
    p.set_defaults(invert_y=True)
    # Emisor del cursor desacoplado del frame rate de la cámara
    p.add_argument("--cursor-fps", type=float, default=120.0)
    p.add_argument("--anim-tau-ms", type=float, default=12.0, help="Constante de tiempo del suavizado de animación (ms)")
    # Parámetros del filtro One Euro (suavizado dependiente de velocidad)
    p.add_argument("--min-cutoff", type=float, default=1.2)
    p.add_argument("--beta", type=float, default=0.007)
    p.add_argument("--d-cutoff", type=float, default=1.0)
    p.add_argument("--move-threshold", type=int, default=4)
    p.add_argument("--preview", type=int, default=0, choices=[0,1])
    # Auto detección de entorno y auto-tuning
    agrp = p.add_mutually_exclusive_group()
    agrp.add_argument("--auto", dest="auto", action="store_true")
    agrp.add_argument("--no-auto", dest="auto", action="store_false")
    p.set_defaults(auto=True)
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

    def detect_monitor_info():
        w, h = get_screen_size()
        hz = None
        hyprctl = shutil.which("hyprctl")
        if hyprctl:
            try:
                proc = subprocess.run([hyprctl, "-j", "monitors"], capture_output=True, text=True)
                if proc.returncode == 0 and proc.stdout:
                    mons = json.loads(proc.stdout)
                    mon = None
                    for m in mons:
                        if m.get("focused"):
                            mon = m
                            break
                    if mon is None and mons:
                        mon = mons[0]
                    if mon:
                        # common keys: width/height or size{x,y}, refreshRate
                        if "width" in mon and "height" in mon:
                            w, h = int(mon["width"]), int(mon["height"])
                        elif isinstance(mon.get("size"), dict):
                            w = int(mon["size"].get("x", w))
                            h = int(mon["size"].get("y", h))
                        if "refreshRate" in mon:
                            hz = float(mon["refreshRate"]) or None
                        elif "refresh_rate" in mon:
                            hz = float(mon["refresh_rate"]) or None
            except Exception:
                pass
        return w, h, hz

    def detect_camera_info(cam_index: int) -> tuple[int, int, float]:
        # Devuelve (width, height, fps_est)
        cw, ch, cfps = args.width, args.height, 0.0
        try:
            cap0 = cv2.VideoCapture(cam_index)
            if cap0.isOpened():
                cap0.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                # No todas las cámaras aceptan set FPS, medir
                cw = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
                ch = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
                # Medición rápida de fps hasta 0.7s o 20 frames
                n, t0 = 0, time.perf_counter()
                while n < 20 and (time.perf_counter() - t0) < 0.7:
                    ok, _ = cap0.read()
                    if not ok:
                        break
                    n += 1
                dt = time.perf_counter() - t0
                if n > 1 and dt > 0:
                    cfps = n / dt
            try:
                cap0.release()
            except Exception:
                pass
        except Exception:
            pass
        return cw, ch, cfps

    screen_w, screen_h, mon_hz = detect_monitor_info()
    cam_w0, cam_h0, cam_fps_est = detect_camera_info(args.camera)

    # Auto-tuning
    if args.auto:
        # Elegir FPS del cursor acorde al monitor
        if mon_hz and mon_hz >= 60:
            args.cursor_fps = float(min(240.0, max(60.0, round(mon_hz))))
        # Ajustar tau acorde a refresh
        if mon_hz:
            # constante de tiempo aprox 1/(2*Hz), acota a [6, 14] ms
            tau_ms = max(6.0, min(14.0, 1000.0 / (2.0 * mon_hz)))
            args.anim_tau_ms = tau_ms
        # Ajustar One Euro en función del fps de cámara
        if cam_fps_est > 0:
            if cam_fps_est <= 30:
                args.min_cutoff = 1.3
                args.beta = 0.008
            elif cam_fps_est <= 60:
                args.min_cutoff = 1.2
                args.beta = 0.009
            else:
                args.min_cutoff = 1.0
                args.beta = 0.012
        # Y suele requerir invertir para sensación natural en escritorio
        # Solo cambiar si el usuario no forzó no-invert-y
        # (ya por defecto viene invertido True)
        # X: si el usuario aplica flip a la imagen, no tocaremos invert-x aquí

    # Resumen de auto-configuración
    print(f"[auto] Monitor: {screen_w}x{screen_h} @ {mon_hz or '??'} Hz, Cámara: {cam_w0}x{cam_h0} ~ {cam_fps_est:.1f} fps")
    print(f"[auto] Cursor FPS: {args.cursor_fps}, tau_ms: {args.anim_tau_ms:.1f}, OneEuro(min_cutoff={args.min_cutoff}, beta={args.beta}, d_cutoff={args.d_cutoff})")
    print(f"[auto] invert_x={args.invert_x}, invert_y={args.invert_y}, flip_input={args.flip}")

    # --- Filtro One Euro para la medición (30 fps aprox) ---
    class LowPass:
        def __init__(self, alpha: float, init: Optional[float] = None):
            self.a = alpha
            self.y = init
            self.s = False
        def filt(self, x: float, alpha: float) -> float:
            self.a = alpha
            if not self.s:
                self.y = x
                self.s = True
            else:
                self.y = self.y + alpha * (x - self.y)
            return self.y

    class OneEuro:
        def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
            self.min_cutoff = float(min_cutoff)
            self.beta = float(beta)
            self.d_cutoff = float(d_cutoff)
            self.x_prev = LowPass(1.0)
            self.dx_prev = LowPass(1.0)
            self.t_prev: Optional[float] = None
        @staticmethod
        def _alpha(cutoff: float, dt: float) -> float:
            # 2*pi*cutoff/(2*pi*cutoff + 1/dt)
            tau = 1.0 / (2.0 * math.pi * max(1e-6, cutoff))
            return 1.0 / (1.0 + tau / max(1e-6, dt))
        def filter(self, t: float, x: float) -> float:
            if self.t_prev is None:
                self.t_prev = t
                self.x_prev.filt(x, 1.0)
                self.dx_prev.filt(0.0, 1.0)
                return x
            dt = max(1e-6, t - self.t_prev)
            # Derivada
            dx = (x - self.x_prev.y) / dt
            ad = OneEuro._alpha(self.d_cutoff, dt)
            edx = self.dx_prev.filt(dx, ad)
            # Corte dinámico
            cutoff = self.min_cutoff + self.beta * abs(edx)
            a = OneEuro._alpha(cutoff, dt)
            ex = self.x_prev.filt(x, a)
            self.t_prev = t
            return ex

    filt_x_euro = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)
    filt_y_euro = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)

    # --- Hilo de animación a 120 Hz para suavizar entre frames ---
    target_lock = threading.Lock()
    anim_target: Tuple[float, float] = (None, None)  # tipo: ignore
    anim_pos: Tuple[float, float] = (None, None)  # tipo: ignore
    last_sent: Optional[Tuple[int, int]] = None
    stop_evt = threading.Event()

    def cursor_animator():
        nonlocal anim_target, anim_pos, last_sent
        fps = max(30.0, float(args.cursor_fps))
        dt = 1.0 / fps
        tau = max(1e-6, args.anim_tau_ms / 1000.0)
        while not stop_evt.is_set():
            with target_lock:
                tx, ty = anim_target
                cx, cy = anim_pos
            if tx is not None and ty is not None:
                if cx is None or cy is None:
                    cx, cy = tx, ty
                # Respuesta exponencial hacia el objetivo (critically-damped approx):
                # factor = 1 - exp(-dt/tau)
                k = 1.0 - math.exp(-dt / tau)
                cx = cx + k * (tx - cx)
                cy = cy + k * (ty - cy)

                ix, iy = int(round(cx)), int(round(cy))
                if last_sent is None or (abs(ix - last_sent[0]) > args.move_threshold or abs(iy - last_sent[1]) > args.move_threshold):
                    move_cursor_abs(ix, iy)
                    last_sent = (ix, iy)

                with target_lock:
                    anim_pos = (cx, cy)
            time.sleep(dt)

    anim_thread = threading.Thread(target=cursor_animator, daemon=True)
    anim_thread.start()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Guardar frame para visualización (si preview)
            vis_frame = frame_bgr.copy()
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
                # Invertir X si está activado
                if args.invert_x:
                    x_norm = 1.0 - x_norm
                # Nota: args.flip solo afecta a la imagen de entrada para el landmarker

                # Filtrado One Euro en coordenadas normalizadas
                tnow = time.monotonic()
                fx = clamp(filt_x_euro.filter(tnow, x_norm), 0.0, 1.0)
                fy = clamp(filt_y_euro.filter(tnow, y_norm), 0.0, 1.0)

                x_scr = fx * screen_w
                y_scr = fy * screen_h
                # Invertir Y si se solicita: arriba de la mano -> arriba del puntero
                y_out = (screen_h - y_scr) if args.invert_y else y_scr
                x_out = x_scr

                with target_lock:
                    anim_target = (float(x_out), float(y_out))

                if args.preview:
                    # Dibujar indicador simple y mostrar preview espejada en X
                    cx = int(tip.x * vis_frame.shape[1])
                    cy = int(tip.y * vis_frame.shape[0])
                    cv2.circle(vis_frame, (cx, cy), 6, (0, 255, 0), -1)
                    disp = cv2.flip(vis_frame, 1)
                    cv2.imshow("Cursor Preview", disp)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            else:
                if args.preview:
                    disp = cv2.flip(vis_frame, 1)
                    cv2.putText(disp, "No hand", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,200), 2)
                    cv2.imshow("Cursor Preview", disp)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

    finally:
        try:
            cap.release()
        except Exception:
            pass
        stop_evt.set()
        try:
            anim_thread.join(timeout=0.2)
        except Exception:
            pass
        try:
            if args.preview:
                cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()
