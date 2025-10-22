#!/usr/bin/env python3
"""
Hyprland (Wayland): Control relativo de ventanas con gesto de pellizco.

Objetivo
- Al iniciar pellizco: registra la posición inicial de la mano (x0,y0) en coordenadas de pantalla.
- Centra el cursor en la ventana activa o seleccionada (si existe) antes de mover.
- Durante el pellizco: mueve la ventana relativo al desplazamiento de la mano Δ=(x-x0, y-y0).
- Extra: pellizco largo para alternar tile/float sin iniciar movimiento.

Notas
- No se cambia la lógica de detección de landmarks (se reutiliza el mismo flujo con MediaPipe que otros scripts).
- Se modifica únicamente la lógica de posicionamiento del cursor y el movimiento de ventana.
- Para coherencia con Hyprland, se leen algunas opciones de ~/.config/hypr/ (gaps) para ajustar el margen de snap.

Requisitos
- Hyprland con hyprctl en PATH; Python 3.11; mediapipe, opencv-python, numpy.
- Wayland + XWayland: se fuerza QT_QPA_PLATFORM=xcb para imshow.
"""
import os, time, argparse, shutil, json, subprocess, math, threading, glob, re
from typing import Optional, Tuple, List, Dict, Any

# Evitar fallo de Qt en Wayland: forzar plugin XCB (XWayland) para imshow
os.environ["QT_QPA_PLATFORM"] = "xcb"
# Reducir verbosidad de TF/Mediapipe en consola
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all,1=info,2=warning,3=error
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Opcional: backend uinput para sostener botones y SUPER
try:
    from evdev import UInput, ecodes  # type: ignore
    _HAS_UINPUT = True
except Exception:
    UInput = None  # type: ignore
    ecodes = None  # type: ignore
    _HAS_UINPUT = False

INDEX_TIP = 8
THUMB_TIP = 4
MIDDLE_TIP = 12
PREVIEW_TITLE = "Pinch Drag Preview"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------------- Hyprland helpers ----------------------

def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)

def hypr_clients() -> List[Dict]:
    hyprctl = shutil.which("hyprctl")
    if not hyprctl:
        return []
    try:
        p = _run([hyprctl, "-j", "clients"])
        if p.returncode == 0 and p.stdout:
            return json.loads(p.stdout)
    except Exception:
        pass
    return []

def hypr_active_window_info() -> Optional[dict]:
    hyprctl = shutil.which("hyprctl")
    if not hyprctl:
        return None
    try:
        p = _run([hyprctl, "-j", "activewindow"])
        if p.returncode == 0 and p.stdout:
            return json.loads(p.stdout)
    except Exception:
        pass
    return None

def hypr_active_workspace() -> Optional[int]:
    hyprctl = shutil.which("hyprctl")
    if not hyprctl:
        return None
    try:
        p = _run([hyprctl, "-j", "activeworkspace"])
        if p.returncode == 0 and p.stdout:
            data = json.loads(p.stdout)
            wid = data.get("id")
            return int(wid) if wid is not None else None
    except Exception:
        pass
    return None

def hypr_focus_window_address(addr: str) -> bool:
    hyprctl = shutil.which("hyprctl")
    if not hyprctl or not addr:
        return False
    try:
        return _run([hyprctl, "dispatch", "focuswindow", f"address:{addr}"]).returncode == 0
    except Exception:
        return False

def hypr_set_floating_address(addr: str, on: bool) -> bool:
    hyprctl = shutil.which("hyprctl")
    if not hyprctl or not addr:
        return False
    try:
        return _run([hyprctl, "dispatch", "setfloating", f"address:{addr}", "yes" if on else "no"]).returncode == 0
    except Exception:
        return False

def hypr_toggle_floating_address(addr: str) -> bool:
    hyprctl = shutil.which("hyprctl")
    if not hyprctl or not addr:
        return False
    try:
        return _run([hyprctl, "dispatch", "togglefloating", f"address:{addr}"]).returncode == 0
    except Exception:
        return False

def hypr_move_tiled_side_address(addr: str, side: str) -> bool:
    """Mueve una ventana (tiled) hacia una dirección del layout: l/r/u/d.

    Requiere que la ventana no esté en floating. Enfoca por address y usa
    `hyprctl dispatch movewindow <side>`.
    """
    if side not in ("l","r","u","d"):
        return False
    hyprctl = shutil.which("hyprctl")
    if not hyprctl or not addr:
        return False
    try:
        hypr_focus_window_address(addr)
        p = _run([hyprctl, "dispatch", "movewindow", side])
        return p.returncode == 0
    except Exception:
        return False

def hypr_move_window_to_address(addr: str, x: int, y: int) -> bool:
    """Mueve una ventana específica identificada por address a (x,y) absolutos.
    Intenta con address; si falla, enfoca y usa movewindowpixel sobre activa.
    """
    if not addr:
        return False
    hyprctl = shutil.which("hyprctl")
    if hyprctl:
        p = _run([hyprctl, "dispatch", "movewindowpixel", "exact", str(int(x)), str(int(y)), f"address:{addr}"])
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode == 0 and "Invalid" not in out:
            return True
    hypr_focus_window_address(addr)
    return _run([shutil.which("hyprctl"), "dispatch", "movewindowpixel", "exact", str(int(x)), str(int(y))]).returncode == 0

def hypr_resize_window_to_address(addr: str, w: int, h: int) -> bool:
    if not addr:
        return False
    hyprctl = shutil.which("hyprctl")
    if hyprctl:
        p = _run([hyprctl, "dispatch", "resizewindowpixel", "exact", str(int(w)), str(int(h)), f"address:{addr}"])
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode == 0 and "Invalid" not in out:
            return True
    hypr_focus_window_address(addr)
    return _run([shutil.which("hyprctl"), "dispatch", "resizewindowpixel", "exact", str(int(w)), str(int(h))]).returncode == 0


def hypr_focused_monitor_geometry() -> Tuple[int, int, Optional[float], int, int]:
    """Devuelve (width, height, hz, ox, oy) del monitor enfocado."""
    w, h, hz = 1920, 1080, None
    ox = oy = 0
    hyprctl = shutil.which("hyprctl")
    if hyprctl:
        try:
            p = _run([hyprctl, "-j", "monitors"])
            if p.returncode == 0 and p.stdout:
                mons = json.loads(p.stdout)
                mon = None
                for m in mons:
                    if m.get("focused"):
                        mon = m
                        break
                if mon is None and mons:
                    mon = mons[0]
                if mon:
                    if "width" in mon and "height" in mon:
                        w, h = int(mon["width"]), int(mon["height"])
                    elif isinstance(mon.get("size"), dict):
                        w = int(mon["size"].get("x", w))
                        h = int(mon["size"].get("y", h))
                    hz = float(mon.get("refreshRate") or mon.get("refresh_rate") or 0.0) or None
                    if "x" in mon and "y" in mon:
                        ox, oy = int(mon["x"]), int(mon["y"])
                    elif isinstance(mon.get("position"), dict):
                        ox = int(mon["position"].get("x", 0))
                        oy = int(mon["position"].get("y", 0))
        except Exception:
            pass
    return w, h, hz, ox, oy


def client_contains_point(cli: Dict, x_abs: int, y_abs: int) -> bool:
    at = cli.get("at") or [0, 0]
    size = cli.get("size") or [0, 0]
    cx, cy = int(at[0]), int(at[1])
    cw, ch = int(size[0]), int(size[1])
    return (cx <= x_abs <= cx + cw) and (cy <= y_abs <= cy + ch)

def pick_window_under_point(x_abs: int, y_abs: int) -> Optional[Dict]:
    wid = hypr_active_workspace()
    clients = [c for c in hypr_clients() if (not wid or (c.get("workspace", {}).get("id") == wid))]
    clients = [c for c in clients if (str(c.get("title") or "") != PREVIEW_TITLE)]
    inside = [c for c in clients if client_contains_point(c, x_abs, y_abs)]
    if not inside:
        return None
    def area(c):
        sz = c.get("size") or [0, 0]
        return int(sz[0]) * int(sz[1])
    inside.sort(key=area)
    return inside[0]


def ydotool_move_abs(x: int, y: int) -> bool:
    ydotool = shutil.which("ydotool")
    if not ydotool:
        return False
    try:
        p = _run([ydotool, "mousemove", "-a", "-x", str(x), "-y", str(y)])
        return p.returncode == 0
    except Exception:
        return False


# ---------------- Hypr config reader (simple) -----------------

HYPR_CONFIG_DIR = os.path.expanduser("~/.config/hypr")

def read_hypr_config_gaps() -> Dict[str, int]:
    """Lee gaps_in/gaps_out de los .conf en ~/.config/hypr.
    Parser sencillo: busca líneas tipo 'gaps_in = N' dentro o fuera de bloques.
    """
    res = {"gaps_in": 0, "gaps_out": 0}
    files = []
    if os.path.isdir(HYPR_CONFIG_DIR):
        files = sorted(glob.glob(os.path.join(HYPR_CONFIG_DIR, "*.conf")))
        main_conf = os.path.join(HYPR_CONFIG_DIR, "hyprland.conf")
        if os.path.exists(main_conf) and main_conf not in files:
            files.insert(0, main_conf)
    key_re = re.compile(r"^\s*(gaps_in|gaps_out)\s*=\s*([0-9]+)\s*$")
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = key_re.match(line)
                    if m:
                        k, v = m.group(1), int(m.group(2))
                        res[k] = v
        except Exception:
            continue
    return res


# ---------------------- Main program -------------------------

def main():
    p = argparse.ArgumentParser(description="Control relativo de ventana (Hyprland) con pellizco")
    p.add_argument("--model", default="hand_landmarker.task")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--inference-scale", type=float, default=0.5)
    p.add_argument("--flip", type=int, default=0, choices=[0,1])
    p.add_argument("--invert-x", dest="invert_x", action="store_true")
    p.add_argument("--no-invert-x", dest="invert_x", action="store_false")
    p.set_defaults(invert_x=True)
    ygrp = p.add_mutually_exclusive_group()
    ygrp.add_argument("--invert-y", dest="invert_y", action="store_true")
    ygrp.add_argument("--no-invert-y", dest="invert_y", action="store_false")
    p.set_defaults(invert_y=True)

    p.add_argument("--cursor-fps", type=float, default=120.0)
    p.add_argument("--anim-tau-ms", type=float, default=10.0)
    p.add_argument("--min-cutoff", type=float, default=1.2)
    p.add_argument("--beta", type=float, default=0.008)
    p.add_argument("--d-cutoff", type=float, default=1.0)

    # Gestos/umbrales (se reutiliza pinch/índice+medio)
    p.add_argument("--pinch-threshold", type=float, default=0.055)
    p.add_argument("--pinch-hysteresis", type=float, default=0.010)
    p.add_argument("--pinch-hold-ms", type=float, default=50.0)
    pr = p.add_mutually_exclusive_group()
    pr.add_argument("--pinch-relative", dest="pinch_relative", action="store_true")
    pr.add_argument("--no-pinch-relative", dest="pinch_relative", action="store_false")
    p.set_defaults(pinch_relative=True)
    p.add_argument("--pinch-rel-threshold", type=float, default=0.12)

    p.add_argument("--preview", type=int, default=0, choices=[0,1])
    p.add_argument("--hud", type=int, default=1, choices=[0,1])
    p.add_argument("--debug", action="store_true")
    p.add_argument("--auto", dest="auto", action="store_true")
    p.add_argument("--no-auto", dest="auto", action="store_false")
    p.set_defaults(auto=True)

    # Modo de arrastre universal por puntero (SUPER+LMB)
    p.add_argument("--drag-mode", choices=["hypr", "pointer"], default="pointer",
                   help="Modo de arrastre: hypr (movewindowpixel) o pointer (SUPER+LMB con uinput)")
    p.add_argument("--super-keycode", type=int, default=125, help="Keycode evdev para SUPER (LeftMeta=125)")
    p.add_argument("--mouse-button", type=int, default=1, help="Botón de mouse para arrastrar (1=izq, 2=medio, 3=der)")
    p.add_argument("--button-backend", choices=["auto", "ydotool", "uinput"], default="uinput",
                   help="Backend para mantener el botón: ydotool o uinput (requiere permisos /dev/uinput)")

    # Selección por gesto (reemplaza la selección anterior por hold)
    p.add_argument("--select-mode", choices=["swipe", "hold"], default="swipe",
                   help="Método de selección: swipe (deslizar índice+medio) o hold (mantener índice+medio)")
    p.add_argument("--select-order", choices=["mru", "position"], default="mru",
                   help="Orden para ciclar ventanas en selección: mru (recientes) o position (x,y)")
    # Parámetros comunes
    sf = p.add_mutually_exclusive_group()
    sf.add_argument("--select-focus", dest="select_focus", action="store_true")
    sf.add_argument("--no-select-focus", dest="select_focus", action="store_false")
    p.set_defaults(select_focus=True)
    # Modo swipe
    p.add_argument("--swipe-threshold-px", type=int, default=120,
                   help="Desplazamiento horizontal mínimo (px) del gesto índice+medio para ciclar ventana")
    p.add_argument("--swipe-cooldown-ms", type=float, default=250.0,
                   help="Tiempo mínimo entre swipes consecutivos")
    # Modo hold (legacy, opcional)
    sg = p.add_mutually_exclusive_group()
    sg.add_argument("--select-relative", dest="select_relative", action="store_true")
    sg.add_argument("--no-select-relative", dest="select_relative", action="store_false")
    p.set_defaults(select_relative=True)
    p.add_argument("--select-rel-threshold", type=float, default=0.10,
                   help="(hold) Factor relativo por diagonal de la mano para índice+medio juntos")
    p.add_argument("--select-threshold", type=float, default=0.060,
                   help="(hold) Umbral absoluto para índice+medio juntos")
    p.add_argument("--select-hold-ms", type=float, default=80.0,
                   help="(hold) Tiempo mínimo (ms) de índice+medio juntos para seleccionar")

    # Long pinch para toggle tile/float si no hay movimiento
    p.add_argument("--toggle-float-long-ms", type=float, default=900.0,
                   help="Duración de pellizco (ms) sin mover para alternar tile/float")

    # Snap margin usando gaps de Hyprland
    gaps = read_hypr_config_gaps()
    default_snap_margin = max(gaps.get("gaps_in", 0), gaps.get("gaps_out", 0)) + 32
    p.add_argument("--snap-margin", type=int, default=default_snap_margin)
    p.add_argument("--tile-heuristic", choices=["none", "axis", "vertical-prefer", "quadrant"], default="none",
                   help="Heurística de ubicación al volver a tile: axis (por eje dominante), vertical-prefer (prioriza vertical), quadrant (dos movimientos u/d y l/r)")

    args = p.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 24)
    if not cap or not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Prueba --camera 1 o revisa V4L2.")

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

    screen_w, screen_h, mon_hz, mon_ox, mon_oy = hypr_focused_monitor_geometry()
    if args.auto and mon_hz and mon_hz >= 60:
        args.cursor_fps = float(min(240.0, max(60.0, round(mon_hz))))
        args.anim_tau_ms = max(6.0, min(14.0, 1000.0 / (2.0 * mon_hz)))

    if args.debug:
        print(f"[auto] Monitor {screen_w}x{screen_h}@{mon_hz or '??'}Hz off=({mon_ox},{mon_oy}) snap_margin={args.snap_margin}")

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
            tau = 1.0 / (2.0 * math.pi * max(1e-6, cutoff))
            return 1.0 / (1.0 + tau / max(1e-6, dt))
        def filter(self, t: float, x: float) -> float:
            if self.t_prev is None:
                self.t_prev = t
                self.x_prev.filt(x, 1.0)
                self.dx_prev.filt(0.0, 1.0)
                return x
            dt = max(1e-6, t - self.t_prev)
            dx = (x - self.x_prev.y) / dt
            ad = OneEuro._alpha(self.d_cutoff, dt)
            edx = self.dx_prev.filt(dx, ad)
            cutoff = self.min_cutoff + self.beta * abs(edx)
            a = OneEuro._alpha(cutoff, dt)
            ex = self.x_prev.filt(x, a)
            self.t_prev = t
            return ex

    filt_x = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)
    filt_y = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)
    filt_mid_x = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)
    filt_mid_y = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)

    target_lock = threading.Lock()
    anim_target: Tuple[float, float] = (None, None)  # type: ignore
    anim_pos: Tuple[float, float] = (None, None)  # type: ignore
    stop_evt = threading.Event()
    mru_lock = threading.Lock()
    mru_by_ws: Dict[int, List[str]] = {}

    # Estados de arranque/relativo
    dragging = False
    hand_start_screen: Optional[Tuple[int,int]] = None  # (x0,y0) pantalla local al monitor
    win_start_at: Optional[Tuple[int,int]] = None       # (wx0, wy0) pos ventana absoluta
    drag_window_addr: Optional[str] = None
    selected_window_addr: Optional[str] = None
    pinch_started_at: Optional[float] = None
    toggled_float_this_hold = False
    drag_last_sent_xy: Optional[Tuple[int,int]] = None
    # Pointer mode state
    pointer_drag_active = False
    pointer_backend: Optional[str] = None  # 'ydotool' | 'uinput'
    uinput_dev: Optional[object] = None if _HAS_UINPUT else None
    pointer_start_abs: Optional[Tuple[int,int]] = None

    # ----------------- Pointer backend helpers -----------------
    def _map_button_to_evdev(btn: int) -> int:
        if btn == 1:
            return 272  # BTN_LEFT
        if btn == 2:
            return 274  # BTN_MIDDLE
        if btn == 3:
            return 273  # BTN_RIGHT
        return btn

    def _ensure_uinput() -> Optional[object]:
        nonlocal uinput_dev
        if not _HAS_UINPUT:
            return None
        if uinput_dev is not None:
            return uinput_dev
        try:
            caps = {ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE, ecodes.KEY_LEFTMETA]}
            uinput_dev = UInput(caps, name="wm-hand-rel-virtual", bustype=0x11)
            if getattr(args, 'debug', False):
                print("[uinput] device created")
            return uinput_dev
        except Exception as e:
            if getattr(args, 'debug', False):
                print(f"[uinput] create failed: {e}")
            uinput_dev = None
            return None

    def uinput_mouse_down(btn: int) -> bool:
        ui = _ensure_uinput()
        if ui is None:
            return False
        try:
            code = ecodes.BTN_LEFT if btn == 1 else (ecodes.BTN_MIDDLE if btn == 2 else ecodes.BTN_RIGHT)
            ui.write(ecodes.EV_KEY, code, 1)
            ui.syn()
            if getattr(args, 'debug', False):
                print(f"[uinput] mouse down {btn}")
            return True
        except Exception:
            return False

    def uinput_mouse_up(btn: int) -> bool:
        ui = _ensure_uinput()
        if ui is None:
            return False
        try:
            code = ecodes.BTN_LEFT if btn == 1 else (ecodes.BTN_MIDDLE if btn == 2 else ecodes.BTN_RIGHT)
            ui.write(ecodes.EV_KEY, code, 0)
            ui.syn()
            if getattr(args, 'debug', False):
                print(f"[uinput] mouse up {btn}")
            return True
        except Exception:
            return False

    def uinput_super_down() -> bool:
        ui = _ensure_uinput()
        if ui is None:
            return False
        try:
            ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 1)
            ui.syn()
            if getattr(args, 'debug', False):
                print("[uinput] SUPER down")
            return True
        except Exception:
            return False

    def uinput_super_up() -> bool:
        ui = _ensure_uinput()
        if ui is None:
            return False
        try:
            ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 0)
            ui.syn()
            if getattr(args, 'debug', False):
                print("[uinput] SUPER up")
            return True
        except Exception:
            return False

    def ydotool_mouse_down(btn: int) -> bool:
        yk = shutil.which("ydotool")
        if not yk:
            if args.button_backend in ("auto", "uinput") and _HAS_UINPUT:
                return uinput_mouse_down(btn)
            return False
        ev = _map_button_to_evdev(btn)
        try:
            # Intentos con diferentes códigos
            cands = [btn, ev]
            if btn == 1:
                cands += [272]
            elif btn == 2:
                cands += [274]
            elif btn == 3:
                cands += [273]
            seen = set()
            for c in [x for x in cands if not (x in seen or seen.add(x))]:
                p = _run([yk, "mousedown", str(c)])
                if p.returncode == 0:
                    return True
            # Click no cuenta como hold → False, pero intentar uinput
            _run([yk, "click", str(btn)])
            if args.button_backend in ("auto", "uinput") and _HAS_UINPUT:
                return uinput_mouse_down(btn)
            return False
        except Exception:
            return False

    def ydotool_mouse_up(btn: int) -> bool:
        yk = shutil.which("ydotool")
        if not yk:
            if pointer_backend == "uinput":
                return uinput_mouse_up(btn)
            return False
        ev = _map_button_to_evdev(btn)
        try:
            cands = [btn, ev]
            if btn == 1:
                cands += [272]
            elif btn == 2:
                cands += [274]
            elif btn == 3:
                cands += [273]
            seen = set()
            for c in [x for x in cands if not (x in seen or seen.add(x))]:
                p = _run([yk, "mouseup", str(c)])
                if p.returncode == 0:
                    return True
            _run([yk, "click", str(btn)])
            if pointer_backend == "uinput":
                return uinput_mouse_up(btn)
            return False
        except Exception:
            return False

    def pointer_super_down_func() -> bool:
        nonlocal pointer_backend
        if args.button_backend in ("auto", "uinput") and _HAS_UINPUT:
            if uinput_super_down():
                pointer_backend = "uinput"
                return True
        # Fallback con ydotool no implementado de forma fiable; devolvemos False
        return False

    def pointer_super_up_func() -> bool:
        if pointer_backend == "uinput":
            return uinput_super_up()
        return False

    def pointer_mouse_down_func(btn: int) -> bool:
        nonlocal pointer_backend
        if args.button_backend == "uinput" and _HAS_UINPUT:
            pointer_backend = "uinput"
            return uinput_mouse_down(btn)
        return ydotool_mouse_down(btn)

    def pointer_mouse_up_func(btn: int) -> bool:
        if pointer_backend == "uinput":
            return uinput_mouse_up(btn)
        return ydotool_mouse_up(btn)

    move_threshold_px = 2

    def decide_tile_direction(win0: Tuple[int,int], winf: Tuple[int,int], margin: int, heuristic: str) -> Optional[str]:
        dx = winf[0] - win0[0]
        dy = winf[1] - win0[1]
        adx, ady = abs(dx), abs(dy)
        if adx < margin and ady < margin:
            return None
        if heuristic == "none":
            return None
        if heuristic == "vertical-prefer":
            if ady >= adx:
                return "d" if dy > 0 else "u"
            else:
                return "r" if dx > 0 else "l"
        # default: axis (eje dominante)
        if adx >= ady:
            return "r" if dx > 0 else "l"
        else:
            return "d" if dy > 0 else "u"

    def animator():
        nonlocal anim_target, anim_pos, dragging
        dt = 1.0 / max(60.0, args.cursor_fps)
        tau = max(1e-6, args.anim_tau_ms / 1000.0)
        last_sent_xy: Optional[Tuple[int,int]] = None
        while not stop_evt.is_set():
            with target_lock:
                tx, ty = anim_target
                cx, cy = anim_pos
            if tx is not None and ty is not None:
                if cx is None or cy is None:
                    cx, cy = tx, ty
                k = 1.0 - math.exp(-dt / tau)
                cx = cx + k * (tx - cx)
                cy = cy + k * (ty - cy)
                ix, iy = int(round(cx)), int(round(cy))

                if dragging and hand_start_screen is not None:
                    # Δ relativo entre mano actual (coords de monitor) y mano inicial
                    dx = ix - hand_start_screen[0]
                    dy = iy - hand_start_screen[1]
                    if args.drag_mode == "hypr" and win_start_at is not None and drag_window_addr:
                        # Nueva posición absoluta de la ventana
                        nx = win_start_at[0] + dx
                        ny = win_start_at[1] + dy
                        # Evitar spam
                        if last_sent_xy is None or abs(nx - last_sent_xy[0]) > move_threshold_px or abs(ny - last_sent_xy[1]) > move_threshold_px:
                            if hypr_move_window_to_address(drag_window_addr, nx, ny):
                                last_sent_xy = (nx, ny)
                                drag_last_sent_xy = last_sent_xy
                    elif args.drag_mode == "pointer" and pointer_start_abs is not None:
                        # Mover el cursor absoluto desde su punto inicial + Δ
                        cx_abs = pointer_start_abs[0] + dx
                        cy_abs = pointer_start_abs[1] + dy
                        if last_sent_xy is None or abs(cx_abs - last_sent_xy[0]) > move_threshold_px or abs(cy_abs - last_sent_xy[1]) > move_threshold_px:
                            if ydotool_move_abs(int(cx_abs), int(cy_abs)):
                                last_sent_xy = (int(cx_abs), int(cy_abs))

                with target_lock:
                    anim_pos = (cx, cy)
            time.sleep(dt)

    t_anim = threading.Thread(target=animator, daemon=True)
    t_anim.start()

    def focus_tracker():
        """Mantiene un MRU por workspace a partir de la ventana activa."""
        last_addr = None
        while not stop_evt.is_set():
            info = hypr_active_window_info()
            if info:
                addr = str(info.get("address") or "")
                wobj = info.get("workspace") or {}
                wid = int(wobj.get("id") or (hypr_active_workspace() or -1))
                title = str(info.get("title") or "")
                if addr and title != PREVIEW_TITLE and addr != last_addr and wid is not None:
                    with mru_lock:
                        lst = mru_by_ws.get(wid, [])
                        try:
                            if addr in lst:
                                lst.remove(addr)
                        except Exception:
                            pass
                        lst.insert(0, addr)
                        # Limitar tamaño razonable
                        if len(lst) > 64:
                            lst = lst[:64]
                        mru_by_ws[wid] = lst
                    last_addr = addr
            time.sleep(0.15)

    t_focus = threading.Thread(target=focus_tracker, daemon=True)
    t_focus.start()

    try:
        was_pinch = False
        pinch_hold_t0: Optional[float] = None
        pinch_filter = OneEuro(min_cutoff=2.0, beta=0.0, d_cutoff=1.0)
        # Estado para selección por swipe (índice+medio)
        select_was_together = False
        select_hold_t0: Optional[float] = None  # (legacy hold)
        select_done_this_hold = False           # (legacy hold)
        select_filter = OneEuro(min_cutoff=2.0, beta=0.0, d_cutoff=1.0)  # (legacy hold)
        swipe_active = False
        swipe_start_px: Optional[Tuple[int,int]] = None
        last_swipe_t = 0.0

        def workspace_clients_excl_preview() -> List[Dict]:
            wid = hypr_active_workspace()
            return [c for c in hypr_clients() if (not wid or (c.get("workspace", {}).get("id") == wid)) and (str(c.get("title") or "") != PREVIEW_TITLE)]

        def cycle_focus(direction: str) -> Optional[str]:
            nonlocal selected_window_addr
            """direction: 'next' o 'prev'. Devuelve address enfocada o None."""
            clis = workspace_clients_excl_preview()
            if not clis:
                return None
            act = hypr_active_window_info()
            cur_addr = str((act or {}).get("address") or "")
            wid = hypr_active_workspace() or -1
            # Filtrar clientes a (addr, x, y)
            addrs = [str(c.get("address") or "") for c in clis]
            if args.select_order == "mru":
                # Construir lista MRU∩clientes (en orden MRU)
                with mru_lock:
                    base = list(mru_by_ws.get(wid, []))
                ordered = [a for a in base if a in addrs]
                # Complementar con clientes no vistos aún
                for a in addrs:
                    if a not in ordered:
                        ordered.append(a)
            else:
                # Orden por posición (x,y)
                clis.sort(key=lambda c: ((c.get("at") or [0,0])[0], (c.get("at") or [0,0])[1]))
                ordered = [str(c.get("address") or "") for c in clis]

            if not ordered:
                return None
            try:
                idx = ordered.index(cur_addr)
            except ValueError:
                idx = -1
            if direction == 'next':
                idx = (idx + 1) % len(ordered)
            else:
                idx = (idx - 1 + len(ordered)) % len(ordered)
            addr = ordered[idx]
            if addr:
                hypr_focus_window_address(addr)
                # Persistir selección para usarla en el pinch START
                selected_window_addr = addr
            return addr or None

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            vis_frame = frame_bgr.copy() if args.preview else frame_bgr
            if args.flip:
                frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            infer_img = frame_rgb
            if 0.3 <= args.inference_scale < 1.0:
                new_w = max(64, int(frame_rgb.shape[1] * args.inference_scale))
                new_h = max(64, int(frame_rgb.shape[0] * args.inference_scale))
                infer_img = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_img)
            t_ms = time.monotonic_ns() // 1_000_000
            result = landmarker.detect_for_video(mp_image, t_ms)

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                try:
                    ix, iy = float(lms[INDEX_TIP].x), float(lms[INDEX_TIP].y)
                    tx, ty = float(lms[THUMB_TIP].x), float(lms[THUMB_TIP].y)
                    mx, my = float(lms[MIDDLE_TIP].x), float(lms[MIDDLE_TIP].y)
                except Exception:
                    ix = iy = tx = ty = mx = my = float('nan')
                valid_thumb = 0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0
                valid_index = 0.0 <= ix <= 1.0 and 0.0 <= iy <= 1.0
                if not valid_thumb:
                    # cancelar y asegurar cleanup del drag actual (si lo hay)
                    if pointer_drag_active:
                        try:
                            pointer_mouse_up_func(args.mouse_button)
                        except Exception:
                            pass
                        try:
                            pointer_super_up_func()
                        except Exception:
                            pass
                        pointer_drag_active = False
                        pointer_start_abs = None
                    if drag_window_addr:
                        try:
                            hypr_set_floating_address(drag_window_addr, False)
                        except Exception:
                            pass
                        drag_window_addr = None
                    dragging = False
                    was_pinch = False
                    pinch_hold_t0 = None
                    toggled_float_this_hold = False
                    select_was_together = False
                    select_hold_t0 = None
                    select_done_this_hold = False
                    if args.preview:
                        cv2.putText(vis_frame, "Thumb not detected", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,200), 2)
                        disp = cv2.flip(vis_frame, 1)
                        cv2.imshow("Pinch Drag Preview", disp)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                    continue

                if args.invert_x:
                    ix = 1.0 - ix
                    tx = 1.0 - tx
                    mx = 1.0 - mx
                tnow = time.monotonic()
                fx = clamp(filt_x.filter(tnow, ix), 0.0, 1.0)
                fy = clamp(filt_y.filter(tnow, iy), 0.0, 1.0)
                x_scr = fx * screen_w
                y_scr = fy * screen_h
                y_out = y_scr if args.invert_y else (screen_h - y_scr)
                x_out = x_scr
                with target_lock:
                    anim_target = (float(x_out), float(y_out))

                # distancia índice-pulgar
                dxn = ix - tx
                dyn = iy - ty
                raw_dist = math.hypot(dxn, dyn)
                if args.pinch_relative:
                    minx = min(pt.x for pt in lms)
                    maxx = max(pt.x for pt in lms)
                    miny = min(pt.y for pt in lms)
                    maxy = max(pt.y for pt in lms)
                    diag = max(1e-4, math.hypot(maxx - minx, maxy - miny))
                    base_thresh = args.pinch_rel_threshold * diag
                else:
                    base_thresh = args.pinch_threshold
                dist = pinch_filter.filter(tnow, raw_dist)
                thresh_on = base_thresh
                thresh_off = base_thresh + args.pinch_hysteresis
                is_pinch = dist < (thresh_on if not was_pinch else thresh_off)

                # Detección de selección (modo swipe u hold)
                sel_dist_raw = math.hypot(ix - mx, iy - my)
                if args.select_mode == 'hold':
                    if args.select_relative:
                        minx = min(pt.x for pt in lms)
                        maxx = max(pt.x for pt in lms)
                        miny = min(pt.y for pt in lms)
                        maxy = max(pt.y for pt in lms)
                        diag = max(1e-4, math.hypot(maxx - minx, maxy - miny))
                        sel_thresh = args.select_rel_threshold * diag
                    else:
                        sel_thresh = args.select_threshold
                    sel_dist = select_filter.filter(tnow, sel_dist_raw)
                    is_together = sel_dist < sel_thresh
                    if is_together and not select_was_together:
                        select_hold_t0 = time.monotonic()
                        select_done_this_hold = False
                    elif is_together and select_hold_t0 is not None and not select_done_this_hold:
                        if (time.monotonic() - select_hold_t0) * 1000.0 >= args.select_hold_ms:
                            # Seleccionar ventana bajo el dedo (coords absolutas)
                            abs_x = mon_ox + int(x_out)
                            abs_y = mon_oy + int(y_out)
                            picked = pick_window_under_point(abs_x, abs_y) or hypr_active_window_info()
                            if picked:
                                selected_window_addr = str(picked.get("address") or "")
                                if args.select_focus and selected_window_addr:
                                    hypr_focus_window_address(selected_window_addr)
                                if args.debug:
                                    t = picked.get("title")
                                    fl = picked.get("floating", False)
                                    print(f"[select-hold] addr={selected_window_addr} title={t} floating={fl}")
                            select_done_this_hold = True
                    elif not is_together and select_was_together:
                        select_hold_t0 = None
                        select_done_this_hold = False
                    select_was_together = is_together
                else:
                    # swipe: índice+medio juntos Activan modo swipe y un desplazamiento horizontal cicla
                    # Determinar si están "juntos" para activar swipe (umbral relativo sencillo)
                    minx = min(pt.x for pt in lms)
                    maxx = max(pt.x for pt in lms)
                    miny = min(pt.y for pt in lms)
                    maxy = max(pt.y for pt in lms)
                    diag = max(1e-4, math.hypot(maxx - minx, maxy - miny))
                    sel_thresh = 0.10 * diag  # fijo y robusto
                    is_together = sel_dist_raw < sel_thresh

                    # Midpoint índice-medio en pixeles de pantalla (coords del monitor enfocado)
                    midx_n = (ix + mx) * 0.5
                    midy_n = (iy + my) * 0.5
                    if args.invert_x:
                        midx_n = 1.0 - midx_n
                    midx_px = clamp(filt_mid_x.filter(tnow, midx_n), 0.0, 1.0) * screen_w
                    midy_px = clamp(filt_mid_y.filter(tnow, midy_n), 0.0, 1.0) * screen_h
                    midy_px = midy_px if args.invert_y else (screen_h - midy_px)

                    if is_together and not select_was_together and not is_pinch:
                        swipe_active = True
                        swipe_start_px = (int(midx_px), int(midy_px))
                        if args.debug:
                            print(f"[swipe] start at {swipe_start_px}")
                    elif is_together and swipe_active and swipe_start_px is not None and not is_pinch:
                        dx = int(midx_px) - swipe_start_px[0]
                        dy = int(midy_px) - swipe_start_px[1]
                        if abs(dx) >= args.swipe_threshold_px and abs(dx) > abs(dy):
                            nowt = time.monotonic()
                            if (nowt - last_swipe_t) * 1000.0 >= args.swipe_cooldown_ms:
                                addr = cycle_focus('next' if dx > 0 else 'prev')
                                if addr:
                                    selected_window_addr = addr
                                    if args.debug:
                                        print(f"[swipe] {'next' if dx>0 else 'prev'} → {addr}")
                                last_swipe_t = nowt
                                swipe_start_px = (int(midx_px), int(midy_px))
                    elif not is_together and swipe_active:
                        swipe_active = False
                        swipe_start_px = None

                    select_was_together = is_together

                # Transiciones de pellizco
                if is_pinch and not was_pinch:
                    now = time.monotonic()
                    if pinch_hold_t0 is None:
                        pinch_hold_t0 = now
                        # esperar confirmación de hold mínimo
                        if args.preview:
                            disp = cv2.flip(vis_frame, 1)
                            cv2.imshow("Pinch Drag Preview", disp)
                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                        continue
                    elif (now - pinch_hold_t0) * 1000.0 < args.pinch_hold_ms:
                        if args.preview:
                            disp = cv2.flip(vis_frame, 1)
                            cv2.imshow("Pinch Drag Preview", disp)
                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                        continue

                    # Preparar ventana objetivo: SIEMPRE usar selección recordada o activa. Sin fallback por posición.
                    picked = None
                    target_addr = None
                    if selected_window_addr:
                        target_addr = selected_window_addr
                    else:
                        info = hypr_active_window_info()
                        if info and info.get("title") != PREVIEW_TITLE:
                            target_addr = str(info.get("address") or "")
                            # Memorizar selección basada en foco actual
                            if target_addr:
                                selected_window_addr = target_addr

                    if target_addr:
                        # Intentar obtener cliente completo por address
                        clis = hypr_clients()
                        for c in clis:
                            if str(c.get("address") or "") == target_addr:
                                picked = c
                                break
                        if picked is None:
                            # Fallback mínimo: usar activewindow info si coincide address
                            info = hypr_active_window_info()
                            if info and str(info.get("address") or "") == target_addr:
                                picked = info
                    if picked and str(picked.get("address") or ""):
                        drag_window_addr = str(picked.get("address") or "")
                        # Memorizar selección en el inicio del pellizco
                        selected_window_addr = drag_window_addr
                        # Garantizar flotante para mover libremente
                        if not picked.get("floating", False) and drag_window_addr:
                            hypr_set_floating_address(drag_window_addr, True)
                        at = picked.get("at") or [0, 0]
                        size = picked.get("size") or [0, 0]
                        win_start_at = (int(at[0]), int(at[1]))
                        # Guardar mano inicial respecto al monitor local (anim coords)
                        with target_lock:
                            cx, cy = anim_target
                        hand_start_screen = (int(cx), int(cy))
                        # Centrar cursor en la ventana
                        cx_abs = int(at[0]) + int(size[0]) // 2
                        cy_abs = int(at[1]) + int(size[1]) // 2
                        ydotool_move_abs(cx_abs, cy_abs)
                        pointer_start_abs = (cx_abs, cy_abs)
                        # Si modo pointer: presionar SUPER + botón
                        if args.drag_mode == "pointer":
                            if pointer_super_down_func():
                                pointer_backend_local = pointer_backend  # noqa: F841 (debug aid)
                            pointer_mouse_down_func(args.mouse_button)
                            pointer_drag_active = True
                        dragging = True
                        pinch_started_at = now
                        toggled_float_this_hold = False
                        if args.debug:
                            print(f"[pinch] START rel; hand0={hand_start_screen} win0={win_start_at} addr={drag_window_addr}")
                    else:
                        # Sin ventana, no iniciar drag
                        pinch_hold_t0 = None

                elif is_pinch and was_pinch:
                    # Dentro del pellizco ya activo: si no se ha movido y excede toggle-long → alternar float/tile
                    if dragging and pinch_started_at and not toggled_float_this_hold:
                        elapsed_ms = (time.monotonic() - pinch_started_at) * 1000.0
                        # Si la mano prácticamente no se ha movido desde el inicio
                        with target_lock:
                            cx, cy = anim_target
                        if hand_start_screen is not None and math.hypot((cx - hand_start_screen[0]), (cy - hand_start_screen[1])) < max(6, args.snap_margin * 0.05):
                            if elapsed_ms >= args.toggle_float_long_ms and drag_window_addr:
                                # Alternar: si actualmente flotante → tile; si tile → float
                                hypr_toggle_floating_address(drag_window_addr)
                                toggled_float_this_hold = True
                                if args.debug:
                                    print("[pinch] long → toggle float/tile")

                elif not is_pinch and was_pinch:
                    # FIN del pellizco
                    # Si modo pointer: soltar botón y SUPER
                    if pointer_drag_active:
                        try:
                            pointer_mouse_up_func(args.mouse_button)
                        except Exception:
                            pass
                        try:
                            pointer_super_up_func()
                        except Exception:
                            pass
                        pointer_drag_active = False
                        pointer_start_abs = None
                    # Antes de limpiar, si hay addr, devolver a tile y ubicar en lado según desplazamiento
                    if drag_window_addr:
                        # Volver a tile
                        hypr_set_floating_address(drag_window_addr, False)
                        if args.tile_heuristic != "none" and args.drag_mode == "hypr" and win_start_at and drag_last_sent_xy:
                            side = decide_tile_direction(win_start_at, drag_last_sent_xy, args.snap_margin, args.tile_heuristic)
                            if args.tile_heuristic == "quadrant" and side is None:
                                side = None  # no-op
                            if side:
                                hypr_move_tiled_side_address(drag_window_addr, side)
                            elif args.tile_heuristic == "quadrant" and win_start_at and drag_last_sent_xy:
                                # Determinar dos ejes
                                dx = drag_last_sent_xy[0] - win_start_at[0]
                                dy = drag_last_sent_xy[1] - win_start_at[1]
                                adx, ady = abs(dx), abs(dy)
                                if ady >= args.snap_margin:
                                    hypr_move_tiled_side_address(drag_window_addr, "d" if dy > 0 else "u")
                                if adx >= args.snap_margin:
                                    hypr_move_tiled_side_address(drag_window_addr, "r" if dx > 0 else "l")
                        elif args.tile_heuristic != "none":
                            # pointer mode: usar Δ de mano actual respecto a inicio
                            if hand_start_screen is not None and anim_pos[0] is not None:
                                cx, cy = anim_pos
                                dx = int(cx - hand_start_screen[0])
                                dy = int(cy - hand_start_screen[1])
                                adx, ady = abs(dx), abs(dy)
                                if args.tile_heuristic == "quadrant":
                                    if ady >= args.snap_margin:
                                        hypr_move_tiled_side_address(drag_window_addr, "d" if dy > 0 else "u")
                                    if adx >= args.snap_margin:
                                        hypr_move_tiled_side_address(drag_window_addr, "r" if dx > 0 else "l")
                                else:
                                    # 'axis' o 'vertical-prefer'
                                    # Construir winf sintético para reutilizar decide_tile_direction
                                    winf = (win_start_at[0] + dx, win_start_at[1] + dy) if win_start_at else (dx, dy)
                                    side = decide_tile_direction(win_start_at or (0,0), winf, args.snap_margin, args.tile_heuristic)
                                    if side:
                                        hypr_move_tiled_side_address(drag_window_addr, side)
                    dragging = False
                    hand_start_screen = None
                    win_start_at = None
                    drag_window_addr = None
                    drag_last_sent_xy = None
                    pinch_hold_t0 = None
                    pinch_started_at = None
                    toggled_float_this_hold = False

                was_pinch = is_pinch

                if args.preview:
                    for lm in [lms[INDEX_TIP], lms[THUMB_TIP]]:
                        cxp = int(lm.x * vis_frame.shape[1])
                        cyp = int(lm.y * vis_frame.shape[0])
                        cv2.circle(vis_frame, (cxp, cyp), 6, (0, 255, 0), -1)
                    p1 = (int(lms[INDEX_TIP].x * vis_frame.shape[1]), int(lms[INDEX_TIP].y * vis_frame.shape[0]))
                    p2 = (int(lms[THUMB_TIP].x * vis_frame.shape[1]), int(lms[THUMB_TIP].y * vis_frame.shape[0]))
                    cv2.line(vis_frame, p1, p2, (0, 200, 255), 2)
                    msg = f"PINCH {'ON' if is_pinch else 'OFF'} dist={dist:.3f}"
                    cv2.putText(vis_frame, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50) if is_pinch else (0, 0, 200), 2)
                    # Dibujar estado selección
                    mode = args.select_mode
                    selmsg = f"SEL-{mode} {'ON' if select_was_together else 'OFF'}"
                    cv2.putText(vis_frame, selmsg, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (200, 200, 0) if select_was_together else (120,120,120), 2)
            else:
                if args.preview:
                    cv2.putText(vis_frame, "No hand", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

            if args.preview:
                disp = cv2.flip(vis_frame, 1)
                cv2.imshow("Pinch Drag Preview", disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        try:
            cap.release()
        except Exception:
            pass
        stop_evt.set()
        try:
            t_anim.join(timeout=0.2)
        except Exception:
            pass
        try:
            t_focus.join(timeout=0.2)
        except Exception:
            pass
        # Cleanup adicional: si quedó arrastre a medias, liberar y asegurar tile
        try:
            if pointer_drag_active:
                try:
                    pointer_mouse_up_func(args.mouse_button)
                except Exception:
                    pass
                try:
                    pointer_super_up_func()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if drag_window_addr:
                hypr_set_floating_address(drag_window_addr, False)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ----------------- Pointer backend helpers -----------------
    # Nota: definimos aquí para acceder a 'args' y variables cerradas.
    def _map_button_to_evdev(btn: int) -> int:
        if btn == 1:
            return 272  # BTN_LEFT
        if btn == 2:
            return 274  # BTN_MIDDLE
        if btn == 3:
            return 273  # BTN_RIGHT
        return btn

    def _ensure_uinput() -> Optional[object]:
        nonlocal uinput_dev
        if not _HAS_UINPUT:
            return None
        if uinput_dev is not None:
            return uinput_dev
        try:
            caps = {ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE, ecodes.KEY_LEFTMETA]}
            uinput_dev = UInput(caps, name="wm-hand-rel-virtual", bustype=0x11)
            if getattr(args, 'debug', False):
                print("[uinput] device created")
            return uinput_dev
        except Exception as e:
            if getattr(args, 'debug', False):
                print(f"[uinput] create failed: {e}")
            uinput_dev = None
            return None

    def uinput_mouse_down(btn: int) -> bool:
        ui = _ensure_uinput()
        if ui is None:
            return False
        try:
            code = ecodes.BTN_LEFT if btn == 1 else (ecodes.BTN_MIDDLE if btn == 2 else ecodes.BTN_RIGHT)
            ui.write(ecodes.EV_KEY, code, 1)
            ui.syn()
            if getattr(args, 'debug', False):
                print(f"[uinput] mouse down {btn}")
            return True
        except Exception:
            return False

    def uinput_mouse_up(btn: int) -> bool:
        ui = _ensure_uinput()
        if ui is None:
            return False
        try:
            code = ecodes.BTN_LEFT if btn == 1 else (ecodes.BTN_MIDDLE if btn == 2 else ecodes.BTN_RIGHT)
            ui.write(ecodes.EV_KEY, code, 0)
            ui.syn()
            if getattr(args, 'debug', False):
                print(f"[uinput] mouse up {btn}")
            return True
        except Exception:
            return False

    def uinput_super_down() -> bool:
        ui = _ensure_uinput()
        if ui is None:
            return False
        try:
            ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 1)
            ui.syn()
            if getattr(args, 'debug', False):
                print("[uinput] SUPER down")
            return True
        except Exception:
            return False

    def uinput_super_up() -> bool:
        ui = _ensure_uinput()
        if ui is None:
            return False
        try:
            ui.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 0)
            ui.syn()
            if getattr(args, 'debug', False):
                print("[uinput] SUPER up")
            return True
        except Exception:
            return False

    def ydotool_mouse_down(btn: int) -> bool:
        yk = shutil.which("ydotool")
        if not yk:
            if args.button_backend in ("auto", "uinput") and _HAS_UINPUT:
                return uinput_mouse_down(btn)
            return False
        ev = _map_button_to_evdev(btn)
        try:
            # Intentos con diferentes códigos
            cands = [btn, ev]
            if btn == 1:
                cands += [272]
            elif btn == 2:
                cands += [274]
            elif btn == 3:
                cands += [273]
            seen = set()
            for c in [x for x in cands if not (x in seen or seen.add(x))]:
                p = _run([yk, "mousedown", str(c)])
                if p.returncode == 0:
                    return True
            # Click no cuenta como hold → False, pero intentar uinput
            _run([yk, "click", str(btn)])
            if args.button_backend in ("auto", "uinput") and _HAS_UINPUT:
                return uinput_mouse_down(btn)
            return False
        except Exception:
            return False

    def ydotool_mouse_up(btn: int) -> bool:
        yk = shutil.which("ydotool")
        if not yk:
            if pointer_backend == "uinput":
                return uinput_mouse_up(btn)
            return False
        ev = _map_button_to_evdev(btn)
        try:
            cands = [btn, ev]
            if btn == 1:
                cands += [272]
            elif btn == 2:
                cands += [274]
            elif btn == 3:
                cands += [273]
            seen = set()
            for c in [x for x in cands if not (x in seen or seen.add(x))]:
                p = _run([yk, "mouseup", str(c)])
                if p.returncode == 0:
                    return True
            _run([yk, "click", str(btn)])
            if pointer_backend == "uinput":
                return uinput_mouse_up(btn)
            return False
        except Exception:
            return False

    def pointer_super_down_func() -> bool:
        nonlocal pointer_backend
        if args.button_backend in ("auto", "uinput") and _HAS_UINPUT:
            if uinput_super_down():
                pointer_backend = "uinput"
                return True
        # Fallback con ydotool no implementado de forma fiable; devolvemos False
        return False

    def pointer_super_up_func() -> bool:
        if pointer_backend == "uinput":
            return uinput_super_up()
        return False

    def pointer_mouse_down_func(btn: int) -> bool:
        if args.button_backend == "uinput" and _HAS_UINPUT:
            return uinput_mouse_down(btn)
        return ydotool_mouse_down(btn)

    def pointer_mouse_up_func(btn: int) -> bool:
        if pointer_backend == "uinput":
            return uinput_mouse_up(btn)
        return ydotool_mouse_up(btn)

if __name__ == "__main__":
    main()
