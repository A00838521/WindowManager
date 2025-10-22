#!/usr/bin/env python3
"""
Linux (Wayland/Hyprland): Arrastrar ventanas con gesto de pellizco (índice + pulgar).

Qué hace
- Detecta un pellizco entre la punta del índice y el pulgar usando MediaPipe Hands.
- Con pellizco activo: “agarra” la ventana objetivo y la mueve siguiendo la mano.
- Al soltar: detiene el arrastre. Incluye selección de ventana con índice+medio y snap opcional.

Recomendado (por compatibilidad en Wayland)
- Modo pointer con backend uinput: mantiene SUPER y BTN_LEFT pulsados de forma fiable.
    Ejemplo: QT_QPA_PLATFORM=xcb python linux/hand_pinch_window_drag.py --preview 1 --debug --drag-mode pointer --button-backend uinput

Requisitos
- Hyprland con hyprctl en PATH; Python 3.11 con mediapipe, opencv-python, numpy.
- Para pointer+uinput: paquete python-evdev y permisos de /dev/uinput.
- En Wayland, para evitar errores de Qt en imshow, se fuerza QT_QPA_PLATFORM=xcb.

Contrato de E/S
- Entrada: frames de cámara y posiciones normalizadas de landmarks (0..1).
- Salida: en modo pointer (recomendado), eventos de SUPER+BTN_LEFT sostenidos y mousemove absoluto; en modo hypr, movewindowpixel/resizewindowpixel.
- Errores esperados: cámara no disponible; hyprctl/evdev/ydotool no encontrados; falta de landmarks.
"""
import os, time, argparse, shutil, json, subprocess, math, threading
from typing import Optional, Tuple, List, Dict, Any

# Evitar fallo de Qt en Wayland: forzar plugin XCB (XWayland) para imshow
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Opcional: backend uinput para sostener botones cuando ydotool no soporta hold
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


# (El helper hypr_active_monitor_info ha sido eliminado; se usa hypr_focused_monitor_geometry.)

def hypr_focused_monitor_geometry() -> Tuple[int, int, Optional[float], int, int]:
    """Como hypr_active_monitor_info, pero incluye el offset absoluto del monitor.

    Returns:
        (width, height, hz, ox, oy) donde (ox,oy) es la posición del monitor en el
        espacio global (útil para convertir a coords absolutas de Hyprland).
    """
    w, h, hz = 1920, 1080, None
    ox = oy = 0
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
                    if "width" in mon and "height" in mon:
                        w, h = int(mon["width"]), int(mon["height"])
                    elif isinstance(mon.get("size"), dict):
                        w = int(mon["size"].get("x", w))
                        h = int(mon["size"].get("y", h))
                    hz = float(mon.get("refreshRate") or mon.get("refresh_rate") or 0.0) or None
                    # offsets
                    if "x" in mon and "y" in mon:
                        ox, oy = int(mon["x"]), int(mon["y"])
                    elif isinstance(mon.get("position"), dict):
                        ox = int(mon["position"].get("x", 0))
                        oy = int(mon["position"].get("y", 0))
        except Exception:
            pass
    return w, h, hz, ox, oy


def hypr_move_active_window_to(x: int, y: int) -> bool:
    """Mueve la ventana activa al punto indicado en píxeles de pantalla.

    Usa el dispatcher de Hyprland: `hyprctl dispatch movewindowpixel exact X Y`.

    Args:
        x: Coordenada X de pantalla destino.
        y: Coordenada Y de pantalla destino.

    Returns:
        True si el comando se ejecutó sin errores, False en caso contrario.
    """
    hyprctl = shutil.which("hyprctl")
    if not hyprctl:
        return False
    try:
        # Hyprland admite movewindowpixel exact X Y
        proc = subprocess.run([hyprctl, "dispatch", "movewindowpixel", "exact", str(x), str(y)], capture_output=True, text=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        if "Invalid" in out or proc.returncode != 0:
            return False
        return True
    except Exception:
        return False

def hypr_move_window_to_address(addr: str, x: int, y: int) -> bool:
    """Mueve una ventana específica identificada por address a (x,y) absolutos.

    Estrategia segura: enfoca la ventana por address y luego usa movewindowpixel exact
    (que actúa sobre la activa). Evita que el preview sea el que se mueva.
    """
    if not addr:
        return False
    hyprctl = shutil.which("hyprctl")
    if hyprctl:
        try:
            # Intento 1: mover por address explícito (si la versión lo soporta)
            proc = subprocess.run(
                [hyprctl, "dispatch", "movewindowpixel", "exact", str(int(x)), str(int(y)), f"address:{addr}"],
                capture_output=True,
                text=True,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0 and "Invalid" not in out:
                return True
        except Exception:
            pass
    # Fallback: enfocar y mover como activa
    hypr_focus_window_address(addr)
    return hypr_move_active_window_to(x, y)

def hypr_resize_window_to_address(addr: str, w: int, h: int) -> bool:
    if not addr:
        return False
    hyprctl = shutil.which("hyprctl")
    if hyprctl:
        try:
            proc = subprocess.run(
                [hyprctl, "dispatch", "resizewindowpixel", "exact", str(int(w)), str(int(h)), f"address:{addr}"],
                capture_output=True,
                text=True,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0 and "Invalid" not in out:
                return True
        except Exception:
            pass
    hypr_focus_window_address(addr)
    return hypr_resize_active_window_to(w, h)


def hypr_active_window_info() -> Optional[dict]:
    """Consulta la ventana activa en Hyprland.

    Returns:
        Diccionario JSON con metadatos de la ventana (title, class, at, size, address),
        o None si falla la consulta.
    """
    hyprctl = shutil.which("hyprctl")
    if not hyprctl:
        return None
    try:
        proc = subprocess.run([hyprctl, "-j", "activewindow"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout:
            data = json.loads(proc.stdout)
            return data
    except Exception:
        pass
    return None

def hypr_clients() -> List[Dict]:
    """Lista de clientes (ventanas) de Hyprland como dicts."""
    hyprctl = shutil.which("hyprctl")
    if not hyprctl:
        return []
    try:
        proc = subprocess.run([hyprctl, "-j", "clients"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout:
            return json.loads(proc.stdout)
    except Exception:
        pass
    return []

def hypr_active_workspace() -> Optional[int]:
    hyprctl = shutil.which("hyprctl")
    if not hyprctl:
        return None
    try:
        proc = subprocess.run([hyprctl, "-j", "activeworkspace"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout:
            data = json.loads(proc.stdout)
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
        proc = subprocess.run([hyprctl, "dispatch", "focuswindow", f"address:{addr}"], capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False

def hypr_toggle_floating_address(addr: str) -> bool:
    hyprctl = shutil.which("hyprctl")
    if not hyprctl or not addr:
        return False
    try:
        proc = subprocess.run([hyprctl, "dispatch", "togglefloating", f"address:{addr}"], capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False

def hypr_set_floating_address(addr: str, on: bool) -> bool:
    hyprctl = shutil.which("hyprctl")
    if not hyprctl or not addr:
        return False
    try:
        proc = subprocess.run([hyprctl, "dispatch", "setfloating", f"address:{addr}", "yes" if on else "no"], capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False

def hypr_move_window_to_current_workspace(addr: str) -> bool:
    """Mueve una ventana (por address) al workspace actual, de forma silenciosa.

    Intenta usar el dispatcher con address; si falla, enfoca la ventana y mueve la activa.
    """
    if not addr:
        return False
    hyprctl = shutil.which("hyprctl")
    if not hyprctl:
        return False
    try:
        # Intento por address (si la versión lo soporta)
        proc = subprocess.run([hyprctl, "dispatch", "movetoworkspacesilent", "current", f"address:{addr}"], capture_output=True, text=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0 and "Invalid" not in out:
            return True
    except Exception:
        pass
    # Fallback: enfocar, luego mover la activa
    try:
        hypr_focus_window_address(addr)
        proc2 = subprocess.run([hyprctl, "dispatch", "movetoworkspacesilent", "current"], capture_output=True, text=True)
        return proc2.returncode == 0
    except Exception:
        return False

def hypr_resize_active_window_to(w: int, h: int) -> bool:
    hyprctl = shutil.which("hyprctl")
    if not hyprctl:
        return False
    try:
        proc = subprocess.run([hyprctl, "dispatch", "resizewindowpixel", "exact", str(int(w)), str(int(h))], capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False

def client_contains_point(cli: Dict, x_abs: int, y_abs: int) -> bool:
    at = cli.get("at") or [0, 0]
    size = cli.get("size") or [0, 0]
    cx, cy = int(at[0]), int(at[1])
    cw, ch = int(size[0]), int(size[1])
    return (cx <= x_abs <= cx + cw) and (cy <= y_abs <= cy + ch)

def pick_window_under_point(x_abs: int, y_abs: int) -> Optional[Dict]:
    """Selecciona la ventana visible bajo el punto absoluto (x,y).

    Se filtra por workspace activo y se elige la que contenga el punto. Si hay
    múltiples, se escoge la de menor área (asunción de topmost aproximada).
    """
    wid = hypr_active_workspace()
    clients = [c for c in hypr_clients() if (not wid or (c.get("workspace", {}).get("id") == wid))]
    # Excluir la ventana del preview de OpenCV para no seleccionarla por accidente
    clients = [c for c in clients if (str(c.get("title") or "") != PREVIEW_TITLE)]
    inside = [c for c in clients if client_contains_point(c, x_abs, y_abs)]
    if not inside:
        return None
    # Elegir la de menor área para aproximar topmost
    def area(c):
        sz = c.get("size") or [0, 0]
        return int(sz[0]) * int(sz[1])
    inside.sort(key=area)
    return inside[0]


def try_ydotool_move_abs(x: int, y: int) -> bool:
    """Mueve el cursor en modo absoluto con ydotool.

    Requiere que el demonio ydotoold esté corriendo en Wayland.

    Args:
        x: Coordenada X objetivo (px).
        y: Coordenada Y objetivo (px).

    Returns:
        True si el comando se ejecutó correctamente; False en caso de error o si
        ydotool no está instalado.
    """
    ydotool = shutil.which("ydotool")
    if not ydotool:
        return False
    try:
        proc = subprocess.run([ydotool, "mousemove", "-a", "-x", str(x), "-y", str(y)], capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False


# (Elos helpers hypr_begin_drag/hypr_end_drag ya no son necesarios.)


def main():
    """Punto de entrada del script de arrastre por pellizco.

    - Configura captura de cámara, modelo de MediaPipe y parámetros CLI.
    - Lanza un hilo animador que interpola suavemente hacia el objetivo.
    - Detecta el gesto de pellizco y produce comandos a Hyprland.
    """
    p = argparse.ArgumentParser(description="Arrastrar ventana con pellizco (Hyprland)")
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
    p.add_argument("--pinch-threshold", type=float, default=0.055, help="Umbral de distancia normalizada índice-pulgar para considerar pellizco")
    p.add_argument("--pinch-hysteresis", type=float, default=0.010, help="Histeresis adicional al soltar para evitar parpadeo")
    p.add_argument("--pinch-hold-ms", type=float, default=50.0, help="Tiempo mínimo (ms) de pellizco estable antes de iniciar arrastre")
    pr = p.add_mutually_exclusive_group()
    pr.add_argument("--pinch-relative", dest="pinch_relative", action="store_true", help="Umbral relativo al tamaño de la mano")
    pr.add_argument("--no-pinch-relative", dest="pinch_relative", action="store_false")
    p.set_defaults(pinch_relative=True)
    p.add_argument("--pinch-rel-threshold", type=float, default=0.12, help="Factor relativo por la diagonal del bbox de la mano")
    p.add_argument("--preview", type=int, default=0, choices=[0,1])
    p.add_argument("--focus-under-finger", action="store_true", help="Enfoca y selecciona la ventana bajo el dedo al iniciar pellizco")
    p.add_argument("--toggle-floating", action="store_true", help="Forzar ventana a flotante al iniciar drag")
    p.add_argument("--snap", dest="snap", action="store_true", help="Activar snap a esquinas/edges al soltar")
    p.add_argument("--no-snap", dest="snap", action="store_false")
    p.set_defaults(snap=True, focus_under_finger=True, toggle_floating=True)
    p.add_argument("--snap-margin", type=int, default=64, help="Margen (px) desde bordes para activar snap")
    p.add_argument("--hud", type=int, default=1, choices=[0,1], help="Dibujar HUD de ventana seleccionada en preview")
    p.add_argument("--hide-preview-on-drag", dest="hide_preview_on_drag", action="store_true", help="Oculta el preview durante el arrastre para evitar que tome el foco")
    p.add_argument("--show-preview-on-drag", dest="hide_preview_on_drag", action="store_false")
    p.set_defaults(hide_preview_on_drag=True)
    # Modo de arrastre universal por puntero (SUPER+LMB con ydotool)
    p.add_argument("--drag-mode", choices=["hypr", "pointer"], default="pointer", help="Modo de arrastre: hypr (movewindowpixel) o pointer (SUPER+LMB con ydotool)")
    p.add_argument("--super-keycode", type=int, default=125, help="Keycode evdev para SUPER (LeftMeta=125)")
    p.add_argument("--mouse-button", type=int, default=1, help="Botón de mouse para arrastrar (1=izq, 2=medio, 3=der). Se intentará fallback a evdev si es necesario")
    p.add_argument("--button-backend", choices=["auto", "ydotool", "uinput"], default="uinput", help="Backend para mantener el botón: ydotool o uinput (requiere permisos /dev/uinput). Auto prueba ydotool y cae a uinput")
    # Selección con índice+medio juntos (como 'click' de selección)
    sg = p.add_mutually_exclusive_group()
    sg.add_argument("--select-relative", dest="select_relative", action="store_true", help="Umbral relativo al tamaño de la mano para selección")
    sg.add_argument("--no-select-relative", dest="select_relative", action="store_false")
    p.set_defaults(select_relative=True)
    p.add_argument("--select-rel-threshold", type=float, default=0.10, help="Factor relativo para índice+medio juntos")
    p.add_argument("--select-threshold", type=float, default=0.060, help="Umbral absoluto para índice+medio juntos")
    p.add_argument("--select-hold-ms", type=float, default=80.0, help="Tiempo mínimo (ms) para confirmar selección")
    sf = p.add_mutually_exclusive_group()
    sf.add_argument("--select-focus", dest="select_focus", action="store_true")
    sf.add_argument("--no-select-focus", dest="select_focus", action="store_false")
    p.set_defaults(select_focus=True)
    st = p.add_mutually_exclusive_group()
    st.add_argument("--select-toggle-floating", dest="select_toggle_floating", action="store_true")
    st.add_argument("--no-select-toggle-floating", dest="select_toggle_floating", action="store_false")
    p.set_defaults(select_toggle_floating=True)
    # Traer al frente: mover al workspace actual y enfocar
    sm = p.add_mutually_exclusive_group()
    sm.add_argument("--select-move-to-current", dest="select_move_to_current", action="store_true", help="Mover al workspace actual la app seleccionada y ponerla al frente")
    sm.add_argument("--no-select-move-to-current", dest="select_move_to_current", action="store_false")
    p.set_defaults(select_move_to_current=True)
    p.add_argument("--move-cursor-together", action="store_true", help="Mueve también el cursor junto con la ventana (ydotool)")
    p.add_argument("--debug", action="store_true", help="Muestra información de ventana y gesto en consola")
    agrp = p.add_mutually_exclusive_group()
    agrp.add_argument("--auto", dest="auto", action="store_true")
    agrp.add_argument("--no-auto", dest="auto", action="store_false")
    p.set_defaults(auto=True)
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

    print(f"[auto] Monitor: {screen_w}x{screen_h} @ {mon_hz or '??'} Hz (offset {mon_ox},{mon_oy}); Cursor FPS: {args.cursor_fps}, tau_ms: {args.anim_tau_ms}")

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

    # Animador
    target_lock = threading.Lock()
    anim_target: Tuple[float, float] = (None, None)  # tipo: ignore
    anim_pos: Tuple[float, float] = (None, None)  # tipo: ignore
    stop_evt = threading.Event()
    dragging = False
    drag_anchor_offset: Tuple[int, int] = (0, 0)
    drag_window_snapshot: Optional[dict] = None
    drag_window_addr: Optional[str] = None
    selected_window_snapshot: Optional[dict] = None
    selected_window_addr: Optional[str] = None
    move_threshold_px = 2
    preview_destroyed = False
    pointer_drag_active = False
    pointer_backend: Optional[str] = None  # 'ydotool' | 'uinput'
    uinput_dev: Optional[object] = None if _HAS_UINPUT else None

    # Utilidades para ydotool: mapear botón a evdev y ejecutar con fallback
    def _map_button_to_evdev(btn: int) -> int:
        # BTN_LEFT=272, BTN_RIGHT=273, BTN_MIDDLE=274
        if btn == 1:
            return 272
        if btn == 2:
            return 274
        if btn == 3:
            return 273
        return btn

    # Backend uinput (si disponible)
    def _ensure_uinput() -> Optional[object]:
        nonlocal uinput_dev
        if not _HAS_UINPUT:
            return None
        if uinput_dev is not None:
            return uinput_dev
        try:
            caps = {
                ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE, ecodes.KEY_LEFTMETA],
            }
            uinput_dev = UInput(caps, name="wm-hand-drag-virtual", bustype=0x11)
            if getattr(args, 'debug', False):
                print("[pointer] uinput device creado")
            return uinput_dev
        except Exception as e:
            if getattr(args, 'debug', False):
                print(f"[pointer] uinput NO disponible: {e}")
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
                print(f"[pointer] uinput mousedown code={code}")
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
                print(f"[pointer] uinput mouseup code={code}")
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
                print("[pointer] uinput SUPER down")
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
                print("[pointer] uinput SUPER up")
            return True
        except Exception:
            return False

    def ydotool_mouse_down(btn: int) -> bool:
        yk = shutil.which("ydotool")
        if not yk:
            # Intentar fallback a uinput si backend permite
            if args.button_backend in ("auto", "uinput") and _HAS_UINPUT:
                return uinput_mouse_down(btn)
            return False
        ev = _map_button_to_evdev(btn)
        try:
            candidates = []
            # Índices típicos
            candidates += [btn]
            # Evdev mapeado
            candidates += [ev]
            # Alternativas comunes
            if btn == 1:
                candidates += [0, 272]
            elif btn == 2:
                candidates += [2, 274]
            elif btn == 3:
                candidates += [3, 273]
            # Intentar en orden, evitando duplicados
            seen = set()
            for c in [x for x in candidates if not (x in seen or seen.add(x))]:
                p = subprocess.run([yk, "mousedown", str(c)], capture_output=True, text=True)
                if p.returncode == 0:
                    if 'args' in globals() and getattr(args, 'debug', False):
                        print(f"[pointer] mousedown OK with code {c}")
                    return True
            # Último recurso: click (no mantiene pulsado). Lo intentamos pero devolvemos False
            p3 = subprocess.run([yk, "click", str(btn)], capture_output=True, text=True)
            if getattr(args, 'debug', False):
                if p3.returncode == 0:
                    print(f"[pointer] click used as fallback for down, btn={btn}")
                else:
                    print(f"[pointer] click fallback FAILED for down, btn={btn}")
            # Intentar fallback a uinput si backend permite
            if args.button_backend in ("auto", "uinput") and _HAS_UINPUT:
                return uinput_mouse_down(btn)
            return False
        except Exception:
            return False

    def ydotool_mouse_up(btn: int) -> bool:
        yk = shutil.which("ydotool")
        if not yk:
            # Intentar fallback a uinput si backend permite
            if args.button_backend in ("auto", "uinput") and _HAS_UINPUT:
                return uinput_mouse_up(btn)
            return False
        ev = _map_button_to_evdev(btn)
        try:
            candidates = []
            candidates += [btn, ev]
            if btn == 1:
                candidates += [0, 272]
            elif btn == 2:
                candidates += [2, 274]
            elif btn == 3:
                candidates += [3, 273]
            seen = set()
            for c in [x for x in candidates if not (x in seen or seen.add(x))]:
                p = subprocess.run([yk, "mouseup", str(c)], capture_output=True, text=True)
                if p.returncode == 0:
                    if getattr(args, 'debug', False):
                        print(f"[pointer] mouseup OK with code {c}")
                    return True
            # Fallback: click (no garantiza soltar si nunca hubo hold). Ejecutar pero devolver False
            p3 = subprocess.run([yk, "click", str(btn)], capture_output=True, text=True)
            if getattr(args, 'debug', False):
                if p3.returncode == 0:
                    print(f"[pointer] click used as fallback for up, btn={btn}")
                else:
                    print(f"[pointer] click fallback FAILED for up, btn={btn}")
            # Intentar fallback a uinput si backend permite
            if args.button_backend in ("auto", "uinput") and _HAS_UINPUT:
                return uinput_mouse_up(btn)
            return False
        except Exception:
            return False

    # Envolturas de backend para SUPER y mouse
    def pointer_super_down_func() -> bool:
        nonlocal pointer_backend
        # Preferir uinput si se solicita o en auto está disponible
        if args.button_backend in ("auto", "uinput") and _HAS_UINPUT:
            if uinput_super_down():
                pointer_backend = "uinput"
                return True
        # Fallback: ydotool key
        yk = shutil.which("ydotool")
        if yk:
            try:
                proc = subprocess.run([yk, "key", f"{args.super_keycode}:1"], capture_output=True)
                if proc.returncode == 0:
                    pointer_backend = "ydotool"
                    if getattr(args, 'debug', False):
                        print("[pointer] SUPER down (ydotool)")
                    return True
            except Exception:
                pass
        return False

    def pointer_super_up_func() -> bool:
        # Liberar según el backend que funcionó
        if pointer_backend == "uinput":
            return uinput_super_up()
        yk = shutil.which("ydotool")
        if yk:
            try:
                subprocess.run([yk, "key", f"{args.super_keycode}:0"], capture_output=True)
                if getattr(args, 'debug', False):
                    print("[pointer] SUPER up (ydotool)")
                return True
            except Exception:
                return False
        return False

    def pointer_mouse_down_func(btn: int) -> bool:
        if args.button_backend == "uinput" and _HAS_UINPUT:
            return uinput_mouse_down(btn)
        return ydotool_mouse_down(btn)

    def pointer_mouse_up_func(btn: int) -> bool:
        if pointer_backend == "uinput":
            return uinput_mouse_up(btn)
        return ydotool_mouse_up(btn)

    def animator():
        nonlocal anim_target, anim_pos, dragging, drag_window_addr
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
                if dragging:
                    # Convertir a coordenadas absolutas
                    tx = mon_ox + ix - drag_anchor_offset[0]
                    ty = mon_oy + iy - drag_anchor_offset[1]
                    if args.drag_mode == "pointer":
                        # Mover el cursor absoluto (usar offset del monitor)
                        try_ydotool_move_abs(mon_ox + ix, mon_oy + iy)
                    else:
                        # Evitar spam si el movimiento es mínimo
                        if last_sent_xy is None or abs(tx - last_sent_xy[0]) > move_threshold_px or abs(ty - last_sent_xy[1]) > move_threshold_px:
                            moved = False
                            if drag_window_addr:
                                moved = hypr_move_window_to_address(drag_window_addr, tx, ty)
                            else:
                                moved = hypr_move_active_window_to(tx, ty)
                            if moved:
                                last_sent_xy = (tx, ty)
                        if args.move_cursor_together:
                            try_ydotool_move_abs(ix, iy)
                with target_lock:
                    anim_pos = (cx, cy)
            time.sleep(dt)

    t_anim = threading.Thread(target=animator, daemon=True)
    t_anim.start()

    try:
        was_pinch = False
        last_dbg_t = 0.0
        pinch_hold_t0: Optional[float] = None
        pinch_filter = OneEuro(min_cutoff=2.0, beta=0.0, d_cutoff=1.0)
        # Estado para selección con índice+medio
        select_was_together = False
        select_hold_t0: Optional[float] = None
        select_filter = OneEuro(min_cutoff=2.0, beta=0.0, d_cutoff=1.0)
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
                # Validación de índice y pulgar; si no hay pulgar válido, no seleccionar
                try:
                    ix, iy = float(lms[INDEX_TIP].x), float(lms[INDEX_TIP].y)
                    tx, ty = float(lms[THUMB_TIP].x), float(lms[THUMB_TIP].y)
                    mx, my = float(lms[MIDDLE_TIP].x), float(lms[MIDDLE_TIP].y)
                except Exception:
                    ix = iy = tx = ty = mx = my = float('nan')
                valid_index = 0.0 <= ix <= 1.0 and 0.0 <= iy <= 1.0
                valid_thumb = 0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0
                valid_middle = 0.0 <= mx <= 1.0 and 0.0 <= my <= 1.0
                if not valid_thumb:
                    # Cancelar pellizco/arrastre si el pulgar no es válido
                    if was_pinch or dragging:
                        if args.debug:
                            print("[pinch] OFF (thumb invalid)")
                        dragging = False
                        was_pinch = False
                        pinch_hold_t0 = None
                    # Actualizar target con índice si es válido para mantener seguimiento
                    if valid_index:
                        tnow = time.monotonic()
                        fx = clamp(filt_x.filter(tnow, ix), 0.0, 1.0)
                        fy = clamp(filt_y.filter(tnow, iy), 0.0, 1.0)
                        x_scr = fx * screen_w
                        y_scr = fy * screen_h
                        y_out = y_scr if args.invert_y else (screen_h - y_scr)
                        x_out = x_scr
                        with target_lock:
                            anim_target = (float(x_out), float(y_out))
                    if args.preview:
                        cv2.putText(vis_frame, "Thumb not detected", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
                        disp = cv2.flip(vis_frame, 1)
                        cv2.imshow("Pinch Drag Preview", disp)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                    continue
                if args.invert_x:
                    ix = 1.0 - ix
                    tx = 1.0 - tx
                    mx = 1.0 - mx
                # OneEuro
                tnow = time.monotonic()
                fx = clamp(filt_x.filter(tnow, ix), 0.0, 1.0)
                fy = clamp(filt_y.filter(tnow, iy), 0.0, 1.0)
                x_scr = fx * screen_w
                y_scr = fy * screen_h
                # Invertimos el mapeo de Y para que el movimiento se sienta natural al arrastrar
                y_out = y_scr if args.invert_y else (screen_h - y_scr)
                x_out = x_scr
                with target_lock:
                    anim_target = (float(x_out), float(y_out))

                # Pinch detection: distancia Euclídea normalizada entre índice y pulgar
                dx = ix - tx
                dy = iy - ty
                raw_dist = math.hypot(dx, dy)
                # Umbral relativo al tamaño de la mano si está activo
                if args.pinch_relative:
                    minx = min(pt.x for pt in lms)
                    maxx = max(pt.x for pt in lms)
                    miny = min(pt.y for pt in lms)
                    maxy = max(pt.y for pt in lms)
                    diag = max(1e-4, math.hypot(maxx - minx, maxy - miny))
                    base_thresh = args.pinch_rel_threshold * diag
                else:
                    base_thresh = args.pinch_threshold
                # Suavizado del valor de distancia para estabilidad
                dist = pinch_filter.filter(tnow, raw_dist)
                thresh_on = base_thresh
                thresh_off = base_thresh + args.pinch_hysteresis
                is_pinch = dist < (thresh_on if not was_pinch else thresh_off)

                # Selección: índice+medio juntos (cuando NO estamos pellizcando)
                if valid_index and valid_middle and not is_pinch:
                    sdx = ix - mx
                    sdy = iy - my
                    s_raw = math.hypot(sdx, sdy)
                    if args.select_relative:
                        minx = min(pt.x for pt in lms)
                        maxx = max(pt.x for pt in lms)
                        miny = min(pt.y for pt in lms)
                        maxy = max(pt.y for pt in lms)
                        diag = max(1e-4, math.hypot(maxx - minx, maxy - miny))
                        s_thresh = args.select_rel_threshold * diag
                    else:
                        s_thresh = args.select_threshold
                    s_dist = select_filter.filter(tnow, s_raw)
                    together = s_dist < s_thresh if not select_was_together else s_dist < (s_thresh + 0.01)
                    if together and not select_was_together:
                        # iniciar hold
                        select_hold_t0 = time.monotonic()
                    if together and select_hold_t0 is not None and (time.monotonic() - select_hold_t0) * 1000.0 >= args.select_hold_ms:
                        # Confirmar selección bajo el dedo
                        abs_x = mon_ox + int(x_out)
                        abs_y = mon_oy + int(y_out)
                        picked = pick_window_under_point(abs_x, abs_y)
                        if picked:
                            selected_window_snapshot = picked
                            selected_window_addr = str(picked.get("address") or "")
                            # Primero mover al workspace actual (si se pide), luego enfocar
                            if args.select_move_to_current and selected_window_addr:
                                if hypr_move_window_to_current_workspace(selected_window_addr) and args.debug:
                                    print("[select] Movida a workspace actual")
                            if args.select_focus and selected_window_addr:
                                hypr_focus_window_address(selected_window_addr)
                            if args.select_toggle_floating and not picked.get("floating", False) and selected_window_addr:
                                hypr_toggle_floating_address(selected_window_addr)
                            if args.debug:
                                ttl = picked.get("title") or "?"
                                print(f"[select] Ventana seleccionada: '{ttl}' addr={selected_window_addr}")
                        # Evitar re-disparo inmediato
                        select_hold_t0 = None
                    select_was_together = together
                else:
                    select_was_together = False
                    select_hold_t0 = None

                if is_pinch and not was_pinch:
                    # Debounce temporal: exigir pellizco estable durante pinch_hold_ms
                    now = time.monotonic()
                    if pinch_hold_t0 is None:
                        pinch_hold_t0 = now
                        # esperar siguiente iteración para acumular tiempo
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
                    # Determinar ventana objetivo: usar selección previa si existe; si no, fallback
                    info = None
                    # Seguridad: evitar usar selección previa si accidentalmente era el preview
                    if selected_window_snapshot is not None and str(selected_window_snapshot.get("title") or "") == PREVIEW_TITLE:
                        selected_window_snapshot = None
                        selected_window_addr = None
                    if selected_window_snapshot is not None and selected_window_addr:
                        info = selected_window_snapshot
                        drag_window_addr = selected_window_addr
                        # Asegurar foco/flotante
                        # Ocultar preview ANTES de enfocar/mover para que no robe foco
                        if args.preview and args.hide_preview_on_drag and not preview_destroyed:
                            try:
                                cv2.destroyWindow("Pinch Drag Preview")
                            except Exception:
                                pass
                            preview_destroyed = True
                        if args.select_focus:
                            hypr_focus_window_address(drag_window_addr)
                        # Garantizar flotante (en ambos modos) para evitar comportamiento de tile
                        if not info.get("floating", False):
                            hypr_set_floating_address(drag_window_addr, True)
                    else:
                        abs_x = mon_ox + int(x_out)
                        abs_y = mon_oy + int(y_out)
                        picked = pick_window_under_point(abs_x, abs_y) if args.focus_under_finger else hypr_active_window_info()
                        # Si la candidata es el preview, intenta elegir la ventana bajo el dedo en su lugar
                        if picked is not None:
                            ttl = str(picked.get("title") or "") if isinstance(picked, dict) else str(picked)
                            if ttl == PREVIEW_TITLE:
                                alt = pick_window_under_point(abs_x, abs_y)
                                if alt is not None:
                                    picked = alt
                        if picked:
                            info = picked
                            drag_window_addr = str(picked.get("address") or "")
                            # Ocultar preview ANTES de enfocar/mover para que no robe foco
                            if args.preview and args.hide_preview_on_drag and not preview_destroyed:
                                try:
                                    cv2.destroyWindow("Pinch Drag Preview")
                                except Exception:
                                    pass
                                preview_destroyed = True
                            if args.focus_under_finger and drag_window_addr:
                                hypr_focus_window_address(drag_window_addr)
                            if not picked.get("floating", False) and drag_window_addr:
                                hypr_set_floating_address(drag_window_addr, True)
                    if args.debug:
                        print(f"[pinch] ON (dist={dist:.3f}) at ({int(x_out)},{int(y_out)})")
                    if info:
                        drag_window_snapshot = info
                        # guardar offset dedo respecto al borde superior-izquierdo de la ventana
                        at = info.get("at") or [0, 0]
                        # Hyprland 'at' suele ser [x,y] top-left
                        # convert anim coords (monitor local) a absolutos y luego calcular offset
                        drag_anchor_offset = ((mon_ox + int(x_out)) - int(at[0]), (mon_oy + int(y_out)) - int(at[1]))
                        if args.debug:
                            title = info.get("title") or "?"
                            cls = info.get("class") or "?"
                            addr = info.get("address") or "?"
                            size = info.get("size") or [0, 0]
                            print(f"[window] Active: '{title}' ({cls}) addr={addr} at={at} size={size}")
                    # Solo iniciar arrastre si tenemos una ventana válida (no el preview)
                    if info is not None:
                        dragging = True
                        # Iniciar arrastre por puntero si se eligió ese modo
                        if args.drag_mode == "pointer":
                            # Mover primero el cursor al objetivo inicial
                            try_ydotool_move_abs(mon_ox + int(x_out), mon_oy + int(y_out))
                            time.sleep(0.01)
                            # SUPER down + mousedown con backend disponible
                            if not pointer_super_down_func():
                                pointer_drag_active = False
                                if args.debug:
                                    print("[pointer] SUPER down FALLÓ")
                            else:
                                if not pointer_mouse_down_func(args.mouse_button):
                                    pointer_super_up_func()
                                    pointer_drag_active = False
                                    if args.debug:
                                        print("[pointer] mousedown FALLÓ")
                                else:
                                    pointer_drag_active = True
                                    if args.debug:
                                        print(f"[pointer] SUPER down + mousedown at {mon_ox + int(x_out)},{mon_oy + int(y_out)}")
                    else:
                        if args.debug:
                            print("[pinch] ON pero sin ventana objetivo; no se inicia arrastre")
                elif not is_pinch and was_pinch:
                    if args.debug:
                        print("[pinch] OFF")
                    dragging = False
                    # Terminar arrastre por puntero si estaba activo
                    if args.drag_mode == "pointer" and pointer_drag_active:
                        # Primero soltar SUPER (para evitar interacciones del compositor)
                        pointer_super_up_func()
                        # Luego mouseup con backend
                        if not pointer_mouse_up_func(args.mouse_button) and args.debug:
                            print("[pointer] mouseup FALLÓ")
                        pointer_drag_active = False
                    # Se puede restaurar el preview en el próximo ciclo de imshow
                    # Snap a esquinas/edges al soltar
                    if args.snap and drag_window_snapshot is not None:
                            # Usar última posición objetivo
                            with target_lock:
                                endx, endy = anim_pos
                            endx = int(endx) if endx is not None else int(x_out)
                            endy = int(endy) if endy is not None else int(y_out)
                            abs_x = mon_ox + endx
                            abs_y = mon_oy + endy
                            # Determinar región
                            M = int(args.snap_margin)
                            left = abs_x <= mon_ox + M
                            right = abs_x >= mon_ox + screen_w - M
                            top = abs_y <= mon_oy + M
                            bottom = abs_y >= mon_oy + screen_h - M
                            # Calcular top-left y size destino
                            rx, ry, rw, rh = None, None, None, None
                            if left and top:
                                rx, ry = mon_ox, mon_oy
                                rw, rh = screen_w // 2, screen_h // 2
                            elif right and top:
                                rw, rh = screen_w // 2, screen_h // 2
                                rx, ry = mon_ox + screen_w - rw, mon_oy
                            elif left and bottom:
                                rw, rh = screen_w // 2, screen_h // 2
                                rx, ry = mon_ox, mon_oy + screen_h - rh
                            elif right and bottom:
                                rw, rh = screen_w // 2, screen_h // 2
                                rx, ry = mon_ox + screen_w - rw, mon_oy + screen_h - rh
                            elif left:
                                rx, ry = mon_ox, mon_oy
                                rw, rh = screen_w // 2, screen_h
                            elif right:
                                rw, rh = screen_w // 2, screen_h
                                rx, ry = mon_ox + screen_w - rw, mon_oy
                            elif top:
                                rx, ry = mon_ox, mon_oy
                                rw, rh = screen_w, screen_h // 2
                            elif bottom:
                                rw, rh = screen_w, screen_h // 2
                                rx, ry = mon_ox, mon_oy + screen_h - rh
                            if rx is not None and ry is not None and rw is not None and rh is not None:
                                if args.drag_mode == "pointer":
                                    # En modo puntero, dejar que el compositor maneje layout; no forzamos snap
                                    pass
                                else:
                                    if drag_window_addr:
                                        hypr_move_window_to_address(drag_window_addr, rx, ry)
                                        time.sleep(0.01)
                                        hypr_resize_window_to_address(drag_window_addr, rw, rh)
                                    else:
                                        hypr_move_active_window_to(rx, ry)
                                        time.sleep(0.01)
                                        hypr_resize_active_window_to(rw, rh)
                    pinch_hold_t0 = None
                was_pinch = is_pinch

                if args.preview:
                    # Dibujar landmarks en el frame de visualización sin afectar el procesamiento
                    for lm in [lms[INDEX_TIP], lms[THUMB_TIP]]:
                        cx = int(lm.x * vis_frame.shape[1])
                        cy = int(lm.y * vis_frame.shape[0])
                        cv2.circle(vis_frame, (cx, cy), 6, (0, 255, 0), -1)
                    p1 = (int(lms[INDEX_TIP].x * vis_frame.shape[1]), int(lms[INDEX_TIP].y * vis_frame.shape[0]))
                    p2 = (int(lms[THUMB_TIP].x * vis_frame.shape[1]), int(lms[THUMB_TIP].y * vis_frame.shape[0]))
                    cv2.line(vis_frame, p1, p2, (0, 200, 255), 2)
                    msg = f"PINCH {'ON' if is_pinch else 'OFF'} dist={dist:.3f}"
                    cv2.putText(vis_frame, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50) if is_pinch else (0, 0, 200), 2)
            else:
                if args.preview:
                    cv2.putText(vis_frame, "No hand", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

            if args.preview and not (args.hide_preview_on_drag and dragging):
                # HUD: dibujar mini-mapa del monitor y sombrear ventana seleccionada
                if args.hud and drag_window_snapshot is not None:
                    hud_w = 220
                    hud_h = max(80, int(hud_w * (screen_h / max(1, screen_w))))
                    hud = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
                    hud[:] = (30, 30, 30)
                    # Borde monitor
                    cv2.rectangle(hud, (1, 1), (hud_w - 2, hud_h - 2), (200, 200, 200), 1)
                    at = drag_window_snapshot.get("at") or [0, 0]
                    sz = drag_window_snapshot.get("size") or [0, 0]
                    # Convertir a coords relativas del monitor
                    rel_x = int(at[0]) - mon_ox
                    rel_y = int(at[1]) - mon_oy
                    # Escala al HUD
                    wx = int(rel_x * (hud_w / max(1, screen_w)))
                    wy = int(rel_y * (hud_h / max(1, screen_h)))
                    ww = max(1, int(int(sz[0]) * (hud_w / max(1, screen_w))))
                    wh = max(1, int(int(sz[1]) * (hud_h / max(1, screen_h))))
                    # Sombrear
                    overlay = hud.copy()
                    cv2.rectangle(overlay, (wx, wy), (min(hud_w - 2, wx + ww), min(hud_h - 2, wy + wh)), (0, 180, 255), -1)
                    hud = cv2.addWeighted(overlay, 0.3, hud, 0.7, 0.0)
                    # Título
                    title = str(drag_window_snapshot.get("title") or "")
                    cv2.putText(hud, title[:22], (6, hud_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
                    # Pegar HUD en esquina del preview
                    ph, pw = vis_frame.shape[:2]
                    x0, y0 = pw - hud_w - 8, 8
                    vis_frame[y0:y0+hud_h, x0:x0+hud_w] = cv2.resize(hud, (hud_w, hud_h))

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
        # Safety: si quedó activo el arrastre por puntero, soltar botones/mod
        try:
            if pointer_drag_active:
                pointer_super_up_func()
                pointer_mouse_up_func(args.mouse_button)
        except Exception:
            pass
        # Cerrar uinput si se creó
        try:
            if uinput_dev is not None:
                uinput_dev.close()  # type: ignore
        except Exception:
            pass
        try:
            if args.preview:
                cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()
