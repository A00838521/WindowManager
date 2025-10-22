#!/usr/bin/env python3
import os, time, argparse, math, threading
from typing import Optional, Tuple, List, Dict

# Forzar plugin XCB para imshow en Wayland
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .filters import clamp, OneEuro
from . import hypr
from .input_backend import PointerController
from .tracker import FocusMRUTracker


INDEX_TIP = 8
THUMB_TIP = 4
MIDDLE_TIP = 12
PREVIEW_TITLE = "Pinch Drag Preview"


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

    # Gestos/umbrales
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

    # Arrastre universal por puntero (SUPER+LMB)
    p.add_argument("--drag-mode", choices=["hypr", "pointer"], default="pointer",
                   help="Modo de arrastre: hypr (movewindowpixel) o pointer (SUPER+LMB con uinput)")
    p.add_argument("--super-keycode", type=int, default=125)
    p.add_argument("--mouse-button", type=int, default=1)
    p.add_argument("--button-backend", choices=["auto", "ydotool", "uinput"], default="uinput")

    # Selección por gesto
    p.add_argument("--select-mode", choices=["swipe", "hold"], default="swipe",
                   help="Método de selección: swipe (deslizar índice+medio) o hold (mantener índice+medio)")
    p.add_argument("--select-order", choices=["mru", "position"], default="mru",
                   help="Orden de ciclo en selección: mru (recientes) o position (x,y)")
    sf = p.add_mutually_exclusive_group()
    sf.add_argument("--select-focus", dest="select_focus", action="store_true")
    sf.add_argument("--no-select-focus", dest="select_focus", action="store_false")
    p.set_defaults(select_focus=True)
    # Swipe
    p.add_argument("--swipe-threshold-px", type=int, default=120)
    p.add_argument("--swipe-cooldown-ms", type=float, default=250.0)
    # Hold (legacy)
    sg = p.add_mutually_exclusive_group()
    sg.add_argument("--select-relative", dest="select_relative", action="store_true")
    sg.add_argument("--no-select-relative", dest="select_relative", action="store_false")
    p.set_defaults(select_relative=True)
    p.add_argument("--select-rel-threshold", type=float, default=0.10)
    p.add_argument("--select-threshold", type=float, default=0.060)
    p.add_argument("--select-hold-ms", type=float, default=80.0)

    # Long pinch toggle
    p.add_argument("--toggle-float-long-ms", type=float, default=900.0)

    # Snap margin
    gaps = hypr.read_hypr_config_gaps()
    default_snap_margin = max(gaps.get("gaps_in", 0), gaps.get("gaps_out", 0)) + 32
    p.add_argument("--snap-margin", type=int, default=default_snap_margin)
    p.add_argument("--tile-heuristic", choices=["none", "axis", "vertical-prefer", "quadrant"], default="none")

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

    screen_w, screen_h, mon_hz, mon_ox, mon_oy = hypr.hypr_focused_monitor_geometry()
    if args.auto and mon_hz and mon_hz >= 60:
        args.cursor_fps = float(min(240.0, max(60.0, round(mon_hz))))
        args.anim_tau_ms = max(6.0, min(14.0, 1000.0 / (2.0 * mon_hz)))

    if args.debug:
        print(f"[auto] Monitor {screen_w}x{screen_h}@{mon_hz or '??'}Hz off=({mon_ox},{mon_oy}) snap_margin={args.snap_margin}")

    # Filtros y estado compartido
    filt_x = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)
    filt_y = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)
    filt_mid_x = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)
    filt_mid_y = OneEuro(args.min_cutoff, args.beta, args.d_cutoff)

    target_lock = threading.Lock()
    anim_target: Tuple[float, float] = (None, None)  # type: ignore
    anim_pos: Tuple[float, float] = (None, None)  # type: ignore
    stop_evt = threading.Event()

    # Tracking MRU
    tracker = FocusMRUTracker(PREVIEW_TITLE)
    tracker.start()

    # Estados de arrastre/relativo
    dragging = False
    hand_start_screen: Optional[Tuple[int,int]] = None
    win_start_at: Optional[Tuple[int,int]] = None
    drag_window_addr: Optional[str] = None
    selected_window_addr: Optional[str] = None
    pinch_started_at: Optional[float] = None
    toggled_float_this_hold = False
    drag_last_sent_xy: Optional[Tuple[int,int]] = None

    # Puntero
    pointer = PointerController(debug=args.debug, button_backend=args.button_backend)
    pointer_drag_active = False
    pointer_start_abs: Optional[Tuple[int,int]] = None

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
        if adx >= ady:
            return "r" if dx > 0 else "l"
        else:
            return "d" if dy > 0 else "u"

    def animator():
        nonlocal anim_target, anim_pos, dragging, drag_last_sent_xy
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
                    dx = ix - hand_start_screen[0]
                    dy = iy - hand_start_screen[1]
                    if args.drag_mode == "hypr" and win_start_at is not None and drag_window_addr:
                        nx = win_start_at[0] + dx
                        ny = win_start_at[1] + dy
                        if last_sent_xy is None or abs(nx - last_sent_xy[0]) > move_threshold_px or abs(ny - last_sent_xy[1]) > move_threshold_px:
                            if hypr.hypr_move_window_to_address(drag_window_addr, nx, ny):
                                last_sent_xy = (nx, ny)
                                drag_last_sent_xy = last_sent_xy
                    elif args.drag_mode == "pointer" and pointer_start_abs is not None:
                        cx_abs = pointer_start_abs[0] + dx
                        cy_abs = pointer_start_abs[1] + dy
                        if last_sent_xy is None or abs(cx_abs - last_sent_xy[0]) > move_threshold_px or abs(cy_abs - last_sent_xy[1]) > move_threshold_px:
                            if pointer.move_abs(int(cx_abs), int(cy_abs)):
                                last_sent_xy = (int(cx_abs), int(cy_abs))

                with target_lock:
                    anim_pos = (cx, cy)
            time.sleep(dt)

    t_anim = threading.Thread(target=animator, daemon=True)
    t_anim.start()

    try:
        was_pinch = False
        pinch_hold_t0: Optional[float] = None
        pinch_filter = OneEuro(min_cutoff=2.0, beta=0.0, d_cutoff=1.0)
        # Selección
        select_was_together = False
        select_hold_t0: Optional[float] = None
        select_done_this_hold = False
        select_filter = OneEuro(min_cutoff=2.0, beta=0.0, d_cutoff=1.0)
        swipe_active = False
        swipe_start_px: Optional[Tuple[int,int]] = None
        last_swipe_t = 0.0

        def workspace_clients_excl_preview() -> List[Dict]:
            wid = hypr.hypr_active_workspace()
            return [c for c in hypr.hypr_clients() if (not wid or (c.get("workspace", {}).get("id") == wid)) and (str(c.get("title") or "") != PREVIEW_TITLE)]

        def cycle_focus(direction: str) -> Optional[str]:
            nonlocal selected_window_addr
            clis = workspace_clients_excl_preview()
            if not clis:
                return None
            act = hypr.hypr_active_window_info()
            cur_addr = str((act or {}).get("address") or "")
            wid = hypr.hypr_active_workspace() or -1
            addrs = [str(c.get("address") or "") for c in clis]
            if args.select_order == "mru":
                ordered = tracker.ordered_for_workspace(wid, addrs, mode="mru")
            else:
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
                hypr.hypr_focus_window_address(addr)
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
                if not valid_thumb:
                    # Cancelar y cleanup
                    if pointer_drag_active:
                        try:
                            pointer.mouse_up(args.mouse_button)
                        except Exception:
                            pass
                        try:
                            pointer.super_up()
                        except Exception:
                            pass
                        pointer_drag_active = False
                        pointer_start_abs = None
                    if drag_window_addr:
                        try:
                            hypr.hypr_set_floating_address(drag_window_addr, False)
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
                        cv2.imshow(PREVIEW_TITLE, disp)
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

                # Pinch
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

                # Selección (swipe/hold)
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
                            # Selección por hold → usar ventana activa (y memorizar)
                            info = hypr.hypr_active_window_info()
                            if info and info.get("title") != PREVIEW_TITLE:
                                selected_window_addr = str(info.get("address") or "")
                                if args.select_focus and selected_window_addr:
                                    hypr.hypr_focus_window_address(selected_window_addr)
                                if args.debug:
                                    print(f"[select-hold] addr={selected_window_addr}")
                            select_done_this_hold = True
                    elif not is_together and select_was_together:
                        select_hold_t0 = None
                        select_done_this_hold = False
                    select_was_together = is_together
                else:
                    # Swipe
                    minx = min(pt.x for pt in lms)
                    maxx = max(pt.x for pt in lms)
                    miny = min(pt.y for pt in lms)
                    maxy = max(pt.y for pt in lms)
                    diag = max(1e-4, math.hypot(maxx - minx, maxy - miny))
                    sel_thresh = 0.10 * diag
                    is_together = sel_dist_raw < sel_thresh

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
                        if args.preview:
                            disp = cv2.flip(vis_frame, 1)
                            cv2.imshow(PREVIEW_TITLE, disp)
                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                        continue
                    elif (now - pinch_hold_t0) * 1000.0 < args.pinch_hold_ms:
                        if args.preview:
                            disp = cv2.flip(vis_frame, 1)
                            cv2.imshow(PREVIEW_TITLE, disp)
                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                        continue

                    # Objetivo: siempre selección previa o activa
                    picked = None
                    target_addr = selected_window_addr
                    if not target_addr:
                        info = hypr.hypr_active_window_info()
                        if info and info.get("title") != PREVIEW_TITLE:
                            target_addr = str(info.get("address") or "")
                            if target_addr:
                                selected_window_addr = target_addr
                    if target_addr:
                        clis = hypr.hypr_clients()
                        for c in clis:
                            if str(c.get("address") or "") == target_addr:
                                picked = c
                                break
                        if picked is None:
                            info = hypr.hypr_active_window_info()
                            if info and str(info.get("address") or "") == target_addr:
                                picked = info
                    if picked and str(picked.get("address") or ""):
                        drag_window_addr = str(picked.get("address") or "")
                        selected_window_addr = drag_window_addr
                        if not picked.get("floating", False) and drag_window_addr:
                            hypr.hypr_set_floating_address(drag_window_addr, True)
                        at = picked.get("at") or [0, 0]
                        size = picked.get("size") or [0, 0]
                        win_start_at = (int(at[0]), int(at[1]))
                        with target_lock:
                            cx, cy = anim_target
                        hand_start_screen = (int(cx), int(cy))
                        cx_abs = int(at[0]) + int(size[0]) // 2
                        cy_abs = int(at[1]) + int(size[1]) // 2
                        pointer.move_abs(cx_abs, cy_abs)
                        pointer_start_abs = (cx_abs, cy_abs)
                        if args.drag_mode == "pointer":
                            try:
                                if pointer.super_down():
                                    pass
                                pointer.mouse_down(args.mouse_button)
                                pointer_drag_active = True
                            except Exception:
                                pointer_drag_active = False
                        dragging = True
                        pinch_started_at = now
                        toggled_float_this_hold = False
                        if args.debug:
                            print(f"[pinch] START rel; hand0={hand_start_screen} win0={win_start_at} addr={drag_window_addr}")
                    else:
                        pinch_hold_t0 = None

                elif is_pinch and was_pinch:
                    if dragging and pinch_started_at and not toggled_float_this_hold:
                        elapsed_ms = (time.monotonic() - pinch_started_at) * 1000.0
                        with target_lock:
                            cx, cy = anim_target
                        if hand_start_screen is not None and math.hypot((cx - hand_start_screen[0]), (cy - hand_start_screen[1])) < max(6, args.snap_margin * 0.05):
                            if elapsed_ms >= args.toggle_float_long_ms and drag_window_addr:
                                hypr.hypr_toggle_floating_address(drag_window_addr)
                                toggled_float_this_hold = True
                                if args.debug:
                                    print("[pinch] long → toggle float/tile")

                elif not is_pinch and was_pinch:
                    # FIN pellizco: liberar y tile SIEMPRE
                    if pointer_drag_active:
                        try:
                            pointer.mouse_up(args.mouse_button)
                        except Exception:
                            pass
                        try:
                            pointer.super_up()
                        except Exception:
                            pass
                        pointer_drag_active = False
                        pointer_start_abs = None
                    if drag_window_addr:
                        hypr.hypr_set_floating_address(drag_window_addr, False)
                        if args.tile_heuristic != "none" and args.drag_mode == "hypr" and win_start_at and drag_last_sent_xy:
                            side = decide_tile_direction(win_start_at, drag_last_sent_xy, args.snap_margin, args.tile_heuristic)
                            if args.tile_heuristic == "quadrant" and side is None:
                                side = None
                            if side:
                                hypr.hypr_move_tiled_side_address(drag_window_addr, side)
                            elif args.tile_heuristic == "quadrant" and win_start_at and drag_last_sent_xy:
                                dx = drag_last_sent_xy[0] - win_start_at[0]
                                dy = drag_last_sent_xy[1] - win_start_at[1]
                                adx, ady = abs(dx), abs(dy)
                                if ady >= args.snap_margin:
                                    hypr.hypr_move_tiled_side_address(drag_window_addr, "d" if dy > 0 else "u")
                                if adx >= args.snap_margin:
                                    hypr.hypr_move_tiled_side_address(drag_window_addr, "r" if dx > 0 else "l")
                        elif args.tile_heuristic != "none":
                            if hand_start_screen is not None and anim_pos[0] is not None:
                                cx, cy = anim_pos
                                dx = int(cx - hand_start_screen[0])
                                dy = int(cy - hand_start_screen[1])
                                adx, ady = abs(dx), abs(dy)
                                if args.tile_heuristic == "quadrant":
                                    if ady >= args.snap_margin:
                                        hypr.hypr_move_tiled_side_address(drag_window_addr, "d" if dy > 0 else "u")
                                    if adx >= args.snap_margin:
                                        hypr.hypr_move_tiled_side_address(drag_window_addr, "r" if dx > 0 else "l")
                                else:
                                    winf = (win_start_at[0] + dx, win_start_at[1] + dy) if win_start_at else (dx, dy)
                                    side = decide_tile_direction(win_start_at or (0,0), winf, args.snap_margin, args.tile_heuristic)
                                    if side:
                                        hypr.hypr_move_tiled_side_address(drag_window_addr, side)
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
                    mode = args.select_mode
                    selmsg = f"SEL-{mode} {'ON' if select_was_together else 'OFF'}"
                    cv2.putText(vis_frame, selmsg, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (200, 200, 0) if select_was_together else (120,120,120), 2)
            else:
                if args.preview:
                    cv2.putText(vis_frame, "No hand", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

            if args.preview:
                disp = cv2.flip(vis_frame, 1)
                cv2.imshow(PREVIEW_TITLE, disp)
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
            tracker.stop()
            tracker.join(timeout=0.2)
        except Exception:
            pass
        # Cleanup adicional: asegurar tile y liberar botón/SUPER
        try:
            if pointer_drag_active:
                try:
                    pointer.mouse_up(args.mouse_button)
                except Exception:
                    pass
                try:
                    pointer.super_up()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if drag_window_addr:
                hypr.hypr_set_floating_address(drag_window_addr, False)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
