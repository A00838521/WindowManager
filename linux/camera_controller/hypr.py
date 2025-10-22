import os, shutil, subprocess, json, glob, re
from typing import List, Dict, Optional, Tuple


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
    if side not in ("l", "r", "u", "d"):
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


HYPR_CONFIG_DIR = os.path.expanduser("~/.config/hypr")


def read_hypr_config_gaps() -> Dict[str, int]:
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
