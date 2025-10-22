import shutil
from typing import Optional


class PointerController:
    """Control de puntero y tecla SUPER, con preferencia por uinput si está disponible.
    Fallback a ydotool donde aplique.
    """
    def __init__(self, debug: bool = False, button_backend: str = "uinput"):
        self.debug = debug
        self.button_backend_pref = button_backend  # 'uinput' | 'ydotool' | 'auto'
        self.pointer_backend: Optional[str] = None  # elegido en runtime
        self.uinput_dev = None
        try:
            from evdev import UInput, ecodes  # type: ignore
            self._HAS_UINPUT = True
            self._evdev = (UInput, ecodes)
        except Exception:
            self._HAS_UINPUT = False
            self._evdev = (None, None)

    # ------------- Cursores absolutos (ydotool) -------------
    def move_abs(self, x: int, y: int) -> bool:
        ydotool = shutil.which("ydotool")
        if not ydotool:
            return False
        import subprocess
        try:
            p = subprocess.run([ydotool, "mousemove", "-a", "-x", str(int(x)), "-y", str(int(y))], capture_output=True, text=True)
            return p.returncode == 0
        except Exception:
            return False

    # ------------- UInput helpers -------------
    def _ensure_uinput(self):
        if not self._HAS_UINPUT:
            return None
        if self.uinput_dev is not None:
            return self.uinput_dev
        UInput, ecodes = self._evdev
        try:
            caps = {ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE, ecodes.KEY_LEFTMETA]}
            self.uinput_dev = UInput(caps, name="wm-hand-rel-virtual", bustype=0x11)
            if self.debug:
                print("[uinput] device created")
            return self.uinput_dev
        except Exception as e:
            if self.debug:
                print(f"[uinput] create failed: {e}")
            self.uinput_dev = None
            return None

    def _uinput_mouse_down(self, btn: int) -> bool:
        dev = self._ensure_uinput()
        if dev is None:
            return False
        _, ecodes = self._evdev
        try:
            code = ecodes.BTN_LEFT if btn == 1 else (ecodes.BTN_MIDDLE if btn == 2 else ecodes.BTN_RIGHT)
            dev.write(ecodes.EV_KEY, code, 1)
            dev.syn()
            if self.debug:
                print(f"[uinput] mouse down {btn}")
            return True
        except Exception:
            return False

    def _uinput_mouse_up(self, btn: int) -> bool:
        dev = self._ensure_uinput()
        if dev is None:
            return False
        _, ecodes = self._evdev
        try:
            code = ecodes.BTN_LEFT if btn == 1 else (ecodes.BTN_MIDDLE if btn == 2 else ecodes.BTN_RIGHT)
            dev.write(ecodes.EV_KEY, code, 0)
            dev.syn()
            if self.debug:
                print(f"[uinput] mouse up {btn}")
            return True
        except Exception:
            return False

    def _uinput_super_down(self) -> bool:
        dev = self._ensure_uinput()
        if dev is None:
            return False
        _, ecodes = self._evdev
        try:
            dev.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 1)
            dev.syn()
            if self.debug:
                print("[uinput] SUPER down")
            return True
        except Exception:
            return False

    def _uinput_super_up(self) -> bool:
        dev = self._ensure_uinput()
        if dev is None:
            return False
        _, ecodes = self._evdev
        try:
            dev.write(ecodes.EV_KEY, ecodes.KEY_LEFTMETA, 0)
            dev.syn()
            if self.debug:
                print("[uinput] SUPER up")
            return True
        except Exception:
            return False

    # ------------- ydotool mouse helpers -------------
    @staticmethod
    def _map_button_to_evdev(btn: int) -> int:
        if btn == 1:
            return 272  # BTN_LEFT
        if btn == 2:
            return 274  # BTN_MIDDLE
        if btn == 3:
            return 273  # BTN_RIGHT
        return btn

    def _ydotool_mouse_down(self, btn: int) -> bool:
        yk = shutil.which("ydotool")
        if not yk:
            if self.button_backend_pref in ("auto", "uinput") and self._HAS_UINPUT:
                return self._uinput_mouse_down(btn)
            return False
        import subprocess
        ev = self._map_button_to_evdev(btn)
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
                p = subprocess.run([yk, "mousedown", str(c)], capture_output=True, text=True)
                if p.returncode == 0:
                    return True
            subprocess.run([yk, "click", str(btn)], capture_output=True, text=True)
            if self.button_backend_pref in ("auto", "uinput") and self._HAS_UINPUT:
                return self._uinput_mouse_down(btn)
            return False
        except Exception:
            return False

    def _ydotool_mouse_up(self, btn: int) -> bool:
        yk = shutil.which("ydotool")
        if not yk:
            if self.pointer_backend == "uinput":
                return self._uinput_mouse_up(btn)
            return False
        import subprocess
        ev = self._map_button_to_evdev(btn)
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
                p = subprocess.run([yk, "mouseup", str(c)], capture_output=True, text=True)
                if p.returncode == 0:
                    return True
            subprocess.run([yk, "click", str(btn)], capture_output=True, text=True)
            if self.pointer_backend == "uinput":
                return self._uinput_mouse_up(btn)
            return False
        except Exception:
            return False

    # ------------- API pública de arrastre -------------
    def super_down(self) -> bool:
        if self.button_backend_pref in ("auto", "uinput") and self._HAS_UINPUT:
            if self._uinput_super_down():
                self.pointer_backend = "uinput"
                return True
        return False

    def super_up(self) -> bool:
        if self.pointer_backend == "uinput":
            return self._uinput_super_up()
        return False

    def mouse_down(self, btn: int) -> bool:
        if self.button_backend_pref == "uinput" and self._HAS_UINPUT:
            self.pointer_backend = "uinput"
            return self._uinput_mouse_down(btn)
        return self._ydotool_mouse_down(btn)

    def mouse_up(self, btn: int) -> bool:
        if self.pointer_backend == "uinput":
            return self._uinput_mouse_up(btn)
        return self._ydotool_mouse_up(btn)
