import threading
import time
from typing import Dict, List

from . import hypr


class FocusMRUTracker:
    """Rastrea la ventana activa y mantiene un MRU por workspace.

    Excluye una ventana de preview por título exacto.
    """

    def __init__(self, preview_title: str):
        self.preview_title = preview_title
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None
        self._lock = threading.Lock()
        self._mru_by_ws: Dict[int, List[str]] = {}

    def start(self):
        if self._thr and self._thr.is_alive():
            return
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout: float | None = None):
        if self._thr:
            self._thr.join(timeout=timeout)

    def _run(self):
        last_addr = None
        while not self._stop.is_set():
            info = hypr.hypr_active_window_info()
            if info:
                addr = str(info.get("address") or "")
                wobj = info.get("workspace") or {}
                wid = int(wobj.get("id") or (hypr.hypr_active_workspace() or -1))
                title = str(info.get("title") or "")
                if addr and title != self.preview_title and addr != last_addr and wid is not None:
                    with self._lock:
                        lst = self._mru_by_ws.get(wid, [])
                        try:
                            if addr in lst:
                                lst.remove(addr)
                        except Exception:
                            pass
                        lst.insert(0, addr)
                        if len(lst) > 64:
                            lst = lst[:64]
                        self._mru_by_ws[wid] = lst
                    last_addr = addr
            time.sleep(0.15)

    def ordered_for_workspace(self, wid: int, addrs: List[str], mode: str) -> List[str]:
        """Devuelve addrs ordenadas por MRU o por defecto conservando las no vistas al final."""
        if mode != "mru":
            return list(addrs)
        with self._lock:
            base = list(self._mru_by_ws.get(wid, []))
        ordered = [a for a in base if a in addrs]
        for a in addrs:
            if a not in ordered:
                ordered.append(a)
        return ordered
