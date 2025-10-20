#!/usr/bin/env python3
# coding: utf-8
"""
Overlay de sombreado de ventana (macOS) controlado por stdin con JSON por línea.

Entrada (una por línea, JSON):
  {"x": int, "y": int, "w": int, "h": int, "visible": bool}

Coordenadas esperadas: bottom-left (Cocoa). Si tienes top-left, convierte con: y' = screen_h - (y + h)
"""
import sys
import json
import threading

from AppKit import (
    NSApplication,
    NSWindow,
    NSView,
    NSColor,
    NSMakeRect,
    NSWindowStyleMaskBorderless,
    NSBackingStoreBuffered,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSScreenSaverWindowLevel,
    NSFloatingWindowLevel,
)
from PyObjCTools import AppHelper


class OverlayView(NSView):
    def initWithFrame_(self, frame):
        self = NSView.initWithFrame_(self, frame)
        if self is None:
            return None
        self.setWantsLayer_(True)
        return self

    def drawRect_(self, rect):
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.2, 0.2, 0.18).set()
        rect.fill()
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.0, 0.0, 0.6).set()
        rect.frame()


class OverlayController:
    def __init__(self):
        self.app = NSApplication.sharedApplication()
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(100, 100, 100, 100),
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setOpaque_(False)
        self.window.setIgnoresMouseEvents_(True)
        self.window.setLevel_(max(NSFloatingWindowLevel, NSScreenSaverWindowLevel - 1))
        self.window.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces)
        view = OverlayView.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 100))
        self.window.setContentView_(view)
        self.window.orderOut_(None)

    def set_visible(self, visible: bool):
        if visible:
            self.window.orderFrontRegardless()
        else:
            self.window.orderOut_(None)

    def set_frame(self, x: int, y: int, w: int, h: int):
        self.window.setFrame_display_(NSMakeRect(x, y, w, h), True)


def reader_loop(ctrl: OverlayController):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            x = int(msg.get("x", 0))
            y = int(msg.get("y", 0))
            w = int(msg.get("w", 0))
            h = int(msg.get("h", 0))
            vis = bool(msg.get("visible", True))
        except Exception:
            continue

        AppHelper.callAfter(ctrl.set_frame, x, y, w, h)
        AppHelper.callAfter(ctrl.set_visible, vis)


def main():
    ctrl = OverlayController()
    t = threading.Thread(target=reader_loop, args=(ctrl,), daemon=True)
    t.start()
    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
