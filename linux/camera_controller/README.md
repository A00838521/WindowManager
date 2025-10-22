# Controlador por cámara para Hyprland (Linux)

Controla ventanas con gestos de mano usando la cámara, compatible con Hyprland. Implementa:

- Movimiento relativo con pellizco (pinch) del índice y pulgar
- Arrastre universal por puntero (SUPER+LMB) vía uinput o ydotool
- Selección por gesto:
  - Swipe (deslizar índice+medio) para ciclar enfoque (como Alt+Tab), por defecto orden MRU
  - Hold (mantener índice+medio) como modo legado
- Limpieza robusta: siempre vuelve a tiled al soltar o salir del programa
- Respeta siempre la selección previa o la ventana activa, sin “fallback bajo el dedo”

## Requisitos

- Hyprland y `hyprctl`
- Python 3.10+
- Paquetes Python (ver `requirements.txt`):
  - mediapipe, opencv-python, evdev (solo Linux), etc.
- Recomendado para arrastre por puntero:
  - uinput (kernel) + permisos para crear dispositivo (python-evdev)
  - y/o `ydotool` instalado en el sistema para movimientos absolutos de cursor

## Uso rápido

Ejecuta desde la raíz del repo:

```bash
# Módulo directo
python -m linux.camera_controller.controller --preview 1 --drag-mode pointer

# Script de conveniencia
python linux/hand_controller.py --preview 1
```

Notas:
- El modelo `hand_landmarker.task` se descarga automáticamente si no existe.
- En Wayland se fuerza `QT_QPA_PLATFORM=xcb` para estabilidad del preview.

## Flags principales

- `--drag-mode [pointer|hypr]` (default pointer):
  - pointer: simula SUPER+LMB con uinput/ydotool para arrastrar ventanas universalmente
  - hypr: usa `hyprctl dispatch movewindowpixel exact`
- Selección:
  - `--select-mode [swipe|hold]` (default swipe)
  - `--select-order [mru|position]` (default mru)
  - `--swipe-threshold-px`, `--swipe-cooldown-ms`
- Pinch:
  - `--pinch-threshold`, `--pinch-hysteresis`, `--pinch-hold-ms`
  - `--pinch-relative` (default on) y `--pinch-rel-threshold`
- Limpieza y tile:
  - `--toggle-float-long-ms` para toggle float/tile con pinch largo sin mover
  - `--snap-margin` (auto lee gaps de Hypr), `--tile-heuristic [none|axis|vertical-prefer|quadrant]`

## Comportamiento clave

- Al iniciar pinch, se usa la ventana seleccionada previamente (vía swipe/hold) o la activa; nunca detecta “bajo el dedo”.
- Al soltar pinch, se fuerza `setfloating no` siempre. Si `--tile-heuristic` está activo, puede despachar `movewindow` hacia un lado.
- El swipe cicla enfoque en el workspace actual respetando MRU cuando está habilitado.

## Solución de problemas

- Si el preview (OpenCV) no aparece en Wayland: asegúrate de que `QT_QPA_PLATFORM=xcb` esté disponible (se configura automáticamente en el script).
- Si no arrastra en modo pointer:
  - Instala `python-evdev` (vía pip: `evdev`) y asegúrate de permisos para uinput.
  - Alternativamente instala `ydotool` y ejecútalo con privilegios según tu distro.
- Si hyprctl falla: verifica que Hyprland esté corriendo y que `hyprctl -j` retorne JSON válido.
