## Cómo ejecutarlo (resumen)
- Requisitos básicos: Python 3.10+, Hyprland (`hyprctl`), OpenCV y MediaPipe.
- Instala dependencias con `pip install -r requirements.txt` (se recomienda entorno virtual).
- Ejecuta el controlador (Wayland):
  - `QT_QPA_PLATFORM=xcb python linux/hand_controller.py --preview 1`

Para más detalles técnicos y flags avanzadas, consulta `linux/camera_controller/README.txt`.

# WindowManager — Proyecto personal
Este repositorio contiene un experimento personal para controlar ventanas en Linux (Hyprland) usando gestos de mano capturados por cámara.

El objetivo es explorar interacción natural con la UI: seleccionar aplicaciones con un gesto de “swipe” y mover ventanas con un “pellizco” (pinch). Está pensado para aprendizaje y uso personal; no es un producto terminado ni soportado formalmente.

## ¿Qué hay aquí?
- Código principal para Linux en `linux/`, con un controlador modular en `linux/camera_controller/`.
- Un lanzador sencillo: `linux/hand_controller.py`.
- Scripts de prueba y utilidades (OpenCV/MediaPipe).

## Estado y alcance
- Proyecto personal/educativo, sujeto a cambios frecuentes.
- Probado en Wayland/Hyprland; pueden existir limitaciones en otros entornos.
- Úsalo bajo tu propio criterio y revisa licencias de dependencias (MediaPipe, OpenCV, etc.).

## Licencia

Uso personal/educativo. Revisa las licencias de MediaPipe y OpenCV.

## Problemas comunes

- "Could not find the Qt platform plugin 'wayland'": exporta `QT_QPA_PLATFORM=xcb`.
- Si no mueve: asegúrate que la ventana esté flotante; el script puede forzarlo con flags.
- Si `--move-cursor-together` no mueve el cursor: confirma que `ydotoold` esté activo.
- Si el hold de botón falla con `ydotool`: usa `--button-backend uinput` (recomendado).

## Funcionamiento interno (resumen)

- Pointer + uinput (por defecto): crea un dispositivo virtual vía `/dev/uinput` para sostener `KEY_LEFTMETA` (SUPER) y `BTN_LEFT` sin soltarlos.
- Envolturas de eventos: funciones que intentan primero uinput; si no, usan `ydotool` como fallback cuando aplica.
- Exclusión del preview: se usa un título fijo de la ventana de preview para que nunca sea “seleccionada”.
- Traer al frente: al seleccionar, puede mover la ventana al workspace actual y enfocarla.
- Multi-monitor: usa el monitor enfocado y su offset absoluto para convertir coordenadas globales.
- Suavizado: One Euro filter y animación exponencial con `--cursor-fps` y `--anim-tau-ms`.

## uinput (recomendado para pointer)

Instala `evdev` y asegúrate de tener acceso a `/dev/uinput`.

Regla udev típica (opcional):

- Archivo: `/etc/udev/rules.d/99-uinput.rules`
- Contenido: `KERNEL=="uinput", MODE="0660", GROUP="input"`
- Aplica: `sudo udevadm control --reload-rules && sudo udevadm trigger` (reingresa sesión)

Prueba rápida:

```bash
QT_QPA_PLATFORM=xcb python linux/hand_pinch_window_drag.py --preview 1 --debug --drag-mode pointer --button-backend uinput
```

## Flags clave

- Orientación: `--invert-x` (on), `--invert-y` (on), `--flip`
- Auto-ajuste: `--auto` (on), detecta Hz del monitor y ajusta `--cursor-fps` y `--anim-tau-ms`
- Suavizado: `--cursor-fps`, `--anim-tau-ms`, `--min-cutoff`, `--beta`, `--d-cutoff`
- Pellizco: `--pinch-relative` (on), `--pinch-rel-threshold 0.12`, `--pinch-threshold`, `--pinch-hysteresis`, `--pinch-hold-ms`
- Selección índice+medio: `--select-relative` (on), `--select-rel-threshold 0.10`, `--select-hold-ms 80`, `--select-focus` (on), `--select-toggle-floating` (on)
- Traer al frente: `--select-move-to-current` (on)
- Snap: `--snap` (on), `--snap-margin 64`
- HUD/preview: `--hud 1` (on), `--hide-preview-on-drag` (on)
- Modo de arrastre: `--drag-mode pointer` (default) o `--drag-mode hypr`
- Backend de botón: `--button-backend uinput` (default) o `ydotool`

## Gestos y flujo

- Selección: junta índice + medio (sin pellizco) por ~80 ms → selecciona ventana bajo el dedo. Opciones para enfocar y forzar flotante.
- Arrastre: pellizco (índice + pulgar) → mueve la ventana seleccionada. Si no hay selección previa, toma la ventana bajo el dedo.
- Traer al frente: opcionalmente mueve la app seleccionada al workspace actual y la enfoca.
- Snap: al soltar cerca de bordes/esquinas, ajusta a mitades/cuadrantes.
- Preview: se oculta durante el drag para evitar robar foco; además se excluye por título de la selección.

```bash
QT_QPA_PLATFORM=xcb python linux/hand_pinch_window_drag.py --preview 1 --debug --drag-mode pointer --button-backend uinput
```

### Ejecución recomendada (Wayland/Hyprland)

Usa modo pointer con backend uinput para hold fiable de SUPER + botón izquierdo.

```bash
# Recomendado: Python 3.11 y venv
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -U pip wheel
pip install mediapipe==0.10.21 opencv-python numpy evdev
```

### Instalación rápida

## Requisitos

- Linux con Hyprland (`hyprctl` en PATH)
- Python 3.11
- Paquetes: `mediapipe==0.10.21`, `opencv-python`, `numpy`, `evdev` (para uinput)
- Wayland: forzar `QT_QPA_PLATFORM=xcb` para que `cv2.imshow` funcione (usa XWayland)

## Qué hace

- Detecta un pellizco entre índice y pulgar con MediaPipe Hands.
- Con pellizco activo: “agarra” la ventana objetivo y la mueve siguiendo la mano.
- Selección de ventana juntando índice + medio por unos milisegundos.
- Snap opcional a esquinas/mitades al soltar.
- Recomendado: modo pointer con backend uinput para mantener SUPER + botón izquierdo de forma fiable en Wayland.