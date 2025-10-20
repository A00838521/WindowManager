#!/usr/bin/env python3
"""
Detección en tiempo real de landmarks de manos con la cámara del MacBook
usando MediaPipe Tasks.

Requisitos:
  - Python 3.9+ (recomendado)
  - pip install mediapipe opencv-python
  - Archivo de modelo hand_landmarker.task en el mismo directorio
    (si no está presente, el script intentará descargarlo automáticamente).

Controles en ventana:
  - q: salir
  - s: guardar un fotograma anotado en ./captures/
  - barra espaciadora: pausar/reanudar
"""

import os
import time
import argparse
from datetime import datetime
from urllib.request import urlretrieve

import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ------------------- Utilidades de dibujo -------------------
MARGIN = 10  # píxeles
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # verde vivo


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result):
    """Dibuja landmarks y conexiones de manos en una imagen RGB.

    Args:
        rgb_image: np.ndarray en formato RGB (H, W, 3).
        detection_result: resultado devuelto por HandLandmarker.

    Returns:
        np.ndarray RGB con anotaciones.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Recorre las manos detectadas.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Dibuja los landmarks de la mano.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Esquina superior izquierda del bounding box aproximado (para texto).
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Etiqueta de lateralidad (Left/Right).
        label = handedness[0].category_name
        cv2.putText(
            annotated_image,
            f"{label}",
            (text_x, max(text_y, MARGIN + 5)),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


# ------------------- Descarga del modelo si falta -------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    print("Modelo no encontrado. Descargando modelo pre-entrenado…")
    urlretrieve(MODEL_URL, model_path)
    print(f"Modelo guardado en: {model_path}")
    return model_path


# ------------------- Lógica principal -------------------

def main():
    parser = argparse.ArgumentParser(description="Detección en tiempo real de manos con MediaPipe Tasks")
    parser.add_argument("--model", default="hand_landmarker.task", help="Ruta al archivo .task del modelo")
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara (por defecto 0)")
    parser.add_argument("--width", type=int, default=1280, help="Ancho de captura")
    parser.add_argument("--height", type=int, default=720, help="Alto de captura")
    parser.add_argument("--num-hands", type=int, default=2, help="Número máximo de manos a detectar")
    parser.add_argument("--min-detection", type=float, default=0.5, help="Confianza mínima de detección")
    parser.add_argument("--min-presence", type=float, default=0.5, help="Confianza mínima de presencia")
    parser.add_argument("--min-tracking", type=float, default=0.5, help="Confianza mínima de tracking")
    parser.add_argument("--flip", type=int, default=1, choices=[0, 1], help="Reflejar horizontalmente (selfie). 1=Sí, 0=No")
    parser.add_argument("--max-frames", type=int, default=0, help="Procesar como mucho N frames y salir (0 = ilimitado)")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "avf", "any"],
        help="Backend de captura de cámara: auto (intenta AVFoundation y ANY), avf (CAP_AVFOUNDATION), any (CAP_ANY)",
    )
    args = parser.parse_args()

    model_path = ensure_model(args.model)

    # Configura la cámara con reintentos y distintos backends
    def try_open_camera(index: int, backend_flag: int):
        cap_local = cv2.VideoCapture(index, backend_flag)
        if cap_local.isOpened():
            cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        return cap_local

    backend_order = []
    if args.backend == "avf":
        backend_order = [cv2.CAP_AVFOUNDATION]
    elif args.backend == "any":
        backend_order = [cv2.CAP_ANY]
    else:
        backend_order = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

    cap = None
    indices_to_try = [args.camera]
    # Si falla el índice dado, probamos 1 y 2 como alternativa
    indices_to_try += [i for i in [1, 2] if i != args.camera]

    for backend_flag in backend_order:
        for idx in indices_to_try:
            cap = try_open_camera(idx, backend_flag)
            if cap.isOpened():
                print(f"Cámara abierta con índice {idx} usando backend {backend_flag}.")
                break
        if cap and cap.isOpened():
            break

    if cap is None or not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la cámara. Sugerencias: "
            "1) Asegúrate de cerrar apps que usen la cámara (FaceTime/Zoom). "
            "2) Otorga permisos de Cámara a tu aplicación (Terminal o VS Code) en Ajustes del Sistema > Privacidad y seguridad > Cámara. "
            "3) Prueba ejecutar desde la app Terminal y usa --backend avf y distintos --camera (0,1,2)."
        )

    # Configuración del landmarker en modo VIDEO para frames secuenciales
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_detection,
        min_hand_presence_confidence=args.min_presence,
        min_tracking_confidence=args.min_tracking,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    paused = False
    os.makedirs("captures", exist_ok=True)

    print("Ventana activa: 'q' para salir, 's' para guardar, 'espacio' para pausar")

    processed = 0
    try:
        while True:
            if not paused:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("No se pudo leer un frame de la cámara. Saliendo…")
                    break

                # Opcional: vista tipo selfie
                if args.flip:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # timestamp en milisegundos (monótono creciente)
                timestamp_ms = time.monotonic_ns() // 1_000_000
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                annotated_rgb = draw_landmarks_on_image(frame_rgb, detection_result)
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                # Mostrar un estimado simple de FPS usando el tiempo entre frames
                if not hasattr(main, "_last_ts"):
                    main._last_ts = time.monotonic()
                    main._fps = 0.0
                else:
                    now = time.monotonic()
                    dt = now - main._last_ts
                    main._last_ts = now
                    if dt > 0:
                        main._fps = 0.9 * getattr(main, "_fps", 0.0) + 0.1 * (1.0 / dt)
                cv2.putText(
                    annotated_bgr,
                    f"FPS: {getattr(main, '_fps', 0.0):.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                # Si está en pausa, muestra el último frame anotado (si existe)
                annotated_bgr = annotated_bgr if 'annotated_bgr' in locals() else np.zeros((args.height, args.width, 3), dtype=np.uint8)

            cv2.imshow("Hand Landmarks - MediaPipe Tasks", annotated_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                out_path = os.path.join("captures", f"hand_landmarks_{ts}.png")
                cv2.imwrite(out_path, annotated_bgr)
                print(f"Imagen guardada en: {out_path}")
            elif key == 32:  # barra espaciadora
                paused = not paused

            # Límite de frames si se indicó
            if not paused and args.max_frames and args.max_frames > 0:
                processed += 1
                if processed >= args.max_frames:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
