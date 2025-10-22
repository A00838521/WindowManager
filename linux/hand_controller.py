#!/usr/bin/env python3
import os, sys

# Permite ejecución directa: python linux/hand_controller.py
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from camera_controller.controller import main  # type: ignore
else:
    # Ejecución como módulo: python -m linux.hand_controller
    from .camera_controller.controller import main

if __name__ == "__main__":
    main()
