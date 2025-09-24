"""Configuration constants for AI Movie Maker."""
from __future__ import annotations

import platform
from pathlib import Path

APP_NAME = "AI Movie Maker"
APP_VERSION = "1.3.2"
OS_TYPE = platform.system().lower()

BASE_DIR = Path.home() / ".ai_movie_maker"
PROJECTS_DIR = BASE_DIR / "projects"
BIN_DIR = BASE_DIR / "bin"
MODELS_DIR = BASE_DIR / "models"
COMFYUI_DIR = BIN_DIR / ("ComfyUI_windows_portable" if OS_TYPE == "windows" else "ComfyUI")
COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = BASE_DIR / "output"

# Default model names used by the renderer
ANIMATEDIFF_CHECKPOINT = "v1-5-pruned-emaonly.ckpt"
ANIMATEDIFF_MOTION_MODEL = "mm_sd_v15_v2.ckpt"