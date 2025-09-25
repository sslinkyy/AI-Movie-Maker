"""General utility helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Optional

from .config import BIN_DIR, COMFYUI_DIR, OS_TYPE


def safe_subprocess(command: Iterable[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run a subprocess safely with helpful errors."""

    command_list = list(command)
    try:
        return subprocess.run(
            command_list,
            cwd=str(cwd) if cwd else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Command '{' '.join(command_list)}' failed with code {exc.returncode}: {exc.stderr.strip()}"
        ) from exc


def get_executable(path: Path) -> Optional[Path]:
    if OS_TYPE == "windows":
        exe = path.with_suffix(path.suffix + ".exe" if path.suffix else ".exe")
        return exe if exe.exists() else None
    return path if path.exists() else None


def ffmpeg_path() -> Optional[Path]:
    base = BIN_DIR / "ffmpeg" / ("bin" if OS_TYPE == "windows" else "")
    exe = base / ("ffmpeg.exe" if OS_TYPE == "windows" else "ffmpeg")
    return exe if exe.exists() else None


def rife_path() -> Optional[Path]:
    exe = BIN_DIR / "rife" / ("rife-ncnn-vulkan.exe" if OS_TYPE == "windows" else "rife-ncnn-vulkan")
    return exe if exe.exists() else None


def realesrgan_path() -> Optional[Path]:
    exe = BIN_DIR / "realesrgan" / (
        "realesrgan-ncnn-vulkan.exe" if OS_TYPE == "windows" else "realesrgan-ncnn-vulkan"
    )
    return exe if exe.exists() else None


def wkhtmltopdf_path() -> Optional[Path]:
    exe = BIN_DIR / "wkhtmltopdf" / ("bin/wkhtmltopdf.exe" if OS_TYPE == "windows" else "wkhtmltopdf")
    return exe if exe.exists() else None


def comfyui_run_script() -> Optional[list[str]]:
    if not COMFYUI_DIR.exists():
        return None
    if OS_TYPE == "windows":
        gpu = COMFYUI_DIR / "run_nvidia_gpu.bat"
        cpu = COMFYUI_DIR / "run_cpu.bat"
        if gpu.exists():
            return [str(gpu)]
        if cpu.exists():
            return [str(cpu)]
        return None
    python_exe = COMFYUI_DIR / "python_embedded" / "python.exe"
    if not python_exe.exists():
        python_exe = COMFYUI_DIR / "venv" / "bin" / "python"
    main_py = COMFYUI_DIR / "main.py"
    if python_exe.exists() and main_py.exists():
        return [str(python_exe), str(main_py), "--listen"]
    return None
