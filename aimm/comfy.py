"""ComfyUI integration helpers."""
from __future__ import annotations

import subprocess
import time
from typing import Dict, Optional

import requests

from .config import COMFYUI_DIR, COMFYUI_URL
from .utils import comfyui_run_script


def comfy_running() -> bool:
    try:
        requests.get(f"{COMFYUI_URL}/queue", timeout=1)
        return True
    except requests.RequestException:
        return False


def start_comfy() -> Optional[subprocess.Popen[str]]:
    if comfy_running():
        return None
    cmd = comfyui_run_script()
    if not cmd:
        raise RuntimeError(f"ComfyUI executable not found under {COMFYUI_DIR}")
    process = subprocess.Popen(  # noqa: S603 - trusted local invocation
        cmd,
        cwd=str(COMFYUI_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    for _ in range(60):
        if comfy_running():
            return process
        time.sleep(1)
        if process.poll() is not None:
            stderr_output = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"ComfyUI failed to start: {stderr_output[:500]}")
    raise RuntimeError("Timed out waiting for ComfyUI to start")


def comfy_queue(prompt_workflow: Dict) -> Optional[Dict]:
    try:
        response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": prompt_workflow}, timeout=30)
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        for _ in range(300):  # 5-minute timeout
            history = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30).json()
            if prompt_id in history and history[prompt_id].get("outputs"):
                return history[prompt_id]["outputs"]
            time.sleep(1)
        raise RuntimeError(f"Timed out waiting for ComfyUI prompt {prompt_id}")
    except Exception as exc:  # pragma: no cover - network heavy
        print(f"[comfy_queue] {exc}")
        return None
