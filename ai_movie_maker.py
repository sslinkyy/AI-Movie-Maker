# ai_movie_maker.py - v1.3.2 (Cross-Platform Standalone)
"""AI Movie Maker: CLI + GUI tool for AI-assisted video creation."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import shlex
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

# --- Third-party packages (installed via requirements.txt) ---
try:
    import gradio as gr
    import keyring
    import numpy as np
    from huggingface_hub import hf_hub_download
    from moviepy.editor import (
        AudioFileClip,
        CompositeVideoClip,
        ImageClip,
        TextClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    from moviepy.video.fx import all as vfx
    from PIL import Image
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    import speech_recognition as sr
except ImportError as exc:  # pragma: no cover - handled during runtime
    print(
        f"❌ Missing dependency '{exc.name}'. Please install dependencies via requirements.txt or the installer.",
        file=sys.stderr,
    )
    raise

# Optional imports with graceful fallback
with contextlib.suppress(ImportError):
    from elevenlabs.client import ElevenLabs
with contextlib.suppress(ImportError):
    import openai
with contextlib.suppress(ImportError):
    import anthropic

APP_NAME = "AI Movie Maker"
APP_VERSION = "1.3.2"
OS_TYPE = platform.system().lower()

BASE_DIR = Path.home() / ".ai_movie_maker"
PROJECTS_DIR = BASE_DIR / "projects"
BIN_DIR = BASE_DIR / "bin"
MODELS_DIR = BASE_DIR / "models"
COMFYUI_DIR = BIN_DIR / "ComfyUI"
if OS_TYPE == "windows":
    COMFYUI_DIR = BIN_DIR / "ComfyUI_windows_portable"

COMFYUI_URL = "http://127.0.0.1:8188"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_subprocess(command: Iterable[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run a subprocess safely using shlex-provided arguments."""

    command_list = list(command)
    try:
        result = subprocess.run(
            command_list,
            cwd=str(cwd) if cwd else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command '{' '.join(command_list)}' failed with code {exc.returncode}: {exc.stderr.strip()}"
        ) from exc


def get_executable(path: Path) -> Optional[Path]:
    if OS_TYPE == "windows":
        exe = path.with_suffix(path.suffix + ".exe" if path.suffix else ".exe")
        return exe if exe.exists() else None
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_db_connection(project_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(project_path / "project.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(project_path: Path) -> None:
    with get_db_connection(project_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS project (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scenes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_number INTEGER UNIQUE,
                description TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS shots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id INTEGER,
                shot_number INTEGER,
                description TEXT,
                status TEXT DEFAULT 'pending',
                render_mode TEXT DEFAULT 'animatediff',
                duration_frames INTEGER DEFAULT 48,
                fps INTEGER DEFAULT 12,
                width INTEGER DEFAULT 512,
                height INTEGER DEFAULT 512,
                prompt TEXT,
                negative_prompt TEXT,
                camera_movement TEXT,
                voiceover_text TEXT,
                voiceover_file TEXT,
                subtitles TEXT,
                transition_to_next TEXT,
                output_path TEXT,
                preview_path TEXT,
                audio_path TEXT,
                FOREIGN KEY(scene_id) REFERENCES scenes(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS keyframes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shot_id INTEGER,
                frame_number INTEGER,
                prompt TEXT,
                FOREIGN KEY(shot_id) REFERENCES shots(id)
            )
            """
        )
        conn.commit()


# ---------------------------------------------------------------------------
# API key storage helpers
# ---------------------------------------------------------------------------


def get_api_key(service: str) -> Optional[str]:
    with contextlib.suppress(Exception):
        return keyring.get_password(APP_NAME, service)
    return None


def set_api_key(service: str, value: str) -> None:
    keyring.set_password(APP_NAME, service, value)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def get_llm_client(provider: str = "openai") -> Any:
    api_key = get_api_key(provider)
    if not api_key:
        raise ValueError(f"{provider.title()} API key is not configured.")
    if provider == "openai":
        if "openai" not in sys.modules:
            raise ImportError("openai package is required for OpenAI provider")
        client = openai.OpenAI(api_key=api_key)
        return client
    if provider == "anthropic":
        if "anthropic" not in sys.modules:
            raise ImportError("anthropic package is required for Anthropic provider")
        client = anthropic.Anthropic(api_key=api_key)
        return client
    raise NotImplementedError(f"Unsupported LLM provider: {provider}")


def llm_plan_keyframes(shot_description: str, num_keyframes: int = 3, context: str = "") -> List[str]:
    """Generate keyframe prompts using the configured LLM."""

    try:
        client = get_llm_client()
        system_prompt = (
            "You are an award-winning storyboard artist assisting a director."
            " Create vivid visual prompts for an image generation model."
            f" Provide exactly {num_keyframes} prompts in JSON format with the key 'prompts'."
        )
        user_prompt = json.dumps(
            {
                "context": context,
                "shot_description": shot_description,
                "instructions": "Focus on cinematic detail, camera placement, and mood."
            },
            indent=2,
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        parsed = json.loads(response.choices[0].message.content)
        prompts = parsed.get("prompts", [])
        if not prompts:
            raise ValueError("LLM returned empty prompt list")
        return prompts[:num_keyframes]
    except Exception as exc:
        print(f"⚠️ LLM keyframe planning failed: {exc}. Falling back to heuristic prompts.")
        return [
            f"{shot_description} – establishing shot with cinematic lighting",
            f"{shot_description} – mid-action, dynamic angle",
            f"{shot_description} – dramatic close-up finale",
        ][:num_keyframes]


def llm_refine_captions(raw_text: str, context: str = "") -> str:
    try:
        client = get_llm_client()
        system_prompt = (
            "You are a professional subtitle editor. Improve punctuation, grammar, and readability"
            " while preserving intent. Return only the cleaned text."
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps({"context": context, "raw": raw_text})
                },
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"⚠️ Caption refinement skipped: {exc}")
        return raw_text


# ---------------------------------------------------------------------------
# External tools & binaries
# ---------------------------------------------------------------------------


def ffmpeg_path() -> Optional[Path]:
    base = BIN_DIR / "ffmpeg" / ("bin" if OS_TYPE == "windows" else "")
    exe = base / ("ffmpeg.exe" if OS_TYPE == "windows" else "ffmpeg")
    return exe if exe.exists() else None


def rife_path() -> Optional[Path]:
    exe = BIN_DIR / "rife" / ("rife-ncnn-vulkan.exe" if OS_TYPE == "windows" else "rife-ncnn-vulkan")
    return exe if exe.exists() else None


def realesrgan_path() -> Optional[Path]:
    exe = BIN_DIR / "realesrgan" / ("realesrgan-ncnn-vulkan.exe" if OS_TYPE == "windows" else "realesrgan-ncnn-vulkan")
    return exe if exe.exists() else None


def wkhtmltopdf_path() -> Optional[Path]:
    exe = BIN_DIR / "wkhtmltopdf" / ("bin/wkhtmltopdf.exe" if OS_TYPE == "windows" else "wkhtmltopdf")
    return exe if exe.exists() else None


def comfyui_run_script() -> Optional[List[str]]:
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
    python_exe = COMFYUI_DIR / "python_embeded" / "python.exe"
    if not python_exe.exists():
        python_exe = COMFYUI_DIR / "venv" / "bin" / "python"
    main_py = COMFYUI_DIR / "main.py"
    if python_exe.exists() and main_py.exists():
        return [str(python_exe), str(main_py), "--listen"]
    return None


def is_comfyui_running() -> bool:
    try:
        requests.get(f"{COMFYUI_URL}/queue", timeout=1)
        return True
    except requests.RequestException:
        return False


def start_comfyui_server() -> Optional[subprocess.Popen]:
    command = comfyui_run_script()
    if not command:
        print("❌ ComfyUI run script not found. Please install ComfyUI via installer.")
        return None
    print("🚀 Starting ComfyUI server …")
    process = subprocess.Popen(
        command,
        cwd=str(COMFYUI_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0,
    )
    for _ in range(60):
        if is_comfyui_running():
            print("✅ ComfyUI server ready.")
            return process
        if process.poll() is not None:
            err = process.stderr.read().decode("utf-8", "ignore") if process.stderr else ""
            print(f"❌ ComfyUI exited early: {err[:500]}")
            return None
        time.sleep(1)
    print("❌ Timed out waiting for ComfyUI.")
    process.terminate()
    return None


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class ShotRenderResult:
    shot_id: int
    output_path: Path
    preview_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# AIMovieMaker core class
# ---------------------------------------------------------------------------


class AIMovieMaker:
    def __init__(self) -> None:
        self.project_name: Optional[str] = None
        self.project_path: Optional[Path] = None
        self._schema_checked: bool = False
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "output").mkdir(parents=True, exist_ok=True)

    # --- Workspace & dependency management -------------------------------------------------
    def init_workspace(self) -> None:
        print("🔧 Initialising workspace …")
        BIN_DIR.mkdir(exist_ok=True)
        MODELS_DIR.mkdir(exist_ok=True)
        print(f" Workspace: {BASE_DIR}")
        print(" Downloading base models via setup_models.py if available …")
        setup_script = Path(__file__).with_name("setup_models.py")
        if setup_script.exists():
            try:
                safe_subprocess([sys.executable, str(setup_script), "--auto"], cwd=setup_script.parent)
            except Exception as exc:
                print(f"⚠️ Model setup skipped: {exc}")
        else:
            print("⚠️ setup_models.py not found. Models must be installed manually.")

    def check_dependencies(self) -> Dict[str, bool]:
        checks = {
            "ffmpeg": ffmpeg_path() is not None,
            "rife": rife_path() is not None,
            "realesrgan": realesrgan_path() is not None,
            "wkhtmltopdf": wkhtmltopdf_path() is not None,
            "comfyui": comfyui_run_script() is not None,
        }
        for name, state in checks.items():
            print(f" - {name}: {'✅' if state else '❌'}")
        return checks

    # --- Project management ---------------------------------------------------------------
    def _require_project(self) -> None:
        if not self.project_path or not self.project_name:
            raise RuntimeError("No project loaded. Use load-project or gui.")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        if not self.project_path or self._schema_checked:
            return
        with get_db_connection(self.project_path) as conn:
            shot_columns = {row[1] for row in conn.execute("PRAGMA table_info(shots)")}
            if "fps" not in shot_columns:
                conn.execute("ALTER TABLE shots ADD COLUMN fps INTEGER DEFAULT 12")
            if "preview_path" not in shot_columns:
                conn.execute("ALTER TABLE shots ADD COLUMN preview_path TEXT")
            if "audio_path" not in shot_columns:
                conn.execute("ALTER TABLE shots ADD COLUMN audio_path TEXT")
            conn.commit()
        self._schema_checked = True

    def create_project(self, name: str) -> None:
        project_path = PROJECTS_DIR / name
        if project_path.exists():
            raise ValueError(f"Project '{name}' already exists")
        project_path.mkdir(parents=True)
        (project_path / "assets").mkdir()
        (project_path / "renders").mkdir()
        init_db(project_path)
        with get_db_connection(project_path) as conn:
            conn.execute("INSERT OR REPLACE INTO project (key, value) VALUES (?, ?)", ("name", name))
            conn.execute(
                "INSERT OR REPLACE INTO project (key, value) VALUES (?, ?)",
                ("script", "# My Movie\n\n## SCENE 1\nA mysterious door opens."),
            )
            conn.commit()
        print(f"🎉 Project '{name}' created at {project_path}")
        self.project_name = name
        self.project_path = project_path
        self._schema_checked = False

    def load_project(self, name: str) -> None:
        project_path = PROJECTS_DIR / name
        if not project_path.exists():
            raise ValueError(f"Project '{name}' not found")
        self.project_name = name
        self.project_path = project_path
        self._schema_checked = False
        print(f"📂 Loaded project '{name}'")

    def list_projects(self) -> List[str]:
        projects = [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]
        for project in projects:
            print(f" - {project}")
        return projects

    def list_scenes(self) -> List[Dict[str, Any]]:
        self._require_project()
        with get_db_connection(self.project_path) as conn:
            rows = conn.execute(
                "SELECT id, scene_number, description FROM scenes ORDER BY scene_number"
            ).fetchall()
        return [
            {"id": row["id"], "scene_number": row["scene_number"], "description": row["description"]}
            for row in rows
        ]

    def list_shots(self) -> List[Dict[str, Any]]:
        self._require_project()
        with get_db_connection(self.project_path) as conn:
            rows = conn.execute(
                """
                SELECT shots.*, scenes.scene_number, scenes.description AS scene_description
                FROM shots
                JOIN scenes ON scenes.id = shots.scene_id
                ORDER BY scenes.scene_number, shots.shot_number
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_shot_details(self, shot_id: int) -> Dict[str, Any]:
        self._require_project()
        with get_db_connection(self.project_path) as conn:
            row = conn.execute(
                """
                SELECT shots.*, scenes.scene_number, scenes.description AS scene_description
                FROM shots
                JOIN scenes ON scenes.id = shots.scene_id
                WHERE shots.id=?
                """,
                (shot_id,),
            ).fetchone()
        if not row:
            raise ValueError(f"Shot {shot_id} not found")
        return dict(row)

    def update_shot_settings(
        self,
        shot_id: int,
        *,
        description: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        render_mode: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        duration_frames: Optional[int] = None,
        camera_movement: Optional[str] = None,
        transition_to_next: Optional[str] = None,
    ) -> None:
        self._require_project()
        updates: Dict[str, Any] = {}
        numeric_fields = {"width", "height", "fps", "duration_frames"}
        for field, value in {
            "description": description,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "render_mode": render_mode,
            "width": width,
            "height": height,
            "fps": fps,
            "duration_frames": duration_frames,
            "camera_movement": camera_movement,
            "transition_to_next": transition_to_next,
        }.items():
            if value is None:
                continue
            if field in numeric_fields:
                updates[field] = int(value)
            else:
                updates[field] = value
        if not updates:
            return
        allowed_modes = {"animatediff", "kenburns", "cloud"}
        if "render_mode" in updates and updates["render_mode"] not in allowed_modes:
            raise ValueError(f"Invalid render mode: {updates['render_mode']}")
        with get_db_connection(self.project_path) as conn:
            assignments = ", ".join(f"{field}=?" for field in updates)
            conn.execute(
                f"UPDATE shots SET {assignments} WHERE id=?",
                tuple(updates.values()) + (shot_id,),
            )
            conn.commit()

    def get_keyframes_for_shot(self, shot_id: int) -> List[Dict[str, Any]]:
        self._require_project()
        with get_db_connection(self.project_path) as conn:
            rows = conn.execute(
                "SELECT frame_number, prompt FROM keyframes WHERE shot_id=? ORDER BY frame_number",
                (shot_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def generate_keyframes_for_shot(self, shot_id: int) -> List[str]:
        self._require_project()
        with get_db_connection(self.project_path) as conn:
            row = conn.execute("SELECT description FROM shots WHERE id=?", (shot_id,)).fetchone()
            if not row:
                raise ValueError(f"Shot {shot_id} not found")
            script = conn.execute("SELECT value FROM project WHERE key='script'").fetchone()[0]
        prompts = llm_plan_keyframes(row["description"], context=script)
        with get_db_connection(self.project_path) as conn:
            conn.execute("DELETE FROM keyframes WHERE shot_id=?", (shot_id,))
            for idx, prompt in enumerate(prompts):
                conn.execute(
                    "INSERT INTO keyframes (shot_id, frame_number, prompt) VALUES (?, ?, ?)",
                    (shot_id, idx, prompt),
                )
            if prompts:
                conn.execute("UPDATE shots SET prompt=? WHERE id=?", (prompts[0], shot_id))
            conn.commit()
        return prompts

    def save_project_as(self, destination: Path) -> Path:
        self._require_project()
        destination = destination.with_suffix(".zip")
        print(f"💾 Saving project to {destination}")
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as zipf:
            for path in self.project_path.rglob("*"):
                if path.is_file():
                    zipf.write(path, path.relative_to(self.project_path))
        return destination

    def load_project_from_zip(self, archive: Path) -> None:
        destination = PROJECTS_DIR / archive.stem
        if destination.exists():
            raise ValueError("Destination project already exists")
        print(f"📥 Importing project from {archive}")
        with zipfile.ZipFile(archive, "r") as zipf:
            zipf.extractall(destination)
        self.load_project(destination.name)

    # --- Script & storyboard --------------------------------------------------------------
    def _parse_script(self, script_text: str) -> List[Tuple[int, str, List[str]]]:
        sections = re.split(r"\n##\s+", script_text)
        parsed: List[Tuple[int, str, List[str]]] = []
        for section in sections:
            lines = [line.strip() for line in section.strip().splitlines() if line.strip()]
            if not lines:
                continue
            header = lines[0]
            match = re.search(r"(\d+)", header)
            if not match:
                continue
            scene_number = int(match.group(1))
            description = header
            shots = lines[1:] or ["Default establishing shot"]
            parsed.append((scene_number, description, shots))
        return parsed

    def sync_script(self, script_text: str) -> None:
        self._require_project()
        parsed = self._parse_script(script_text)
        with get_db_connection(self.project_path) as conn:
            conn.execute("DELETE FROM keyframes")
            conn.execute("DELETE FROM shots")
            conn.execute("DELETE FROM scenes")
            for scene_number, description, shots in parsed:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO scenes (scene_number, description) VALUES (?, ?)",
                    (scene_number, description),
                )
                scene_id = cur.lastrowid
                for shot_number, shot_desc in enumerate(shots, start=1):
                    conn.execute(
                        """
                        INSERT INTO shots (
                            scene_id,
                            shot_number,
                            description,
                            prompt,
                            camera_movement,
                            duration_frames,
                            fps,
                            width,
                            height
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            scene_id,
                            shot_number,
                            shot_desc,
                            shot_desc,
                            "static",
                            48,
                            12,
                            512,
                            512,
                        ),
                    )
            conn.execute(
                "INSERT OR REPLACE INTO project (key, value) VALUES (?, ?)",
                ("script", script_text),
            )
            conn.commit()
        print("📘 Script synchronised with database")

    def storyboard(self) -> List[Dict[str, Any]]:
        self._require_project()
        story: List[Dict[str, Any]] = []
        with get_db_connection(self.project_path) as conn:
            rows = conn.execute(
                """
                SELECT scenes.scene_number, scenes.description AS scene_desc,
                       shots.id AS shot_id, shots.shot_number, shots.description AS shot_desc,
                       shots.status, shots.output_path
                FROM scenes
                JOIN shots ON shots.scene_id = scenes.id
                ORDER BY scenes.scene_number, shots.shot_number
                """
            ).fetchall()
        for row in rows:
            story.append({
                "scene": row["scene_number"],
                "scene_description": row["scene_desc"],
                "shot_id": row["shot_id"],
                "shot_number": row["shot_number"],
                "shot_description": row["shot_desc"],
                "status": row["status"],
                "output_path": row["output_path"],
            })
        return story

    # --- Keyframes -----------------------------------------------------------------------
    def generate_keyframes(self) -> None:
        self._require_project()
        with get_db_connection(self.project_path) as conn:
            script = conn.execute("SELECT value FROM project WHERE key='script'").fetchone()[0]
            shots = conn.execute("SELECT id, description FROM shots").fetchall()
            for shot in shots:
                prompts = llm_plan_keyframes(shot["description"], context=script)
                conn.execute("DELETE FROM keyframes WHERE shot_id=?", (shot["id"],))
                for idx, prompt in enumerate(prompts):
                    conn.execute(
                        "INSERT INTO keyframes (shot_id, frame_number, prompt) VALUES (?, ?, ?)",
                        (shot["id"], idx, prompt),
                    )
                conn.execute(
                    "UPDATE shots SET prompt=? WHERE id=?",
                    (prompts[0] if prompts else shot["description"], shot["id"]),
                )
            conn.commit()
        print("✨ Keyframes generated for all shots")

    # --- Audio pipeline ------------------------------------------------------------------
    def generate_voiceover(self, shot_id: int, text: str) -> Path:
        self._require_project()
        api_key = get_api_key("elevenlabs")
        if not api_key:
            raise RuntimeError("ElevenLabs API key not configured")
        if "elevenlabs.client" not in sys.modules:
            raise ImportError("elevenlabs package required for voiceover generation")
        client = ElevenLabs(api_key=api_key)
        audio_bytes = client.generate(text=text, voice="Rachel", model="eleven_multilingual_v2")
        asset_path = self.project_path / "assets"
        asset_path.mkdir(exist_ok=True)
        audio_file = asset_path / f"shot_{shot_id}_voiceover.mp3"
        audio_file.write_bytes(audio_bytes)
        processed = self._auto_edit_audio(audio_file)
        with get_db_connection(self.project_path) as conn:
            conn.execute(
                "UPDATE shots SET voiceover_text=?, voiceover_file=? WHERE id=?",
                (text, str(processed.relative_to(self.project_path)), shot_id),
            )
            conn.commit()
        return processed

    def _auto_edit_audio(self, audio_path: Path) -> Path:
        sound = AudioSegment.from_file(audio_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=sound.dBFS - 18, keep_silence=200)
        processed = sum(chunks) if chunks else sound
        edited_path = audio_path.with_name(audio_path.stem + "_edited.mp3")
        processed.export(edited_path, format="mp3")
        return edited_path

    def generate_captions(self, shot_id: int) -> str:
        self._require_project()
        with get_db_connection(self.project_path) as conn:
            row = conn.execute("SELECT voiceover_file FROM shots WHERE id=?", (shot_id,)).fetchone()
        if not row or not row[0]:
            raise RuntimeError("No voiceover available for captioning")
        audio_path = self.project_path / row[0]
        recognizer = sr.Recognizer()
        with sr.AudioFile(str(audio_path)) as source:
            audio = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio)
        except sr.UnknownValueError as exc:
            raise RuntimeError("Speech recognition could not understand audio") from exc
        refined = llm_refine_captions(transcript)
        with get_db_connection(self.project_path) as conn:
            conn.execute("UPDATE shots SET subtitles=? WHERE id=?", (refined, shot_id))
            conn.commit()
        return refined

    def generate_music_stub(self, scene_id: int) -> Path:
        self._require_project()
        music_dir = self.project_path / "assets"
        music_dir.mkdir(exist_ok=True)
        music_path = music_dir / f"scene_{scene_id}_music_stub.mp3"
        if not music_path.exists():
            music_path.write_bytes(b"Suno AI music placeholder")
        return music_path

    # --- Rendering ----------------------------------------------------------------------
    def _render_ken_burns(self, shot_row: sqlite3.Row) -> ShotRenderResult:
        prompt = shot_row["prompt"] or "Ken Burns still"
        image = Image.new("RGB", (shot_row["width"], shot_row["height"]), color=(10, 10, 10))
        tmp_image = self.project_path / "assets" / f"shot_{shot_row['id']}_kenburns.png"
        tmp_image.parent.mkdir(exist_ok=True)
        image.save(tmp_image)
        fps = max(int(shot_row["fps"] or 12), 1)
        duration = max(shot_row["duration_frames"] / fps if shot_row["duration_frames"] else 1, 1.0)
        clip = ImageClip(str(tmp_image)).set_duration(duration)
        clip = clip.fx(vfx.resize, width=shot_row["width"] * 1.1)
        clip = clip.fx(vfx.crop, width=shot_row["width"], height=shot_row["height"], x_center=shot_row["width"] / 2)
        output = self.project_path / "renders" / f"shot_{shot_row['id']}.mp4"
        clip.write_videofile(str(output), fps=fps, codec="libx264", audio=False, verbose=False, logger=None)
        return ShotRenderResult(shot_id=shot_row["id"], output_path=output)

    def _render_animatediff(self, shot_row: sqlite3.Row) -> ShotRenderResult:
        if not is_comfyui_running():
            start_comfyui_server()
        payload = {
            "prompt": {
                "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.ckpt"}},
                "4": {"class_type": "AnimateDiffLoaderV1", "inputs": {"model_name": "mm_sd_v15_v2.ckpt"}},
                "6": {"class_type": "CLIPTextEncode", "inputs": {"text": shot_row["prompt"], "clip": ["3", 1]}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {"text": shot_row["negative_prompt"] or "", "clip": ["3", 1]}},
                "10": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": np.random.randint(0, 2 ** 32 - 1),
                        "steps": 25,
                        "cfg": 7,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "model": ["14", 0],
                        "positive": ["6", 0],
                        "negative": ["7", 0],
                        "latent_image": ["15", 0],
                    },
                },
                "12": {
                    "class_type": "ADE_AnimateDiffUniformContextOptions",
                    "inputs": {
                        "context_length": shot_row["duration_frames"],
                        "context_stride": 1,
                        "context_overlap": max(shot_row["duration_frames"] // 4, 1),
                        "closed_loop": "false",
                    },
                },
                "14": {
                    "class_type": "AnimateDiffModelSettings_V2",
                    "inputs": {"motion_model": ["4", 0], "context_options": ["12", 0], "model": ["3", 0]},
                },
                "15": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {"width": shot_row["width"], "height": shot_row["height"], "batch_size": 1},
                },
                "16": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 2]}},
                "17": {
                    "class_type": "VideoCombine",
                    "inputs": {
                        "images": ["16", 0],
                        "frame_rate": max(int(shot_row["fps"] or 12), 1),
                        "filename_prefix": f"shot_{shot_row['id']}",
                        "format": "image/gif",
                    },
                },
            }
        }
        response = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=30)
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        while True:
            history = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30).json()
            if prompt_id in history and history[prompt_id]["outputs"]:
                break
            time.sleep(1)
        outputs = history[prompt_id]["outputs"]
        gifs = outputs.get("17", {}).get("gifs", [])
        if not gifs:
            raise RuntimeError("ComfyUI did not return a GIF output")
        comfy_output = COMFYUI_DIR / "output" / gifs[0]["filename"]
        output = self.project_path / "renders" / f"shot_{shot_row['id']}.mp4"
        clip = VideoFileClip(str(comfy_output))
        fps = max(int(shot_row["fps"] or 12), 1)
        clip.write_videofile(str(output), codec="libx264", audio=False, verbose=False, logger=None, fps=fps)
        return ShotRenderResult(shot_id=shot_row["id"], output_path=output, preview_path=comfy_output)

    def _render_cloud_stub(self, shot_row: sqlite3.Row) -> ShotRenderResult:
        output = self.project_path / "renders" / f"shot_{shot_row['id']}_cloud.mp4"
        output.write_bytes(b"RunwayML cloud render placeholder")
        return ShotRenderResult(shot_id=shot_row["id"], output_path=output)

    def render_shot(self, shot_id: int) -> ShotRenderResult:
        self._require_project()
        with get_db_connection(self.project_path) as conn:
            shot_row = conn.execute("SELECT * FROM shots WHERE id=?", (shot_id,)).fetchone()
        if not shot_row:
            raise ValueError("Shot not found")
        mode = shot_row["render_mode"]
        if mode == "kenburns":
            result = self._render_ken_burns(shot_row)
        elif mode == "cloud":
            result = self._render_cloud_stub(shot_row)
        else:
            result = self._render_animatediff(shot_row)
        with get_db_connection(self.project_path) as conn:
            conn.execute(
                "UPDATE shots SET status='rendered', output_path=?, preview_path=? WHERE id=?",
                (
                    str(result.output_path.relative_to(self.project_path)),
                    str(result.preview_path.relative_to(self.project_path)) if result.preview_path else None,
                    shot_id,
                ),
            )
            conn.commit()
        return result

    def render_scene(self, scene_number: int) -> List[ShotRenderResult]:
        self._require_project()
        with get_db_connection(self.project_path) as conn:
            rows = conn.execute(
                """
                SELECT shots.* FROM shots
                JOIN scenes ON scenes.id = shots.scene_id
                WHERE scenes.scene_number=?
                ORDER BY shots.shot_number
                """,
                (scene_number,),
            ).fetchall()
        results = [self.render_shot(row["id"]) for row in rows]
        return results

    def render_project(self) -> List[ShotRenderResult]:
        self._require_project()
        with get_db_connection(self.project_path) as conn:
            rows = conn.execute("SELECT id FROM shots ORDER BY scene_id, shot_number").fetchall()
        return [self.render_shot(row["id"]) for row in rows]

    # --- Export --------------------------------------------------------------------------
    def _apply_transitions(self, clips: List[VideoFileClip], transitions: List[str]) -> List[VideoFileClip]:
        final_clips: List[VideoFileClip] = []
        for idx, clip in enumerate(clips):
            transition = transitions[idx] if idx < len(transitions) else "none"
            if transition == "fade" and clip.duration > 1:
                clip = clip.fx(vfx.fadein, 0.5).fx(vfx.fadeout, 0.5)
            elif transition == "crossfade" and idx + 1 < len(clips):
                clip = clip.crossfadeout(0.7)
            final_clips.append(clip)
        return final_clips

    def export(self, fmt: str = "mp4") -> Path:
        self._require_project()
        with get_db_connection(self.project_path) as conn:
            rows = conn.execute(
                """
                SELECT output_path, voiceover_file, subtitles, transition_to_next
                FROM shots WHERE status='rendered' AND output_path IS NOT NULL
                ORDER BY scene_id, shot_number
                """
            ).fetchall()
        if not rows:
            raise RuntimeError("No rendered shots available")
        clips: List[VideoFileClip] = []
        transitions: List[str] = []
        for row in rows:
            clip_path = self.project_path / row["output_path"]
            clip = VideoFileClip(str(clip_path))
            if row["voiceover_file"]:
                audio_path = self.project_path / row["voiceover_file"]
                if audio_path.exists():
                    clip = clip.set_audio(AudioFileClip(str(audio_path)))
            if row["subtitles"]:
                txt = TextClip(row["subtitles"], fontsize=26, color="white", bg_color="black", size=(clip.w * 0.9, None), method="caption")
                txt = txt.set_position(("center", "bottom")).set_duration(clip.duration)
                clip = CompositeVideoClip([clip, txt])
            clips.append(clip)
            transitions.append(row["transition_to_next"] or "none")
        clips = self._apply_transitions(clips, transitions)
        final = concatenate_videoclips(clips, method="compose")
        output = BASE_DIR / "output" / f"{self.project_name}.{fmt}"
        output.parent.mkdir(exist_ok=True)
        if fmt == "mp4":
            final.write_videofile(str(output), codec="libx264", audio_codec="aac")
        elif fmt == "gif":
            final.write_gif(str(output))
        elif fmt == "avi":
            final.write_videofile(str(output), codec="png", audio_codec="pcm_s16le")
        else:
            raise ValueError("Unsupported export format")
        return output

    def export_pdf(self, destination: Path) -> Path:
        self._require_project()
        wkhtml = wkhtmltopdf_path()
        if not wkhtml:
            raise RuntimeError("wkhtmltopdf not available")
        html_file = self.project_path / "storyboard.html"
        story = self.storyboard()
        html_content = ["<html><body><h1>Storyboard</h1>"]
        for entry in story:
            html_content.append(f"<h2>Scene {entry['scene']}</h2><p>{entry['scene_description']}</p>")
            html_content.append(f"<strong>Shot {entry['shot_number']}:</strong> {entry['shot_description']}<br/>")
        html_content.append("</body></html>")
        html_file.write_text("\n".join(html_content), encoding="utf-8")
        destination = destination.with_suffix(".pdf")
        safe_subprocess([str(wkhtml), str(html_file), str(destination)])
        return destination

    # --- Collaboration -------------------------------------------------------------------
    def share_project_stub(self) -> str:
        self._require_project()
        share_path = self.project_path / "share_link.txt"
        url = f"https://dropbox.example.com/{uuid.uuid4()}"
        share_path.write_text(url, encoding="utf-8")
        return url

    def fetch_shared_project_stub(self, url: str) -> Path:
        destination = BASE_DIR / "shared" / url.split("/")[-1]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("Shared project placeholder", encoding="utf-8")
        return destination


# ---------------------------------------------------------------------------
# Gradio GUI
# ---------------------------------------------------------------------------


def launch_gui():
    app = AIMovieMaker()

    def _project_choices() -> List[str]:
        return app.list_projects()

    def _project_payload() -> Tuple[str, List[Dict[str, Any]], List[Tuple[str, int]], Optional[int], List[Tuple[str, int]], Optional[int]]:
        if not app.project_name:
            return "", [], [], None, [], None
        with get_db_connection(app.project_path) as conn:
            row = conn.execute("SELECT value FROM project WHERE key='script'").fetchone()
            script = row[0] if row else ""
        storyboard = app.storyboard()
        shot_rows = app.list_shots()
        shot_choices = [
            (
                f"Scene {row['scene_number']} • Shot {row['shot_number']}: {row['description'][:40]}",
                row['id'],
            )
            for row in shot_rows
        ]
        default_shot = shot_choices[0][1] if shot_choices else None
        scenes = app.list_scenes()
        scene_choices = [
            (f"Scene {scene['scene_number']}: {scene['description']}", scene['scene_number'])
            for scene in scenes
        ]
        scene_default = scene_choices[0][1] if scene_choices else None
        return script, storyboard, shot_choices, default_shot, scene_choices, scene_default

    def _keyframe_table_data(shot_id: Optional[int]) -> List[List[Any]]:
        if not shot_id:
            return []
        frames = app.get_keyframes_for_shot(int(shot_id))
        return [[row["frame_number"], row["prompt"]] for row in frames]

    def handle_create_project(name: str):
        if not name:
            raise gr.Error("Project name required")
        app.create_project(name)
        script, storyboard, shot_choices, default_shot, scene_choices, scene_default = _project_payload()
        return (
            gr.update(choices=_project_choices(), value=name),
            script,
            gr.update(value=storyboard),
            gr.update(choices=shot_choices, value=default_shot),
            gr.update(choices=scene_choices, value=scene_default),
            f"Created project {name}",
        )

    def handle_load_project(name: str):
        if not name:
            raise gr.Error("Select a project")
        app.load_project(name)
        script, storyboard, shot_choices, default_shot, scene_choices, scene_default = _project_payload()
        return (
            script,
            gr.update(value=storyboard),
            gr.update(choices=shot_choices, value=default_shot),
            gr.update(choices=scene_choices, value=scene_default),
            f"Loaded project {name}",
        )

    def handle_save_script(script_text: str):
        if not app.project_name:
            raise gr.Error("Load a project first")
        app.sync_script(script_text)
        script, storyboard, shot_choices, default_shot, scene_choices, scene_default = _project_payload()
        return (
            script_text,
            gr.update(value=storyboard),
            gr.update(choices=shot_choices, value=default_shot),
            gr.update(choices=scene_choices, value=scene_default),
            "Script saved and shots updated.",
        )

    def handle_generate_keyframes_all(current_shot: Optional[int]):
        if not app.project_name:
            raise gr.Error("Load a project first")
        app.generate_keyframes()
        table = _keyframe_table_data(current_shot)
        return "Generated keyframes for all shots.", gr.update(value=table)

    def handle_shot_change(shot_id: Optional[int]):
        if not shot_id:
            return (
                "",
                "",
                "",
                "animatediff",
                512,
                512,
                12,
                48,
                "",
                "none",
                None,
                "",
                "",
                [],
                None,
            )
        data = app.get_shot_details(int(shot_id))
        preview = None
        if data.get("output_path"):
            preview_path = app.project_path / data["output_path"]
            if preview_path.exists():
                preview = str(preview_path)
        table = _keyframe_table_data(int(shot_id))
        transition = data.get("transition_to_next") or "none"
        return (
            data.get("description", ""),
            data.get("prompt", ""),
            data.get("negative_prompt", ""),
            data.get("render_mode", "animatediff"),
            int(data.get("width") or 512),
            int(data.get("height") or 512),
            int(data.get("fps") or 12),
            int(data.get("duration_frames") or 48),
            data.get("camera_movement") or "",
            transition,
            preview,
            data.get("voiceover_text") or "",
            data.get("subtitles") or "",
            table,
            data.get("scene_number"),
        )

    def handle_save_shot(
        shot_id: Optional[int],
        description: str,
        prompt: str,
        negative: str,
        render_mode: str,
        width: int,
        height: int,
        fps: int,
        duration_frames: int,
        camera: str,
        transition: str,
    ):
        if not shot_id:
            raise gr.Error("Select a shot to save")
        app.update_shot_settings(
            int(shot_id),
            description=description,
            prompt=prompt,
            negative_prompt=negative,
            render_mode=render_mode,
            width=int(width),
            height=int(height),
            fps=int(fps),
            duration_frames=int(duration_frames),
            camera_movement=camera or None,
            transition_to_next=None if transition == "none" else transition,
        )
        storyboard = app.storyboard()
        shot_values = handle_shot_change(shot_id)
        return (
            f"Shot {shot_id} saved.",
            gr.update(value=storyboard),
            *shot_values,
        )

    def handle_generate_keyframes_shot(shot_id: Optional[int]):
        if not shot_id:
            raise gr.Error("Select a shot")
        app.generate_keyframes_for_shot(int(shot_id))
        table = _keyframe_table_data(int(shot_id))
        return (
            f"Generated keyframes for shot {shot_id}.",
            gr.update(value=table),
        )

    def handle_voiceover(shot_id: Optional[int], text: str):
        if not shot_id:
            raise gr.Error("Select a shot")
        if not text.strip():
            raise gr.Error("Enter voiceover text")
        path = app.generate_voiceover(int(shot_id), text.strip())
        shot_values = handle_shot_change(shot_id)
        return (
            f"Voiceover saved to {path}",
            shot_values[11],
            shot_values[12],
        )

    def handle_captions(shot_id: Optional[int]):
        if not shot_id:
            raise gr.Error("Select a shot")
        captions = app.generate_captions(int(shot_id))
        return f"Captions generated for shot {shot_id}.", captions

    def handle_music(scene_number: Optional[int]):
        if scene_number is None:
            raise gr.Error("Select a scene")
        path = app.generate_music_stub(int(scene_number))
        return f"Music stub saved to {path}"

    def handle_render_shot(shot_id: Optional[int]):
        if not shot_id:
            raise gr.Error("Select a shot")
        result = app.render_shot(int(shot_id))
        storyboard = app.storyboard()
        shot_values = handle_shot_change(shot_id)
        return (
            f"Rendered shot {shot_id} → {result.output_path}",
            gr.update(value=storyboard),
            shot_values[10],
        )

    def handle_render_scene(scene_number: Optional[float]):
        if scene_number is None:
            raise gr.Error("Enter a scene number")
        results = app.render_scene(int(scene_number))
        storyboard = app.storyboard()
        return f"Rendered {len(results)} shots for scene {int(scene_number)}.", gr.update(value=storyboard)

    def handle_render_project():
        results = app.render_project()
        storyboard = app.storyboard()
        return f"Rendered {len(results)} shots for project.", gr.update(value=storyboard)

    def handle_export(fmt: str):
        output = app.export(fmt)
        return f"Exported project to {output}"

    def handle_check():
        checks = app.check_dependencies()
        return "\n".join(f"{name}: {'OK' if state else 'missing'}" for name, state in checks.items())

    def handle_save_keys(openai_key: str, anthropic_key: str, eleven_key: str):
        set_api_key("openai", openai_key)
        set_api_key("anthropic", anthropic_key)
        set_api_key("elevenlabs", eleven_key)
        return "API keys saved."

    def handle_share():
        url = app.share_project_stub()
        return url, f"Share link generated: {url}"

    def handle_fetch(url: str):
        if not url:
            raise gr.Error("Enter a share URL")
        path = app.fetch_shared_project_stub(url)
        return f"Fetched shared project placeholder at {path}"

    def handle_save_project_zip(path: str):
        if not path:
            raise gr.Error("Enter a destination path")
        saved = app.save_project_as(Path(path))
        return f"Saved project archive to {saved}"

    def handle_load_project_zip(path: str):
        if not path:
            raise gr.Error("Enter a project archive path")
        app.load_project_from_zip(Path(path))
        script, storyboard, shot_choices, default_shot, scene_choices, scene_default = _project_payload()
        return (
            gr.update(choices=_project_choices(), value=app.project_name),
            script,
            gr.update(value=storyboard),
            gr.update(choices=shot_choices, value=default_shot),
            gr.update(choices=scene_choices, value=scene_default),
            f"Loaded project {app.project_name} from archive.",
        )

    with gr.Blocks(title=f"{APP_NAME} {APP_VERSION}") as demo:
        gr.Markdown(f"# 🎬 {APP_NAME} v{APP_VERSION}")
        with gr.Row():
            project_selector = gr.Dropdown(_project_choices(), label="Project", interactive=True)
            new_project = gr.Textbox(label="Create Project", placeholder="my_movie")
            create_btn = gr.Button("Create & Load")
            status_bar = gr.Textbox(label="Status", interactive=False)

        with gr.Tabs():
            with gr.TabItem("Checklist"):
                check_btn = gr.Button("Run Dependency Check")
                check_output = gr.Textbox(label="Results", lines=6)

            with gr.TabItem("Settings"):
                openai_key = gr.Textbox(label="OpenAI API Key", type="password", value=get_api_key("openai") or "")
                anthropic_key = gr.Textbox(label="Anthropic API Key", type="password", value=get_api_key("anthropic") or "")
                eleven_key = gr.Textbox(label="ElevenLabs API Key", type="password", value=get_api_key("elevenlabs") or "")
                save_keys_btn = gr.Button("Save API Keys")

            with gr.TabItem("Script & Storyboard"):
                script_editor = gr.Code(language="markdown", label="Script")
                save_script_btn = gr.Button("Save & Sync Script")
                generate_all_btn = gr.Button("Generate Keyframes for All Shots")
                storyboard_table = gr.DataFrame(headers=["scene", "scene_description", "shot_number", "shot_description", "status"], interactive=False)

            with gr.TabItem("Shot Editor"):
                shot_dropdown = gr.Dropdown(choices=[], label="Shot")
                with gr.Row():
                    shot_desc = gr.Textbox(label="Description")
                    shot_prompt = gr.Textbox(label="Prompt")
                    shot_negative = gr.Textbox(label="Negative Prompt")
                with gr.Row():
                    shot_render_mode = gr.Radio(["animatediff", "kenburns", "cloud"], label="Render Mode", value="animatediff")
                    shot_width = gr.Number(label="Width", precision=0, value=512)
                    shot_height = gr.Number(label="Height", precision=0, value=512)
                    shot_fps = gr.Number(label="FPS", precision=0, value=12)
                    shot_duration = gr.Number(label="Duration (frames)", precision=0, value=48)
                with gr.Row():
                    shot_camera = gr.Textbox(label="Camera Movement")
                    shot_transition = gr.Radio(["none", "fade", "crossfade"], label="Transition → Next", value="none")
                shot_preview = gr.Video(label="Latest Render Preview")
                with gr.Row():
                    save_shot_btn = gr.Button("Save Shot")
                    generate_shot_keyframes_btn = gr.Button("Generate Keyframes for Shot")
                keyframe_table = gr.DataFrame(headers=["Frame", "Prompt"], interactive=False)
                shot_status = gr.Textbox(label="Shot Status", interactive=False)

            with gr.TabItem("Audio / Captions / Music"):
                voiceover_text = gr.Textbox(label="Voiceover Text", lines=4)
                generate_voiceover_btn = gr.Button("Generate Voiceover")
                generate_captions_btn = gr.Button("Generate Captions")
                subtitles_box = gr.Textbox(label="Subtitles", lines=4)
                music_scene_dropdown = gr.Dropdown(label="Scene for Music Stub", choices=[])
                generate_music_btn = gr.Button("Generate Music Stub")
                audio_status = gr.Textbox(label="Audio Status", interactive=False)

            with gr.TabItem("Render & Export"):
                render_shot_btn = gr.Button("Render Selected Shot")
                render_scene_number = gr.Number(label="Scene #", precision=0)
                render_scene_btn = gr.Button("Render Scene")
                render_project_btn = gr.Button("Render Entire Project")
                export_format = gr.Radio(["mp4", "gif", "avi"], label="Export Format", value="mp4")
                export_btn = gr.Button("Export Project")
                render_status = gr.Textbox(label="Render Status", interactive=False)

            with gr.TabItem("Collaboration"):
                save_project_path = gr.Textbox(label="Save Project (.zip)")
                save_project_btn = gr.Button("Save Project Archive")
                load_project_path = gr.Textbox(label="Load Project (.zip)")
                load_project_btn = gr.Button("Load Project Archive")
                share_btn = gr.Button("Create Share Link")
                share_output = gr.Textbox(label="Share Link", interactive=False)
                fetch_url = gr.Textbox(label="Fetch Shared Project URL")
                fetch_btn = gr.Button("Fetch Shared Project")
                collab_status = gr.Textbox(label="Collaboration Status", interactive=False)

            with gr.TabItem("Help"):
                gr.Markdown(
                    """
                    ### Tips
                    - Use **Script & Storyboard** to edit your screenplay and automatically create shots.
                    - Configure per-shot prompts, camera moves, and render options in **Shot Editor**.
                    - Generate ElevenLabs voiceovers and refined captions from the **Audio** tab.
                    - Render locally with AnimateDiff/Ken Burns or queue cloud stubs from the **Render** tab.
                    - Share and archive projects directly from the **Collaboration** tab.
                    """
                )

        # Event wiring
        create_btn.click(
            handle_create_project,
            inputs=new_project,
            outputs=[project_selector, script_editor, storyboard_table, shot_dropdown, music_scene_dropdown, status_bar],
        )

        project_selector.change(
            handle_load_project,
            inputs=project_selector,
            outputs=[script_editor, storyboard_table, shot_dropdown, music_scene_dropdown, status_bar],
        )

        save_script_btn.click(
            handle_save_script,
            inputs=script_editor,
            outputs=[script_editor, storyboard_table, shot_dropdown, music_scene_dropdown, status_bar],
        )

        generate_all_btn.click(
            handle_generate_keyframes_all,
            inputs=shot_dropdown,
            outputs=[status_bar, keyframe_table],
        )

        shot_dropdown.change(
            handle_shot_change,
            inputs=shot_dropdown,
            outputs=[
                shot_desc,
                shot_prompt,
                shot_negative,
                shot_render_mode,
                shot_width,
                shot_height,
                shot_fps,
                shot_duration,
                shot_camera,
                shot_transition,
                shot_preview,
                voiceover_text,
                subtitles_box,
                keyframe_table,
                music_scene_dropdown,
            ],
        )

        save_shot_btn.click(
            handle_save_shot,
            inputs=[
                shot_dropdown,
                shot_desc,
                shot_prompt,
                shot_negative,
                shot_render_mode,
                shot_width,
                shot_height,
                shot_fps,
                shot_duration,
                shot_camera,
                shot_transition,
            ],
            outputs=[
                shot_status,
                storyboard_table,
                shot_desc,
                shot_prompt,
                shot_negative,
                shot_render_mode,
                shot_width,
                shot_height,
                shot_fps,
                shot_duration,
                shot_camera,
                shot_transition,
                shot_preview,
                voiceover_text,
                subtitles_box,
                keyframe_table,
                music_scene_dropdown,
            ],
        )

        generate_shot_keyframes_btn.click(
            handle_generate_keyframes_shot,
            inputs=shot_dropdown,
            outputs=[shot_status, keyframe_table],
        )

        generate_voiceover_btn.click(
            handle_voiceover,
            inputs=[shot_dropdown, voiceover_text],
            outputs=[audio_status, voiceover_text, subtitles_box],
        )

        generate_captions_btn.click(
            handle_captions,
            inputs=shot_dropdown,
            outputs=[audio_status, subtitles_box],
        )

        generate_music_btn.click(
            handle_music,
            inputs=music_scene_dropdown,
            outputs=[audio_status],
        )

        render_shot_btn.click(
            handle_render_shot,
            inputs=shot_dropdown,
            outputs=[render_status, storyboard_table, shot_preview],
        )

        render_scene_btn.click(
            handle_render_scene,
            inputs=render_scene_number,
            outputs=[render_status, storyboard_table],
        )

        render_project_btn.click(
            handle_render_project,
            outputs=[render_status, storyboard_table],
        )

        export_btn.click(
            handle_export,
            inputs=export_format,
            outputs=[render_status],
        )

        check_btn.click(handle_check, outputs=[check_output])

        save_keys_btn.click(
            handle_save_keys,
            inputs=[openai_key, anthropic_key, eleven_key],
            outputs=[status_bar],
        )

        share_btn.click(handle_share, outputs=[share_output, collab_status])

        fetch_btn.click(handle_fetch, inputs=fetch_url, outputs=[collab_status])

        save_project_btn.click(handle_save_project_zip, inputs=save_project_path, outputs=[collab_status])

        load_project_btn.click(
            handle_load_project_zip,
            inputs=load_project_path,
            outputs=[project_selector, script_editor, storyboard_table, shot_dropdown, music_scene_dropdown, collab_status],
        )

    demo.launch()


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialise workspace")
    sub.add_parser("check", help="Check dependencies")
    sub.add_parser("list", help="List projects")

    create_parser = sub.add_parser("create", help="Create project")
    create_parser.add_argument("name")

    load_parser = sub.add_parser("load", help="Load project")
    load_parser.add_argument("name")

    sync_parser = sub.add_parser("sync-script", help="Sync script from file")
    sync_parser.add_argument("name")
    sync_parser.add_argument("script_file")

    sub.add_parser("gui", help="Launch GUI")

    keyframes_parser = sub.add_parser("gen-keyframes", help="Generate keyframes")
    keyframes_parser.add_argument("name")

    render_parser = sub.add_parser("run", help="Render project or specific shot/scene")
    render_parser.add_argument("name")
    render_parser.add_argument("target")

    export_parser = sub.add_parser("export", help="Export project")
    export_parser.add_argument("name")
    export_parser.add_argument("format", choices=["mp4", "gif", "avi"])

    pdf_parser = sub.add_parser("export-pdf", help="Export storyboard PDF")
    pdf_parser.add_argument("name")
    pdf_parser.add_argument("destination")

    save_parser = sub.add_parser("save-project", help="Save project to archive")
    save_parser.add_argument("name")
    save_parser.add_argument("destination")

    loadzip_parser = sub.add_parser("load-project", help="Load project from archive")
    loadzip_parser.add_argument("archive")

    share_parser = sub.add_parser("share", help="Generate collaboration link stub")
    share_parser.add_argument("name")

    fetch_parser = sub.add_parser("fetch-share", help="Fetch shared project stub")
    fetch_parser.add_argument("url")

    storyboard_parser = sub.add_parser("storyboard", help="Print storyboard entries")
    storyboard_parser.add_argument("name")

    edit_parser = sub.add_parser("edit-shot", help="Update shot settings")
    edit_parser.add_argument("name")
    edit_parser.add_argument("shot_id", type=int)
    edit_parser.add_argument("field")
    edit_parser.add_argument("value")

    voice_parser = sub.add_parser("voiceover", help="Generate AI voiceover for a shot")
    voice_parser.add_argument("name")
    voice_parser.add_argument("shot_id", type=int)
    voice_parser.add_argument("text", nargs="+")

    captions_parser = sub.add_parser("captions", help="Generate captions for a shot")
    captions_parser.add_argument("name")
    captions_parser.add_argument("shot_id", type=int)

    music_parser = sub.add_parser("music-stub", help="Generate music placeholder for a scene")
    music_parser.add_argument("name")
    music_parser.add_argument("scene", type=int)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    app = AIMovieMaker()

    if args.command == "init":
        app.init_workspace()
    elif args.command == "check":
        app.check_dependencies()
    elif args.command == "list":
        app.list_projects()
    elif args.command == "create":
        app.create_project(args.name)
    elif args.command == "load":
        app.load_project(args.name)
    elif args.command == "sync-script":
        app.load_project(args.name)
        script_text = Path(args.script_file).read_text(encoding="utf-8")
        app.sync_script(script_text)
    elif args.command == "gen-keyframes":
        app.load_project(args.name)
        app.generate_keyframes()
    elif args.command == "run":
        app.load_project(args.name)
        target = args.target
        if target == "project":
            app.render_project()
        elif target.startswith("scene:"):
            scene_number = int(target.split(":", 1)[1])
            app.render_scene(scene_number)
        elif target.startswith("shot:"):
            shot_id = int(target.split(":", 1)[1])
            app.render_shot(shot_id)
        else:
            raise ValueError("Unknown target. Use project, scene:<n>, or shot:<id>.")
    elif args.command == "export":
        app.load_project(args.name)
        app.export(args.format)
    elif args.command == "export-pdf":
        app.load_project(args.name)
        app.export_pdf(Path(args.destination))
    elif args.command == "save-project":
        app.load_project(args.name)
        app.save_project_as(Path(args.destination))
    elif args.command == "load-project":
        app.load_project_from_zip(Path(args.archive))
    elif args.command == "share":
        app.load_project(args.name)
        url = app.share_project_stub()
        print(f"Share link: {url}")
    elif args.command == "fetch-share":
        path = app.fetch_shared_project_stub(args.url)
        print(f"Fetched shared project placeholder at {path}")
    elif args.command == "storyboard":
        app.load_project(args.name)
        for entry in app.storyboard():
            print(
                f"Scene {entry['scene']}: {entry['scene_description']} | "
                f"Shot {entry['shot_number']} ({entry['status']}): {entry['shot_description']}"
            )
    elif args.command == "edit-shot":
        app.load_project(args.name)
        field = args.field.replace("-", "_")
        value: Any = args.value
        numeric = {"width", "height", "fps", "duration_frames"}
        if field in numeric:
            value = int(value)
        if field == "transition_to_next" and value == "none":
            value = None
        app.update_shot_settings(args.shot_id, **{field: value})
        print(f"Updated shot {args.shot_id}: {field}")
    elif args.command == "voiceover":
        app.load_project(args.name)
        text = " ".join(args.text)
        path = app.generate_voiceover(args.shot_id, text)
        print(f"Voiceover saved to {path}")
    elif args.command == "captions":
        app.load_project(args.name)
        text = app.generate_captions(args.shot_id)
        print(text)
    elif args.command == "music-stub":
        app.load_project(args.name)
        path = app.generate_music_stub(args.scene)
        print(f"Music stub saved to {path}")
    elif args.command == "gui" or args.command is None:
        launch_gui()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
