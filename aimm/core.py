"""Core application logic for AI Movie Maker."""
from __future__ import annotations

import contextlib
import re
import sqlite3
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Third-party imports installed via requirements.txt
    import numpy as np
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
except ImportError as exc:  # pragma: no cover - runtime guard
    print(
        f"❌ Missing dependency '{exc.name}'. Install requirements via requirements.txt or the installer.",
        file=sys.stderr,
    )
    raise

from .config import (
    ANIMATEDIFF_CHECKPOINT,
    ANIMATEDIFF_MOTION_MODEL,
    BASE_DIR,
    BIN_DIR,
    COMFYUI_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PROJECTS_DIR,
)
from .comfy import comfy_queue, comfy_running, start_comfy
from .db import get_db_connection, init_db
from .llm import llm_plan_keyframes, llm_refine_captions
from .secrets import get_api_key
from .utils import (
    comfyui_run_script,
    ffmpeg_path,
    realesrgan_path,
    rife_path,
    safe_subprocess,
    wkhtmltopdf_path,
)

with contextlib.suppress(ImportError):
    from elevenlabs.client import ElevenLabs


@dataclass
class ShotRenderResult:
    shot_id: int
    output_path: Path
    preview_path: Optional[Path] = None


class WorkspaceManager:
    """Manage workspace folders and dependency checks."""

    def __init__(self) -> None:
        self._ensure_base_dirs()

    def _ensure_base_dirs(self) -> None:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def init_workspace(self) -> None:
        """Initialise the workspace and trigger optional model setup."""
        print("🔧 Initialising workspace …")
        self._ensure_base_dirs()
        BIN_DIR.mkdir(exist_ok=True)
        MODELS_DIR.mkdir(exist_ok=True)
        setup_script = Path(__file__).resolve().parent.parent / "setup_models.py"
        if setup_script.exists():
            try:
                safe_subprocess([sys.executable, str(setup_script), "--auto"], cwd=setup_script.parent)
            except RuntimeError as exc:
                print(f"⚠️ Model setup skipped: {exc}")
        else:
            print("⚠️ setup_models.py not found. Install models manually if required.")

    def check_dependencies(self) -> Dict[str, bool]:
        """Return availability of optional binaries."""
        checks = {
            "ffmpeg": ffmpeg_path() is not None,
            "rife": rife_path() is not None,
            "realesrgan": realesrgan_path() is not None,
            "wkhtmltopdf": wkhtmltopdf_path() is not None,
            "comfyui": comfyui_run_script() is not None,
        }
        for name, available in checks.items():
            print(f" - {name}: {'✅' if available else '❌'}")
        return checks


class ProjectManager:
    """Handle project lifecycle, database access, and story structure."""

    def __init__(self, workspace: WorkspaceManager) -> None:
        self.workspace = workspace
        self.project_name: Optional[str] = None
        self.project_path: Optional[Path] = None
        self._schema_checked = False

    # ------------------------------------------------------------------
    # Project metadata helpers
    # ------------------------------------------------------------------
    def require_project(self) -> Path:
        if not self.project_path or not self.project_name:
            raise RuntimeError("No project loaded. Use load-project or gui.")
        self._ensure_schema()
        return self.project_path

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

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Storyboard helpers
    # ------------------------------------------------------------------
    def list_scenes(self) -> List[Dict[str, Any]]:
        path = self.require_project()
        with get_db_connection(path) as conn:
            rows = conn.execute(
                "SELECT id, scene_number, description FROM scenes ORDER BY scene_number"
            ).fetchall()
        return [
            {"id": row["id"], "scene_number": row["scene_number"], "description": row["description"]}
            for row in rows
        ]

    def list_shots(self) -> List[Dict[str, Any]]:
        path = self.require_project()
        with get_db_connection(path) as conn:
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
        path = self.require_project()
        with get_db_connection(path) as conn:
            row = conn.execute(
                """
                SELECT shots.*, scenes.scene_number, scenes.description AS scene_description
                FROM shots
                JOIN scenes ON scenes.scene_id = shots.scene_id
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
        path = self.require_project()
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
            updates[field] = int(value) if field in numeric_fields else value
        if not updates:
            return
        allowed_modes = {"animatediff", "kenburns", "cloud"}
        if "render_mode" in updates and updates["render_mode"] not in allowed_modes:
            raise ValueError(f"Invalid render mode: {updates['render_mode']}")
        with get_db_connection(path) as conn:
            assignments = ", ".join(f"{field}=?" for field in updates)
            conn.execute(
                f"UPDATE shots SET {assignments} WHERE id=?",
                tuple(updates.values()) + (shot_id,),
            )
            conn.commit()

    def get_keyframes_for_shot(self, shot_id: int) -> List[Dict[str, Any]]:
        path = self.require_project()
        with get_db_connection(path) as conn:
            rows = conn.execute(
                "SELECT frame_number, prompt FROM keyframes WHERE shot_id=? ORDER BY frame_number",
                (shot_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def generate_keyframes_for_shot(self, shot_id: int) -> List[str]:
        path = self.require_project()
        with get_db_connection(path) as conn:
            script_row = conn.execute("SELECT value FROM project WHERE key='script'").fetchone()
            shot_row = conn.execute("SELECT description FROM shots WHERE id=?", (shot_id,)).fetchone()
            if not shot_row:
                raise ValueError("Shot not found")
            prompts = llm_plan_keyframes(shot_row["description"], context=script_row["value"] if script_row else "")
            conn.execute("DELETE FROM keyframes WHERE shot_id=?", (shot_id,))
            for idx, prompt in enumerate(prompts):
                conn.execute(
                    "INSERT INTO keyframes (shot_id, frame_number, prompt) VALUES (?, ?, ?)",
                    (shot_id, idx, prompt),
                )
            conn.execute(
                "UPDATE shots SET prompt=? WHERE id=?",
                (prompts[0] if prompts else shot_row["description"], shot_id),
            )
            conn.commit()
        return prompts

    def sync_script(self, script_text: str) -> None:
        path = self.require_project()
        scenes = re.split(r"\n##?\s*", script_text)
        with get_db_connection(path) as conn:
            conn.execute("DELETE FROM keyframes")
            conn.execute("DELETE FROM shots")
            conn.execute("DELETE FROM scenes")
            scene_counter = 1
            for scene_text in scenes:
                lines = [line.strip() for line in scene_text.strip().splitlines() if line.strip()]
                if not lines:
                    continue
                if not lines[0].lower().startswith("scene"):
                    continue
                scene_title = lines[0]
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO scenes (scene_number, description) VALUES (?, ?)",
                    (scene_counter, scene_title),
                )
                scene_id = cur.lastrowid
                for shot_idx, line in enumerate(lines[1:], start=1):
                    conn.execute(
                        "INSERT INTO shots (scene_id, shot_number, description, prompt) VALUES (?, ?, ?, ?)",
                        (scene_id, shot_idx, line, line),
                    )
                scene_counter += 1
            conn.execute("UPDATE project SET value=? WHERE key='script'", (script_text,))
            conn.commit()

    def generate_keyframes(self) -> None:
        path = self.require_project()
        with get_db_connection(path) as conn:
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

    def storyboard(self) -> List[Dict[str, Any]]:
        path = self.require_project()
        story: List[Dict[str, Any]] = []
        with get_db_connection(path) as conn:
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
            story.append(
                {
                    "scene": row["scene_number"],
                    "scene_description": row["scene_desc"],
                    "shot_id": row["shot_id"],
                    "shot_number": row["shot_number"],
                    "shot_description": row["shot_desc"],
                    "status": row["status"],
                    "output_path": row["output_path"],
                }
            )
        return story

    # ------------------------------------------------------------------
    # Collaboration & archiving
    # ------------------------------------------------------------------
    def save_project_as(self, destination: Path) -> Path:
        path = self.require_project()
        destination = destination.with_suffix(".zip")
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for file_path in path.rglob("*"):
                archive.write(file_path, file_path.relative_to(path))
        return destination

    def load_project_from_zip(self, archive: Path) -> None:
        project_name = archive.stem
        target_dir = PROJECTS_DIR / project_name
        if target_dir.exists():
            raise ValueError(f"Project '{project_name}' already exists")
        with zipfile.ZipFile(archive, "r") as zipf:
            zipf.extractall(target_dir)
        self.load_project(project_name)

    def share_project_stub(self) -> str:
        path = self.require_project()
        share_path = path / "share_link.txt"
        url = f"https://dropbox.example.com/{uuid.uuid4()}"
        share_path.write_text(url, encoding="utf-8")
        return url

    def fetch_shared_project_stub(self, url: str) -> Path:
        destination = BASE_DIR / "shared" / url.split("/")[-1]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("Shared project placeholder", encoding="utf-8")
        return destination


class AudioProcessor:
    """Generate and post-process audio assets."""

    def __init__(self, projects: ProjectManager) -> None:
        self.projects = projects

    def generate_voiceover(self, shot_id: int, text: str) -> Path:
        project_path = self.projects.require_project()
        api_key = get_api_key("elevenlabs")
        if not api_key:
            raise RuntimeError("ElevenLabs API key not configured")
        if "elevenlabs.client" not in sys.modules:
            raise ImportError("elevenlabs package required for voiceover generation")
        client = ElevenLabs(api_key=api_key)
        try:
            audio_bytes = client.generate(text=text, voice="Rachel", model="eleven_multilingual_v2")
        except Exception as exc:
            raise RuntimeError(f"ElevenLabs API call failed: {exc}") from exc
        assets = project_path / "assets"
        assets.mkdir(exist_ok=True)
        raw_audio = assets / f"shot_{shot_id}_voiceover.mp3"
        raw_audio.write_bytes(audio_bytes)
        processed = self._auto_edit_audio(raw_audio)
        with get_db_connection(project_path) as conn:
            conn.execute(
                "UPDATE shots SET voiceover_text=?, voiceover_file=? WHERE id=?",
                (text, str(processed.relative_to(project_path)), shot_id),
            )
            conn.commit()
        return processed

    def _auto_edit_audio(self, audio_path: Path) -> Path:
        sound = AudioSegment.from_file(audio_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=sound.dBFS - 18, keep_silence=200)
        processed = sum(chunks) if chunks else sound
        edited = audio_path.with_name(audio_path.stem + "_edited.mp3")
        processed.export(edited, format="mp3")
        return edited

    def generate_captions(self, shot_id: int) -> str:
        project_path = self.projects.require_project()
        with get_db_connection(project_path) as conn:
            row = conn.execute("SELECT voiceover_file FROM shots WHERE id=?", (shot_id,)).fetchone()
        if not row or not row[0]:
            raise RuntimeError("No voiceover available for captioning")
        audio_path = project_path / row[0]
        recognizer = sr.Recognizer()
        with sr.AudioFile(str(audio_path)) as source:
            audio = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio)
        except sr.UnknownValueError as exc:
            raise RuntimeError("Speech recognition could not understand audio") from exc
        except sr.RequestError as exc:
            raise RuntimeError(f"Speech recognition service failed: {exc}") from exc
        refined = llm_refine_captions(transcript)
        with get_db_connection(project_path) as conn:
            conn.execute("UPDATE shots SET subtitles=? WHERE id=?", (refined, shot_id))
            conn.commit()
        return refined

    def generate_music_stub(self, scene_id: int) -> Path:
        project_path = self.projects.require_project()
        assets = project_path / "assets"
        assets.mkdir(exist_ok=True)
        music_path = assets / f"scene_{scene_id}_music_stub.mp3"
        if not music_path.exists():
            music_path.write_bytes(b"Suno AI music placeholder")
        return music_path


class Renderer:
    """Rendering pipeline and export helpers."""

    def __init__(self, projects: ProjectManager) -> None:
        self.projects = projects

    def _render_ken_burns(self, shot_row: sqlite3.Row) -> ShotRenderResult:
        project_path = self.projects.require_project()
        image = Image.new("RGB", (shot_row["width"], shot_row["height"]), color=(10, 10, 10))
        still = project_path / "assets" / f"shot_{shot_row['id']}_kenburns.png"
        still.parent.mkdir(exist_ok=True)
        image.save(still)
        fps = max(int(shot_row["fps"] or 12), 1)
        duration = max(shot_row["duration_frames"] / fps if shot_row["duration_frames"] else 1, 1.0)
        clip = ImageClip(str(still)).set_duration(duration)
        clip = clip.fx(vfx.resize, width=shot_row["width"] * 1.1)
        clip = clip.fx(vfx.crop, width=shot_row["width"], height=shot_row["height"], x_center=shot_row["width"] / 2)
        output = project_path / "renders" / f"shot_{shot_row['id']}.mp4"
        clip.write_videofile(str(output), fps=fps, codec="libx264", audio=False, verbose=False, logger=None)
        return ShotRenderResult(shot_id=shot_row["id"], output_path=output)

    def _render_animatediff(self, shot_row: sqlite3.Row) -> ShotRenderResult:
        project_path = self.projects.require_project()
        if not comfy_running():
            start_comfy()
        workflow = {
            "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ANIMATEDIFF_CHECKPOINT}},
            "4": {"class_type": "AnimateDiffLoaderV1", "inputs": {"model_name": ANIMATEDIFF_MOTION_MODEL}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": shot_row["prompt"], "clip": ["3", 1]}},
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": shot_row["negative_prompt"] or "blurry, low quality, bad anatomy", "clip": ["3", 1]},
            },
            "10": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": int(np.random.randint(0, 2 ** 32 - 1)),
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
        outputs = comfy_queue(workflow)
        gifs = outputs.get("17", {}).get("gifs", [])
        if not gifs:
            raise RuntimeError("ComfyUI did not return a GIF output")
        comfy_output = COMFYUI_DIR / "output" / gifs[0]["filename"]
        output = project_path / "renders" / f"shot_{shot_row['id']}.mp4"
        fps = max(int(shot_row["fps"] or 12), 1)
        clip = VideoFileClip(str(comfy_output))
        clip.write_videofile(str(output), codec="libx264", audio=False, verbose=False, logger=None, fps=fps)
        return ShotRenderResult(shot_id=shot_row["id"], output_path=output, preview_path=comfy_output)

    def _render_cloud_stub(self, shot_row: sqlite3.Row) -> ShotRenderResult:
        project_path = self.projects.require_project()
        output = project_path / "renders" / f"shot_{shot_row['id']}_cloud.mp4"
        output.write_bytes(b"RunwayML cloud render placeholder")
        return ShotRenderResult(shot_id=shot_row["id"], output_path=output)

    def render_shot(self, shot_id: int) -> ShotRenderResult:
        project_path = self.projects.require_project()
        with get_db_connection(project_path) as conn:
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
        with get_db_connection(project_path) as conn:
            conn.execute(
                "UPDATE shots SET status='rendered', output_path=?, preview_path=? WHERE id=?",
                (
                    str(result.output_path.relative_to(project_path)),
                    str(result.preview_path.relative_to(project_path)) if result.preview_path else None,
                    shot_id,
                ),
            )
            conn.commit()
        return result

    def render_scene(self, scene_number: int) -> List[ShotRenderResult]:
        project_path = self.projects.require_project()
        with get_db_connection(project_path) as conn:
            rows = conn.execute(
                """
                SELECT shots.* FROM shots
                JOIN scenes ON scenes.id = shots.scene_id
                WHERE scenes.scene_number=?
                ORDER BY shots.shot_number
                """,
                (scene_number,),
            ).fetchall()
        return [self.render_shot(row["id"]) for row in rows]

    def render_project(self) -> List[ShotRenderResult]:
        project_path = self.projects.require_project()
        with get_db_connection(project_path) as conn:
            rows = conn.execute("SELECT id FROM shots ORDER BY scene_id, shot_number").fetchall()
        return [self.render_shot(row["id"]) for row in rows]

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
        project_path = self.projects.require_project()
        with get_db_connection(project_path) as conn:
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
            clip_path = project_path / row["output_path"]
            clip = VideoFileClip(str(clip_path))
            if row["voiceover_file"]:
                audio_path = project_path / row["voiceover_file"]
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
        output = OUTPUT_DIR / f"{self.projects.project_name}.{fmt}"
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
        project_path = self.projects.require_project()
        wkhtml = wkhtmltopdf_path()
        if not wkhtml:
            raise RuntimeError("wkhtmltopdf not available")
        html_file = project_path / "storyboard.html"
        story = self.projects.storyboard()
        html_content = ["<html><body><h1>Storyboard</h1>"]
        for entry in story:
            html_content.append(f"<h2>Scene {entry['scene']}</h2><p>{entry['scene_description']}</p>")
            html_content.append(f"<strong>Shot {entry['shot_number']}:</strong> {entry['shot_description']}<br/>")
        html_content.append("</body></html>")
        html_file.write_text("\n".join(html_content), encoding="utf-8")
        destination = destination.with_suffix(".pdf")
        safe_subprocess([str(wkhtml), str(html_file), str(destination)])
        return destination


class AIMovieMaker:
    """Facade that composes specialised managers for CLI/GUI use."""

    def __init__(self) -> None:
        self.workspace = WorkspaceManager()
        self.projects = ProjectManager(self.workspace)
        self.audio = AudioProcessor(self.projects)
        self.renderer = Renderer(self.projects)

    # Convenience properties -------------------------------------------------
    @property
    def project_name(self) -> Optional[str]:
        return self.projects.project_name

    @property
    def project_path(self) -> Optional[Path]:
        return self.projects.project_path

    # Workspace --------------------------------------------------------------
    def init_workspace(self) -> None:
        self.workspace.init_workspace()

    def check_dependencies(self) -> Dict[str, bool]:
        return self.workspace.check_dependencies()

    # Project wrappers -------------------------------------------------------
    def list_projects(self) -> List[str]:
        return self.projects.list_projects()

    def create_project(self, name: str) -> None:
        self.projects.create_project(name)

    def load_project(self, name: str) -> None:
        self.projects.load_project(name)

    def list_scenes(self) -> List[Dict[str, Any]]:
        return self.projects.list_scenes()

    def list_shots(self) -> List[Dict[str, Any]]:
        return self.projects.list_shots()

    def get_shot_details(self, shot_id: int) -> Dict[str, Any]:
        return self.projects.get_shot_details(shot_id)

    def update_shot_settings(self, shot_id: int, **kwargs: Any) -> None:
        self.projects.update_shot_settings(shot_id, **kwargs)

    def get_keyframes_for_shot(self, shot_id: int) -> List[Dict[str, Any]]:
        return self.projects.get_keyframes_for_shot(shot_id)

    def generate_keyframes_for_shot(self, shot_id: int) -> List[str]:
        return self.projects.generate_keyframes_for_shot(shot_id)

    def sync_script(self, script_text: str) -> None:
        self.projects.sync_script(script_text)

    def generate_keyframes(self) -> None:
        self.projects.generate_keyframes()

    def storyboard(self) -> List[Dict[str, Any]]:
        return self.projects.storyboard()

    def save_project_as(self, destination: Path) -> Path:
        return self.projects.save_project_as(destination)

    def load_project_from_zip(self, archive: Path) -> None:
        self.projects.load_project_from_zip(archive)

    def share_project_stub(self) -> str:
        return self.projects.share_project_stub()

    def fetch_shared_project_stub(self, url: str) -> Path:
        return self.projects.fetch_shared_project_stub(url)

    # Audio -----------------------------------------------------------------
    def generate_voiceover(self, shot_id: int, text: str) -> Path:
        return self.audio.generate_voiceover(shot_id, text)

    def generate_captions(self, shot_id: int) -> str:
        return self.audio.generate_captions(shot_id)

    def generate_music_stub(self, scene_id: int) -> Path:
        return self.audio.generate_music_stub(scene_id)

    # Rendering --------------------------------------------------------------
    def render_shot(self, shot_id: int) -> ShotRenderResult:
        return self.renderer.render_shot(shot_id)

    def render_scene(self, scene_number: int) -> List[ShotRenderResult]:
        return self.renderer.render_scene(scene_number)

    def render_project(self) -> List[ShotRenderResult]:
        return self.renderer.render_project()

    def export(self, fmt: str = "mp4") -> Path:
        return self.renderer.export(fmt)

    def export_pdf(self, destination: Path) -> Path:
        return self.renderer.export_pdf(destination)

