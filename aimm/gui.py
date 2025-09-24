"""Gradio GUI for AI Movie Maker."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .config import APP_NAME, APP_VERSION
from .core import AIMovieMaker
from .db import get_db_connection
from .secrets import get_api_key, set_api_key


def launch_gui(app: Optional[AIMovieMaker] = None) -> None:
    """Launch the Gradio interface."""

    app = app or AIMovieMaker()

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
            raise gr.Error("Load or create a project first")
        app.sync_script(script_text)
        script, storyboard, shot_choices, default_shot, scene_choices, scene_default = _project_payload()
        return (
            script,
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
                shot_dropdown = gr.Dropdown(label="Shot", choices=[])
                shot_desc = gr.Textbox(label="Shot Description")
                shot_prompt = gr.Textbox(label="Prompt")
                shot_negative = gr.Textbox(label="Negative Prompt")
                shot_render_mode = gr.Radio(["animatediff", "kenburns", "cloud"], value="animatediff", label="Render Mode")
                shot_width = gr.Number(label="Width", value=512)
                shot_height = gr.Number(label="Height", value=512)
                shot_fps = gr.Number(label="FPS", value=12)
                shot_duration = gr.Number(label="Duration (frames)", value=48)
                shot_camera = gr.Textbox(label="Camera Movement")
                shot_transition = gr.Radio(["none", "fade", "crossfade"], value="none", label="Transition → next")
                save_shot_btn = gr.Button("Save Shot")
                keyframe_table = gr.DataFrame(headers=["Frame", "Prompt"], interactive=False)
                generate_shot_keyframes_btn = gr.Button("Generate Keyframes for Shot")

            with gr.TabItem("Audio & Captions"):
                voiceover_text = gr.Textbox(label="Voiceover Text", lines=3)
                generate_voiceover_btn = gr.Button("Generate Voiceover")
                generate_captions_btn = gr.Button("Generate Captions")
                audio_status = gr.Textbox(label="Status", interactive=False)
                subtitles_box = gr.Textbox(label="Subtitles", lines=4)
                music_scene_dropdown = gr.Dropdown(label="Scene", choices=[])
                generate_music_btn = gr.Button("Generate Music Stub")

            with gr.TabItem("Rendering"):
                render_shot_btn = gr.Button("Render Selected Shot")
                render_scene_number = gr.Number(label="Scene Number")
                render_scene_btn = gr.Button("Render Scene")
                render_project_btn = gr.Button("Render Project")
                render_status = gr.Textbox(label="Render Status", interactive=False)
                shot_preview = gr.Video(label="Shot Preview")

            with gr.TabItem("Export & Collaboration"):
                export_format = gr.Radio(["mp4", "gif", "avi"], value="mp4", label="Format")
                export_btn = gr.Button("Export")
                share_btn = gr.Button("Generate Share Link (stub)")
                share_output = gr.Textbox(label="Share Link", interactive=False)
                fetch_url = gr.Textbox(label="Fetch Shared Project (stub)")
                fetch_btn = gr.Button("Fetch")
                save_project_path = gr.Textbox(label="Save Project As (.zip)")
                save_project_btn = gr.Button("Save Project Archive")
                load_project_path = gr.Textbox(label="Load Project From (.zip)")
                load_project_btn = gr.Button("Load Project Archive")
                collab_status = gr.Textbox(label="Collaboration Status", interactive=False)

        create_btn.click(handle_create_project, inputs=new_project, outputs=[project_selector, script_editor, storyboard_table, shot_dropdown, music_scene_dropdown, status_bar])
        project_selector.change(handle_load_project, inputs=project_selector, outputs=[script_editor, storyboard_table, shot_dropdown, music_scene_dropdown, status_bar])
        save_script_btn.click(handle_save_script, inputs=script_editor, outputs=[script_editor, storyboard_table, shot_dropdown, music_scene_dropdown, status_bar])
        generate_all_btn.click(handle_generate_keyframes_all, inputs=shot_dropdown, outputs=[status_bar, keyframe_table])
        shot_dropdown.change(handle_shot_change, inputs=shot_dropdown, outputs=[shot_desc, shot_prompt, shot_negative, shot_render_mode, shot_width, shot_height, shot_fps, shot_duration, shot_camera, shot_transition, shot_preview, voiceover_text, subtitles_box, keyframe_table, music_scene_dropdown])
        save_shot_btn.click(handle_save_shot, inputs=[shot_dropdown, shot_desc, shot_prompt, shot_negative, shot_render_mode, shot_width, shot_height, shot_fps, shot_duration, shot_camera, shot_transition], outputs=[status_bar, storyboard_table, shot_desc, shot_prompt, shot_negative, shot_render_mode, shot_width, shot_height, shot_fps, shot_duration, shot_camera, shot_transition, shot_preview, voiceover_text, subtitles_box, keyframe_table, music_scene_dropdown])
        generate_shot_keyframes_btn.click(handle_generate_keyframes_shot, inputs=shot_dropdown, outputs=[status_bar, keyframe_table])
        generate_voiceover_btn.click(handle_voiceover, inputs=[shot_dropdown, voiceover_text], outputs=[audio_status, voiceover_text, subtitles_box])
        generate_captions_btn.click(handle_captions, inputs=shot_dropdown, outputs=[audio_status, subtitles_box])
        generate_music_btn.click(handle_music, inputs=music_scene_dropdown, outputs=[audio_status])
        render_shot_btn.click(handle_render_shot, inputs=shot_dropdown, outputs=[render_status, storyboard_table, shot_preview])
        render_scene_btn.click(handle_render_scene, inputs=render_scene_number, outputs=[render_status, storyboard_table])
        render_project_btn.click(handle_render_project, outputs=[render_status, storyboard_table])
        export_btn.click(handle_export, inputs=export_format, outputs=[render_status])
        check_btn.click(handle_check, outputs=[check_output])
        save_keys_btn.click(handle_save_keys, inputs=[openai_key, anthropic_key, eleven_key], outputs=[status_bar])
        share_btn.click(handle_share, outputs=[share_output, collab_status])
        fetch_btn.click(handle_fetch, inputs=fetch_url, outputs=[collab_status])
        save_project_btn.click(handle_save_project_zip, inputs=save_project_path, outputs=[collab_status])
        load_project_btn.click(handle_load_project_zip, inputs=load_project_path, outputs=[project_selector, script_editor, storyboard_table, shot_dropdown, music_scene_dropdown, collab_status])

    demo.launch()
