"""Command-line entry points for AI Movie Maker."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional

from .core import AIMovieMaker
from .gui import launch_gui


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Movie Maker CLI")
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

    keyframe_parser = sub.add_parser("gen-keyframes", help="Generate keyframes for project")
    keyframe_parser.add_argument("name")

    run_parser = sub.add_parser("run", help="Render project/scene/shot")
    run_parser.add_argument("name")
    run_parser.add_argument("target")

    export_parser = sub.add_parser("export", help="Export project video")
    export_parser.add_argument("name")
    export_parser.add_argument("format", nargs="?", default="mp4")

    export_pdf_parser = sub.add_parser("export-pdf", help="Export storyboard PDF")
    export_pdf_parser.add_argument("name")
    export_pdf_parser.add_argument("destination")

    save_parser = sub.add_parser("save-project", help="Archive project")
    save_parser.add_argument("name")
    save_parser.add_argument("destination")

    load_zip_parser = sub.add_parser("load-project", help="Load project from archive")
    load_zip_parser.add_argument("archive")

    share_parser = sub.add_parser("share", help="Create share link (stub)")
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

    sub.add_parser("gui", help="Launch GUI")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    app = AIMovieMaker()

    def with_project(handler):
        def wrapper(ns):
            app.load_project(ns.name)
            return handler(ns)

        return wrapper

    def handle_sync_script(ns):
        script_text = Path(ns.script_file).read_text(encoding="utf-8")
        app.sync_script(script_text)

def handle_run(ns):
    target = ns.target
    if target == "project":
        app.render_project()
    elif target.startswith("scene:"):
        try:
            scene_number = int(target.split(":", 1)[1])
            app.render_scene(scene_number)
        except (ValueError, IndexError):
            raise ValueError("Invalid scene format. Use scene:<number>.")
    elif target.startswith("shot:"):
        try:
            shot_id = int(target.split(":", 1)[1])
            app.render_shot(shot_id)
        except (ValueError, IndexError):
            raise ValueError("Invalid shot format. Use shot:<id>.")
    else:
        raise ValueError("Unknown target. Use project, scene:<n>, or shot:<id>.")

    def handle_export(ns):
        app.export(ns.format)

    def handle_export_pdf(ns):
        app.export_pdf(Path(ns.destination))

    def handle_save_project(ns):
        app.save_project_as(Path(ns.destination))

    def handle_share(ns):
        url = app.share_project_stub()
        print(f"Share link: {url}")

    def handle_storyboard(ns):
        for entry in app.storyboard():
            print(
                f"Scene {entry['scene']}: {entry['scene_description']} | "
                f"Shot {entry['shot_number']} ({entry['status']}): {entry['shot_description']}"
            )

    def handle_edit_shot(ns):
        field = ns.field.replace("-", "_")
        value: Any = ns.value
        numeric = {"width", "height", "fps", "duration_frames"}
        if field in numeric:
            value = int(value)
        if field == "transition_to_next" and value == "none":
            value = None
        app.update_shot_settings(ns.shot_id, **{field: value})
        print(f"Updated shot {ns.shot_id}: {field}")

    def handle_voiceover(ns):
        text = " ".join(ns.text)
        path = app.generate_voiceover(ns.shot_id, text)
        print(f"Voiceover saved to {path}")

    def handle_captions(ns):
        print(app.generate_captions(ns.shot_id))

    def handle_music_stub(ns):
        path = app.generate_music_stub(ns.scene)
        print(f"Music stub saved to {path}")

    handlers = {
        "init": lambda ns: app.init_workspace(),
        "check": lambda ns: app.check_dependencies(),
        "list": lambda ns: app.list_projects(),
        "create": lambda ns: app.create_project(ns.name),
        "load": lambda ns: app.load_project(ns.name),
        "sync-script": with_project(handle_sync_script),
        "gen-keyframes": with_project(lambda ns: app.generate_keyframes()),
        "run": with_project(handle_run),
        "export": with_project(handle_export),
        "export-pdf": with_project(handle_export_pdf),
        "save-project": with_project(handle_save_project),
        "load-project": lambda ns: app.load_project_from_zip(Path(ns.archive)),
        "share": with_project(handle_share),
        "fetch-share": lambda ns: print(
            f"Fetched shared project placeholder at {app.fetch_shared_project_stub(ns.url)}"
        ),
        "storyboard": with_project(handle_storyboard),
        "edit-shot": with_project(handle_edit_shot),
        "voiceover": with_project(handle_voiceover),
        "captions": with_project(handle_captions),
        "music-stub": with_project(handle_music_stub),
        "gui": lambda ns: launch_gui(app),
    }

    command = args.command or "gui"
    handler = handlers.get(command)
    if not handler:
        parser.print_help()
        return
    handler(args)
