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
        launch_gui(app)
    else:
        parser.print_help()
