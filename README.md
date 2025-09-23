# AI Movie Maker

AI Movie Maker is a single-file Python application (`ai_movie_maker.py`) that combines a command line interface and a Gradio GUI to orchestrate AI-assisted storyboarding, animation, and video production workflows. The tool coordinates Stable Diffusion / AnimateDiff rendering through ComfyUI, automates voiceovers and captions, keeps project state in SQLite, and exports final edits with moviepy-powered transitions. This repository also contains a Windows installer definition and helper scripts for dependency bootstrapping.

## Key Capabilities

* **Hybrid CLI + GUI** – Drive the entire workflow from scripted commands or through a multi-tab Gradio interface that covers setup, scripting, keyframe planning, shot controls, audio tools, rendering, review, and collaboration.
* **Project management** – Each project lives inside `~/.ai_movie_maker/projects/<name>` with scenes, shots, keyframes, assets, and renders tracked in `project.db` (SQLite). CLI verbs and GUI actions both keep the database in sync.
* **Rendering pipelines** – Support for AnimateDiff (via ComfyUI queue API), Ken Burns-style moves on stills, and a RunwayML-style cloud stub. Shot-level width, height, frame count, FPS, prompts, and transitions are editable.
* **Audio automation** – ElevenLabs-powered voiceovers with automatic silence trimming (pydub), speech_recognition transcription with LLM caption polishing, and a Suno music generation stub.
* **Post-processing helpers** – Optional RIFE frame interpolation, Real-ESRGAN upscaling hooks, transition-aware export (fade / crossfade) to MP4, GIF, or AVI, and ffmpeg utilities for additional processing.
* **Collaboration stubs** – Quick share-link and project ZIP helpers to coordinate work through file-sharing services.

## Repository Layout

| Path | Description |
| --- | --- |
| `ai_movie_maker.py` | Primary application script (CLI + GUI). |
| `requirements.txt` | Python packages required by the app and installer. |
| `setup_models.py` | Hugging Face download helper invoked during workspace setup. |
| `installer.iss` | Inno Setup script that produces `AI_Movie_Maker_Setup.exe` on Windows. |

## Installation

### Windows (recommended)
1. Install [Inno Setup 6.2+](https://jrsoftware.org/isdl.php).
2. Clone this repository and review the binary download URLs and SHA256 hashes defined near the top of `installer.iss`.
3. Compile the installer via **Build → Compile** inside the Inno Setup Compiler. The resulting `AI_Movie_Maker_Setup.exe` appears in `Output`.
4. Run the installer. It downloads FFmpeg, RIFE, Real-ESRGAN, wkhtmltopdf, ComfyUI Portable, installs Python packages into ComfyUI’s embedded interpreter, and creates Start Menu / Desktop shortcuts alongside a `run.bat` launcher.
5. Re-run the installer with `/SILENT` for unattended deployments. Optional components (interpolation, upscaling, wkhtmltopdf, model sets) can be toggled through installer checkboxes or preseeded via silent-mode parameters.

### Linux / macOS
1. Ensure `python3`, `pip`, `git`, `curl`, and `unzip` are available.
2. Create and activate a virtual environment, then install dependencies: `pip install -r requirements.txt`.
3. Place ComfyUI inside `~/.ai_movie_maker/bin/ComfyUI` (or adjust the path in the script) and install its requirements.
4. Run `python setup_models.py` to fetch default Stable Diffusion / AnimateDiff checkpoints via `huggingface_hub`.
5. Launch the app with `python ai_movie_maker.py init` followed by `python ai_movie_maker.py gui`.

## Quick Start

1. **Initialise workspace**: `python ai_movie_maker.py init`
2. **Create a project**: `python ai_movie_maker.py create my_movie`
3. **Generate keyframes**: `python ai_movie_maker.py gen-keyframes my_movie`
4. **Render a shot**: `python ai_movie_maker.py run my_movie shot:1`
5. **Export final video**: `python ai_movie_maker.py export my_movie mp4`
6. **Launch GUI**: `python ai_movie_maker.py gui`

All commands use secure key storage via `keyring`. Set API keys inside the GUI Settings tab or through your OS keychain utility before invoking LLM or ElevenLabs features.

## Gradio GUI Overview

The GUI organises tasks into dedicated tabs:

* **Checklist** – Run dependency checks (FFmpeg, RIFE, Real-ESRGAN, ComfyUI).
* **Settings** – Configure API keys for OpenAI, Anthropic, and ElevenLabs.
* **Script / Storyboard** – Edit the script, sync scenes/shots, and manage prompts.
* **Shot Editor** – Update shot metadata, camera settings, transitions, and assets.
* **Audio / Captions / Music** – Trigger voiceover generation, auto-captioning, and music stubs.
* **Render** – Submit renders (AnimateDiff, Ken Burns, cloud stub) and preview outputs.
* **Review & Export** – Combine shots with transitions and export to MP4/GIF/AVI.
* **Collaboration** – Save or load project ZIPs and retrieve share links.
* **Help** – Quick usage reminders and support resources.

## CLI Reference

| Command | Purpose |
| --- | --- |
| `init` | Create workspace directories and sentinel file. |
| `check` | Print dependency status for FFmpeg / RIFE / Real-ESRGAN / ComfyUI. |
| `list` | Display all projects found in the workspace. |
| `create <name>` | Scaffold a new project with default script content. |
| `load <name>` | Load an existing project (required before most verbs). |
| `sync-script <name> <path>` | Replace the stored script with the contents of `<path>`. |
| `gen-keyframes <name>` | LLM-driven keyframe prompts for every shot. |
| `run <name> project|scene:<n>|shot:<id>` | Render the full project, a scene, or a single shot. |
| `export <name> <mp4|gif|avi>` | Export rendered shots to the chosen format. |
| `export-pdf <name> <path>` | Produce a PDF storyboard via wkhtmltopdf (if installed). |
| `save-project <name> <zip>` | Zip the project folder for sharing/backups. |
| `load-project <zip>` | Restore a project from a ZIP archive. |
| `storyboard <name>` | Print a text storyboard summarising scenes and shots. |
| `edit-shot <name> <shot_id> <field> <value>` | Update shot metadata (width, height, fps, etc.). |
| `voiceover <name> <shot_id> <text...>` | Generate an ElevenLabs voiceover for a shot. |
| `captions <name> <shot_id>` | Transcribe + refine captions for the associated voiceover. |
| `music-stub <name> <scene>` | Produce a Suno placeholder music clip for a scene. |
| `share <name>` | Return a collaboration stub link. |
| `fetch-share <url>` | Demonstrate retrieving a shared project link. |
| `gui` | Launch the Gradio interface. |

Additional helpers include `run my_movie shot:3` (shot aliases), `run my_movie scene:1` (batch render a scene), audio utilities, music stubs, and share-link retrieval. Use `python ai_movie_maker.py --help` to view the latest command list.

## Frequently Asked Questions

**Do I need a GPU?**  
No. AnimateDiff runs faster on NVIDIA GPUs, but the script falls back to CPU-friendly modes (Ken Burns, cloud stub) if GPU-only dependencies are unavailable.

**What happens if LLM APIs are unreachable?**  
Keyframe planning and caption refinement gracefully fall back to the shot description so you can continue working offline.

**Where are projects stored?**  
Projects live under `~/.ai_movie_maker/projects/`. Each project folder contains `assets/`, `renders/`, and a `project.db` file.

**Can I use custom models?**  
Yes. Place additional checkpoints under `~/.ai_movie_maker/models/` and update ComfyUI workflows or CLI parameters as needed. The installer exposes a custom model path option for Windows builds.

**How do I update binaries?**  
Re-run the Windows installer or re-fetch the binaries manually and replace the contents of `~/.ai_movie_maker/bin/`. The `check` command reports missing components.

**Is there a way to collaborate?**  
Use the Collaboration tab or the `save-project` / `load-project` commands to exchange ZIP archives via Dropbox, Google Drive, or similar services. A stub share-link helper is included for quick handoffs.

**Where are logs stored?**  
CLI output prints to stdout/stderr. ComfyUI logs live under `~/.ai_movie_maker/bin/ComfyUI/logs/`, and FFmpeg errors are surfaced through gradio alerts or CLI messages.

## Support & Contributions

Please open issues or pull requests if you encounter bugs, need installer tweaks, or want to contribute enhancements. Include detailed logs and reproduction steps to help triage quickly.

