# AI Movie Maker

This repository contains the source code and installer assets for **AI Movie Maker v1.3.2**,
a cross-platform CLI + GUI application for AI-assisted storyboarding and video creation.

## Contents

- `ai_movie_maker.py` – standalone application script (CLI + Gradio GUI).
- `requirements.txt` – Python dependencies required by the app.
- `setup_models.py` – helper to download Stable Diffusion / AnimateDiff models.
- `installer.iss` – Windows installer definition (Inno Setup 6.2+).

## Build Steps (Windows Installer)

1. Install **Inno Setup 6.2 or later**.
2. Clone or download this repository to your workstation.
3. Verify the download URLs in `installer.iss` match the desired versions and update
   the SHA256 hashes if necessary.
4. Launch the Inno Setup Compiler, open `installer.iss`, and select *Build → Compile*.
5. The compiled `AI_Movie_Maker_Setup.exe` will be placed in the `dist` directory.

## Testing Notes

After installation on Windows 10+:

1. Open **Command Prompt** and run `ai_movie_maker.py init` to initialise the workspace.
2. Execute `ai_movie_maker.py create demo_project` then `ai_movie_maker.py gen-keyframes demo_project`.
3. Render a test shot with `ai_movie_maker.py run demo_project shot:1`.
4. Export a preview using `ai_movie_maker.py export demo_project mp4`.
5. Launch the GUI with `ai_movie_maker.py gui` to verify interactive features.