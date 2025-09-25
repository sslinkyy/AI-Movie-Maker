"""Model bootstrap script for AI Movie Maker."""

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {
    "sd15": ("runwayml/stable-diffusion-v1-5", "v1-5-pruned-emaonly.ckpt"),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
    "animatediff": ("guoyww/animatediff-motion-adapter-v1-5", "mm_sd_v15_v2.ckpt"),
    "ipadapter": ("h94/IP-Adapter", "ip-adapter_sd15.bin"),
}


def download_model(target_dir: Path, repo: str, filename: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Downloading {filename} from {repo}")
    file_path = Path(
        hf_hub_download(repo_id=repo, filename=filename, local_dir=target_dir, local_dir_use_symlinks=False)
    )
    print(f"   -> {file_path}")
    return file_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download ComfyUI models")
    parser.add_argument("--model", choices=list(MODELS.keys()), nargs="*", help="Specific models to download")
    parser.add_argument("--target", type=Path, default=Path.home() / ".ai_movie_maker" / "models", help="Model directory")
    parser.add_argument("--auto", action="store_true", help="Download recommended defaults")
    args = parser.parse_args(argv)

    selection = set(args.model or [])
    if args.auto or not selection:
        selection.update({"sd15", "animatediff"})

    for key in selection:
        repo, filename = MODELS[key]
        download_model(args.target, repo, filename)


if __name__ == "__main__":
    main(sys.argv[1:])
