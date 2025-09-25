"""AI Movie Maker modular package."""
from __future__ import annotations

from .core import AIMovieMaker
from .cli import build_cli_parser, main as cli_main
from .gui import launch_gui

__all__ = ["AIMovieMaker", "build_cli_parser", "cli_main", "launch_gui"]
