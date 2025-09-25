"""Database helpers for AI Movie Maker."""
from __future__ import annotations

import sqlite3
from pathlib import Path


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
