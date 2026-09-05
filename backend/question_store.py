# -*- coding: utf-8 -*-
"""SQLite question log for admin review."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

try:
    from .config import CFG
except ImportError:  # 兼容在 backend 目录内直接运行脚本
    from config import CFG


def _connect() -> sqlite3.Connection:
    path = Path(CFG.question_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            ip TEXT,
            user_agent TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_question(question: str, ip: str | None, user_agent: str | None) -> None:
    text = question.strip()
    if not text:
        return
    with _connect() as conn:
        conn.execute(
            "INSERT INTO questions(question, ip, user_agent, created_at) VALUES (?, ?, ?, ?)",
            (text, ip or "", user_agent or "", datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def list_questions(limit: int = 300) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, question, ip, user_agent, created_at FROM questions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def count_questions() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()
    return int(row["n"])
