# -*- coding: utf-8 -*-
"""SQLite question log for admin review."""
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from .config import CFG
except ImportError:  # 兼容在 backend 目录内直接运行脚本
    from config import CFG

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


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
            answer TEXT,
            ip TEXT,
            user_agent TEXT,
            created_at TEXT NOT NULL,
            answered_at TEXT
        )
        """
    )
    conn.commit()
    _ensure_columns(conn)
    _normalize_stored_times(conn)
    return conn


def _ensure_columns(conn: sqlite3.Connection) -> None:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(questions)").fetchall()}
    if "answer" not in cols:
        conn.execute("ALTER TABLE questions ADD COLUMN answer TEXT")
    if "answered_at" not in cols:
        conn.execute("ALTER TABLE questions ADD COLUMN answered_at TEXT")
    conn.commit()


def _beijing_now() -> str:
    return datetime.now(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _format_beijing(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    try:
        if "T" in text:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=SHANGHAI_TZ)
            return dt.astimezone(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S")
        dt = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return text[:19]


def _normalize_stored_times(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, created_at, answered_at FROM questions").fetchall()
    changed = False
    for row in rows:
        normalized = _format_beijing(row["created_at"])
        if normalized and normalized != row["created_at"]:
            conn.execute("UPDATE questions SET created_at = ? WHERE id = ?", (normalized, row["id"]))
            changed = True
        answered_at = row["answered_at"]
        if answered_at:
            normalized_answered = _format_beijing(answered_at)
            if normalized_answered and normalized_answered != answered_at:
                conn.execute("UPDATE questions SET answered_at = ? WHERE id = ?", (normalized_answered, row["id"]))
                changed = True
    if changed:
        conn.commit()


def save_question(question: str, ip: str | None, user_agent: str | None) -> int | None:
    text = question.strip()
    if not text:
        return None
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO questions(question, ip, user_agent, created_at) VALUES (?, ?, ?, ?)",
            (text, ip or "", user_agent or "", _beijing_now()),
        )
        conn.commit()
        return int(cur.lastrowid)


def save_answer(question_id: int | None, answer: str) -> None:
    if not question_id:
        return
    text = answer.strip()
    with _connect() as conn:
        conn.execute(
            "UPDATE questions SET answer = ?, answered_at = ? WHERE id = ?",
            (text, _beijing_now(), question_id),
        )
        conn.commit()


def list_questions(limit: int = 300) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, question, answer, ip, user_agent, created_at, answered_at FROM questions ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            **dict(row),
            "created_at": _format_beijing(row["created_at"]),
            "answered_at": _format_beijing(row["answered_at"] or ""),
        }
        for row in rows
    ]


def count_questions() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()
    return int(row["n"])
