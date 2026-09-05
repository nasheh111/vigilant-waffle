# -*- coding: utf-8 -*-
"""Question log storage for admin review.

Uses DATABASE_URL/Postgres when configured, otherwise falls back to local SQLite.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

try:
    from .config import CFG
except ImportError:  # 兼容在 backend 目录内直接运行脚本
    from config import CFG

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def _use_postgres() -> bool:
    return bool(getattr(CFG, "database_url", ""))


def store_backend() -> str:
    return "postgres" if _use_postgres() else "sqlite"


def _beijing_now() -> str:
    return datetime.now(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _format_beijing(value: str | None) -> str:
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


def _normalize_row(row: dict) -> dict:
    return {
        "id": row.get("id"),
        "question": row.get("question") or "",
        "answer": row.get("answer") or "",
        "ip": row.get("ip") or "",
        "user_agent": row.get("user_agent") or "",
        "created_at": _format_beijing(row.get("created_at")),
        "answered_at": _format_beijing(row.get("answered_at")),
    }


def _sqlite_connect() -> sqlite3.Connection:
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
    _sqlite_ensure_columns(conn)
    _sqlite_normalize_stored_times(conn)
    return conn


def _sqlite_ensure_columns(conn: sqlite3.Connection) -> None:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(questions)").fetchall()}
    if "answer" not in cols:
        conn.execute("ALTER TABLE questions ADD COLUMN answer TEXT")
    if "answered_at" not in cols:
        conn.execute("ALTER TABLE questions ADD COLUMN answered_at TEXT")
    conn.commit()


def _sqlite_normalize_stored_times(conn: sqlite3.Connection) -> None:
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


@contextmanager
def _postgres_connect() -> Iterator[object]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("DATABASE_URL 已配置，但缺少 psycopg 依赖，请确认 requirements.txt 已安装。") from exc

    with psycopg.connect(CFG.database_url, row_factory=dict_row) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id BIGSERIAL PRIMARY KEY,
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
        yield conn


def save_question(question: str, ip: str | None, user_agent: str | None) -> int | None:
    text = question.strip()
    if not text:
        return None
    if _use_postgres():
        with _postgres_connect() as conn:
            row = conn.execute(
                """
                INSERT INTO questions(question, ip, user_agent, created_at)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (text, ip or "", user_agent or "", _beijing_now()),
            ).fetchone()
            conn.commit()
            return int(row["id"]) if row else None

    with _sqlite_connect() as conn:
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
    if _use_postgres():
        with _postgres_connect() as conn:
            conn.execute(
                "UPDATE questions SET answer = %s, answered_at = %s WHERE id = %s",
                (text, _beijing_now(), question_id),
            )
            conn.commit()
        return

    with _sqlite_connect() as conn:
        conn.execute(
            "UPDATE questions SET answer = ?, answered_at = ? WHERE id = ?",
            (text, _beijing_now(), question_id),
        )
        conn.commit()


def list_questions(limit: int = 300) -> list[dict]:
    if _use_postgres():
        with _postgres_connect() as conn:
            rows = conn.execute(
                """
                SELECT id, question, answer, ip, user_agent, created_at, answered_at
                FROM questions
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [_normalize_row(dict(row)) for row in rows]

    with _sqlite_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, question, answer, ip, user_agent, created_at, answered_at
            FROM questions
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_normalize_row(dict(row)) for row in rows]


def count_questions() -> int:
    if _use_postgres():
        with _postgres_connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()
        return int(row["n"])

    with _sqlite_connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()
    return int(row["n"])
