"""Хранилище на SQLite.

Время событий хранится «настенным» (без смещения) вместе с часовым поясом
календаря: так «каждый понедельник в 8:00» остаётся в 8:00 и после перехода
на летнее время, а .ics получает TZID.

Текст переписки лежит в pending_messages только до обработки: после
извлечения событий он удаляется, остаются события и короткое резюме чата.
"""
from __future__ import annotations

import re
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    start_at TEXT NOT NULL,
    end_at TEXT,
    all_day INTEGER NOT NULL DEFAULT 0,
    rrule TEXT,
    location TEXT,
    notes TEXT,
    status TEXT NOT NULL DEFAULT 'confirmed',
    source TEXT NOT NULL DEFAULT 'manual',
    source_refs TEXT NOT NULL DEFAULT '',
    confidence REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_at);
CREATE TABLE IF NOT EXISTS business_connections(
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    user_chat_id INTEGER NOT NULL,
    enabled INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_messages(
    connection_id TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    date INTEGER NOT NULL,
    from_owner INTEGER NOT NULL,
    sender TEXT NOT NULL,
    chat_title TEXT NOT NULL,
    text TEXT NOT NULL,
    received_at REAL NOT NULL,
    PRIMARY KEY(connection_id, chat_id, message_id)
);
CREATE TABLE IF NOT EXISTS processed_messages(
    connection_id TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    processed_at REAL NOT NULL,
    PRIMARY KEY(connection_id, chat_id, message_id)
);
CREATE TABLE IF NOT EXISTS chat_state(
    connection_id TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    PRIMARY KEY(connection_id, chat_id)
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

EVENT_FIELDS = ("title", "start", "end", "all_day", "rrule", "location", "notes", "status", "source", "confidence")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def fmt_dt(v: datetime | date | None) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.replace(tzinfo=None).isoformat(timespec="minutes")
    return v.isoformat()


def parse_dt(s: str) -> datetime:
    """'2026-10-19' или '2026-10-19T08:00' → datetime (для дня — полночь)."""
    return datetime.fromisoformat(s) if "T" in s else datetime.combine(date.fromisoformat(s), datetime.min.time())


def _row_event(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"], "uid": r["uid"], "title": r["title"], "start": r["start_at"], "end": r["end_at"],
        "all_day": bool(r["all_day"]), "rrule": r["rrule"], "location": r["location"], "notes": r["notes"],
        "status": r["status"], "source": r["source"],
        "source_refs": [x for x in r["source_refs"].split(" ") if x], "confidence": r["confidence"],
        "created_at": r["created_at"], "updated_at": r["updated_at"],
    }


def title_tokens(title: str) -> set[str]:
    """Грубая основа слов: первые 5 букв. «Кино с Машей» и «кино» дают общий токен."""
    words = re.findall(r"\w+", title.lower().replace("ё", "е"))
    return {w[:5] for w in words if len(w) > 2 or w.isdigit()}


def similar_titles(a: str, b: str) -> bool:
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta or not tb:
        return a.strip().lower() == b.strip().lower()
    if ta <= tb or tb <= ta:
        return True
    return len(ta & tb) / len(ta | tb) >= 0.5


@dataclass
class Connection:
    id: str
    user_id: int
    user_chat_id: int
    enabled: bool


class Repo:
    def __init__(self, path: str | Path):
        on_disk = str(path) != ":memory:"
        if on_disk:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Бот обрабатывает чаты в отдельном потоке, поэтому соединение общее под блокировкой.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if on_disk:
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.RLock()

    def _q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self.lock:
            cur = self.conn.execute(sql, args)
            rows = cur.fetchall()
            self.conn.commit()
            return rows

    # ---- meta ----
    def get_meta(self, key: str) -> str | None:
        rows = self._q("SELECT value FROM meta WHERE key=?", (key,))
        return rows[0]["value"] if rows else None

    def set_meta(self, key: str, value) -> None:
        self._q("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))

    def ics_token(self, configured: str = "") -> str:
        if configured:
            return configured
        tok = self.get_meta("ics_token")
        if not tok:
            tok = secrets.token_urlsafe(16)
            self.set_meta("ics_token", tok)
        return tok

    # ---- события ----
    def add_event(self, title: str, start, end=None, all_day=False, rrule=None, location=None, notes=None,
                  status="confirmed", source="manual", source_refs: list[str] | None = None, confidence=None) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO events(uid,title,start_at,end_at,all_day,rrule,location,notes,status,source,source_refs,"
                "confidence,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"{uuid.uuid4()}@tg-calendar", title, fmt_dt(start) if not isinstance(start, str) else start,
                 fmt_dt(end) if not isinstance(end, str) else end, int(all_day), rrule, location, notes, status,
                 source, " ".join(source_refs or []), confidence, _now(), _now()),
            )
            self.conn.commit()
            return cur.lastrowid

    def get_event(self, event_id: int) -> dict | None:
        rows = self._q("SELECT * FROM events WHERE id=?", (event_id,))
        return _row_event(rows[0]) if rows else None

    def update_event(self, event_id: int, **changes) -> dict | None:
        cols = {"start": "start_at", "end": "end_at"}
        sets, args = [], []
        for k, v in changes.items():
            if k == "source_refs":
                v = " ".join(v)
            elif k in ("start", "end") and not isinstance(v, (str, type(None))):
                v = fmt_dt(v)
            elif k == "all_day":
                v = int(v)
            sets.append(f"{cols.get(k, k)}=?")
            args.append(v)
        if sets:
            self._q(f"UPDATE events SET {', '.join(sets)}, updated_at=? WHERE id=?", (*args, _now(), event_id))
        return self.get_event(event_id)

    def delete_event(self, event_id: int) -> bool:
        with self.lock:
            cur = self.conn.execute("DELETE FROM events WHERE id=?", (event_id,))
            self.conn.commit()
            return cur.rowcount > 0

    def list_events(self, statuses=("confirmed", "draft")) -> list[dict]:
        marks = ",".join("?" * len(statuses))
        return [_row_event(r) for r in self._q(f"SELECT * FROM events WHERE status IN ({marks}) ORDER BY start_at", statuses)]

    def find_duplicate(self, title: str, start: datetime | date, all_day: bool, minutes: int = 60) -> dict | None:
        """Похожее название и близкое время (±minutes) или тот же день, если одно из событий на весь день.

        Отклонённые тоже ищем: если владелец уже сказал «нет», тот же план из
        соседнего сообщения не должен всплыть снова.
        """
        day = start.date() if isinstance(start, datetime) else start
        lo, hi = (day - timedelta(days=1)).isoformat(), (day + timedelta(days=2)).isoformat()
        rows = self._q("SELECT * FROM events WHERE start_at >= ? AND start_at < ? ORDER BY id", (lo, hi))
        for r in rows:
            ev = _row_event(r)
            if not similar_titles(title, ev["title"]):
                continue
            other = parse_dt(ev["start"])
            if all_day or ev["all_day"]:
                if other.date() == day:
                    return ev
            elif abs((other - start).total_seconds()) <= minutes * 60:
                return ev
        return None

    # ---- бизнес-подключения ----
    def upsert_connection(self, c: Connection) -> None:
        self._q(
            "INSERT INTO business_connections(id,user_id,user_chat_id,enabled,updated_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET user_id=excluded.user_id, user_chat_id=excluded.user_chat_id, "
            "enabled=excluded.enabled, updated_at=excluded.updated_at",
            (c.id, c.user_id, c.user_chat_id, int(c.enabled), _now()),
        )

    def get_connection(self, cid: str) -> Connection | None:
        rows = self._q("SELECT * FROM business_connections WHERE id=?", (cid,))
        if not rows:
            return None
        r = rows[0]
        return Connection(r["id"], r["user_id"], r["user_chat_id"], bool(r["enabled"]))

    # ---- буфер сообщений ----
    def add_pending(self, connection_id: str, chat_id: int, message_id: int, date_ts: int, from_owner: bool,
                    sender: str, chat_title: str, text: str, edited: bool = False, now: float | None = None) -> bool:
        """False — сообщение уже обработано (повторная доставка апдейта)."""
        now = time.time() if now is None else now
        key = (connection_id, chat_id, message_id)
        with self.lock:
            done = self.conn.execute(
                "SELECT 1 FROM processed_messages WHERE connection_id=? AND chat_id=? AND message_id=?", key
            ).fetchone()
            if done and not edited:
                return False
            if done:
                self.conn.execute("DELETE FROM processed_messages WHERE connection_id=? AND chat_id=? AND message_id=?", key)
            self.conn.execute(
                "INSERT OR REPLACE INTO pending_messages VALUES(?,?,?,?,?,?,?,?,?)",
                (*key, date_ts, int(from_owner), sender, chat_title, text, now),
            )
            self.conn.commit()
            return True

    def pending_for_chat(self, connection_id: str, chat_id: int) -> list[sqlite3.Row]:
        return self._q(
            "SELECT * FROM pending_messages WHERE connection_id=? AND chat_id=? ORDER BY date, message_id",
            (connection_id, chat_id),
        )

    def chats_ready(self, idle_seconds: float, max_messages: int, now: float | None = None) -> list[tuple[str, int]]:
        """Чаты, где переписка затихла или накопилось много сообщений."""
        now = time.time() if now is None else now
        rows = self._q(
            "SELECT connection_id, chat_id, COUNT(*) AS n, MAX(received_at) AS last FROM pending_messages "
            "GROUP BY connection_id, chat_id HAVING last <= ? OR n >= ?",
            (now - idle_seconds, max_messages),
        )
        return [(r["connection_id"], r["chat_id"]) for r in rows]

    def mark_processed(self, connection_id: str, chat_id: int, message_ids: list[int]) -> None:
        with self.lock:
            for mid in message_ids:
                self.conn.execute("DELETE FROM pending_messages WHERE connection_id=? AND chat_id=? AND message_id=?",
                                  (connection_id, chat_id, mid))
                self.conn.execute("INSERT OR REPLACE INTO processed_messages VALUES(?,?,?,?)",
                                  (connection_id, chat_id, mid, time.time()))
            self.conn.commit()

    def delete_pending(self, connection_id: str, chat_id: int | None = None, message_ids: list[int] | None = None) -> None:
        with self.lock:
            if chat_id is None:
                self.conn.execute("DELETE FROM pending_messages WHERE connection_id=?", (connection_id,))
            else:
                for mid in message_ids or []:
                    self.conn.execute("DELETE FROM pending_messages WHERE connection_id=? AND chat_id=? AND message_id=?",
                                      (connection_id, chat_id, mid))
            self.conn.commit()

    def prune(self, pending_ttl_hours: float, processed_ttl_days: float = 30, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._q("DELETE FROM pending_messages WHERE received_at < ?", (now - pending_ttl_hours * 3600,))
        self._q("DELETE FROM processed_messages WHERE processed_at < ?", (now - processed_ttl_days * 86400,))

    # ---- резюме диалога ----
    def get_summary(self, connection_id: str, chat_id: int) -> str:
        rows = self._q("SELECT summary FROM chat_state WHERE connection_id=? AND chat_id=?", (connection_id, chat_id))
        return rows[0]["summary"] if rows else ""

    def set_summary(self, connection_id: str, chat_id: int, summary: str) -> None:
        self._q(
            "INSERT INTO chat_state VALUES(?,?,?,?) ON CONFLICT(connection_id, chat_id) DO UPDATE SET "
            "summary=excluded.summary, updated_at=excluded.updated_at",
            (connection_id, chat_id, summary, _now()),
        )
