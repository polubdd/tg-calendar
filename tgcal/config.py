"""Настройки из переменных окружения (и из .env рядом с проектом, если он есть)."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Минимальный разбор .env: KEY=VALUE, комментарии через #. Уже заданные переменные не трогает."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    # общее
    tz: str = "Europe/Moscow"
    db_path: str = "data/calendar.db"
    host: str = "127.0.0.1"
    port: int = 8000
    ics_token: str = ""  # пусто — сгенерируется при первом запуске и сохранится в БД

    # локальная модель
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    num_ctx: int = 8192  # задаём явно: по умолчанию Ollama берёт 2048–4096 и молча обрезает начало промпта
    llm_timeout: float = 180.0
    llm_max_attempts: int = 3

    # бюджет контекста
    reserve_output_tokens: int = 1024
    max_window_tokens: int = 2500
    max_window_chars: int = 6000
    max_message_chars: int = 600
    window_overlap: int = 2
    summary_max_chars: int = 400
    max_messages_per_run: int = 300

    # Telegram
    telegram_bot_token: str = ""
    allowed_owner_ids: str = ""  # через запятую; без этого бот не обрабатывает переписку
    flush_idle_seconds: int = 90
    flush_max_messages: int = 40
    pending_ttl_hours: int = 48
    auto_confirm_threshold: float = 0.8
    notify_auto_added: bool = True
    tg_global_rate: float = 25.0  # сообщений в секунду на бота (лимит Telegram ~30)
    tg_per_chat_rate: float = 1.0  # в один чат — не чаще раза в секунду
    tg_concurrency: int = 4
    tg_max_attempts: int = 5

    @property
    def owner_ids(self) -> set[int]:
        return {int(x) for x in self.allowed_owner_ids.replace(" ", "").split(",") if x.strip().lstrip("-").isdigit()}

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        values = {}
        for f in fields(cls):
            # Не TZ: эту переменную читает C-рантайм (на Windows — в своём формате) и сбивает локальное время.
            raw = os.getenv("CALENDAR_TZ" if f.name == "tz" else f.name.upper())
            if raw is None or raw == "":
                continue
            if f.type in ("int", int):
                values[f.name] = int(raw)
            elif f.type in ("float", float):
                values[f.name] = float(raw)
            elif f.type in ("bool", bool):
                values[f.name] = raw.strip().lower() in ("1", "true", "yes", "on", "да")
            else:
                values[f.name] = raw
        values.update(overrides)
        return cls(**values)
