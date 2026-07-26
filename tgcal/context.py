"""Как не переполнить контекст модели.

Переписка бывает на тысячи сообщений, а у 7B-модели на ноутбуке разумный
num_ctx — 8k токенов. Поэтому:
  * каждое сообщение обрезается до max_message_chars (начало + конец);
  * сообщения режутся на окна с лимитом и по символам, и по оценке токенов;
  * соседние окна перекрываются на пару сообщений, чтобы «давай в 7?» → «ок»
    не разорвались;
  * вместо старой переписки в промпт идёт короткое резюме, которое модель
    сама обновляет в каждом ответе;
  * окна без единого намёка на время в LLM вообще не отправляются.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from .timeparse import MONTH_RE, WD_RE, WD_SHORT

# Токенизаторы Qwen/Llama тратят на кириллицу примерно 1 токен на 2.5–3 символа.
# Берём пессимистичную оценку: лучше недогрузить окно, чем потерять начало промпта.
CHARS_PER_TOKEN = 2.5


@dataclass
class ChatMessage:
    message_id: int
    date: datetime  # aware, UTC
    sender: str
    from_owner: bool
    text: str


def estimate_tokens(text: str | int) -> int:
    n = text if isinstance(text, int) else len(text)
    return math.ceil(n / CHARS_PER_TOKEN)


def clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head - 3
    return text[:head] + " … " + text[-tail:]


def format_message(m: ChatMessage, tz: ZoneInfo) -> str:
    local = m.date.astimezone(tz)
    who = "Я" if m.from_owner else m.sender
    return f"[id={m.message_id}] {local:%Y-%m-%d} {WD_SHORT[local.weekday()]} {local:%H:%M} {who}: {m.text}"


def build_windows(msgs: list[ChatMessage], fmt: Callable[[ChatMessage], str], max_chars: int,
                  max_tokens: int, overlap: int = 2) -> list[list[ChatMessage]]:
    windows: list[list[ChatMessage]] = []
    i, n = 0, len(msgs)
    while i < n:
        cur: list[ChatMessage] = []
        chars = 0
        j = i
        while j < n:
            c = len(fmt(msgs[j])) + 1
            if cur and (chars + c > max_chars or estimate_tokens(chars + c) > max_tokens):
                break
            cur.append(msgs[j])
            chars += c
            j += 1
        windows.append(cur)
        if j >= n:
            break
        i = max(j - overlap, i + 1)  # перекрытие, но всегда с продвижением вперёд
    return windows


def window_budget(num_ctx: int, reserve_output: int, system_prompt: str, summary_max_chars: int,
                  max_window_tokens: int) -> int:
    """Сколько токенов можно отдать под сообщения, чтобы промпт + ответ влезли в num_ctx."""
    overhead = estimate_tokens(system_prompt) + estimate_tokens(summary_max_chars) + 150
    available = num_ctx - reserve_output - overhead
    if available < 300:
        raise ValueError(f"num_ctx={num_ctx} слишком мал: под переписку остаётся {available} токенов")
    return min(max_window_tokens, available)


_CUE = re.compile(
    r"\b(?:сегодня|завтра|послезавтра|через|утром|вечером|днем|ночью|выходн|недел|числ|дедлайн|срок|сдать|сдач|встре|"
    r"созвон|собес|экзамен|зач[её]т|пара|кино|ужин|обед|завтрак|приду|увидимся|давай|напомн|кажд|будн|в\s+\d|к\s+\d)"
    r"|\d{1,2}[:.]\d{2}|" + WD_RE + "|" + MONTH_RE,
    re.I,
)


def has_time_cue(text: str) -> bool:
    return bool(_CUE.search(text.lower().replace("ё", "е")))
