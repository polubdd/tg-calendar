"""Чат с календарём: фраза → событие (с правилом повторения).

Сначала пробуем правила timeparse: «каждый понедельник зарядка в 8» они
разбирают мгновенно и без модели. Если правила не справились («в последний
рабочий день месяца отчёт»), спрашиваем LLM со схемой, проверяем RRULE и дату
и при ошибке переспрашиваем. Результат — предложение, а не запись: в календарь
событие попадает только после «Добавить».
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

from pydantic import BaseModel, Field, field_validator

from .db import fmt_dt
from .extractor import _norm_date, _norm_time
from .llm import LLM, structured
from .timeparse import WD_SHORT, Scan, describe_rrule, find_date, first_occurrence, normalize_rrule, parse_phrase


class NotUnderstood(Exception):
    pass


class ChatEventLLM(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    when_text: str = ""
    date: str | None = None
    time: str | None = None
    end_time: str | None = None
    rrule: str | None = None
    location: str | None = None

    @field_validator("time", "end_time", mode="before")
    @classmethod
    def _time(cls, v):
        return _norm_time(v)

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v):
        return _norm_date(v)

    @field_validator("rrule", mode="before")
    @classmethod
    def _rrule(cls, v):
        if v in (None, "", "null"):
            return None
        return normalize_rrule(str(v))  # ValueError → модель переспросят с текстом ошибки


PROMPT = """Пользователь пишет в свой календарь. Преобразуй фразу в одно событие и верни только JSON.
Сейчас {now:%Y-%m-%d %H:%M}, {wd}, часовой пояс {tz}.
- title: короткое название по-русски, без слов о времени
- when_text: дословный кусок фразы, где названы дата или время
- date: YYYY-MM-DD первого раза (null, если это повтор и дата не важна)
- time, end_time: HH:MM (24 часа) или null для события на весь день
- rrule: правило повторения RFC 5545 без DTSTART, например FREQ=WEEKLY;BYDAY=MO
  или FREQ=MONTHLY;BYDAY=-1FR; null, если событие разовое
- location: место или null

Фраза: «{text}»"""


def _proposal(title, start, end, all_day, rrule, location, ambiguous, source) -> dict:
    if rrule:
        when = describe_rrule(rrule)
        when += "" if all_day else f" в {start:%H:%M}"
        when += f", с {start:%d.%m.%Y}"
    elif all_day:
        when = f"{start:%d.%m.%Y}, {WD_SHORT[start.weekday()]}, весь день"
    else:
        when = f"{start:%d.%m.%Y}, {WD_SHORT[start.weekday()]}, {start:%H:%M}–{end:%H:%M}"
    return {
        "source": source,
        "event": {"title": title, "start": fmt_dt(start), "end": fmt_dt(end), "all_day": all_day,
                  "rrule": rrule, "location": location, "status": "confirmed"},
        "description": when,
        "ambiguous": ambiguous,
    }


def interpret(text: str, now: datetime, llm: LLM | None = None, tz: str = "Europe/Moscow",
              max_attempts: int = 3) -> dict:
    text = " ".join(text.split())
    if not text:
        raise NotUnderstood("пустая фраза")
    p = parse_phrase(text, now)
    if p:
        return _proposal(p.title, p.start, p.end, p.all_day, p.rrule, None, p.ambiguous, "rules")
    if llm is None:
        raise NotUnderstood("Не нашёл дату или повтор. Напишите, например: «завтра в 15 созвон» "
                            "или «каждый понедельник зарядка в 8».")

    msgs = [{"role": "user", "content": PROMPT.format(now=now, wd=WD_SHORT[now.weekday()], tz=tz, text=text)}]
    r = structured(llm, msgs, ChatEventLLM, max_attempts)
    ambiguous: list[str] = []
    day: date | None = None
    dm = find_date(Scan(r.when_text), now.date()) if r.when_text else None
    if dm:
        day = dm.value  # относительные даты считаем сами, модель часто ошибается на день
    elif r.date:
        day = date.fromisoformat(r.date)
    if r.rrule and day is None:
        day = first_occurrence(r.rrule, now.date())
    if day is None:
        raise NotUnderstood("Не понял, на какую дату поставить событие.")
    if r.time:
        start = datetime.combine(day, time.fromisoformat(r.time))
        end = datetime.combine(day, time.fromisoformat(r.end_time)) if r.end_time else start + timedelta(hours=1)
        if end <= start:
            end += timedelta(days=1)
        return _proposal(r.title, start, end, False, r.rrule, r.location, ambiguous, "llm")
    return _proposal(r.title, day, None, True, r.rrule, r.location, ambiguous, "llm")
