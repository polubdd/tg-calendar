"""Извлечение событий из переписки в Telegram.

Разделение труда:
  * LLM читает окно сообщений и решает, что здесь план владельца, а что нет
    («давай в пятницу в 7» → «ок» — план; «они в пятницу идут в кино» — нет),
    и возвращает дословную фразу о времени (when_text);
  * дату по этой фразе считает timeparse относительно даты того сообщения,
    где фраза встретилась, — не «сегодня»;
  * уверенные события сразу в календарь, неуверенные — черновиком на
    подтверждение в личке с ботом;
  * дубли (то же событие из соседнего окна или повторное «ок, в 7») склеиваются.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .context import ChatMessage, build_windows, clip, format_message, has_time_cue, window_budget
from .db import Repo
from .llm import LLM, LLMError, LLMUnavailable, structured
from .timeparse import WD_SHORT, Scan, find_date, find_recurrence, find_time, first_occurrence

log = logging.getLogger(__name__)


def _norm_time(v):
    if v in (None, "", "null"):
        return None
    m = re.fullmatch(r"\s*(\d{1,2})[:.](\d{2})(?::\d{2})?\s*", str(v))
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError("время нужно в формате HH:MM")
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def _norm_date(v):
    if v in (None, "", "null"):
        return None
    date.fromisoformat(str(v)[:10])  # кинет ValueError
    return str(v)[:10]


class ExtractedEvent(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    when_text: str = ""
    date: str | None = None
    time: str | None = None
    duration_min: int | None = Field(default=None, ge=5, le=24 * 60)
    location: str | None = None
    owner_committed: bool
    confidence: float = Field(ge=0, le=1)
    message_ids: list[int] = []

    @field_validator("time", mode="before")
    @classmethod
    def _time(cls, v):
        return _norm_time(v)

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v):
        return _norm_date(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _percent(cls, v):
        # «85» вместо 0.85 — частая ошибка небольших моделей, это не повод переспрашивать.
        if isinstance(v, (int, float)) and 1 < v <= 100:
            return v / 100
        return v


class ExtractionResult(BaseModel):
    events: list[ExtractedEvent] = []
    summary: str = ""

    @field_validator("summary")
    @classmethod
    def _clip(cls, v: str) -> str:
        return v.strip()[:400]


SYSTEM_PROMPT = """Ты извлекаешь из переписки в Telegram события для личного календаря владельца аккаунта.
Сообщения владельца подписаны «Я». Нужны только планы, которые владелец сам назначил или явно принял:
встречи, созвоны, дедлайны, «давай в пятницу в 7» → «ок». Не включай: чужие планы без владельца,
отклонённые и отменённые предложения, прошедшие события, расплывчатое «как-нибудь увидимся».

Для каждого события верни:
- title: коротко по-русски, что за событие («Кино с Машей», «Сдача курсовой»)
- when_text: дословный кусок сообщения, где названо время («в пятницу в 7»)
- date: YYYY-MM-DD, если можешь вычислить, иначе null
- time: HH:MM в 24-часовом формате или null для дедлайна без времени.
  «В 7» про кино, ужин, встречу после учёбы — 19:00; про пробежку или зарядку — 07:00
- duration_min и location — если известны, иначе null
- owner_committed: true, только если владелец предложил или согласился
- confidence: от 0 до 1 — насколько уверен, что событие состоится именно тогда
- message_ids: id сообщений, из которых взято событие

summary: 1–3 предложения о том, что обсуждается и что ещё не решено (до 300 символов).
Его покажут тебе со следующей порцией сообщений вместо старой переписки.
Если событий нет, верни пустой список events. Отвечай только JSON."""


@dataclass
class Candidate:
    title: str
    start: datetime | date
    end: datetime | date | None
    all_day: bool
    rrule: str | None
    status: str
    confidence: float
    location: str | None
    notes: str
    source_ref: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class SaveResult:
    event_id: int
    action: str  # created | merged | skipped
    status: str
    event: dict


class EventExtractor:
    def __init__(self, llm: LLM, settings: Settings, repo: Repo):
        self.llm = llm
        self.s = settings
        self.repo = repo
        self.tz = ZoneInfo(settings.tz)
        self.budget = window_budget(settings.num_ctx, settings.reserve_output_tokens, SYSTEM_PROMPT,
                                    settings.summary_max_chars, settings.max_window_tokens)

    # ---- промпт ----
    def _fmt(self, m: ChatMessage) -> str:
        return format_message(m, self.tz)

    def build_prompt(self, window: list[ChatMessage], summary: str, chat_title: str) -> list[dict]:
        last = window[-1].date.astimezone(self.tz)
        user = (
            f"Собеседник: {chat_title}\n"
            f"Часовой пояс: {self.s.tz}. Дата последнего сообщения: {last:%Y-%m-%d}, {WD_SHORT[last.weekday()]}.\n"
            f"Что было раньше: {summary or 'нет'}\n\n"
            "Сообщения:\n" + "\n".join(self._fmt(m) for m in window)
        )
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def extract_window(self, window: list[ChatMessage], summary: str, chat_title: str) -> ExtractionResult:
        return structured(self.llm, self.build_prompt(window, summary, chat_title), ExtractionResult,
                          self.s.llm_max_attempts)

    # ---- от ответа модели к событию ----
    def resolve(self, ev: ExtractedEvent, window: list[ChatMessage], ref_prefix: str) -> Candidate | None:
        by_id = {m.message_id: m for m in window}
        wt = " ".join(ev.when_text.lower().split())
        src = None
        if wt:
            ids_first = [by_id[i] for i in ev.message_ids if i in by_id] + window
            src = next((m for m in ids_first if wt in " ".join(m.text.lower().split())), None)
        if src is None:
            ids = [i for i in ev.message_ids if i in by_id]
            src = by_id[max(ids)] if ids else window[-1]
        sent = src.date.astimezone(self.tz)
        ref = sent.date()
        reasons: list[str] = []

        scan = Scan(ev.when_text)
        rule = find_recurrence(scan, ref)
        dm = find_date(scan, ref)
        tm = find_time(scan)
        llm_day = date.fromisoformat(ev.date) if ev.date else None

        if dm:
            day = dm.value
            if dm.ambiguous:
                reasons.append("дата понята неоднозначно")
            if llm_day and llm_day != day:
                log.info("дата модели %s расходится с разбором «%s» → %s, берём разбор", llm_day, ev.when_text, day)
        elif rule:
            day = first_occurrence(rule, ref)
        elif llm_day:
            day = llm_day
            reasons.append("дату вычислила модель")
        else:
            return None  # без даты в календарь класть нечего

        t: time | None = None
        if tm and not tm.ambiguous:
            t = tm.start  # явное «19:00» в тексте надёжнее интерпретации модели
        elif ev.time:
            t = time.fromisoformat(ev.time)  # модель видит контекст: «кино в 7» → 19:00
        elif tm:
            t = tm.start
            reasons.append("неясно, утро или вечер")

        if t is not None:
            start = datetime.combine(day, t)
            if tm and tm.end:
                end = datetime.combine(day, tm.end)
            else:
                end = start + timedelta(minutes=ev.duration_min or 60)
            if not rule and start.replace(tzinfo=self.tz) < sent - timedelta(hours=2):
                return None  # прошедшее: «вчера были в кино»
        else:
            start, end = day, None
            if not rule and day < ref:
                return None

        status = "confirmed" if ev.confidence >= self.s.auto_confirm_threshold and not reasons else "draft"
        if ev.confidence < self.s.auto_confirm_threshold:
            reasons.append(f"уверенность модели {ev.confidence:.2f}")
        return Candidate(
            title=ev.title.strip(), start=start, end=end, all_day=t is None, rrule=rule, status=status,
            confidence=ev.confidence, location=ev.location, notes=f"Из переписки: «{clip(ev.when_text, 80)}»",
            source_ref=f"{ref_prefix}:{src.message_id}", reasons=reasons,
        )

    def save(self, c: Candidate) -> SaveResult:
        dup = self.repo.find_duplicate(c.title, c.start, c.all_day)
        if dup:
            if dup["status"] == "rejected":
                return SaveResult(dup["id"], "skipped", "rejected", dup)
            changes: dict = {}
            if c.source_ref not in dup["source_refs"]:
                changes["source_refs"] = dup["source_refs"] + [c.source_ref]
            if dup["status"] == "draft" and c.status == "confirmed":
                changes["status"] = "confirmed"
            ev = self.repo.update_event(dup["id"], **changes) if changes else dup
            return SaveResult(dup["id"], "merged", ev["status"], ev)
        eid = self.repo.add_event(c.title, c.start, c.end, c.all_day, c.rrule, c.location, c.notes, c.status,
                                  "telegram", [c.source_ref], c.confidence)
        return SaveResult(eid, "created", c.status, self.repo.get_event(eid))

    # ---- обработка накопленного чата ----
    def process_chat(self, connection_id: str, chat_id: int) -> list[SaveResult]:
        rows = self.repo.pending_for_chat(connection_id, chat_id)
        if not rows:
            return []
        if len(rows) > self.s.max_messages_per_run:
            # Слишком длинный хвост: старое не обрабатываем, иначе бот часами занимает модель.
            dropped = rows[: len(rows) - self.s.max_messages_per_run]
            log.warning("чат %s: %s старых сообщений пропущено", chat_id, len(dropped))
            self.repo.mark_processed(connection_id, chat_id, [r["message_id"] for r in dropped])
            rows = rows[len(dropped):]

        chat_title = rows[-1]["chat_title"]
        msgs = [ChatMessage(r["message_id"], datetime.fromtimestamp(r["date"], timezone.utc), r["sender"],
                            bool(r["from_owner"]), clip(r["text"], self.s.max_message_chars)) for r in rows]
        windows = build_windows(msgs, self._fmt, self.s.max_window_chars, self.budget, self.s.window_overlap)
        summary = self.repo.get_summary(connection_id, chat_id)
        results: list[SaveResult] = []
        ref_prefix = f"tg:{connection_id}:{chat_id}"

        for k, w in enumerate(windows):
            last_window = k == len(windows) - 1
            # Последние overlap сообщений окна попадут в следующее — их пока не отмечаем.
            done_ids = [m.message_id for m in (w if last_window else w[: max(1, len(w) - self.s.window_overlap)])]
            if not any(has_time_cue(m.text) for m in w):
                self.repo.mark_processed(connection_id, chat_id, done_ids)
                continue
            try:
                res = self.extract_window(w, summary, chat_title)
            except LLMUnavailable:
                raise  # сообщения остаются в буфере, обработаем позже
            except LLMError as e:
                log.warning("чат %s: окно пропущено — %s", chat_id, e)
                self.repo.mark_processed(connection_id, chat_id, done_ids)
                continue
            summary = res.summary[: self.s.summary_max_chars] or summary
            for ev in res.events:
                if not ev.owner_committed:
                    continue
                cand = self.resolve(ev, w, ref_prefix)
                if cand:
                    results.append(self.save(cand))
            self.repo.set_summary(connection_id, chat_id, summary)
            self.repo.mark_processed(connection_id, chat_id, done_ids)
        return results
