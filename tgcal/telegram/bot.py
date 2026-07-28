"""Бот для Telegram Business: читает переписку владельца и ведёт календарь.

Что бот делает:
  * принимает business_connection — владелец подключил бота в
    «Настройки → Telegram для бизнеса → Чат-боты»;
  * складывает business_message в буфер и, когда переписка затихла,
    отдаёт чат в EventExtractor (локальная Ollama);
  * о найденных событиях пишет только владельцу в личный чат с ботом,
    неуверенные — с кнопками «Добавить / Не надо».

Чего бот не делает: не отвечает собеседникам и не пишет от имени владельца
(sendMessage с business_connection_id заблокирован в клиенте), не читает
чужие бизнес-аккаунты (только ALLOWED_OWNER_IDS).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from ..assistant import NotUnderstood, interpret
from ..config import Settings
from ..db import Connection, Repo, parse_dt
from ..extractor import EventExtractor, SaveResult
from ..llm import LLMError, LLMUnavailable
from ..timeparse import WD_SHORT, describe_rrule
from .client import Outbox, TelegramClient, TelegramError

log = logging.getLogger(__name__)

ALLOWED_UPDATES = ["message", "callback_query", "business_connection", "business_message",
                   "edited_business_message", "deleted_business_messages"]

HELP = (
    "Я веду ваш календарь по переписке в Telegram.\n\n"
    "Как подключить: Настройки → Telegram для бизнеса → Чат-боты → выберите меня. "
    "Права на ответы не нужны: я только читаю и никому не пишу от вашего имени.\n\n"
    "Команды:\n/drafts — неподтверждённые события\n/help — эта справка\n\n"
    "Можно написать мне и напрямую: «каждый понедельник зарядка в 8»."
)


def describe_event(e: dict) -> str:
    start = parse_dt(e["start"])
    if e.get("rrule"):
        when = describe_rrule(e["rrule"]) + ("" if e["all_day"] else f" в {start:%H:%M}")
    elif e["all_day"]:
        when = f"{WD_SHORT[start.weekday()]} {start:%d.%m}, весь день"
    else:
        when = f"{WD_SHORT[start.weekday()]} {start:%d.%m} в {start:%H:%M}"
    return f"«{e['title']}» — {when}"


def _keyboard(event_id: int, draft: bool) -> dict:
    if draft:
        row = [{"text": "✅ Добавить", "callback_data": f"ok:{event_id}"},
               {"text": "✖ Не надо", "callback_data": f"no:{event_id}"}]
    else:
        row = [{"text": "Отменить", "callback_data": f"no:{event_id}"}]
    return {"inline_keyboard": [row]}


class CalendarBot:
    def __init__(self, settings: Settings, repo: Repo, extractor: EventExtractor, client: TelegramClient,
                 outbox: Outbox, llm=None):
        self.s = settings
        self.repo = repo
        self.extractor = extractor
        self.client = client
        self.outbox = outbox
        self.llm = llm
        self.tz = ZoneInfo(settings.tz)
        self.llm_lock = asyncio.Lock()  # локальная модель одна: чаты обрабатываются по очереди
        self.stopping = False

    def is_owner(self, user_id: int) -> bool:
        return user_id in self.s.owner_ids

    # ---------- циклы ----------
    async def run(self) -> None:
        if not self.s.owner_ids:
            log.warning("ALLOWED_OWNER_IDS пуст: переписка обрабатываться не будет. "
                        "Напишите боту /start, он покажет ваш id.")
        me = await self.client.call("getMe")
        log.info("бот @%s запущен", me.get("username"))
        await self.outbox.start()
        try:
            await asyncio.gather(self.poll_loop(), self.flush_loop())
        finally:
            await self.outbox.stop(drain=False)

    async def poll_loop(self) -> None:
        offset = int(self.repo.get_meta("tg_offset") or 0)
        backoff = 1.0
        while not self.stopping:
            try:
                updates = await self.client.get_updates(offset, 30, ALLOWED_UPDATES)
            except TelegramError as e:
                log.warning("getUpdates: %s, пауза %.0f с", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            backoff = 1.0
            for u in updates:
                offset = u["update_id"] + 1
                try:
                    await self.handle(u)
                except Exception:
                    log.exception("ошибка обработки апдейта %s", u.get("update_id"))
                self.repo.set_meta("tg_offset", offset)

    async def flush_loop(self, interval: float = 10.0) -> None:
        while not self.stopping:
            await asyncio.sleep(interval)
            try:
                await self.flush()
            except LLMUnavailable as e:
                log.warning("%s — сообщения подождут в буфере", e)
                await asyncio.sleep(60)

    async def flush(self, now: float | None = None) -> list[SaveResult]:
        self.repo.prune(self.s.pending_ttl_hours, now=now)
        all_results: list[SaveResult] = []
        for conn_id, chat_id in self.repo.chats_ready(self.s.flush_idle_seconds, self.s.flush_max_messages, now):
            conn = self.repo.get_connection(conn_id)
            if not conn or not conn.enabled:
                self.repo.delete_pending(conn_id)
                continue
            async with self.llm_lock:
                results = await asyncio.to_thread(self.extractor.process_chat, conn_id, chat_id)
            self.notify(conn, results)
            all_results += results
        return all_results

    # ---------- апдейты ----------
    async def handle(self, u: dict) -> None:
        if "business_connection" in u:
            await self.on_connection(u["business_connection"])
        elif "business_message" in u:
            await self.on_business_message(u["business_message"])
        elif "edited_business_message" in u:
            await self.on_business_message(u["edited_business_message"], edited=True)
        elif "deleted_business_messages" in u:
            d = u["deleted_business_messages"]
            self.repo.delete_pending(d["business_connection_id"], d["chat"]["id"], d.get("message_ids", []))
        elif "callback_query" in u:
            await self.on_callback(u["callback_query"])
        elif "message" in u:
            await self.on_private(u["message"])

    async def on_connection(self, bc: dict) -> None:
        owner = bc["user"]["id"]
        allowed = self.is_owner(owner)
        enabled = bool(bc.get("is_enabled", True)) and allowed
        self.repo.upsert_connection(Connection(bc["id"], owner, bc["user_chat_id"], enabled))
        if not allowed:
            log.warning("бизнес-подключение от %s отклонено: его нет в ALLOWED_OWNER_IDS", owner)
            self.outbox.submit("sendMessage", chat_id=bc["user_chat_id"],
                               text=f"Этот бот настроен для другого владельца. Ваш id: {owner}.")
            return
        if enabled:
            self.outbox.submit("sendMessage", chat_id=bc["user_chat_id"], text=(
                "Подключено. Я читаю переписку только чтобы найти ваши планы и внести их в календарь. "
                "Обработка идёт на вашем компьютере, никому от вашего имени я не пишу."))
        else:
            self.repo.delete_pending(bc["id"])  # отключили — недообработанный текст не храним

    async def _connection(self, cid: str) -> Connection | None:
        conn = self.repo.get_connection(cid)
        if conn is None:
            try:
                await self.on_connection(await self.client.call("getBusinessConnection", business_connection_id=cid))
            except TelegramError as e:
                log.warning("не удалось получить подключение %s: %s", cid, e)
                return None
            conn = self.repo.get_connection(cid)
        return conn

    async def on_business_message(self, m: dict, edited: bool = False) -> None:
        conn = await self._connection(m["business_connection_id"])
        if not conn or not conn.enabled:
            return
        if m.get("sender_business_bot"):
            return  # сообщения, отправленные ботами от имени владельца, — не его планы
        text = m.get("text") or m.get("caption")
        if not text:
            return
        chat = m["chat"]
        sender = m.get("from", {})
        title = " ".join(x for x in (chat.get("first_name"), chat.get("last_name")) if x) or chat.get("title") \
            or chat.get("username") or str(chat["id"])
        self.repo.add_pending(conn.id, chat["id"], m["message_id"], m["date"], sender.get("id") == conn.user_id,
                              sender.get("first_name") or title, title, text, edited=edited)

    def notify(self, conn: Connection, results: list[SaveResult]) -> None:
        for r in results:
            if r.action != "created":
                continue
            draft = r.status == "draft"
            if not draft and not self.s.notify_auto_added:
                continue
            head = "Похоже, вы договорились:" if draft else "Добавил в календарь:"
            text = f"{head}\n{describe_event(r.event)}\n{r.event.get('notes') or ''}".strip()
            self.outbox.submit("sendMessage", chat_id=conn.user_chat_id, text=text,
                               reply_markup=_keyboard(r.event_id, draft))

    async def on_callback(self, cq: dict) -> None:
        if not self.is_owner(cq["from"]["id"]):
            self.outbox.submit("answerCallbackQuery", callback_query_id=cq["id"], text="Нет доступа")
            return
        action, _, raw_id = (cq.get("data") or "").partition(":")
        event = self.repo.get_event(int(raw_id)) if raw_id.isdigit() else None
        if not event or action not in ("ok", "no"):
            self.outbox.submit("answerCallbackQuery", callback_query_id=cq["id"], text="Событие не найдено")
            return
        status = "confirmed" if action == "ok" else "rejected"
        self.repo.update_event(event["id"], status=status)
        self.outbox.submit("answerCallbackQuery", callback_query_id=cq["id"],
                           text="Добавлено" if status == "confirmed" else "Убрал")
        msg = cq.get("message")
        if msg:
            mark = "✅ в календаре" if status == "confirmed" else "✖ не добавлено"
            self.outbox.submit("editMessageText", chat_id=msg["chat"]["id"], message_id=msg["message_id"],
                               text=f"{describe_event(event)}\n{mark}")

    async def on_private(self, m: dict) -> None:
        chat = m.get("chat", {})
        if chat.get("type") != "private":
            return  # в группах бот молчит
        user_id = m.get("from", {}).get("id")
        text = (m.get("text") or "").strip()
        if text.startswith("/start") or text.startswith("/help"):
            extra = "" if self.is_owner(user_id) else (
                f"\n\nВаш id: {user_id}. Добавьте его в ALLOWED_OWNER_IDS в .env и перезапустите бота.")
            self.outbox.submit("sendMessage", chat_id=chat["id"], text=HELP + extra)
            return
        if not self.is_owner(user_id):
            return
        if text.startswith("/drafts"):
            drafts = [e for e in self.repo.list_events(("draft",))][:10]
            if not drafts:
                self.outbox.submit("sendMessage", chat_id=chat["id"], text="Черновиков нет.")
            for e in drafts:
                self.outbox.submit("sendMessage", chat_id=chat["id"], text=describe_event(e),
                                   reply_markup=_keyboard(e["id"], True))
            return
        if text and not text.startswith("/"):
            now = datetime.now(self.tz).replace(tzinfo=None)
            try:
                async with self.llm_lock:
                    p = await asyncio.to_thread(interpret, text, now, self.llm, self.s.tz, self.s.llm_max_attempts)
            except (NotUnderstood, LLMError) as e:
                self.outbox.submit("sendMessage", chat_id=chat["id"], text=f"Не понял: {e}")
                return
            ev = p["event"]
            eid = self.repo.add_event(ev["title"], ev["start"], ev["end"], ev["all_day"], ev["rrule"],
                                      ev["location"], None, "draft", "chat")
            self.outbox.submit("sendMessage", chat_id=chat["id"], text=f"«{ev['title']}» — {p['description']}",
                               reply_markup=_keyboard(eid, True))


async def run_bot(settings: Settings, repo: Repo, llm) -> None:
    from .ratelimit import RateLimiter, RetryPolicy

    if not settings.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан — см. .env.example")
    limiter = RateLimiter(settings.tg_global_rate, settings.tg_per_chat_rate)
    client = TelegramClient(settings.telegram_bot_token, limiter, RetryPolicy(max_attempts=settings.tg_max_attempts))
    outbox = Outbox(client, settings.tg_concurrency)
    bot = CalendarBot(settings, repo, EventExtractor(llm, settings, repo), client, outbox, llm)
    try:
        await bot.run()
    finally:
        await client.close()

