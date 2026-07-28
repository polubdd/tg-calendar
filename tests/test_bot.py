import asyncio
import time

from tgcal.config import Settings
from tgcal.db import Repo
from tgcal.extractor import EventExtractor
from tgcal.llm import FakeLLM
from tgcal.telegram.bot import CalendarBot

OWNER, OWNER_CHAT, FRIEND = 1001, 1001, 2002


class FakeOutbox:
    def __init__(self):
        self.sent = []

    def submit(self, method, **params):
        self.sent.append((method, params))


class FakeClient:
    async def call(self, method, **params):
        if method == "getBusinessConnection":
            return {"id": params["business_connection_id"], "user": {"id": OWNER}, "user_chat_id": OWNER_CHAT,
                    "is_enabled": True}
        raise AssertionError(method)


def make_bot(responses, owners=str(OWNER)):
    s = Settings(allowed_owner_ids=owners)
    repo = Repo(":memory:")
    llm = FakeLLM(responses)
    bot = CalendarBot(s, repo, EventExtractor(llm, s, repo), FakeClient(), FakeOutbox(), llm)
    return bot, repo, llm


def bmsg(mid, from_id, text, date=1791216000, **extra):  # 2026-10-05 19:00 МСК
    m = {"business_connection_id": "bc1", "message_id": mid, "date": date, "text": text,
         "chat": {"id": FRIEND, "type": "private", "first_name": "Маша"},
         "from": {"id": from_id, "first_name": "Я" if from_id == OWNER else "Маша"}}
    m.update(extra)
    return {"business_message": m}


EVENT = {"events": [{"title": "Кино с Машей", "when_text": "в пятницу в 7", "date": None, "time": "19:00",
                     "duration_min": None, "location": None, "owner_committed": True, "confidence": 0.6,
                     "message_ids": [2]}], "summary": "кино в пятницу"}


def test_full_flow_draft_then_confirm():
    bot, repo, llm = make_bot([EVENT])

    async def go():
        await bot.handle(bmsg(1, FRIEND, "Пойдём в кино?"))
        await bot.handle(bmsg(2, OWNER, "давай в пятницу в 7", date=1791216060))
        await bot.handle(bmsg(3, 999, "реклама", sender_business_bot={"id": 999}))  # бот от имени владельца
        assert len(repo.pending_for_chat("bc1", FRIEND)) == 2
        return await bot.flush(now=time.time() + 3600)

    res = asyncio.run(go())
    assert res[0].status == "draft"
    method, params = bot.outbox.sent[-1]
    assert method == "sendMessage" and params["chat_id"] == OWNER_CHAT  # пишем только владельцу в личку с ботом
    assert "callback_data" in str(params["reply_markup"])
    assert all("business_connection_id" not in p for _, p in bot.outbox.sent)

    eid = res[0].event_id
    cq = {"id": "q1", "from": {"id": OWNER}, "data": f"ok:{eid}",
          "message": {"chat": {"id": OWNER_CHAT}, "message_id": 50}}
    asyncio.run(bot.handle({"callback_query": cq}))
    assert repo.get_event(eid)["status"] == "confirmed"


def test_foreign_business_account_is_ignored():
    bot, repo, llm = make_bot([], owners="555")

    async def go():
        await bot.handle({"business_connection": {"id": "bc1", "user": {"id": OWNER}, "user_chat_id": OWNER_CHAT,
                                                  "is_enabled": True}})
        await bot.handle(bmsg(1, OWNER, "давай завтра в 7"))

    asyncio.run(go())
    assert repo.pending_for_chat("bc1", FRIEND) == []
    assert "другого владельца" in bot.outbox.sent[0][1]["text"]


def test_callbacks_from_strangers_are_refused():
    bot, repo, _ = make_bot([])
    eid = repo.add_event("Кино", "2026-10-09T19:00", status="draft")
    asyncio.run(bot.handle({"callback_query": {"id": "q", "from": {"id": 42}, "data": f"ok:{eid}"}}))
    assert repo.get_event(eid)["status"] == "draft"


def test_deleted_messages_leave_buffer():
    bot, repo, _ = make_bot([])

    async def go():
        await bot.handle(bmsg(1, OWNER, "давай завтра в 7"))
        await bot.handle({"deleted_business_messages": {"business_connection_id": "bc1", "chat": {"id": FRIEND},
                                                        "message_ids": [1]}})

    asyncio.run(go())
    assert repo.pending_for_chat("bc1", FRIEND) == []


def test_private_chat_creates_recurring_draft_without_llm():
    bot, repo, llm = make_bot([])
    msg = {"message_id": 1, "chat": {"id": OWNER_CHAT, "type": "private"}, "from": {"id": OWNER},
           "text": "каждую пятницу сдача ведомости"}
    asyncio.run(bot.handle({"message": msg}))
    e = repo.list_events(("draft",))[0]
    assert e["rrule"] == "FREQ=WEEKLY;BYDAY=FR" and llm.calls == []
    assert "каждую пятницу" in bot.outbox.sent[-1][1]["text"]
