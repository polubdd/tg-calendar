import json
from datetime import date, datetime, timezone

import httpx
import pytest

from tgcal.config import Settings
from tgcal.context import ChatMessage, build_windows, clip, estimate_tokens, has_time_cue
from tgcal.db import Repo
from tgcal.extractor import EventExtractor, ExtractionResult
from tgcal.llm import FakeLLM, LLMError, LLMUnavailable, OllamaLLM, structured

CONN, CHAT = "bc1", 777


def ts(y, mo, d, h, mi=0):
    """Время сообщения по Москве → unix (как в Telegram)."""
    return int(datetime(y, mo, d, h - 3, mi, tzinfo=timezone.utc).timestamp())


def ev(**kw):
    base = {"title": "Кино с Машей", "when_text": "в пятницу в 7", "date": None, "time": "19:00",
            "duration_min": None, "location": None, "owner_committed": True, "confidence": 0.9, "message_ids": [2]}
    base.update(kw)
    return base


def answer(events, summary="Договорились о кино"):
    return {"events": events, "summary": summary}


@pytest.fixture
def repo():
    return Repo(":memory:")


def feed(repo, msgs):
    for mid, when, owner, text in msgs:
        repo.add_pending(CONN, CHAT, mid, when, owner, "Маша", "Маша", text)


KINO = [
    (1, ts(2026, 10, 5, 18), False, "Пойдём в кино на неделе?"),
    (2, ts(2026, 10, 5, 18, 2), True, "давай в пятницу в 7"),
    (3, ts(2026, 10, 5, 18, 3), False, "ок, договорились"),
]


def test_date_is_relative_to_message_not_today(repo):
    # Сообщение от понедельника 5 октября: «в пятницу» — это 9 октября, хотя модель
    # ошиблась в арифметике, а обрабатываем мы переписку гораздо позже.
    feed(repo, KINO)
    llm = FakeLLM([answer([ev(date="2026-10-10")])])
    res = EventExtractor(llm, Settings(), repo).process_chat(CONN, CHAT)
    assert len(res) == 1 and res[0].action == "created"
    e = res[0].event
    assert e["start"] == "2026-10-09T19:00" and e["status"] == "confirmed" and e["source"] == "telegram"
    assert repo.pending_for_chat(CONN, CHAT) == []  # текст переписки после обработки не хранится
    assert repo.get_summary(CONN, CHAT) == "Договорились о кино"
    assert "Я: давай в пятницу в 7" in llm.calls[0][1]["content"]


def test_not_owner_plans_are_dropped(repo):
    feed(repo, KINO)
    llm = FakeLLM([answer([ev(owner_committed=False)])])
    assert EventExtractor(llm, Settings(), repo).process_chat(CONN, CHAT) == []
    assert repo.list_events() == []


def test_unsure_events_become_drafts(repo):
    feed(repo, KINO)
    llm = FakeLLM([answer([ev(confidence=0.5), ev(title="Пробежка", when_text="в пятницу в 7", time=None, confidence=0.95)])])
    res = EventExtractor(llm, Settings(), repo).process_chat(CONN, CHAT)
    assert [r.status for r in res] == ["draft", "draft"]  # низкая уверенность; «в 7» без контекста


def test_past_events_are_skipped(repo):
    feed(repo, [(1, ts(2026, 10, 5, 18), True, "вчера в 7 вечера были в кино")])
    llm = FakeLLM([answer([ev(when_text="вчера в 7 вечера", date="2026-10-04")])])
    assert EventExtractor(llm, Settings(), repo).process_chat(CONN, CHAT) == []


def test_deduplication_across_runs(repo):
    feed(repo, KINO)
    ex = EventExtractor(FakeLLM([answer([ev()]), answer([ev(title="Кино", confidence=0.95, message_ids=[5])])]),
                        Settings(), repo)
    ex.process_chat(CONN, CHAT)
    feed(repo, [(4, ts(2026, 10, 6, 10), False, "напоминаю, в пятницу в 7 кино!"),
                (5, ts(2026, 10, 6, 10, 1), True, "да, помню, в пятницу в 7")])
    res = ex.process_chat(CONN, CHAT)
    assert [r.action for r in res] == ["merged"]
    events = repo.list_events()
    assert len(events) == 1 and len(events[0]["source_refs"]) == 2


def test_rejected_event_does_not_come_back(repo):
    feed(repo, KINO)
    ex = EventExtractor(FakeLLM([answer([ev(confidence=0.5)]), answer([ev()])]), Settings(), repo)
    eid = ex.process_chat(CONN, CHAT)[0].event_id
    repo.update_event(eid, status="rejected")
    feed(repo, [(9, ts(2026, 10, 6, 10), True, "в пятницу в 7 в силе")])
    assert [r.action for r in ex.process_chat(CONN, CHAT)] == ["skipped"]


def test_invalid_json_is_retried_with_error(repo):
    feed(repo, KINO)
    llm = FakeLLM(["Конечно! Вот события: {oops", json.dumps(answer([ev(time="25:00")])), answer([ev()])])
    res = EventExtractor(llm, Settings(), repo).process_chat(CONN, CHAT)
    assert len(llm.calls) == 3 and len(res) == 1
    assert "не прошёл проверку" in llm.calls[1][-1]["content"]
    assert "time" in llm.calls[2][-1]["content"]  # модели сказали, какое поле неверно


def test_retries_are_limited_and_window_skipped(repo):
    feed(repo, KINO)
    llm = FakeLLM(["нет", "нет", "нет"])
    assert EventExtractor(llm, Settings(llm_max_attempts=3), repo).process_chat(CONN, CHAT) == []
    assert len(llm.calls) == 3
    assert repo.pending_for_chat(CONN, CHAT) == []


def test_ollama_down_keeps_messages(repo):
    feed(repo, KINO)
    with pytest.raises(LLMUnavailable):
        EventExtractor(FakeLLM([]), Settings(), repo).process_chat(CONN, CHAT)
    assert len(repo.pending_for_chat(CONN, CHAT)) == 3


def test_windows_without_time_cues_skip_llm(repo):
    feed(repo, [(1, ts(2026, 10, 5, 18), False, "как дела?"), (2, ts(2026, 10, 5, 18, 1), True, "норм, а у тебя")])
    llm = FakeLLM([])  # любой вызов упадёт
    assert EventExtractor(llm, Settings(), repo).process_chat(CONN, CHAT) == []
    assert llm.calls == []


def test_long_chat_is_split_and_summary_carried(repo):
    s = Settings(num_ctx=4096, max_window_tokens=800, max_window_chars=2000, max_message_chars=300)
    long_text = "давай обсудим планы на неделе " + "очень длинное сообщение " * 60
    feed(repo, [(i, ts(2026, 10, 5, 10) + i * 60, i % 2 == 0, long_text) for i in range(1, 41)])
    n_windows = 0

    class CountingLLM(FakeLLM):
        def chat(self, messages, schema=None):
            nonlocal n_windows
            self.calls.append(messages)
            n_windows += 1
            prompt = sum(estimate_tokens(m["content"]) for m in messages)
            assert prompt <= s.num_ctx - s.reserve_output_tokens  # ответу всегда остаётся место
            return json.dumps(answer([], summary=f"резюме окна {n_windows}"), ensure_ascii=False)

    llm = CountingLLM([])
    EventExtractor(llm, s, repo).process_chat(CONN, CHAT)
    assert n_windows > 3
    assert "резюме окна 1" in llm.calls[1][1]["content"]  # следующее окно видит резюме, а не старую переписку
    assert repo.pending_for_chat(CONN, CHAT) == []


def test_build_windows_overlap_and_progress():
    msgs = [ChatMessage(i, datetime(2026, 10, 5, tzinfo=timezone.utc), "a", False, "x" * 90) for i in range(20)]
    wins = build_windows(msgs, lambda m: m.text, max_chars=300, max_tokens=10_000, overlap=1)
    assert all(len(w) <= 3 for w in wins)
    assert wins[1][0].message_id == wins[0][-1].message_id  # перекрытие на одно сообщение
    assert wins[-1][-1].message_id == 19


def test_clip_keeps_head_and_tail():
    t = clip("начало " + "а" * 1000 + " конец", 100)
    assert len(t) <= 100 and t.startswith("начало") and t.endswith("конец")


def test_time_cues():
    assert has_time_cue("давай в пятницу")
    assert has_time_cue("созвон в 15:30")
    assert not has_time_cue("спасибо, отлично")


def test_ollama_request_sets_num_ctx_and_schema():
    seen = {}

    def handler(request: httpx.Request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(answer([]))}, "done_reason": "stop",
                                         "prompt_eval_count": 100})

    llm = OllamaLLM("http://ollama", "qwen2.5:7b", num_ctx=8192, http=httpx.Client(transport=httpx.MockTransport(handler)))
    res = structured(llm, [{"role": "user", "content": "привет"}], ExtractionResult)
    assert res.events == []
    assert seen["options"]["num_ctx"] == 8192 and seen["options"]["temperature"] == 0
    assert seen["format"]["type"] == "object" and "events" in seen["format"]["properties"]


def test_ollama_down_is_reported():
    def handler(request):
        raise httpx.ConnectError("refused")

    llm = OllamaLLM("http://ollama", "m", 4096, http=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(LLMUnavailable):
        llm.chat([{"role": "user", "content": "x"}])


def test_truncated_answer_is_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json={"message": {"content": '{"events": [{"tit'}, "done_reason": "length"})
        return httpx.Response(200, json={"message": {"content": '{"events": [], "summary": ""}'}, "done_reason": "stop"})

    llm = OllamaLLM("http://o", "m", 4096, http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert structured(llm, [{"role": "user", "content": "x"}], ExtractionResult).events == []
    assert len(calls) == 2


def test_percent_confidence_is_normalized():
    r = ExtractionResult.model_validate({"events": [ev(confidence=85)]})
    assert r.events[0].confidence == 0.85
    with pytest.raises(Exception):
        ExtractionResult.model_validate({"events": [ev(date="пятница")]})
    assert date.fromisoformat(ExtractionResult.model_validate({"events": [ev(date="2026-10-09")]}).events[0].date)
