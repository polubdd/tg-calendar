from datetime import date

import pytest
from fastapi.testclient import TestClient

from tgcal.config import Settings
from tgcal.db import Repo
from tgcal.ics import build_ics
from tgcal.llm import FakeLLM
from tgcal.web import create_app, occurrences


@pytest.fixture
def ctx():
    repo = Repo(":memory:")
    llm = FakeLLM([])
    client = TestClient(create_app(Settings(ics_token="secret"), repo, llm))
    return client, repo, llm


def test_chat_rules_then_add(ctx):
    client, repo, llm = ctx
    r = client.post("/api/chat", json={"text": "каждый понедельник зарядка в 8"})
    assert r.status_code == 200
    p = r.json()
    assert p["source"] == "rules" and p["event"]["rrule"] == "FREQ=WEEKLY;BYDAY=MO"
    assert p["description"].startswith("каждый понедельник в 08:00")
    assert llm.calls == []  # простая фраза — без модели
    assert client.post("/api/events", json=p["event"]).status_code == 201

    occ = client.get("/api/events", params={"start": "2026-11-01", "end": "2026-12-01"}).json()
    assert [o["start"] for o in occ] == ["2026-11-02T08:00", "2026-11-09T08:00", "2026-11-16T08:00",
                                         "2026-11-23T08:00", "2026-11-30T08:00"]


def test_chat_falls_back_to_llm_and_validates_rrule(ctx):
    client, repo, llm = ctx
    llm.responses += [
        '{"title": "Отчёт", "when_text": "", "date": null, "time": "18:00", "end_time": null, '
        '"rrule": "каждый последний", "location": null}',
        '{"title": "Отчёт", "when_text": "", "date": null, "time": "18:00", "end_time": null, '
        '"rrule": "FREQ=MONTHLY;BYDAY=-1FR", "location": null}',
    ]
    # Ни даты, ни знакомого повтора: правила сдаются, фразу разбирает модель.
    p = client.post("/api/chat", json={"text": "отчёт по итогам месяца, последний пятничный вечер"}).json()
    assert p["source"] == "llm" and p["event"]["rrule"] == "FREQ=MONTHLY;BYDAY=-1FR"
    assert len(llm.calls) == 2 and "не прошёл проверку" in llm.calls[1][-1]["content"]


def test_chat_without_llm_explains(ctx):
    client, repo, llm = ctx
    r = client.post("/api/chat", json={"text": "сходить куда-нибудь"})
    assert r.status_code == 503  # FakeLLM пуст — как будто Ollama не запущена


def test_crud_and_drafts(ctx):
    client, repo, _ = ctx
    e = client.post("/api/events", json={"title": "Кино", "start": "2026-10-09T19:00", "end": "2026-10-09T21:00",
                                         "status": "draft"}).json()
    assert [d["id"] for d in client.get("/api/drafts").json()] == [e["id"]]
    client.post(f"/api/events/{e['id']}/confirm")
    assert client.get("/api/drafts").json() == []
    assert client.patch(f"/api/events/{e['id']}", json={"title": "Кино с Машей"}).json()["title"] == "Кино с Машей"
    client.post(f"/api/events/{e['id']}/reject")
    assert client.get("/api/events", params={"start": "2026-10-01", "end": "2026-11-01"}).json() == []
    assert client.delete(f"/api/events/{e['id']}").status_code == 204
    assert client.get(f"/api/events/{e['id']}").status_code == 404


def test_bad_input_rejected(ctx):
    client, _, _ = ctx
    assert client.post("/api/events", json={"title": "x", "start": "завтра"}).status_code == 422
    assert client.post("/api/events", json={"title": "x", "start": "2026-10-09", "rrule": "FREQ=OFTEN"}).status_code == 422


def test_ics_export_and_subscription(ctx):
    client, repo, _ = ctx
    repo.add_event("Зарядка", "2026-10-19T08:00", "2026-10-19T08:30", rrule="FREQ=WEEKLY;BYDAY=MO")
    repo.add_event("Сдача ведомости", "2026-10-16", all_day=True, rrule="FREQ=WEEKLY;BYDAY=FR")
    repo.add_event("Черновик", "2026-10-20T10:00", status="draft")
    body = client.get("/api/export.ics").text
    assert "DTSTART;TZID=Europe/Moscow:20261019T080000" in body
    assert "RRULE:FREQ=WEEKLY;BYDAY=MO" in body
    assert "DTSTART;VALUE=DATE:20261016" in body and "DTEND;VALUE=DATE:20261017" in body
    assert "Черновик" not in body  # в подписку идут только подтверждённые
    assert body.count("BEGIN:VEVENT") == 2 and "\r\n" in body
    assert client.get("/ics/secret.ics").status_code == 200
    assert client.get("/ics/guess.ics").status_code == 404


def test_ics_escaping_and_folding():
    ev = {"uid": "u1", "title": "Встреча; важно, " + "очень " * 30, "start": "2026-10-19T08:00", "end": None,
          "all_day": False, "rrule": None, "location": None, "notes": None, "status": "confirmed"}
    text = build_ics([ev], "Europe/Moscow")
    assert "Встреча\\; важно\\," in text
    assert all(len(line.encode()) <= 75 for line in text.split("\r\n"))


def test_occurrences_multi_day_and_recurring_all_day():
    events = [
        {"id": 1, "title": "Конференция", "start": "2026-10-30T10:00", "end": "2026-11-01T18:00", "all_day": False,
         "rrule": None, "status": "confirmed", "source": "manual"},
        {"id": 2, "title": "Ведомость", "start": "2026-10-16", "end": None, "all_day": True,
         "rrule": "FREQ=WEEKLY;BYDAY=FR", "status": "confirmed", "source": "manual"},
    ]
    occ = occurrences(events, date(2026, 11, 1), date(2026, 11, 15))
    assert [o["id"] for o in occ] == [1, 2, 2]  # конференция ещё идёт 1 ноября; пятницы 6 и 13
    assert [o["start"] for o in occ if o["id"] == 2] == ["2026-11-06", "2026-11-13"]


def test_index_served(ctx):
    client, _, _ = ctx
    assert "Чат с календарём" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
