"""Веб-интерфейс и API календаря.

Сервер по умолчанию слушает только 127.0.0.1: календарь собран из личной
переписки, наружу его выставлять не нужно. Подписка .ics защищена случайным
токеном в адресе — на случай, если порт всё-таки откроют для телефона в той же сети.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .assistant import NotUnderstood, interpret
from .config import Settings
from .db import Repo, parse_dt
from .ics import build_ics
from .llm import LLMError, OllamaLLM
from .timeparse import describe_rrule, expand, normalize_rrule

STATIC = Path(__file__).parent / "static"


def _check_dt(v):
    if v in (None, ""):
        return None
    parse_dt(v)
    return v[:16]


def _check_rrule(v):
    return normalize_rrule(v) if v else None


class EventIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    start: str
    end: str | None = None
    all_day: bool = False
    rrule: str | None = None
    location: str | None = None
    notes: str | None = None
    status: Literal["confirmed", "draft"] = "confirmed"

    _dt = field_validator("start", "end")(classmethod(lambda cls, v: _check_dt(v)))
    _rr = field_validator("rrule")(classmethod(lambda cls, v: _check_rrule(v)))


class EventPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    start: str | None = None
    end: str | None = None
    all_day: bool | None = None
    rrule: str | None = None
    location: str | None = None
    notes: str | None = None
    status: Literal["confirmed", "draft", "rejected"] | None = None

    _dt = field_validator("start", "end")(classmethod(lambda cls, v: _check_dt(v)))
    _rr = field_validator("rrule")(classmethod(lambda cls, v: _check_rrule(v)))


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


def occurrences(events: list[dict], start: date, end: date) -> list[dict]:
    """Разворачивает повторы в конкретные даты внутри [start, end)."""
    rs, re_ = datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time())
    out = []
    for e in events:
        s = parse_dt(e["start"])
        dur = (parse_dt(e["end"]) - s) if e.get("end") else (timedelta(days=1) if e["all_day"] else timedelta(hours=1))
        starts = expand(e["rrule"], s, rs - dur, re_ - timedelta(seconds=1)) if e.get("rrule") else [s]
        for st in starts:
            if st + dur <= rs or st >= re_:
                continue
            out.append({
                "id": e["id"], "title": e["title"], "all_day": e["all_day"], "status": e["status"],
                "source": e["source"], "recurring": bool(e.get("rrule")), "location": e.get("location"),
                "start": st.date().isoformat() if e["all_day"] else st.isoformat(timespec="minutes"),
                "end": None if e["all_day"] else (st + dur).isoformat(timespec="minutes"),
            })
    return sorted(out, key=lambda o: (o["start"], o["title"]))


def create_app(settings: Settings | None = None, repo: Repo | None = None, llm=None) -> FastAPI:
    s = settings or Settings.from_env()
    repo = repo or Repo(s.db_path)
    llm = llm if llm is not None else OllamaLLM(s.ollama_url, s.ollama_model, s.num_ctx, s.llm_timeout)
    tz = ZoneInfo(s.tz)
    app = FastAPI(title="tg-calendar")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def now() -> datetime:
        return datetime.now(tz).replace(tzinfo=None)

    def get_or_404(event_id: int) -> dict:
        e = repo.get_event(event_id)
        if not e:
            raise HTTPException(404, "событие не найдено")
        return e

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        token = repo.ics_token(s.ics_token)
        return {
            "tz": s.tz, "model": s.ollama_model, "num_ctx": s.num_ctx,
            "llm": getattr(llm, "available", lambda: True)(),
            "bot_configured": bool(s.telegram_bot_token), "owners": len(s.owner_ids),
            "ics_url": f"/ics/{token}.ics", "today": now().date().isoformat(),
        }

    @app.get("/api/events")
    def list_occurrences(start: date = Query(...), end: date = Query(...), drafts: bool = True):
        if (end - start).days > 400:
            raise HTTPException(400, "слишком большой диапазон")
        statuses = ("confirmed", "draft") if drafts else ("confirmed",)
        return occurrences(repo.list_events(statuses), start, end)

    @app.get("/api/drafts")
    def drafts():
        return [dict(e, when=describe_rrule(e["rrule"]) if e["rrule"] else None) for e in repo.list_events(("draft",))]

    @app.get("/api/events/{event_id}")
    def get_event(event_id: int):
        e = get_or_404(event_id)
        return dict(e, when=describe_rrule(e["rrule"]) if e["rrule"] else None)

    @app.post("/api/events", status_code=201)
    def create_event(body: EventIn):
        if body.end and parse_dt(body.end) < parse_dt(body.start):
            raise HTTPException(422, "конец раньше начала")
        eid = repo.add_event(body.title, body.start, body.end, body.all_day, body.rrule, body.location, body.notes,
                             body.status, "manual")
        return repo.get_event(eid)

    @app.patch("/api/events/{event_id}")
    def patch_event(event_id: int, body: EventPatch):
        get_or_404(event_id)
        return repo.update_event(event_id, **body.model_dump(exclude_unset=True))

    @app.post("/api/events/{event_id}/confirm")
    def confirm(event_id: int):
        get_or_404(event_id)
        return repo.update_event(event_id, status="confirmed")

    @app.post("/api/events/{event_id}/reject")
    def reject(event_id: int):
        # Не удаляем, а помечаем: иначе бот найдёт то же событие в следующем сообщении.
        get_or_404(event_id)
        return repo.update_event(event_id, status="rejected")

    @app.delete("/api/events/{event_id}", status_code=204)
    def delete(event_id: int):
        if not repo.delete_event(event_id):
            raise HTTPException(404, "событие не найдено")

    @app.post("/api/chat")
    def chat(body: ChatIn):
        try:
            return interpret(body.text, now(), llm, s.tz, s.llm_max_attempts)
        except NotUnderstood as e:
            raise HTTPException(422, str(e))
        except LLMError as e:
            raise HTTPException(503, f"Правилами не разобрать, а модель недоступна или ответила неверно: {e}")

    @app.get("/api/export.ics")
    def export():
        body = build_ics(repo.list_events(("confirmed",)), s.tz)
        return Response(body, media_type="text/calendar; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="calendar.ics"'})

    @app.get("/ics/{token}.ics")
    def subscribe(token: str):
        if token != repo.ics_token(s.ics_token):
            raise HTTPException(404)
        return Response(build_ics(repo.list_events(("confirmed",)), s.tz), media_type="text/calendar; charset=utf-8")

    return app
