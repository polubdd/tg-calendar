"""Экспорт в iCalendar (RFC 5545) без сторонних библиотек.

Время пишется «настенным» с TZID (DTSTART;TZID=Europe/Moscow:20261019T080000):
так повторы сохраняют час при переходах на летнее время. Google, Apple и
Outlook понимают IANA-имена поясов в TZID; дополнительно ставим X-WR-TIMEZONE.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .db import parse_dt


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line: str) -> list[str]:
    """Строки длиннее 75 октетов переносятся с пробелом в начале продолжения."""
    out, cur = [], ""
    for ch in line:
        if len((cur + ch).encode()) > (75 if not out else 74):
            out.append(cur)
            cur = ""
        cur += ch
    out.append(cur)
    return [out[0]] + [" " + x for x in out[1:]]


def build_ics(events: list[dict], tz: str, name: str = "Календарь") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//tg-calendar//RU", "CALSCALE:GREGORIAN",
             f"X-WR-CALNAME:{_escape(name)}", f"X-WR-TIMEZONE:{tz}"]
    for e in events:
        start = parse_dt(e["start"])
        lines += ["BEGIN:VEVENT", f"UID:{e['uid']}", f"DTSTAMP:{stamp}", f"SUMMARY:{_escape(e['title'])}"]
        if e["all_day"]:
            end = parse_dt(e["end"]) if e.get("end") else start + timedelta(days=1)
            lines += [f"DTSTART;VALUE=DATE:{start:%Y%m%d}", f"DTEND;VALUE=DATE:{end:%Y%m%d}"]
        else:
            end = parse_dt(e["end"]) if e.get("end") else start + timedelta(hours=1)
            lines += [f"DTSTART;TZID={tz}:{start:%Y%m%dT%H%M%S}", f"DTEND;TZID={tz}:{end:%Y%m%dT%H%M%S}"]
        if e.get("rrule"):
            lines.append(f"RRULE:{e['rrule']}")
        if e.get("location"):
            lines.append(f"LOCATION:{_escape(e['location'])}")
        if e.get("notes"):
            lines.append(f"DESCRIPTION:{_escape(e['notes'])}")
        lines.append("STATUS:TENTATIVE" if e["status"] == "draft" else "STATUS:CONFIRMED")
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(x for line in lines for x in _fold(line)) + "\r\n"
