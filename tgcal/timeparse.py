"""Русские даты, время и повторы — правилами, без LLM.

Модели плохо считают даты: «в пятницу» из сообщения, отправленного в среду,
7B-модель легко превращает не в то число. Поэтому арифметику дат делает этот
модуль, а LLM только находит фразу («в пятницу в 7») и понимает смысл
(«в 7» про кино — это вечер). Простые фразы для календаря («каждый понедельник
зарядка в 8») разбираются здесь целиком, и модель вообще не вызывается.

Все даты считаются от опорной даты `ref` — для переписки это дата сообщения,
а не сегодняшний день.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrulestr

MONTH_RE = (
    r"(январ[яеь]|феврал[яеь]|марта?|апрел[яеь]|ма[яй]|июн[яеь]|июл[яеь]|августа?"
    r"|сентябр[яеь]|октябр[яеь]|ноябр[яеь]|декабр[яеь])(?!\w)"
)
_MONTH_PREFIX = ["янв", "фев", "мар", "апр", "ма", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]

_WD = [
    (0, r"понедельник(?:а|ам|ах|у)?|пн"),
    (1, r"вторник(?:а|ам|ах|у)?|вт"),
    (2, r"сред(?:а|у|ы|ам|ах)|ср"),
    (3, r"четверг(?:а|ам|ах|у)?|чт"),
    (4, r"пятниц(?:а|у|ы|ам|ах)|пт"),
    (5, r"суббот(?:а|у|ы|ам|ах)|сб"),
    (6, r"воскресень(?:е|я|ям|ях)|вс"),
]
WD_RE = "(?:" + "|".join(p for _, p in _WD) + r")(?!\w)"
BYDAY = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
WD_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]

_NUM_WORDS = {"один": 1, "одну": 1, "одна": 1, "два": 2, "две": 2, "пару": 2, "три": 3, "четыре": 4, "пять": 5}
_DAYPART = r"(утра|дня|вечера|ночи)"


def month_num(word: str) -> int:
    w = word.lower()
    for i, p in enumerate(_MONTH_PREFIX, 1):
        if w.startswith(p):
            return i
    raise ValueError(word)


def weekday_of(word: str) -> int:
    for n, p in _WD:
        if re.fullmatch(p, word.lower()):
            return n
    raise ValueError(word)


def _num(s: str | None, default: int = 1) -> int:
    if not s:
        return default
    return int(s) if s.isdigit() else _NUM_WORDS.get(s, default)


class Scan:
    """Текст, из которого по очереди вырезаются найденные куски.

    Вырезанное заменяется пробелами той же длины, чтобы позиции не съезжали,
    а остаток потом становится названием события.
    """

    def __init__(self, text: str):
        self.orig = text
        self.low = text.lower().replace("ё", "е")

    def search(self, pattern: str) -> re.Match | None:
        return re.search(pattern, self.low)

    def cut(self, m: re.Match) -> None:
        a, b = m.span()
        pad = " " * (b - a)
        self.orig = self.orig[:a] + pad + self.orig[b:]
        self.low = self.low[:a] + pad + self.low[b:]


# ---------- даты ----------

@dataclass
class DateMatch:
    value: date
    ambiguous: bool = False


def _year_for(day: int, month: int, ref: date) -> date:
    """«15 октября» без года: ближайшее такое число, не раньше опорной даты."""
    d = date(ref.year, month, day)
    return d if d >= ref else date(ref.year + 1, month, day)


def _explicit_date(day: str, month: int, year: str | None, ref: date) -> date | None:
    try:
        if year:
            y = int(year)
            return date(y + 2000 if y < 100 else y, month, int(day))
        return _year_for(int(day), month, ref)
    except ValueError:
        return None


def find_date(scan: Scan, ref: date) -> DateMatch | None:
    m = scan.search(r"\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b")
    if m and 1 <= int(m.group(2)) <= 12:
        d = _explicit_date(m.group(1), int(m.group(2)), m.group(3), ref)
        if d:
            scan.cut(m)
            return DateMatch(d)

    m = scan.search(r"\b(?:(?:на|к|ко|до)\s+)?(\d{1,2})(?:-?го|-?е)?\s+" + MONTH_RE + r"(?:\s+(\d{4})(?:\s*г(?:ода|\.)?)?)?")
    if m:
        d = _explicit_date(m.group(1), month_num(m.group(2)), m.group(3), ref)
        if d:
            scan.cut(m)
            return DateMatch(d)

    for word, shift in (("послезавтра", 2), ("завтра", 1), ("сегодня", 0)):
        m = scan.search(rf"\b(?:на\s+)?{word}\b")
        if m:
            scan.cut(m)
            return DateMatch(ref + timedelta(days=shift))

    m = scan.search(r"\bчерез\s+(\d+|одну|один|два|две|три|четыре|пять|пару)?\s*(дн(?:я|ей)|день|недел[июь]|месяц(?:а|ев)?)\b")
    if m:
        n, unit = _num(m.group(1)), m.group(2)
        scan.cut(m)
        if unit.startswith("недел"):
            return DateMatch(ref + timedelta(weeks=n))
        if unit.startswith("месяц"):
            return DateMatch(ref + relativedelta(months=n))
        return DateMatch(ref + timedelta(days=n))

    m = scan.search(
        r"\b(?:(?:в|во|на|к|ко)\s+)?(?:(эт[оуи]т?|эту|следующ(?:ий|ую|ее|ей)|ближайш(?:ий|ую|ее))\s+)?(" + WD_RE + ")"
    )
    if m:
        wd = weekday_of(m.group(2))
        mod = m.group(1) or ""
        scan.cut(m)
        delta = (wd - ref.weekday()) % 7
        if mod.startswith("следующ"):
            # «в следующую пятницу» обычно значит пятницу следующей недели,
            # но так говорят и про ближайшую — поэтому отмечаем как неуверенное.
            monday_next = ref + timedelta(days=7 - ref.weekday())
            return DateMatch(monday_next + timedelta(days=wd), ambiguous=True)
        if delta == 0:
            # «в пятницу», сказанное в пятницу: скорее про следующую неделю.
            return DateMatch(ref + timedelta(days=7), ambiguous=not mod)
        return DateMatch(ref + timedelta(days=delta))

    m = scan.search(r"\b(?:(?:на|к|до)\s+)?(\d{1,2})(?:-?го|\s+числа)(?!\w)")
    if m:
        day = int(m.group(1))
        scan.cut(m)
        try:
            d = ref.replace(day=day)
            if d < ref:
                d = (ref.replace(day=1) + relativedelta(months=1)).replace(day=day)
            return DateMatch(d)
        except ValueError:
            return None

    m = scan.search(r"\bна\s+(?:этих\s+)?выходных\b")
    if m:
        scan.cut(m)
        return DateMatch(ref + timedelta(days=(5 - ref.weekday()) % 7), ambiguous=True)

    m = scan.search(r"\bна\s+следующей\s+неделе\b")
    if m:
        scan.cut(m)
        return DateMatch(ref + timedelta(days=7 - ref.weekday()), ambiguous=True)
    return None


# ---------- время ----------

@dataclass
class TimeMatch:
    start: time
    end: time | None = None
    ambiguous: bool = False


def _hour(h: int, part: str | None, bare: bool) -> tuple[int, bool]:
    """Час с учётом «утра/дня/вечера/ночи». bare — время без минут и без уточнения."""
    if part == "утра":
        return (0 if h == 12 else h), False
    if part in ("дня", "вечера"):
        return (h + 12 if h < 12 else h), False
    if part == "ночи":
        return (0 if h == 12 else h), False
    if bare and 1 <= h <= 6:
        return h + 12, True  # «в 3» почти всегда 15:00, но это догадка
    if bare and 7 <= h <= 11:
        return h, True  # «в 7» — утро или вечер, без контекста не понять
    return h, False


def _mk(h: int, mnt: int) -> time | None:
    return time(h, mnt) if 0 <= h <= 23 and 0 <= mnt <= 59 else None


def find_time(scan: Scan) -> TimeMatch | None:
    m = scan.search(
        r"\b(?:с\s+)?(\d{1,2})(?:[:.](\d{2}))?\s*(?:до|-|–|—)\s*(\d{1,2})(?:[:.](\d{2}))?(?:\s+" + _DAYPART + r")?(?!\w)"
    )
    if m and (m.group(2) or m.group(4) or m.group(0).lstrip().startswith("с")):
        part = m.group(5)
        h1, _ = _hour(int(m.group(1)), part if part in ("дня", "вечера") and int(m.group(1)) < 12 else None, False)
        h2, _ = _hour(int(m.group(3)), part, False)
        t1, t2 = _mk(h1, int(m.group(2) or 0)), _mk(h2, int(m.group(4) or 0))
        if t1 and t2:
            scan.cut(m)
            return TimeMatch(t1, t2)

    m = scan.search(r"\b(?:(?:в|к|около|на)\s+)?(\d{1,2}):(\d{2})(?!\d)(?:\s+" + _DAYPART + r")?")
    if m:
        h, _ = _hour(int(m.group(1)), m.group(3), False)
        t = _mk(h, int(m.group(2)))
        if t:
            scan.cut(m)
            return TimeMatch(t)

    m = scan.search(
        r"\b(?:в|к|около)\s+(\d{1,2})(?:\.(\d{2}))?(?:\s*час(?:а|ов)?)?(?:\s+" + _DAYPART + r")?(?!\w)"
        r"(?!\s*(?:числ|-|раз\b|дн|недел|месяц|лет|год|%|человек|минут|мин\b|рубл|р\b|" + MONTH_RE + "))"
    )
    if m:
        bare = not m.group(2) and not m.group(3)
        h, amb = _hour(int(m.group(1)), m.group(3), bare)
        t = _mk(h, int(m.group(2) or 0))
        if t:
            scan.cut(m)
            return TimeMatch(t, ambiguous=amb)

    m = scan.search(r"\b(?:в|к)\s+(полдень|полночь)\b")
    if m:
        scan.cut(m)
        return TimeMatch(time(12, 0) if m.group(1) == "полдень" else time(0, 0))
    return None


# ---------- повторы ----------

_FREQ_BY_UNIT = {"д": "DAILY", "н": "WEEKLY", "м": "MONTHLY", "г": "YEARLY", "л": "YEARLY"}


def find_recurrence(scan: Scan, ref: date) -> str | None:
    """Ищет правило повторения и возвращает тело RRULE (без DTSTART)."""
    freq: str | None = None
    interval = 1
    byday: list[str] = []
    bymonthday: int | None = None
    bymonth: int | None = None

    if m := scan.search(r"\bкажд(?:ый|ого)\s+будн(?:ий|его)\s+(?:день|дня)\b|\bпо\s+будн(?:ям|им|ие|и)\b"):
        freq, byday = "WEEKLY", BYDAY[:5]
        scan.cut(m)
    if m := scan.search(r"\bпо\s+выходн(?:ым|ые)\b|\bкажд(?:ые|ых)\s+выходн(?:ые|ых)\b"):
        freq, byday = "WEEKLY", BYDAY[5:]
        scan.cut(m)

    wd_list = WD_RE + r"(?:\s*(?:,|\bи\b)\s*(?:(?:кажд\w*|по)\s+)?" + WD_RE + ")*"
    if m := scan.search(r"\b(?:кажд(?:ый|ую|ое|ые)|по)\s+(?:(втор(?:ой|ую|ое|ые))\s+)?(" + wd_list + ")"):
        freq = "WEEKLY"
        if m.group(1):
            interval = 2
        for w in re.findall(WD_RE, m.group(2)):
            code = BYDAY[weekday_of(w)]
            if code not in byday:
                byday.append(code)
        scan.cut(m)

    if m := scan.search(r"\bкажд(?:ое|ый)\s+(\d{1,2})(?:-?е|-?го)?(?:\s+числ[оа])?(?!\w)"):
        freq, bymonthday = "MONTHLY", int(m.group(1))
        scan.cut(m)

    if m := scan.search(
        r"\b(?:каждые|каждых|раз\s+в)\s+(\d+|два|две|три|четыре|пять|пару)?\s*(дн(?:я|ей)|день|недел[июь]|месяц(?:а|ев)?|год(?:а)?|лет)\b"
    ):
        freq = freq or _FREQ_BY_UNIT[m.group(2)[0]]
        interval = _num(m.group(1))
        scan.cut(m)

    for pattern, f in (
        (r"\bкажд(?:ый|ого)\s+(?:день|дня)\b|\bежедневно\b", "DAILY"),
        (r"\bкажд(?:ую|ой)\s+недел[июь]\b|\bеженедельно\b", "WEEKLY"),
        (r"\bкажд(?:ый|ого)\s+месяц[а]?\b|\bежемесячно\b", "MONTHLY"),
        (r"\bкажд(?:ый|ого)\s+год[а]?\b|\bежегодно\b", "YEARLY"),
    ):
        if m := scan.search(pattern):
            freq = freq or f
            scan.cut(m)

    if not freq:
        return None

    # Ограничения ищем до уточнений даты, иначе «до 25 декабря» станет днём повтора.
    until: date | None = None
    if m := scan.search(r"\bдо\s+(\d{1,2})(?:-?го)?\s+" + MONTH_RE + r"(?:\s+(\d{4}))?"):
        until = _explicit_date(m.group(1), month_num(m.group(2)), m.group(3), ref)
        scan.cut(m)
    elif m := scan.search(r"\bдо\s+(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b"):
        if 1 <= int(m.group(2)) <= 12:
            until = _explicit_date(m.group(1), int(m.group(2)), m.group(3), ref)
            scan.cut(m)
    count: int | None = None
    if m := scan.search(r"\b(\d{1,3})\s+раз(?:а)?\b(?!\s+в\b)"):
        count = int(m.group(1))
        scan.cut(m)

    if freq == "WEEKLY" and not byday:
        if m := scan.search(r"\b(?:в|во)\s+(" + WD_RE + ")"):
            byday = [BYDAY[weekday_of(m.group(1))]]
            scan.cut(m)
    if freq == "MONTHLY" and bymonthday is None:
        if m := scan.search(r"\b(\d{1,2})(?:-?го|-?е|\s+числ[ао]?)(?!\w)"):
            bymonthday = int(m.group(1))
            scan.cut(m)
    if freq == "YEARLY":
        if m := scan.search(r"\b(\d{1,2})(?:-?го)?\s+" + MONTH_RE):
            bymonthday, bymonth = int(m.group(1)), month_num(m.group(2))
            scan.cut(m)

    parts = [f"FREQ={freq}"]
    if interval > 1:
        parts.append(f"INTERVAL={interval}")
    if byday:
        parts.append("BYDAY=" + ",".join(byday))
    if bymonth:
        parts.append(f"BYMONTH={bymonth}")
    if bymonthday:
        parts.append(f"BYMONTHDAY={bymonthday}")
    if until:
        parts.append(f"UNTIL={until:%Y%m%d}T235959")
    if count:
        parts.append(f"COUNT={count}")
    return ";".join(parts)


def normalize_rrule(rule: str) -> str:
    """Проверяет RRULE (в том числе от LLM) и приводит к виду без DTSTART."""
    s = rule.strip()
    if s.upper().startswith("RRULE:"):
        s = s[6:]
    s = s.upper().replace(" ", "").strip(";")
    # Даты храним «настенными» без пояснов, а dateutil не смешивает UNTIL с Z и наивный DTSTART.
    s = re.sub(r"(UNTIL=\d{8}(?:T\d{6})?)Z", r"\1", s)
    if not s.startswith("FREQ=") and "FREQ=" not in s:
        raise ValueError("в правиле нет FREQ")
    if "DTSTART" in s or "\n" in s:
        raise ValueError("DTSTART в правиле не нужен")
    try:
        rrulestr("RRULE:" + s, dtstart=datetime(2026, 1, 1))
    except Exception as e:  # dateutil кидает ValueError, TypeError, KeyError
        raise ValueError(f"некорректное правило: {e}") from e
    return s


def first_occurrence(rule: str, ref: date) -> date | None:
    # INTERVAL отсчитывается от DTSTART: «каждые 2 недели во вторник», сказанное в среду,
    # dateutil начал бы через две недели. Первое занятие — ближайший подходящий день.
    rule = re.sub(r";?INTERVAL=\d+", "", rule)
    r = rrulestr("RRULE:" + rule, dtstart=datetime.combine(ref, time.min))
    nxt = r.after(datetime.combine(ref, time.min), inc=True)
    return nxt.date() if nxt else None


def expand(rule: str, start: datetime, range_start: datetime, range_end: datetime, limit: int = 1000) -> list[datetime]:
    r = rrulestr("RRULE:" + rule, dtstart=start)
    out = []
    for dt in r.xafter(range_start, inc=True):
        if dt > range_end or len(out) >= limit:
            break
        out.append(dt)
    return out


# ---------- человеческое описание ----------

_WD_ACC = ["каждый понедельник", "каждый вторник", "каждую среду", "каждый четверг", "каждую пятницу", "каждую субботу", "каждое воскресенье"]
_WD_DAT = ["понедельникам", "вторникам", "средам", "четвергам", "пятницам", "субботам", "воскресеньям"]
_MONTH_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]


def describe_rrule(rule: str) -> str:
    p = dict(kv.split("=", 1) for kv in rule.split(";") if "=" in kv)
    freq, interval = p.get("FREQ"), int(p.get("INTERVAL", 1))
    days = [BYDAY.index(d[-2:]) for d in p.get("BYDAY", "").split(",") if d[-2:] in BYDAY]
    if freq == "WEEKLY" and days == [0, 1, 2, 3, 4]:
        text = "по будням"
    elif freq == "WEEKLY" and days == [5, 6]:
        text = "по выходным"
    elif freq == "WEEKLY" and len(days) == 1 and interval == 1:
        text = _WD_ACC[days[0]]
    elif freq == "WEEKLY" and days:
        text = "по " + ", ".join(_WD_DAT[d] for d in days)
        if interval > 1:
            text = f"раз в {interval} недели, " + text
    else:
        unit = {"DAILY": ("каждый день", "дня"), "WEEKLY": ("каждую неделю", "недели"),
                "MONTHLY": ("каждый месяц", "месяца"), "YEARLY": ("каждый год", "года")}.get(freq, (freq or "?", ""))
        text = unit[0] if interval == 1 else f"раз в {interval} {unit[1]}"
        if freq == "MONTHLY" and "BYMONTHDAY" in p:
            text += f", {p['BYMONTHDAY']} числа"
        if freq == "YEARLY" and "BYMONTHDAY" in p and "BYMONTH" in p:
            text += f", {p['BYMONTHDAY']} {_MONTH_GEN[int(p['BYMONTH']) - 1]}"
    if "UNTIL" in p:
        u = p["UNTIL"]
        text += f", до {u[6:8]}.{u[4:6]}.{u[:4]}"
    if "COUNT" in p:
        text += f", {p['COUNT']} раз"
    return text


# ---------- фраза целиком ----------

_COMMANDS = r"\b(?:напомни(?:ть)?|напоминай|добавь|поставь|запиши|создай|запланируй|занеси|внеси)\b"
_EDGE_STOP = {
    "давай", "давайте", "мне", "меня", "у", "надо", "нужно", "в", "во", "на", "к", "ко", "и", "по", "с", "со",
    "до", "а", "пожалуйста", "будет", "есть", "это", "что", "ок", "окей", "го", "про", "о", "об", "-", "—",
}


def clean_title(text: str) -> str:
    s = re.sub(_COMMANDS, " ", text, flags=re.I)
    s = re.sub(r"[«»\"!?;:()]", " ", s)
    tokens = [t.strip(",.") for t in s.split()]
    tokens = [t for t in tokens if t]
    while tokens and tokens[0].lower() in _EDGE_STOP:
        tokens.pop(0)
    while tokens and tokens[-1].lower() in _EDGE_STOP:
        tokens.pop()
    title = " ".join(tokens).strip(" ,.-–")
    return (title[:1].upper() + title[1:]) if title else "Событие"


@dataclass
class ParsedEvent:
    title: str
    start: datetime | date
    end: datetime | date | None
    all_day: bool
    rrule: str | None = None
    ambiguous: list[str] = field(default_factory=list)


def parse_phrase(text: str, now: datetime, default_minutes: int = 60) -> ParsedEvent | None:
    """Разбор фразы целиком. None — правилами не понять, нужен LLM."""
    scan = Scan(text)
    ref = now.date()
    ambiguous: list[str] = []
    rule = find_recurrence(scan, ref)
    dm = find_date(scan, ref)
    tm = find_time(scan)
    if rule:
        day = dm.value if dm else first_occurrence(rule, ref)
        if day is None:
            return None
    elif dm:
        day = dm.value
    elif tm:
        day = ref if datetime.combine(ref, tm.start) > now.replace(tzinfo=None) else ref + timedelta(days=1)
        ambiguous.append("день не указан — поставил ближайший")
    else:
        return None
    if dm and dm.ambiguous:
        ambiguous.append("дата понята неоднозначно")
    if tm and tm.ambiguous:
        ambiguous.append(f"«{tm.start:%H:%M}» — утро или вечер? проверьте время")

    title = clean_title(scan.orig)
    if tm:
        start = datetime.combine(day, tm.start)
        end = datetime.combine(day, tm.end) if tm.end else start + timedelta(minutes=default_minutes)
        if end <= start:
            end += timedelta(days=1)
        return ParsedEvent(title, start, end, False, rule, ambiguous)
    return ParsedEvent(title, day, None, True, rule, ambiguous)
