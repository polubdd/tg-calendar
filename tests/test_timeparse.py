from datetime import date, datetime, time

import pytest

from tgcal.timeparse import (Scan, describe_rrule, expand, find_date, find_time, normalize_rrule, parse_phrase)

WED = datetime(2026, 10, 14, 10, 0)  # среда


@pytest.mark.parametrize("text, rule, title, start, all_day", [
    ("каждый понедельник зарядка в 8", "FREQ=WEEKLY;BYDAY=MO", "Зарядка", datetime(2026, 10, 19, 8, 0), False),
    ("каждую пятницу сдача ведомости", "FREQ=WEEKLY;BYDAY=FR", "Сдача ведомости", date(2026, 10, 16), True),
    ("по будням в 9:30 стендап", "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR", "Стендап", datetime(2026, 10, 14, 9, 30), False),
    ("каждый будний день в 9:00 стендап", "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR", "Стендап", datetime(2026, 10, 14, 9, 0), False),
    ("по выходным в 11:00 бассейн", "FREQ=WEEKLY;BYDAY=SA,SU", "Бассейн", datetime(2026, 10, 17, 11, 0), False),
    ("по понедельникам и средам с 18:30 до 20:00 английский", "FREQ=WEEKLY;BYDAY=MO,WE", "Английский",
     datetime(2026, 10, 14, 18, 30), False),
    ("каждые 2 недели во вторник в 18:00 встреча клуба", "FREQ=WEEKLY;INTERVAL=2;BYDAY=TU", "Встреча клуба",
     datetime(2026, 10, 20, 18, 0), False),
    ("каждый день в 23:00 читать", "FREQ=DAILY", "Читать", datetime(2026, 10, 14, 23, 0), False),
    ("каждое 15 число оплата интернета", "FREQ=MONTHLY;BYMONTHDAY=15", "Оплата интернета", date(2026, 10, 15), True),
    ("ежегодно 3 марта день рождения мамы", "FREQ=YEARLY;BYMONTH=3;BYMONTHDAY=3", "День рождения мамы",
     date(2027, 3, 3), True),
    ("каждую пятницу до 25 декабря сдача ведомости", "FREQ=WEEKLY;BYDAY=FR;UNTIL=20261225T235959",
     "Сдача ведомости", date(2026, 10, 16), True),
    ("напомни каждый вторник в 19:00 тренировка 10 раз", "FREQ=WEEKLY;BYDAY=TU;COUNT=10", "Тренировка",
     datetime(2026, 10, 20, 19, 0), False),
])
def test_recurring_phrases(text, rule, title, start, all_day):
    p = parse_phrase(text, WED)
    assert p is not None
    assert p.rrule == rule
    assert p.title == title
    assert p.start == start
    assert p.all_day is all_day
    normalize_rrule(p.rrule)  # правило валидно для dateutil


def test_bare_morning_hour_is_flagged():
    p = parse_phrase("каждый понедельник зарядка в 8", WED)
    assert p.start.time() == time(8, 0)
    assert any("утро или вечер" in a for a in p.ambiguous)


@pytest.mark.parametrize("text, start, title", [
    ("завтра в 15 созвон с Ильёй", datetime(2026, 10, 15, 15, 0), "Созвон с Ильёй"),
    ("через 2 дня в 7 вечера ужин", datetime(2026, 10, 16, 19, 0), "Ужин"),
    ("напомни 20.10 в 9 утра стоматолог", datetime(2026, 10, 20, 9, 0), "Стоматолог"),
    ("15 октября дедлайн курсовой", date(2026, 10, 15), "Дедлайн курсовой"),
    ("встреча с 10 до 12 послезавтра", datetime(2026, 10, 16, 10, 0), "Встреча"),
])
def test_single_events(text, start, title):
    p = parse_phrase(text, WED)
    assert p.rrule is None
    assert p.start == start
    assert p.title == title


def test_not_an_event():
    assert parse_phrase("купить молоко", WED) is None


@pytest.mark.parametrize("phrase, ref, expected, ambiguous", [
    ("в пятницу", date(2026, 10, 14), date(2026, 10, 16), False),     # от среды
    ("в пятницу", date(2026, 10, 10), date(2026, 10, 16), False),     # от субботы — следующая неделя
    ("в пятницу", date(2026, 10, 16), date(2026, 10, 23), True),      # сказано в пятницу
    ("в следующую пятницу", date(2026, 10, 14), date(2026, 10, 23), True),
    ("завтра", date(2026, 12, 31), date(2027, 1, 1), False),
    ("15 октября", date(2026, 10, 20), date(2027, 10, 15), False),    # уже прошло — следующий год
    ("к 5-го", date(2026, 10, 20), date(2026, 11, 5), False),
    ("через неделю", date(2026, 10, 14), date(2026, 10, 21), False),
    ("на выходных", date(2026, 10, 14), date(2026, 10, 17), True),
])
def test_relative_dates(phrase, ref, expected, ambiguous):
    m = find_date(Scan(phrase), ref)
    assert m.value == expected
    assert m.ambiguous is ambiguous


@pytest.mark.parametrize("phrase, t, amb", [
    ("в 7 вечера", time(19, 0), False),
    ("в 7", time(7, 0), True),
    ("в 3", time(15, 0), True),
    ("к 18:00", time(18, 0), False),
    ("в 2 часа дня", time(14, 0), False),
    ("в полдень", time(12, 0), False),
])
def test_times(phrase, t, amb):
    m = find_time(Scan(phrase))
    assert (m.start, m.ambiguous) == (t, amb)


def test_number_before_month_is_not_time():
    assert find_time(Scan("к 15 октября")) is None
    assert find_time(Scan("в 2026 году")) is None


def test_normalize_rrule_from_llm():
    assert normalize_rrule("RRULE:freq=weekly;byday=mo") == "FREQ=WEEKLY;BYDAY=MO"
    assert normalize_rrule("FREQ=WEEKLY;UNTIL=20261225T000000Z") == "FREQ=WEEKLY;UNTIL=20261225T000000"
    for bad in ("каждый понедельник", "FREQ=SOMETIMES", "FREQ=WEEKLY;BYDAY=XX"):
        with pytest.raises(ValueError):
            normalize_rrule(bad)


def test_expand_weekly():
    got = expand("FREQ=WEEKLY;BYDAY=MO", datetime(2026, 10, 19, 8), datetime(2026, 11, 1), datetime(2026, 11, 30, 23, 59))
    assert [d.day for d in got] == [2, 9, 16, 23, 30]
    assert all(d.hour == 8 for d in got)


def test_describe():
    assert describe_rrule("FREQ=WEEKLY;BYDAY=MO") == "каждый понедельник"
    assert describe_rrule("FREQ=WEEKLY;BYDAY=FR") == "каждую пятницу"
    assert describe_rrule("FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR") == "по будням"
    assert describe_rrule("FREQ=MONTHLY;BYMONTHDAY=15") == "каждый месяц, 15 числа"
