import asyncio
from urllib.parse import parse_qs

import httpx
import pytest

from tgcal.telegram.client import Outbox, SendAsUserForbidden, TelegramClient, TelegramError
from tgcal.telegram.ratelimit import RateLimiter, RetryPolicy, TokenBucket


class FakeClock:
    """Виртуальное время: sleep не ждёт, а двигает часы и запоминает задержку."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def run(coro):
    return asyncio.run(coro)


def test_token_bucket_burst_then_rate():
    c = FakeClock()
    b = TokenBucket(rate=2, capacity=3, clock=c)
    assert [b.try_acquire() for _ in range(3)] == [0, 0, 0]
    assert b.try_acquire() == pytest.approx(0.5)
    c.now += 0.5
    assert b.try_acquire() == 0


def test_per_chat_limit_one_per_second():
    c = FakeClock()
    lim = RateLimiter(global_rate=30, per_chat_rate=1, clock=c, sleep=c.sleep)

    async def go():
        stamps = []
        for _ in range(4):
            await lim.acquire(42)
            stamps.append(c.now)
        return stamps

    assert run(go()) == pytest.approx([0, 1, 2, 3])


def test_global_limit_across_chats():
    c = FakeClock()
    lim = RateLimiter(global_rate=5, per_chat_rate=1, clock=c, sleep=c.sleep)

    async def go():
        for chat in range(20):  # разные чаты: ограничивает только общее ведро
            await lim.acquire(chat)

    run(go())
    # 5 сразу (ёмкость ведра), остальные 15 со скоростью 5/с
    assert c.now == pytest.approx(3.0)


def test_groups_are_slower():
    c = FakeClock()
    lim = RateLimiter(global_rate=30, per_chat_rate=1, group_per_minute=20, clock=c, sleep=c.sleep)

    async def go():
        for _ in range(3):
            await lim.acquire(-100123)

    run(go())
    assert c.now == pytest.approx(6.0)  # 20 в минуту = раз в 3 секунды


def _client(handler, clock, attempts=5):
    lim = RateLimiter(clock=clock, sleep=clock.sleep)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TelegramClient("TEST", lim, RetryPolicy(max_attempts=attempts, jitter=0), http=http, sleep=clock.sleep,
                          base_url="http://tg")


def test_429_waits_retry_after_and_pauses_everyone():
    c = FakeClock()
    calls = []

    def handler(request):
        calls.append(c.now)
        if len(calls) == 1:
            return httpx.Response(429, json={"ok": False, "error_code": 429,
                                             "description": "Too Many Requests: retry after 7",
                                             "parameters": {"retry_after": 7}})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    client = _client(handler, c)

    async def go():
        res = await client.call("sendMessage", chat_id=1, text="привет")
        # Другой чат тоже не пишет раньше окончания паузы.
        assert client.limiter.paused_until == pytest.approx(7)
        await client.call("sendMessage", chat_id=2, text="второй")
        return res

    assert run(go()) == {"message_id": 1}
    assert calls[0] == 0 and calls[1] >= 7
    assert 7 in c.sleeps


def test_server_errors_use_exponential_backoff():
    c = FakeClock()
    n = []

    def handler(request):
        n.append(1)
        if len(n) < 4:
            return httpx.Response(502, json={"ok": False, "error_code": 502, "description": "Bad Gateway"})
        return httpx.Response(200, json={"ok": True, "result": True})

    assert run(_client(handler, c).call("getMe")) is True
    assert c.sleeps == [1, 2, 4]


def test_network_errors_are_retried_then_raised():
    c = FakeClock()

    def handler(request):
        raise httpx.ConnectError("нет сети")

    with pytest.raises(TelegramError):
        run(_client(handler, c, attempts=3).call("getMe"))
    assert c.sleeps == [1, 2]


def test_persistent_429_gives_up():
    c = FakeClock()

    def handler(request):
        return httpx.Response(429, json={"ok": False, "error_code": 429, "description": "flood",
                                         "parameters": {"retry_after": 30}})

    with pytest.raises(TelegramError) as e:
        run(_client(handler, c, attempts=3).call("sendMessage", chat_id=1, text="x"))
    assert e.value.code == 429 and e.value.retry_after == 30


def test_client_errors_are_not_retried():
    c = FakeClock()
    n = []

    def handler(request):
        n.append(1)
        return httpx.Response(403, json={"ok": False, "error_code": 403, "description": "bot was blocked by the user"})

    with pytest.raises(TelegramError):
        run(_client(handler, c).call("sendMessage", chat_id=1, text="x"))
    assert len(n) == 1


def test_sending_as_business_owner_is_forbidden():
    c = FakeClock()

    def handler(request):
        raise AssertionError("запрос не должен уйти")

    with pytest.raises(SendAsUserForbidden):
        run(_client(handler, c).call("sendMessage", chat_id=5, text="привет", business_connection_id="bc1"))


def test_reply_markup_is_json_encoded():
    c = FakeClock()
    seen = {}

    def handler(request):
        seen.update(parse_qs(request.content.decode()))
        return httpx.Response(200, json={"ok": True, "result": {}})

    run(_client(handler, c).call("sendMessage", chat_id=1, text="x",
                                 reply_markup={"inline_keyboard": [[{"text": "Да", "callback_data": "ok:1"}]]}))
    assert '"callback_data": "ok:1"' in seen["reply_markup"][0]


def test_outbox_limits_concurrency():
    in_flight = 0
    peak = 0

    class SlowClient:
        async def call(self, method, **params):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return params["chat_id"]

    async def go():
        box = Outbox(SlowClient(), concurrency=3)
        await box.start()
        futs = [box.submit("sendMessage", chat_id=i, text="x") for i in range(12)]
        results = await asyncio.gather(*futs)
        await box.stop()
        return results

    assert sorted(run(go())) == list(range(12))
    assert peak == 3


def test_outbox_overflow_fails_fast():
    async def go():
        box = Outbox(None, concurrency=1, maxsize=1)  # воркеры не запущены
        box.submit("sendMessage", chat_id=1)
        fut = box.submit("sendMessage", chat_id=2)
        with pytest.raises(TelegramError):
            await fut

    run(go())
