"""Лимиты Bot API, чтобы бота не заблокировали за флуд.

Официальные ориентиры Telegram: не больше ~30 сообщений в секунду на бота,
не чаще 1 сообщения в секунду в один чат и ~20 в минуту в группу. При
превышении приходит 429 с parameters.retry_after — всё это время бот должен
молчать целиком, а не только в этом чате.

Часы и sleep передаются снаружи, чтобы тесты шли мгновенно и детерминированно.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class TokenBucket:
    """Ведро на capacity токенов, пополняется со скоростью rate в секунду."""

    def __init__(self, rate: float, capacity: float, clock: Clock = time.monotonic):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.clock = clock
        self.updated = clock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def try_acquire(self) -> float:
        """0 — токен взят; иначе сколько секунд подождать до следующего."""
        self._refill()
        if self.ready():
            self.tokens -= 1
            return 0.0
        return self.wait_time()

    def ready(self) -> bool:
        # Допуск на погрешность float: 0.2 * 5 может дать 0.9999999999999999,
        # и без него ожидание «на 1e-17 секунды» зациклилось бы.
        return self.tokens >= 1 - 1e-9

    def wait_time(self) -> float:
        return max((1 - self.tokens) / self.rate, 1e-3)


class RateLimiter:
    def __init__(self, global_rate: float = 25.0, per_chat_rate: float = 1.0, group_per_minute: float = 20.0,
                 clock: Clock = time.monotonic, sleep: Sleep = asyncio.sleep):
        self.clock = clock
        self.sleep = sleep
        self.global_bucket = TokenBucket(global_rate, global_rate, clock)
        self.per_chat_rate = per_chat_rate
        self.group_rate = group_per_minute / 60
        self.chats: dict[int | str, TokenBucket] = {}
        self.paused_until = 0.0
        self.lock = asyncio.Lock()

    def _chat_bucket(self, chat_id: int | str) -> TokenBucket:
        if chat_id not in self.chats:
            if len(self.chats) > 10_000:  # не копим вёдра бесконечно
                self.chats.clear()
            is_group = isinstance(chat_id, int) and chat_id < 0
            rate = self.group_rate if is_group else self.per_chat_rate
            self.chats[chat_id] = TokenBucket(rate, 1, self.clock)
        return self.chats[chat_id]

    def pause(self, seconds: float) -> None:
        """После 429 молчим всем ботом, а не только в одном чате."""
        self.paused_until = max(self.paused_until, self.clock() + seconds)

    async def acquire(self, chat_id: int | str | None = None) -> None:
        async with self.lock:  # очередь честная: кто первый пришёл, тот первый отправил
            while True:
                wait = self.paused_until - self.clock()
                if wait > 0:
                    await self.sleep(wait)
                    continue
                if chat_id is not None:
                    # Сначала проверяем чат, чтобы не тратить глобальный токен впустую.
                    bucket = self._chat_bucket(chat_id)
                    bucket._refill()
                    if not bucket.ready():
                        await self.sleep(bucket.wait_time())
                        continue
                wait = self.global_bucket.try_acquire()
                if wait > 0:
                    await self.sleep(wait)
                    continue
                if chat_id is not None:
                    self._chat_bucket(chat_id).tokens -= 1
                return


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base: float = 1.0
    cap: float = 60.0
    jitter: float = 0.2

    def backoff(self, attempt: int, rnd: Callable[[], float] = random.random) -> float:
        """Экспоненциальная задержка 1, 2, 4, 8… с джиттером, чтобы повторы не шли пачкой."""
        delay = min(self.cap, self.base * 2 ** (attempt - 1))
        return delay * (1 + self.jitter * rnd())
