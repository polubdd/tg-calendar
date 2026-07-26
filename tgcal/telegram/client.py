"""Тонкий клиент Bot API поверх httpx: лимиты, 429, повторы и очередь отправки.

Работаем только через официальный Bot API. Никаких userbot/MTProto-клиентов
под личным аккаунтом: это нарушает правила Telegram и ведёт к бану аккаунта,
а бизнес-подключение даёт ровно нужное — бот видит переписку, которую ему
разрешил владелец, и может отключиться в один клик в настройках Telegram.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from .ratelimit import RateLimiter, RetryPolicy, Sleep

log = logging.getLogger(__name__)

# Методы, которые что-то отправляют или меняют в чатах: только они идут через лимитер.
SEND_METHODS = {
    "sendMessage", "editMessageText", "editMessageReplyMarkup", "deleteMessage", "sendChatAction",
    "answerCallbackQuery", "sendPhoto", "sendDocument", "forwardMessage", "copyMessage",
}


class TelegramError(Exception):
    def __init__(self, method: str, code: int, description: str, retry_after: float | None = None):
        super().__init__(f"{method}: {code} {description}")
        self.code = code
        self.description = description
        self.retry_after = retry_after


class SendAsUserForbidden(PermissionError):
    """Попытка написать от имени владельца бизнес-аккаунта."""


class TelegramClient:
    def __init__(self, token: str, limiter: RateLimiter | None = None, policy: RetryPolicy | None = None,
                 http: httpx.AsyncClient | None = None, sleep: Sleep = asyncio.sleep,
                 base_url: str = "https://api.telegram.org"):
        self.url = f"{base_url}/bot{token}/"
        self.limiter = limiter or RateLimiter(sleep=sleep)
        self.policy = policy or RetryPolicy()
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(40.0, connect=10.0))
        self.sleep = sleep

    async def call(self, method: str, **params: Any) -> Any:
        if method in SEND_METHODS and params.get("business_connection_id"):
            # Сообщение с business_connection_id уйдёт собеседнику от имени владельца.
            # Календарю это не нужно, поэтому запрещено на уровне клиента, а не настройкой.
            raise SendAsUserForbidden(f"{method} от имени владельца бизнес-аккаунта запрещён")
        params = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                  for k, v in params.items() if v is not None}
        timeout = None
        if method == "getUpdates":
            timeout = httpx.Timeout(float(params.get("timeout", 0)) + 15, connect=10.0)

        attempt = 0
        while True:
            attempt += 1
            if method in SEND_METHODS:
                await self.limiter.acquire(params.get("chat_id"))
            try:
                kwargs = {"timeout": timeout} if timeout else {}
                r = await self.http.post(self.url + method, data=params, **kwargs)
                data = r.json()
            except (httpx.TransportError, ValueError) as e:
                if attempt >= self.policy.max_attempts:
                    raise TelegramError(method, 0, f"сеть: {e}") from e
                delay = self.policy.backoff(attempt)
                log.warning("%s: сетевая ошибка (%s), повтор через %.1f с", method, e, delay)
                await self.sleep(delay)
                continue

            if data.get("ok"):
                return data.get("result")
            code = data.get("error_code", r.status_code)
            desc = data.get("description", "")
            retry_after = (data.get("parameters") or {}).get("retry_after")

            if code == 429:
                wait = float(retry_after or self.policy.backoff(attempt))
                self.limiter.pause(wait)
                if attempt >= self.policy.max_attempts:
                    raise TelegramError(method, code, desc, wait)
                log.warning("%s: 429, Telegram просит подождать %s с", method, wait)
                await self.sleep(wait)
                continue
            if code >= 500 and attempt < self.policy.max_attempts:
                delay = self.policy.backoff(attempt)
                log.warning("%s: %s %s, повтор через %.1f с", method, code, desc, delay)
                await self.sleep(delay)
                continue
            # 400/403 и т.п. повторять бессмысленно: неверный запрос или бот заблокирован.
            raise TelegramError(method, code, desc, retry_after)

    async def get_updates(self, offset: int, timeout: int, allowed_updates: list[str]) -> list[dict]:
        return await self.call("getUpdates", offset=offset, timeout=timeout, allowed_updates=allowed_updates)

    async def close(self) -> None:
        await self.http.aclose()


class Outbox:
    """Очередь исходящих с ограниченным числом одновременных запросов.

    Обработчики апдейтов не ждут отправки: кладут задачу и идут дальше,
    а воркеры отправляют с учётом лимитов. Очередь ограничена, чтобы при
    долгой паузе после 429 не копить память бесконечно.
    """

    def __init__(self, client: TelegramClient, concurrency: int = 4, maxsize: int = 1000):
        self.client = client
        self.concurrency = concurrency
        self.queue: asyncio.Queue = asyncio.Queue(maxsize)
        self.workers: list[asyncio.Task] = []

    async def start(self) -> None:
        self.workers = [asyncio.create_task(self._worker()) for _ in range(self.concurrency)]

    async def stop(self, drain: bool = True) -> None:
        if drain:
            await self.queue.join()
        for w in self.workers:
            w.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)

    def submit(self, method: str, **params) -> asyncio.Future:
        fut = asyncio.get_running_loop().create_future()
        # Большинство отправок — «выстрелил и забыл»; ошибка уже залогирована воркером.
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())
        try:
            self.queue.put_nowait((method, params, fut))
        except asyncio.QueueFull:
            fut.set_exception(TelegramError(method, 0, "очередь отправки переполнена"))
        return fut

    async def _worker(self) -> None:
        while True:
            method, params, fut = await self.queue.get()
            try:
                result = await self.client.call(method, **params)
                if not fut.done():
                    fut.set_result(result)
            except Exception as e:
                log.warning("не отправлено %s: %s", method, e)
                if not fut.done():
                    fut.set_exception(e)
            finally:
                self.queue.task_done()
