"""Клиент локальной Ollama и структурированные ответы.

Используется нативный /api/chat, а не OpenAI-совместимый /v1: только в нём
можно задать num_ctx на запрос и передать JSON-схему в `format`, чтобы модель
генерировала ответ по грамматике. Схема — подсказка, а не гарантия, поэтому
ответ всё равно валидируется pydantic и при ошибке переспрашивается.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Модель ответила, но толку нет (невалидный JSON после всех попыток)."""


class LLMUnavailable(LLMError):
    """Ollama не запущена или не отвечает — сообщения стоит обработать позже."""


class LLM(Protocol):
    def chat(self, messages: list[dict], schema: dict | None = None) -> str: ...


class OllamaLLM:
    def __init__(self, base_url: str, model: str, num_ctx: int, timeout: float = 180.0,
                 http: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.http = http or httpx.Client()
        self.last_prompt_tokens: int | None = None

    def chat(self, messages: list[dict], schema: dict | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_ctx": self.num_ctx, "temperature": 0},
        }
        if schema is not None:
            payload["format"] = schema
        try:
            r = self.http.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise LLMUnavailable(f"Ollama недоступна ({self.base_url}): {e}") from e

        # Ollama не падает при переполнении контекста, а молча выкидывает начало промпта.
        # Счётчик токенов промпта — единственный способ это заметить.
        self.last_prompt_tokens = data.get("prompt_eval_count")
        if self.last_prompt_tokens and self.last_prompt_tokens >= self.num_ctx * 0.95:
            log.warning("промпт занял %s токенов из num_ctx=%s — начало могло обрезаться",
                        self.last_prompt_tokens, self.num_ctx)
        content = data.get("message", {}).get("content", "")
        if data.get("done_reason") == "length":
            raise _Truncated(content)
        return content

    def available(self) -> bool:
        try:
            r = self.http.get(f"{self.base_url}/api/tags", timeout=2)
            names = {m.get("name", "") for m in r.json().get("models", [])}
        except Exception:
            return False
        return self.model in names or f"{self.model}:latest" in names


class _Truncated(Exception):
    def __init__(self, partial: str):
        super().__init__("ответ обрезан по лимиту длины")
        self.partial = partial


def extract_json(raw: str) -> str:
    """Модели любят оборачивать JSON в ```json ... ``` и добавлять слова вокруг."""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    a, b = raw.find("{"), raw.rfind("}")
    if a == -1 or b <= a:
        raise ValueError("в ответе нет JSON-объекта")
    return raw[a:b + 1]


def _short_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:5])
    return str(e)[:300]


def structured(llm: LLM, messages: list[dict], model_cls: type[T], max_attempts: int = 3) -> T:
    """Запрос с JSON-схемой, проверка pydantic и повтор с текстом ошибки."""
    schema = model_cls.model_json_schema()
    msgs = list(messages)
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = llm.chat(msgs, schema)
        except _Truncated as e:
            raw, last = e.partial, e
            hint = "Ответ получился слишком длинным и обрезался. Верни короче: только JSON, summary до 300 символов."
        else:
            try:
                return model_cls.model_validate_json(extract_json(raw))
            except (ValidationError, ValueError) as e:
                last = e
                hint = f"Ответ не прошёл проверку: {_short_error(e)}. Исправь и верни только JSON по схеме, без пояснений."
        log.info("LLM попытка %s/%s неудачна: %s", attempt, max_attempts, _short_error(last))
        # В историю кладём обрезанный неудачный ответ, чтобы повторы не раздували контекст.
        msgs = list(messages) + [{"role": "assistant", "content": raw[:1000]}, {"role": "user", "content": hint}]
    raise LLMError(f"модель {max_attempts} раза вернула невалидный ответ: {_short_error(last)}")


class FakeLLM:
    """Для тестов и демо: отдаёт заготовленные ответы по очереди и запоминает запросы."""

    def __init__(self, responses: list[str | dict]):
        self.responses = [r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in responses]
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], schema: dict | None = None) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise LLMUnavailable("FakeLLM: ответы закончились")
        return self.responses.pop(0)
