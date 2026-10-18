"""Запуск: python -m tgcal web | bot | all"""
from __future__ import annotations

import argparse
import asyncio
import logging

import uvicorn

from .config import Settings, load_dotenv
from .db import Repo
from .llm import OllamaLLM
from .telegram.bot import run_bot
from .web import create_app


async def _run_all(settings: Settings, repo: Repo, llm: OllamaLLM) -> None:
    app = create_app(settings, repo, llm)
    server = uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port, log_level="info"))
    bot = asyncio.create_task(run_bot(settings, repo, llm))
    try:
        await server.serve()
    finally:
        bot.cancel()
        await asyncio.gather(bot, return_exceptions=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tgcal", description="Календарь с локальной LLM и Telegram Business")
    p.add_argument("mode", choices=["web", "bot", "all"], nargs="?", default="all",
                   help="web — только календарь, bot — только бот, all — оба (по умолчанию)")
    p.add_argument("--port", type=int, help="порт веб-интерфейса (по умолчанию PORT или 8000)")
    args = p.parse_args(argv)

    load_dotenv()
    settings = Settings.from_env(**({"port": args.port} if args.port else {}))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # иначе в лог попадает URL с токеном бота
    repo = Repo(settings.db_path)
    llm = OllamaLLM(settings.ollama_url, settings.ollama_model, settings.num_ctx, settings.llm_timeout)
    if not llm.available():
        logging.warning("Ollama (%s) или модель %s недоступны: чат работает только на правилах, "
                        "бот копит сообщения до запуска модели", settings.ollama_url, settings.ollama_model)

    if args.mode == "web":
        uvicorn.run(create_app(settings, repo, llm), host=settings.host, port=settings.port)
    elif args.mode == "bot":
        asyncio.run(run_bot(settings, repo, llm))
    else:
        if not settings.telegram_bot_token:
            logging.warning("TELEGRAM_BOT_TOKEN не задан — запускаю только веб-календарь")
            uvicorn.run(create_app(settings, repo, llm), host=settings.host, port=settings.port)
        else:
            asyncio.run(_run_all(settings, repo, llm))


if __name__ == "__main__":
    main()
