#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
import asyncio
import os

import backoff
import openai

from src.core.context import Context
from src.core.app import harness
from src.data import tasks

from typing import Any


CONFIG = {corpus: {"text_column": "input_text"} for corpus in tasks.TASKS}
for corpus in ("goemotions", "goemotions_ekman", "iemocap", "imdb"):
    CONFIG[corpus]["text_column"] = "text"
CONFIG["commitment_bank"]["text_column"] = "cb_target"


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def generate_utterance(
    client: openai.AsyncOpenAI, text: str, voice: str
) -> dict[str, Any]:
    response = await client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        input=text,
        response_format="wav"
    )
    return response


async def generate_utterances(corpus: str, voices: list[str]) -> None:
    client = openai.AsyncOpenAI()
    data = tasks.load_kfold(
        corpus,
        fold=0,
        k=5,
    )
    raise NotImplementedError


def main(ctx: Context) -> None:
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    ctx.parser.add_argument("-c", "--corpus", choices=tasks.TASKS)
    ctx.parser.add_argument("--voices", nargs="+", choices=voices)
    args = ctx.parser.parse_args()
    # Generate audio asynchronously.
    os.environ["OPENAI_API_KEY"] = keychain.get("IACS")
    raise NotImplementedError


if __name__ == "__main__":
    harness(main)
