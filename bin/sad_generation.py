#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate synthetic audio data (SAD) for corpora using OpenAI TTS models.

Note:
    The output directory is structured as follows.

        path/to/outdir/ [default=./data/tts/]
        |- corpus1/
        |  |- voice1/
        |  |  |- uid1.wav
        |  |  |- uid2.wav
        |  |  |- ...
        |  |- voice2/
        |  |- ...
        |- corpus2/
        |- ...

Usage Examples:
    $ sad_generation.py -c commitment_bank --voices nova        # Basic usage.
    $ sad_generation.py -c commitment_bank --voices nova, echo  # Multiple voices.
"""
import asyncio
import os

import backoff
import openai
from tqdm import tqdm

from src.core.app import harness
from src.core.context import Context, get_context
from src.core import keychain
from src.data import tasks

from typing import Any


CONFIG = {corpus: {"text_column": "input_text"} for corpus in tasks.TASKS}
for corpus in ("goemotions", "iemocap", "imdb"):
    CONFIG[corpus]["text_column"] = "text"
CONFIG["commitment_bank"]["text_column"] = "cb_target"


def save_generations(
        corpus: str,
        chunk: tuple[str, dict[str, Any]],
        utterances: list[bytes],
        outdir: str
) -> None:
    log = get_context().log
    for chunk, audio in zip(utterances):
        voice, info = chunk
        outpath = os.path.join(outdir, corpus, voice, info["uid"] + ".wav")
        with open(outpath, "wb") as fd:
            fd.write(audio)
        log.info("wrote: %s", outpath)


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def generate_utterance(
    client: openai.AsyncOpenAI, text: str, voice: str
) -> bytes:
    response = await client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        input=text,
        response_format="wav"
    )
    return response.content


async def generate_utterances(corpus: str, voices: list[str]) -> None:
    client = openai.AsyncOpenAI()
    data_dict = tasks.load_kfold(
        corpus,
        fold=0,
        k=5,
    )
    data = []
    for split in data_dict:
        data += data_dict[split].to_list()
    for chunk in tqdm(chunked(product(voices, data), 10)):
        tasks = [
            generate_utterance(
                row[CONFIG[corpus]["text_column"]], voice
            ) for voice, row in batch
        ]
        results = await asyncio.gather(*tasks)


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "data", "tts"
    )
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    ctx.parser.add_argument("-c", "--corpus", choices=tasks.TASKS, required=True)
    ctx.parser.add_argument("--voices", nargs="+", choices=voices)
    args = ctx.parser.parse_args()
    # Generate audio asynchronously.
    os.environ["OPENAI_API_KEY"] = keychain.get("IACS")
    raise NotImplementedError


if __name__ == "__main__":
    harness(main)
