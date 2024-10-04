#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate synthetic audio data (SAD) for corpora using OpenAI TTS models.

Note:
    The output directory is structured as follows.

        path/to/outdir/ [default=./data/sad/]
        |- task1/
        |  |- voice1/
        |  |  |- uid1.wav
        |  |  |- uid2.wav
        |  |  |- ...
        |  |- voice2/
        |  |- ...
        |- task2/
        |- ...

Usage Examples:
    $ sad_generation.py -t commitment_bank --voices nova        # Basic usage.
    $ sad_generation.py -t commitment_bank --voices nova, echo  # Multiple voices.
"""
import asyncio
import os
from itertools import product

import backoff
import openai
from more_itertools import chunked
from tqdm import tqdm

from src.core.app import harness
from src.core.context import Context, get_context
from src.core import keychain
from src.data import tasks

from typing import Any


def save_generations(
        task: str,
        chunk: tuple[str, dict[str, Any]],
        utterances: list[bytes],
        outdir: str
) -> None:
    log = get_context().log
    for tup, audio in zip(chunk, utterances):
        voice, info = tup
        outpath = os.path.join(outdir, task, voice, f"{info['hexdigest']}.wav")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
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


async def generate_utterances(task: str, voices: list[str], outdir: str) -> None:
    client = openai.AsyncOpenAI()
    text_column = tasks.get_config()[task].text_column
    dataset_dict = tasks.load(task)
    data = []
    for split in dataset_dict:
        data += dataset_dict[split].to_list()
    for chunk in tqdm(chunked(product(voices, data), 10)):
        futs = [
            generate_utterance(
                client, row[text_column], voice
            ) for voice, row in chunk
        ]
        utterances = await asyncio.gather(*futs)
        save_generations(task, chunk, utterances, outdir)


def main(ctx: Context) -> None:
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    ctx.parser.add_argument("-t", "--tasks", nargs="+", required=True)
    ctx.parser.add_argument("--voices", nargs="+", choices=voices, required=True)
    ctx.parser.add_argument("-o", "--outdir", default=tasks.SAD_DIR)
    args = ctx.parser.parse_args()
    # Generate audio asynchronously.
    tasks_config = tasks.get_config()
    os.environ["OPENAI_API_KEY"] = keychain.get("IACS")
    for task in args.tasks:
        if task not in tasks_config:
            ctx.parser.error(f"unknown task: '{task}' {list(tasks_config)}")
        ctx.log.info("Generating SAD for %s", task)
        asyncio.run(
            generate_utterances(
                task,
                args.voices,
                args.outdir
            )
        )
    ctx.log.info("COMPLETE")


if __name__ == "__main__":
    harness(main)
