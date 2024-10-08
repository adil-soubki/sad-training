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

    It will only generate files which do not already exist.

Usage Examples:
    $ sad_generation.py -t commitment_bank --voices nova        # Basic usage.
    $ sad_generation.py -t commitment_bank --voices nova, echo  # Multiple voices.
"""
import asyncio
import os

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
        voice: str,
        chunk: list[dict[str, Any]],
        utterances: list[bytes],
        outdir: str
) -> None:
    log = get_context().log
    assert len(chunk) == len(utterances)
    for row, audio in zip(chunk, utterances):
        outpath = os.path.join(outdir, task, voice, f"{row['hexdigest']}.wav")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, "wb") as fd:
            fd.write(audio)
        log.info("wrote: %s", outpath)


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError))
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


def get_missing_data(
        data: list[dict[str, Any]], task: str, voice: str, outdir: str
) -> list[dict[str, Any]]:
    log = get_context().log
    def sad_is_missing(row: dict[str, Any]) -> bool:
        # TODO: Needs to match path in save_generations. Make shared method?
        return not os.path.exists(
            os.path.join(outdir, task, voice, f"{row['hexdigest']}.wav")
        )
    missing_data = list(filter(sad_is_missing, data))
    log.info(
        f"{task} [voice={voice}] is missing sad for "
        f"{len(missing_data)} / {len(data)} hexdigests."
    )
    return missing_data


async def generate_utterances(task: str, voices: list[str], outdir: str) -> None:
    client = openai.AsyncOpenAI()
    text_column = tasks.get_config()[task].text_column
    dataset_dict = tasks.load(task)
    data = []
    for split in dataset_dict:
        data += dataset_dict[split].to_list()
    for voice in voices:
        missing_data = get_missing_data(data, task, voice, outdir)
        for chunk in tqdm(list(chunked(missing_data, n=60))):
            futs = [
                generate_utterance(
                    client, row[text_column], voice
                ) for row in chunk
            ]
            utterances = await asyncio.gather(*futs)
            save_generations(task, voice, chunk, utterances, outdir)


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


if __name__ == "__main__":
    harness(main)
