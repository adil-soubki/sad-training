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
import tempfile
import shutil

from typing import Any
import subprocess
import glob

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

def generate_matcha_utterances(task: str, outdir: str) -> None:
    text_column = tasks.get_config()[task].text_column
    dataset_dict = tasks.load(task)
    data = []
    for split in dataset_dict:
        data += dataset_dict[split].to_list()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary input file
        temp_input_file = os.path.join(temp_dir, f'{task}.txt')
        with open(temp_input_file, 'w') as f:
            for item in data:
                f.write(f"{item[text_column]}\n")
        
        # Run matcha-tts with temporary output directory
        temp_output_dir = os.path.join(temp_dir, f'{task}_matcha')
        subprocess.run(['matcha-tts', '--file', temp_input_file, '--output_folder', temp_output_dir], capture_output=True, text=True)
        
        # Process and move generated files
        hex_digests = [item['hexdigest'] for item in data]
        wav_files = glob.glob(os.path.join(temp_output_dir, '*.wav'))
        
        log = get_context().log
        for hexdigest, wav_file in zip(hex_digests, wav_files):
            outpath = os.path.join(outdir, task, "matcha", f"{hexdigest}.wav")
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            shutil.copy2(wav_file, outpath)
            log.info("wrote: %s", outpath)

        # Clean up PNG files in the current directory
        current_dir = os.getcwd()
        png_files = glob.glob(os.path.join(current_dir, '*.png'))
        for png_file in png_files:
            os.remove(png_file)
            log.info(f"Removed: {png_file}")

async def generate_utterances(task: str, voices: list[str], outdir: str) -> None:
    client = openai.AsyncOpenAI()
    text_column = tasks.get_config()[task].text_column
    dataset_dict = tasks.load(task)
    data = []
    for split in dataset_dict:
        data += dataset_dict[split].to_list()
    for voice in voices:
        if voice == "matcha":
            generate_matcha_utterances(task, outdir)
        else:
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
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "matcha"]
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
