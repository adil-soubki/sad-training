import asyncio
import os
import sys
import backoff
import pandas as pd
from more_itertools import chunked
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import keychain

import openai
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=keychain.get("IACS"))

def save_audio(audio_contents, corpus_name, voice, filenames):
    os.makedirs(f"audio/clipped/{corpus_name}/{voice}", exist_ok=True)
    for audio, filename in zip(audio_contents, filenames):
        with open(f"audio/clipped/{corpus_name}/{voice}/{voice}_{filename}", "wb") as f:
            f.write(audio)

@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def generate_audio(text: str, voice: str) -> bytes:
    response = await client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        input=text,
        response_format="wav"
    )
    return response.content

async def generate_multiple_audios(texts: list[str], batch_size: int = 10, voice: str = "alloy") -> list[bytes]:
    all_audio_contents = []
    for batch in tqdm(list(chunked(texts, batch_size)), desc="Processing batches"):
        tasks = [generate_audio(text, voice) for text in batch]
        batch_results = await asyncio.gather(*tasks)
        all_audio_contents.extend(batch_results)
    return all_audio_contents

async def generate_and_save_audio(corpus_name: str, texts: list[str], files: list[str], batch_size: int = 10):
    # voices = ["alloy"]
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    for voice in voices:
        audio_contents = await generate_multiple_audios(texts, batch_size, voice)
        save_audio(audio_contents, "cb", voice, files)

async def main():
    cb_data = pd.read_json("../data/cb/annotations.jsonl", lines=True)
    # cb_texts = cb_data["turn"].tolist()
    cb_texts = cb_data["cb_target"].tolist()
    files = cb_data["audio_file"].tolist()
    await generate_and_save_audio("cb", cb_texts, files, batch_size=10)

if __name__ == "__main__":
    asyncio.run(main())