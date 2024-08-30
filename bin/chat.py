#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chat with HuggingFace models on the commandline.

Usage Examples:
    $ chat.py                   # No args needed.
    $ chat.py -o path/to/outdir # Custom outdir.
    $ chat.py --model meta-llama/Meta-Llama-3-8B-Instruct

TODO:
    * Support changing the system message.
    * Support reading in a generation config.
    * Support writing chat logs to file.
"""
import datetime
import hashlib
import operator
import os
import time
from typing import Any

import torch
import pandas as pd
import transformers as tf
from datasets import Dataset
from more_itertools import chunked
from tqdm import tqdm

from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from src.data import prompts, wsj


def get_completions(model_name: str) -> list[dict[str, str]]:
    torch.backends.cuda.matmul.allow_tf32 = True  # Improves inference speed.
    torch.compile(mode="reduce-overhead")         # Improves inference speed.
    # Load the model and tokenizer.
    model = tf.AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
    # Prepare the initial input.
    chat = [
        {"role": "system", "content": "You are a helpful and precise assistent."},
    ]
    while True:
        # Get the next user message.
        user_message = input("user      : ")
        if user_message == "q":
            break
        chat.append({"role": "user", "content": user_message})
        # Apply the chat template
        formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # Tokenize the chat.
        inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        # Move the tokenized inputs to the same device the model is on (GPU/CPU).
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        # Generate text from the model.
        pad_token_id = tokenizer.pad_token_id
        if tokenizer.pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=1.0,
            pad_token_id=pad_token_id,
        )
        # Decode the output back to a string
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].size(1):],
            skip_special_tokens=True
        )
        print(f"assistant : {response}")
        chat.append({"role": "assistant", "content": response})
    return chat


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "outputs", "prompting"
    )
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    ctx.parser.add_argument("-m", "--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = ctx.parser.parse_args()
    # Start chatting.
    completions = pd.DataFrame(get_completions(args.model))


if __name__ == "__main__":
    harness(main)
