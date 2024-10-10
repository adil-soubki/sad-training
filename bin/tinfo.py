#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Print basic task info, check for errors, and summarize the results.

Usage Examples: 
    $ tinfo.py                     # Basic usage.
    $ tinfo.py -t commitment_bank  # Single task.
"""
import logging
import operator
import os
import warnings
from functools import reduce
from typing import Any

import librosa as lr
import numpy as np
import pandas as pd
import transformers as tf
from more_itertools import flatten

from src.core.app import harness
from src.core.context import Context
from src.data import tasks
from src.data.utils import progress_bar_disabled


def get_task_df(task: str, text_model: str, filtered: bool) -> pd.DataFrame:
    ret = []
    cfg = tasks.get_config()[task]
    # Tokenize the dataset.
    tokenizer = tf.AutoTokenizer.from_pretrained(text_model)
    data = tasks.load(task, fold=0, k=5, filtered=filtered)
    def tokenize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return tokenizer(
            rows[cfg.text_column],
            padding=False,
            truncation=False
        )
    data = data.map(tokenize, batched=True, desc="tokenize")
    # Collect a list of entries with errors.
    tts_character_limit = 4096
    text_model_token_limit = tokenizer.model_max_length
    audio_model_length_limit = 30
    for row in flatten([ds.to_list() for ds in data.values()]):
        audio_info = {}
        for key in row:
            if not key.startswith("audio"): continue
            if key == "audio_file": continue
            if not os.path.exists(row[key]["path"]): continue
            voice = key.replace("audio_", "") if key != "audio" else "gold"
            audio_info[f"audio_{voice}_length"] = lr.get_duration(path=row[key]["path"])
            audio_info[f"audio_model_length_limit_{voice}_badness"] = max(
                audio_info[f"audio_{voice}_length"] - audio_model_length_limit, 0
            )
        ret.append({
            "task": task,
            "hexdigest": row["hexdigest"],
            "text": row[cfg.text_column],
            "label": row[cfg.label_column],
            "text_num_chars": len(row[cfg.text_column]),
            "text_num_tokens": len(row["input_ids"]),
            "tts_character_limit": tts_character_limit,
            "text_model_token_limit": tokenizer.model_max_length,
            "audio_model_length_limit": audio_model_length_limit,
            "tts_character_limit_badness": max(
                len(row[cfg.text_column]) - tts_character_limit, 0
            ),
            "text_model_token_limit_badness": max(
                len(row["input_ids"]) - tokenizer.model_max_length, 0
            )
        } | audio_info)
    return pd.DataFrame(ret)


def main(ctx: Context) -> None:
    cfg = tasks.get_config()
    default_tasks = list(cfg)
    default_text_model = "google-bert/bert-base-uncased"
    ctx.parser.add_argument(
        "-t", "--tasks", nargs="+", choices=list(cfg), default=default_tasks
    )
    ctx.parser.add_argument("-m", "--text-model", default=default_text_model)
    ctx.parser.add_argument("-f", "--filtered", action="store_true")
    args = ctx.parser.parse_args()
    # Set up logging.
    tf.logging.set_verbosity_error()
    warnings.filterwarnings(
        action="ignore",
        message="`resume_download` is deprecated and will be removed"
    )
    # Check task data for errors. 
    for task in args.tasks:
        error_types = ["tts_character_limit", "text_model_token_limit"]
        with progress_bar_disabled():
            df = get_task_df(task, args.text_model, args.filtered)
        voices = [c.split("_")[-2] for c in df.columns if "badness" in c and "audio" in c]
        error_types += [f"audio_model_length_limit_{voice}" for voice in voices]
        num_entries = len(df)
        num_errors = len(df[
            reduce(operator.or_, [df[f"{et}_badness"] > 0 for et in error_types])
        ])
        ctx.log.info(f"{'[ ' + task + ' ]':=^80}")
        num_chars, avg_chars = df.text_num_chars.sum(), df.text_num_chars.mean()
        num_tokens, avg_tokens = df.text_num_tokens.sum(), df.text_num_tokens.mean()
        ctx.log.info(f"character count   : {num_chars:,} (avg={avg_chars:,.0f})")
        ctx.log.info(f"token count       : {num_tokens:,} (avg={avg_tokens:,.0f})")
        ctx.log.info(f"OpenAI cost/voice : ${df.text_num_chars.sum() * 3e-5:,.2f}")
        for voice in voices:
            total_secs = df[f"audio_{voice}_length"].sum()
            avg_secs = df[f"audio_{voice}_length"].mean()
            ctx.log.info(
                f"{voice + ' secs':<17} : {total_secs:,.1f} (avg={avg_secs:,.1f})"
            )
        ctx.log.info(
            f"{num_errors} / {num_entries} entries contain an error "
            f"({num_errors / num_entries:.1%})"
        )
        for error_type in error_types:
            errors = df[df[f"{error_type}_badness"] > 0]
            num_errors = len(errors)
            avg_badness = errors[f"{error_type}_badness"].mean()
            avg_badness = 0 if np.isnan(avg_badness) else avg_badness
            ctx.log.info(
                f"    {num_errors} / {num_entries} ({num_errors / num_entries:.1%}) "
                f"contain a {error_type} error (avg_badness={avg_badness:.0f})"
            )
        ctx.log.info('-' * 80)


if __name__ == "__main__":
    harness(main)
