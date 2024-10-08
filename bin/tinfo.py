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
import warnings
from functools import reduce
from typing import Any

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
    for row in flatten([ds.to_list() for ds in data.values()]):
        ret.append({
            "task": task,
            "hexdigest": row["hexdigest"],
            "text": row[cfg.text_column],
            "label": row[cfg.label_column],
            "text_num_chars": len(row[cfg.text_column]),
            "text_num_tokens": len(row["input_ids"]),
            "tts_character_limit": tts_character_limit,
            "text_model_token_limit": tokenizer.model_max_length,
            "tts_character_limit_badness": max(
                len(row[cfg.text_column]) - tts_character_limit, 0
            ),
            "text_model_token_limit_badness": max(
                len(row["input_ids"]) - tokenizer.model_max_length, 0
            )
        })
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
    error_types = ("tts_character_limit", "text_model_token_limit")
    for task in args.tasks:
        with progress_bar_disabled():
            df = get_task_df(task, args.text_model, args.filtered)
        num_entries = len(df)
        num_errors = len(df[
            reduce(operator.or_, [df[f"{et}_badness"] > 0 for et in error_types])
        ])
        ctx.log.info(f"{'[ ' + task + ' ]':=^80}")
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
