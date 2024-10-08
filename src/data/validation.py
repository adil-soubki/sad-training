# -*- coding: utf-8 -*-
import itertools
from typing import Any

import pandas as pd
import transformers as tf
from datasets import DatasetDict
from more_itertools import flatten

from src.core.context import get_context
from src.data import tasks
from src.data.utils import progress_bar_disabled


# TODO: Similar to code in bin/tinfo.py. Combine?
def warn_on_audio_length(data: DatasetDict, audio_max_length: int) -> None:
    ctx = get_context()
    audio_lengths = list(
        map(
            lambda audio: len(audio["array"]) / audio["sampling_rate"],
            itertools.chain(*[data[split]["audio"] for split in data])
        )
    )
    longest_audio_length = max(audio_lengths)
    if longest_audio_length <= audio_max_length:
        return
    truncated_lengths = list(
        filter(lambda l: l > audio_max_length, audio_lengths)
    )
    num_errors, num_entries = len(truncated_lengths), sum(map(len, data.values()))
    ctx.log.warning("=" * 80)
    ctx.log.warning(
        #  f"{len(truncated_lengths)} OUT OF {sum(map(len, data.values()))} "
        f"{num_errors} OUT OF {num_entries} ({num_errors / num_entries:.1%}) "
        f"AUDIO CLIPS ARE LONGER THAN AUDIO_MAX_LENGTH ({audio_max_length} "
        "SECONDS) AND WILL BE TRUNCATED."
    )
    ctx.log.warning(f"TRUNCATED LENGTHS: {truncated_lengths}")
    ctx.log.warning("=" * 80)


# TODO: Similar to code in bin/tinfo.py. Combine?
# TODO: Also similar to code in src/data/tasks.py.
def warn_on_text_length(
    data: DatasetDict, task: str, text_max_length: int, text_model: str
) -> None:
    ctx = get_context()
    cfg = tasks.get_config()[task]
    tokenizer = tf.AutoTokenizer.from_pretrained(text_model)
    def tokenize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return tokenizer(
            rows[cfg.text_column],
            padding=False,
            truncation=False
        )
    with progress_bar_disabled():
        data = data.map(tokenize, batched=True, desc="tokenize")
    rows = []
    for row in itertools.chain(*[ds.to_list() for ds in data.values()]):
        rows.append({
            "hexdigest": row["hexdigest"],
            "text_num_tokens": len(row["input_ids"]),
            "text_badness": max(
                len(row["input_ids"]) - text_max_length, 0
            )
        })
    df = pd.DataFrame(rows)
    assert len(df) == sum(map(len, data.values()))
    num_entries = len(df)
    num_errors = len(df[df.text_num_tokens > text_max_length])
    if not num_errors:
        return  # No issues to report.
    avg_badness = df[df.text_num_tokens > text_max_length].text_badness.mean()
    ctx.log.warning("=" * 80)
    ctx.log.warning(
        f"{num_errors} OUT OF {num_entries} ({num_errors / num_entries:.1%}) "
        f"INPUT TEXTS ARE LONGER THAN TEXT_MAX_LENGTH ({text_max_length} "
        "TOKENS) AND WILL BE TRUNCATED."
    )
    ctx.log.warning(f"AVERAGE TOKENS TRUNCATED: {avg_badness:.0f}")
    ctx.log.warning("=" * 80)
