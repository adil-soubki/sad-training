# -*- coding: utf-8 -*-
import dataclasses
import functools
import json
import os
from argparse import Namespace
from typing import Any, Callable, Self

from datasets import Audio, DatasetDict

from ..data import corpora
from ..core.path import dirparent


_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "tasks.json"
)
SAD_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "sad")


# NOTE: Changing the text or label column for any task in the config will change
#   the synthetic audio file mappings since this are created by hashing those
#   columns. If this happens, it will fail to load the data with FileNotFoundError.
# TODO: Add audio_column for datasets with gold audio?
@dataclasses.dataclass
class TaskConfig:
    name: str
    dataset: str
    dataset_kwargs: dict[str: Any]
    text_column: str
    label_column: str

    @classmethod
    def from_json(cls, dct: dict[str, Any]) -> Self:
        return cls(**{fld.name: dct[fld.name] for fld in dataclasses.fields(cls)})


def load_config(path: str) -> dict[str, TaskConfig]:
    ret = {}
    with open(path, "r") as fd:
        dct = json.load(fd)
    return {k: TaskConfig.from_json(v | {"name": k}) for k, v in dct.items()}


_CONFIG = None


def get_config() -> dict[str, TaskConfig]:
    global _CONFIG
    if not _CONFIG:
        _CONFIG = load_config(_CONFIG_PATH)
    return _CONFIG


def postprocess(
    pfn: Callable[[str, DatasetDict], DatasetDict]
) -> Callable[..., DatasetDict]:
    """
    A decorator that adds a postprocessing step to any function that returns a
    DatasetDict. It requires an argument containing a function that takes in
    a DatasetDict and returns a DatasetDict.

    Args:
        pfn (Callable[[str, DatasetDict], DatasetDict): The postprocessing function.

    Returns:
        A decorator that calls `pfn` on the output of the decorated function.
    """
    def decorator(
        fn: Callable[..., DatasetDict]
    ) -> Callable[..., DatasetDict]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> DatasetDict:
            return pfn(args[0], fn(*args, **kwargs))
        return wrapper
    return decorator


# XXX: I think the hexdigest should not use the label column at all when
#   hashing. That or we should add another hexdigest that just uses the
#   text column. The reason for this is that what actually matters is what
#   we send to the tts system and for tasks that have multiple versions
#   with the same text and different labels (e.g., goemotions and
#   goemotions_ekman) they should share audio files instead of us double
#   paying for generations.
def add_hexdigests(task: str, ddict: DatasetDict) -> DatasetDict:
    """
    Adds hexdigests (sha1 hash ids) to a DatasetDict.

    Note:
        These hexdigests are used to map rows of the dataset to audio files. 
        Rows that exactly match by both `text_column` and `label_column` will
        be assigned the same hexdigest.

    Args:
        task (str): Task identifier
        ddict (DatasetDict): The DatasetDict.

    Returns:
        The given DatasetDict with an added "hexdigest" column.
    """
    import json
    import hashlib

    if task == "commitment_bank":
        # Already has a unique identifier.
        def add_hexdigest(row: dict[str, Any]) -> dict[str, Any]:
            row["hexdigest"] = row["audio_file"].replace(".wav", "")
            return row
        return ddict.map(add_hexdigest, desc="adding hexdigests")
    cfg = get_config()[task]
    text_column, label_column = cfg.text_column, cfg.label_column
    assert all([text_column in cols for cols in ddict.column_names.values()])
    assert all([label_column in cols for cols in ddict.column_names.values()])
    # Dump each row to key-sorted json and add sha1 hexdigest.
    hexdigests = set()
    def add_hexdigest(row: dict[str, Any]) -> dict[str, Any]:
        row["hexdigest"] = hashlib.sha1(
            json.dumps({
                text_column: row[text_column],
                label_column: row[label_column]
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()
        hexdigests.add(row["hexdigest"])
        return row
    return ddict.map(add_hexdigest, desc="adding hexdigests")


def add_sad(task: str, ddict: DatasetDict) -> DatasetDict:
    from glob import glob

    voices = list(map(os.path.basename, glob(os.path.join(SAD_DIR, task, "*"))))
    def add_voices(row: dict[str, Any]) -> dict[str, Any]:
        for voice in voices:
            row[f"audio_{voice}"] = os.path.join(
                SAD_DIR, task, voice, f"{row['hexdigest']}.wav"
            )
        return row
    if voices:
        ddict = ddict.map(add_voices, desc=f"add sad voices {voices}")
    for voice in voices:
        ddict = ddict.cast_column(f"audio_{voice}", Audio(sampling_rate=16_000))
    return ddict


def filter_by_char_limit(
    task: str, ddict: DatasetDict, char_limit: int = 4096
) -> DatasetDict:
    cfg = get_config()[task]
    def is_below_char_limit(row: dict[str, Any]) -> bool:
        return len(row[cfg.text_column]) < char_limit
    return ddict.filter(
        is_below_char_limit,
        desc=f"filter by char_limit ({char_limit})"
    )


def filter_by_token_limit(
    task: str, ddict: DatasetDict, text_model: str = "google-bert/bert-base-uncased"
) -> DatasetDict:
    import transformers as tf

    cfg = get_config()[task]
    tokenizer = tf.AutoTokenizer.from_pretrained(text_model)
    token_limit = tokenizer.model_max_length
    def is_below_token_limit(row: dict[str, Any]) -> bool:
        token_ids = tokenizer(
            row[cfg.text_column], padding=False, truncation=False
        )["input_ids"]
        return len(token_ids) < token_limit
    verbosity_og = tf.logging.get_verbosity()
    tf.logging.set_verbosity_error()
    ret = ddict.filter(
        is_below_token_limit,
        desc=f"filter by token_limit ({token_limit})"
    )
    tf.logging.set_verbosity(verbosity_og)
    return ret


def downsample(task, ddict: DatasetDict) -> DatasetDict:
    text_column = get_config()[task].text_column
    openai_cost_limit = 10.0     # Dollars
    openai_cost_per_char = 3e-5  # Dollars
    num_chars = sum([sum(map(len, ds[text_column])) for ds in ddict.values()])
    num_entries = sum(ddict.num_rows.values())
    avg_chars_per_entry = num_chars / num_entries
    avg_cost_per_entry = avg_chars_per_entry * openai_cost_per_char
    estimated_cost = avg_cost_per_entry * num_entries
    # NOTE: IMDB has long entries so this gets us closer to $10 after filtering.
    if task == "imdb":
        openai_cost_limit = 17.0     # Dollars
    # If it would cost more than $25 sample it down to $10.
    if estimated_cost < 25.0:
        return ddict  # Don't downsample cheaper datasets.
    num_entries_to_sample = round(openai_cost_limit / avg_cost_per_entry)
    for split in ddict:
        split_pct = ddict[split].num_rows / num_entries
        num_entries_wanted = round(num_entries_to_sample * split_pct)
        hexdigests = sorted(ddict[split]["hexdigest"])[:num_entries_wanted]
        is_selected_hexdigest = lambda e: e["hexdigest"] in hexdigests
        ddict[split] = ddict[split].filter(
            is_selected_hexdigest,
            desc=f"downsampling {split}",
        )
        assert sorted(ddict[split]["hexdigest"]) == hexdigests
    return ddict


@postprocess(add_hexdigests)
def load_unfiltered(
    task: str, fold: int = 0, k: int = 5, seed: int = 42
) -> DatasetDict:
    if task not in get_config():
        raise ValueError(f"unknown task: '{task}' not in {list(get_config())}")
    task_config = get_config()[task]
    return corpora.load(
        task_config.dataset,
        **task_config.dataset_kwargs,
        fold=fold,
        k=k,
        seed=seed
    )


@postprocess(filter_by_token_limit)
@postprocess(filter_by_char_limit)
@postprocess(downsample)
def load_filtered(
    task: str, fold: int = 0, k: int = 5, seed: int = 42
) -> DatasetDict:
    return load_unfiltered(task, fold, k, seed)


@postprocess(add_sad)
def load(
    task: str, fold: int = 0, k: int = 5, seed: int = 42, filtered: bool = True
):
    if filtered:
        return load_filtered(task, fold, k, seed)
    return load_unfiltered(task, fold, k, seed)
