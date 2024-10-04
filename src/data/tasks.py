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


@postprocess(add_sad)
@postprocess(add_hexdigests)
def load(task: str, fold: int = 0, k: int = 5, seed: int = 42) -> DatasetDict:
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
