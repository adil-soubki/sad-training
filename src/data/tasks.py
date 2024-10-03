# -*- coding: utf-8 -*-
import dataclasses
import functools
import os
from typing import Any, Callable

import datasets

from ..core.path import dirparent
from ..data import (
    commitment_bank, fact_bank, fantom, goemotions, iemocap, super_glue, wikiface, wsj
)


# TODO: Make a Task interface? Kind of hacky atm.
DMAP = {
    "commitment_bank": commitment_bank,
    "fact_bank": fact_bank,  # NOTE: Not done.
    "fantom": fantom,
    "wikiface": wikiface,  # XXX: Delete me.
    "wsj": wsj,  # XXX: Delete me.
}
TASKS = list(DMAP) + list(super_glue.TASK_TO_FN) + ["imdb", "iemocap", "goemotions"]


# Configuration for adding consistent column names.
TEXT_COLUMN = "task_text"
LABEL_COLUMN = "task_label"
CONFIG = {
    task: {
        "text_column": "input_text",
        "label_column": "label"
    } for task in TASKS
}
for task in ("goemotions", "iemocap", "imdb"):
    CONFIG[task]["text_column"] = "text"
for task in ("fantom", "goemotions"):
    CONFIG[task]["label_column"] = "label_name"
CONFIG["commitment_bank"]["text_column"] = "cb_target"
CONFIG["commitment_bank"]["label_column"] = "cb_val_float"


def postprocess(
    pfn: Callable[[str, datasets.DatasetDict], datasets.DatasetDict]
) -> Callable[..., datasets.DatasetDict]:
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
        fn: Callable[..., datasets.DatasetDict]
    ) -> Callable[..., datasets.DatasetDict]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> datasets.DatasetDict:
            return pfn(args[0], fn(*args, **kwargs))
        return wrapper
    return decorator


def add_consistent_column_names(
    task: str, ddict: datasets.DatasetDict
) -> datasets.DatasetDict:
    def add_columns(row: dict[str, Any]) -> dict[str, Any]:
        row[TEXT_COLUMN] = row[CONFIG[task]["text_column"]]
        row[LABEL_COLUMN] = row[CONFIG[task]["label_column"]]
        return row
    return ddict.map(add_columns, desc="adding consistent columns")


def add_hids(task: str, ddict: datasets.DatasetDict) -> datasets.DatasetDict:
    """
    Adds hash ids (hids) to a DatasetDict.

    Note:
        These hids are used to map rows of the dataset to audio files. Rows
        that exactly match by both TEXT_COLUMN and LABEL_COLUMN will be
        assigned the same hid.

    Args:
        task (str): Task identifier
        ddict (DatasetDict): The DatasetDict.

    Returns:
        The given DatasetDict with an added "hid" column.
    """
    import json
    import hashlib

    if task == "commitment_bank":
        # Already has a unique identifier.
        def add_hid(row: dict[str, Any]) -> dict[str, Any]:
            row["hid"] = row["audio_file"].replace(".wav", "")
            return row
        return ddict.map(add_hid, desc="adding hids")
    assert all([TEXT_COLUMN in cols for cols in ddict.column_names.values()])
    assert all([LABEL_COLUMN in cols for cols in ddict.column_names.values()])
    # Dump each row to key-sorted json and add sha1 hexdigest.
    hexdigests = set()
    def add_hexdigest(row: dict[str, Any]) -> dict[str, Any]:
        row["hexdigest"] = hashlib.sha1(
            json.dumps({
                TEXT_COLUMN: row[TEXT_COLUMN],
                LABEL_COLUMN: row[LABEL_COLUMN]
            }, sort_keys=True).encode("utf-8")
        ).hexdigest()
        hexdigests.add(row["hexdigest"])
        return row
    ddict = ddict.map(add_hexdigest)
    # Sort all generated hexdigests so their indices can be used as hids.
    hexdigests = sorted(hexdigests)
    # NOTE: Duplicates are ok, actually.
    #   assert len(hexdigests) == sum(map(len, ddict.values()))
    # Add hids by index in the sorted hexdigest list.
    def add_hid(row: dict[str, Any]) -> dict[str, Any]:
        row["hid"] = hexdigests.index(row["hexdigest"])
        return row
    return ddict.map(add_hid, desc="adding hids")


# XXX: Use this in bin/sad_generation.py
# XXX: Expects the task name (as in the config) not the dataset name (as in the
#   TASKS list in this file). So, fantom wouldn't work, it would need fantom_bin
#   or fantom_mc. This means it won't currently work for those datasets. I think
#   the solution is to do away with dataset names altogether but broken for now.
SAD_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "sad")
def add_sad(task: str, ddict: datasets.DatasetDict) -> datasets.DatasetDict:
    from glob import glob

    voices = list(map(os.path.basename, glob(os.path.join(SAD_DIR, task, "*"))))
    def add_voices(row: dict[str, Any]) -> dict[str, Any]:
        for voice in voices:
            row[f"audio_{voice}"] = os.path.join(
                SAD_DIR, task, voice, f"{row['hid']}.wav"
            )
        return row
    ddict = ddict.map(add_voices, desc=f"add sad voices {voices}")
    for voice in voices:
        ddict = ddict.cast_column(f"audio_{voice}", datasets.Audio(sampling_rate=16_000))
    return ddict


# TODO: Delete this? We aren't using k-fold cross-validation for this work.
@postprocess(add_sad)
@postprocess(add_hids)
@postprocess(add_consistent_column_names)
def load_kfold(
    name: str, fold: int, k: int = 5, seed: int = 42, **data_kwargs: Any
) -> datasets.DatasetDict:
    if name in DMAP:
        return DMAP[name].load_kfold(**data_kwargs, fold=fold, k=k, seed=seed)
    # If we are asking for the first fold just return the default splits.
    elif name in super_glue.TASKS and fold == 0:
        return load(name, seed, **data_kwargs)
    elif name in ("imdb", "iemocap", "goemotions") and fold == 0:
        return load(name, seed, **data_kwargs)
    raise NotImplementedError


# XXX: In other places this returns Dataset but here it returns a DatasetDict.
def load(name: str, seed: int = 42, **data_kwargs: Any) -> datasets.DatasetDict:
    if name in DMAP:
        return DMAP[name].load_kfold(**data_kwargs, fold=0, k=5, seed=seed)
    elif name in super_glue.TASKS:
        return super_glue.load(name, **data_kwargs)
    elif name == "imdb":
        imdb = datasets.load_dataset("stanfordnlp/imdb")
        del imdb["unsupervised"]  # Unused and unlabeled.
        return imdb
    elif name == "iemocap":
        return iemocap.load()
    elif name == "goemotions":
        return goemotions.load(**data_kwargs)
    raise NotImplementedError
