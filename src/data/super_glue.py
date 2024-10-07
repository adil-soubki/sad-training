# -*- coding: utf-8 -*-
import getpass
import uuid
from typing import Any

import datasets


TASKS = [
    "boolq", "cb", "copa", "multirc", "record", "rte",
    "wic", "wsc", "wsc.fixed", "axb", "axg"
]


# XXX: These are all very similar... condense?
def preprocess_boolq(example: dict[str, Any]) -> dict[str, Any]:
    example["input_text"] = "\n".join([
        f"passage: {example['passage']}",
        f"question: {example['question']}"
    ])
    return example


def preprocess_wic(example: dict[str, Any]) -> dict[str, Any]:
    example["input_text"] = "\n".join([
        f"sentence1: {example['sentence1']}",
        f"sentence2: {example['sentence2']}",
        f"word: {example['word']}"
    ])
    return example


def preprocess_wsc(example: dict[str, Any]) -> dict[str, Any]:
    # NOTE: There are two rows duplicated in this data.
    #   (1) 168 == 216
    #   (2) 374 == 505
    # NOTE: Sometimes the coref has whitespace (e.g., "Mr. Singer")
    #  tkns = example["text"].split()
    #  for sdx in (2, 1):
    #      index, text = example[f"span{sdx}_index"], example[f"span{sdx}_text"]
    #      tkns = tkns[:index] + [text] + tkns[index + len(text.split()):]
    #      assert tkns[index] == text
    #      tkns[index] = f"*{text}*"
    example["input_text"] = "\n".join([
        #  f"text: {' '.join(tkns)}",
        f"text: {example['text']}",
        f"span1: {example['span1_text']}",
        f"span2: {example['span2_text']}"
    ])
    return example


TASK_TO_FN = {
    "boolq": preprocess_boolq,
    "wic": preprocess_wic,
    "wsc": preprocess_wsc,
}


def load(task: str, **data_kwargs: Any) -> datasets.DatasetDict:
    if task not in TASKS:
        raise ValueError(f"{task} is not in super_glue: {TASKS}")
    if task not in TASK_TO_FN:
        raise NotImplementedError(f"preprocessing not implemented for {task}")
    # XXX: NFS was holding the cache file preventing other processes from opening it.
    #      I solved this by just directing the cache directory to a random tmp path.
    data = datasets.load_dataset(
        "super_glue", task,
        # XXX: /tmp/ is full on the cluster so until this is fixed we put it here.
        #  cache_dir=f"/tmp/{uuid.uuid4()}",
        cache_dir=f"/home/{getpass.getuser()}/scratch/tmp/{uuid.uuid4()}",
        trust_remote_code=True,
        **data_kwargs
    ).map(TASK_TO_FN[task], desc="adding input_text column")
    # NOTE: The test splits are unlabeled so we replace test with validation splits.
    data["test"] = data["validation"]
    del data["validation"]
    return data
