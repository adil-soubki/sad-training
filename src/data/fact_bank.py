# -*- coding: utf-8 -*-
import dataclasses
import os

import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..core.path import dirparent


FB_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "fb")


@dataclasses.dataclass
class Label:
    source: str
    target: str
    value: str

    def __post_init__(self):
        assert self.value in ("false", "pfalse", "unknown", "ptrue", "true")


def parse_labels(string: str) -> list[Label]:
    ret = []
    for label in string.split(";"):
        label = label.strip()  # Remove whitespace.
        assert label.startswith("(") and label.endswith(")")
        ret.append(Label(*map(lambda s: s.strip(), label[1:-1].split(","))))
    #  return ret
    return list(map(dataclasses.asdict, ret))


def load_df() -> pd.DataFrame:
    ret = []
    for split in ("train", "dev", "test"):
        ret.append(
            pd.read_csv(os.path.join(FB_DIR, f"{split}.csv")).assign(split=split)
        )
    df = pd.concat(ret).reset_index(drop=True)
    df = df.assign(
        input_text=df.input_text.str.replace("source target factuality: ", ""),
        labels=list(map(parse_labels, df.target_text))
    )
    return df[["input_text", "target_text", "labels", "split"]]


def load() -> datasets.Dataset:
    return datasets.Dataset.from_pandas(load_df(), preserve_index=False)


def load_kfold(
    fold: int, k: int = 5, seed: int = 42
) -> datasets.DatasetDict:
    assert fold >= 0 and fold <= k - 1
    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    fb = load()
    train_idxs, test_idxs = list(kf.split(fb))[fold]
    return datasets.DatasetDict({
        "train": fb.select(train_idxs),
        "test": fb.select(test_idxs),
    })
