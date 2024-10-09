# -*- coding: utf-8 -*-
import os

import datasets
import pandas as pd

from ..core.path import dirparent


FB_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "fb")


def load() -> datasets.DatasetDict:
    df = pd.read_csv(os.path.join(FB_DIR, "span", "corpus.csv"))
    return datasets.DatasetDict({
        split: datasets.Dataset.from_pandas(
            df[df.split == split], preserve_index=False
        ) for split in ("train", "test")  # NOTE: Not using dev yet.
    })


# NOTE: FB has established split. Fold params are not used.
def load_kfold(fold: int, k: int = 5, seed: int = 42) -> datasets.DatasetDict:
    assert fold == 0, "KFold splitting not implemented"
    return load()
