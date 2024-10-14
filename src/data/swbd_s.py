# -*- coding: utf-8 -*-
import os

import datasets
import pandas as pd
from sklearn.model_selection import KFold

from ..core.path import dirparent


SS_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "swbd_s")


def load() -> datasets.Dataset:
    df = pd.read_json(os.path.join(SS_DIR, "annotations.jsonl"), lines=True)
    # Add full audio path.
    df = df.assign(
        audio=os.path.join(SS_DIR, "audio") + os.sep + df.file
    )
    # Convert to dataset
    ds = datasets.Dataset.from_pandas(df, preserve_index=False)
    return ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))


def load_kfold(fold: int, k: int = 5, seed: int = 42) -> datasets.DatasetDict:
    assert fold >= 0 and fold <= k - 1
    ds = load()
    # If they ask for one fold, just return the whole dataset.
    if k == 1:
        return datasets.DatasetDict({
            "train": ds,
            "test": ds.select([]),
        })
    # Otherwise do kfold splitting.
    # XXX: Do not honor the given seed. Moving examples across splits
    #   breaks the downsampling code now. Need to fix this later. This
    #   is because it samples and sorts hexdigests within each split
    #   so when splits change the sorting changes and then the selected
    #   hexdigests change. Should fix this but don't want to mess with
    #   how everything is working currently.
    kf = KFold(n_splits=k, random_state=19, shuffle=True)
    train_idxs, test_idxs = list(kf.split(ds))[fold]
    return datasets.DatasetDict({
        "train": ds.select(train_idxs),
        "test": ds.select(test_idxs),
    })
