# -*- coding: utf-8 -*-
import dataclasses
import functools
import os
from typing import Any, Callable

import datasets

from ..core.path import dirparent
from ..data import (
    commitment_bank, fact_bank, fantom, goemotions,
    iemocap, super_glue, wikiface, wsj
)


CMAP = {
    "commitment_bank": commitment_bank,
    "fact_bank": fact_bank,  # NOTE: Not done.
    "fantom": fantom,
    "wikiface": wikiface,  # XXX: Delete me.
    "wsj": wsj,  # XXX: Delete me.
}
CORPORA = (
    list(CMAP) +
    list(super_glue.TASK_TO_FN) +
    ["imdb", "iemocap", "goemotions"]
)


def load(
    corpus: str, fold: int = 0, k: int = 5, seed: int = 42, **data_kwargs: Any
) -> datasets.DatasetDict:
    if corpus in CMAP:
        return CMAP[corpus].load_kfold(**data_kwargs, fold=fold, k=k, seed=seed)
    elif corpus in super_glue.TASKS:
        return super_glue.load(corpus, **data_kwargs)
    elif corpus == "imdb":
        imdb = datasets.load_dataset("stanfordnlp/imdb")
        del imdb["unsupervised"]  # Unused and unlabeled.
        return imdb
    elif corpus == "iemocap":
        return iemocap.load()
    elif corpus == "goemotions":
        return goemotions.load(**data_kwargs)
    raise NotImplementedError
