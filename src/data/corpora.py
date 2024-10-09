# -*- coding: utf-8 -*-
import dataclasses
import functools
import os
from typing import Any, Callable

import datasets

from ..core.path import dirparent
from ..data import (
    commitment_bank, commitment_bank_text_only, fact_bank,
    fantom, goemotions, iemocap, super_glue, wikiface, wsj
)


CMAP = {
    "commitment_bank": commitment_bank,
    "commitment_bank_text_only": commitment_bank_text_only,
    "fact_bank": fact_bank,
    "fantom": fantom,
    "wikiface": wikiface,  # XXX: Delete me.
    "wsj": wsj,  # XXX: Delete me.
}
CORPORA = (
    list(CMAP) +
    list(super_glue.TASK_TO_FN) +
    ["imdb", "iemocap", "goemotions"]
)


# TODO: Move kfold splitting logic out of the corpus modules. Only require
#   modules to implement a load function that returns a cannonical split.
# TODO: This logic should work automatically using the task config. Right now
#   it requires updating every time a new corpus is added.
# TODO: Check that modules have the right methods implemented.
# NOTE: https://stackoverflow.com/a/8719100
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
