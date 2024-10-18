# -*- coding: utf-8 -*-
import os
import re

import datasets
import pandas as pd

from ..core.path import dirparent


# TODO: This module should be "commitment_bank" and the other module should be
#   "commitment_bank_prosody" since this is the more common naming used.
CB_DIR = os.path.join(
    dirparent(os.path.realpath(__file__), 3), "data", "cb_text_only", "original"
)


# XXX: Move to utils.py? This undoes some undesirable aspects of Penn Treebank
#   tokenization (e.g., contractions being separated by a space).
def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?;:(){}[\]%"\'’%])', r'\1', text)
    contraction_suffixes = [
        r"n['’]t",   # n't
        r"['’]m",    # 'm
        r"['’]s",    # 's
        r"['’]re",   # 're
        r"['’]ll",   # 'll
        r"['’]ve",   # 've
        r"['’]d",    # 'd
        r"['’]em",   # 'em (optional, e.g., 'em for 'them')
    ]
    pattern = r'\s+(' + '|'.join(contraction_suffixes) + r')\b'
    text = re.sub(pattern, r'\1', text)
    text = re.sub(r'\$\s+', r'$', text)
    return text


def load_df() -> pd.DataFrame:
    ret = []
    for split in ("train", "dev", "test"):
        path = os.path.join(CB_DIR, f"{split}.jsonl")
        ret.append(pd.read_json(path, lines=True).assign(split=split))
    # NOTE: Using apply here is much slower than using Dataset.map().
    def normalize(row: pd.Series) -> pd.Series:
        # Copy original text and normalize.
        row["text_og"] = row["text"]
        row["text"] = normalize_text(row["text_og"])
        # Extract the relevant span and label.
        assert len(row["targets"]) == 1
        target = row["targets"][0]
        row["span_text"] = normalize_text(target["span_text"])
        row["label"] = target["label"]
        return row
    return pd.concat(ret).apply(normalize, axis=1)


# XXX: CBTO has established split. Fold params are not used.
def load_kfold(fold: int, k: int = 5, seed: int = 42) -> datasets.DatasetDict:
    assert fold == 0, "KFold splitting not implemented"
    fb = datasets.Dataset.from_pandas(load_df(), preserve_index=False)
    return datasets.DatasetDict({
        "train": fb.filter(lambda x: x["split"] == "train"),  # XXX: Not using dev yet.
        "test": fb.filter(lambda x: x["split"] == "test"),
    })
