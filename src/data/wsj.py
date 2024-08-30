# -*- coding: utf-8 -*-
import os
from glob import glob
from typing import Any, Literal

import datasets
import pandas as pd
import transformers as tf
from sklearn.model_selection import KFold

from ..core.path import dirparent


WSJ_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "wsj")
Version = Literal["raw", "llama3"]


# XXX: Make this return a Dataset.
def load(version: Version) -> pd.DataFrame:
    assert version in Version.__args__
    if version == "llama3":
        return pd.read_csv(os.path.join(WSJ_DIR, "llama3.csv.gz")).dropna(
            subset=["text", "generation"]
        )
    ret = []
    for path in glob(os.path.join(WSJ_DIR, "**", "*"), recursive=False):
        with open(path, "r", encoding="latin-1") as fd:
            text = fd.read().replace(".START", "").strip()
        ret.append({"fid": int(os.path.basename(path).split("_")[1]), "text": text})
    df = pd.DataFrame(ret)
    assert len(df) == 2499
    return df


def load_kfold(
    version: Version, fold: int, k: int = 5, seed: int = 42
) -> datasets.DatasetDict:
    assert fold >= 0 and fold <= k - 1
    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    wsj = datasets.Dataset.from_pandas(load(version), preserve_index=False)
    train_idxs, test_idxs = list(kf.split(wsj))[fold]
    return datasets.DatasetDict({
        "train": wsj.select(train_idxs),
        "test": wsj.select(test_idxs),
    })


def tokenize_text(text: str, tokenizer: Any) -> list[str]:
    # Spacy tokenizer.
    if not hasattr(tokenizer, "convert_ids_to_tokens"):
        tkns = []
        for tkn in tokenizer(rec["text"]):
            tkns.append(tkn.text)
            if tkn.whitespace_:
                tkns.append(tkn.whitespace_)
    # Transformers tokenizer.
    return list(
        map(
            lambda t: tokenizer.convert_tokens_to_string([t]),
            tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
        )
    )


def load_tokenized(tokenizer_name: str = "openai-community/gpt2") -> pd.DataFrame:
    ret = []
    try:
        tokenizer = tf.AutoTokenizer.from_pretrained(tokenizer_name)
    except OSError:
        tokenizer = spacy.load(tokenizer)
    for rec in load().to_dict("records"):
        tkns = tokenize_text(rec["text"], tokenizer)
        for idx in range(len(tkns) + 1):
            ret.append({
                "fid": rec["fid"],
                "stop_token_index": idx,
                "tokenizer_name": tokenizer_name,
                "text": "".join(tkns[:idx])
            })
    return pd.DataFrame.from_records(ret)
