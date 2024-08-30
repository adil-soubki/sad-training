# -*- coding: utf-8 -*-
import os
from glob import glob
from operator import itemgetter
from typing import Any, Iterable

import datasets
import pandas as pd
from more_itertools import windowed
from sklearn.model_selection import StratifiedKFold

from ..core.path import dirparent


WF_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "wikiface")


def set_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"face_act": "face_act_adil"})

    def fn(row: pd.Series) -> pd.Series:
        # No double entries.
        assert (pd.isna(row.face_act_adil) and not pd.isna(row.face_act_shyne)) or (
            not pd.isna(row.face_act_adil) and pd.isna(row.face_act_shyne)
        ), row
        # get the annotation.
        face_act = (
            row.face_act_adil if not pd.isna(row.face_act_adil) else row.face_act_shyne
        )
        assert not pd.isna(face_act)
        row["face_act"] = face_act
        row["label"] = face_act
        return row

    return df.apply(fn, axis=1)


def conversation_windows(
    data: pd.DataFrame, hlen: int
) -> Iterable[tuple[int, list[dict[str, Any]], int]]:
    padding = [None] * (hlen - 1)
    for cid, df in data.groupby("conversation_id"):
        for wdx, win in enumerate(
            windowed(padding + df.to_dict("records"), n=hlen, step=1)
        ):
            win = [w for w in win if w]
            yield cid, win, wdx


def set_hlen(df: pd.DataFrame, hlen: int) -> pd.DataFrame:
    ret = []
    #  df = df.sort_values(
    #      ["conversation_id", "timestamp", "reply_to", "sentence_index"],
    #      na_position="first"
    #  )
    for cid, win, wdx in conversation_windows(df, hlen):
        ret.append(
            {
                "new_input_text": "\n".join(map(itemgetter("utterance"), win)),
                "new_target_text": win[-1]["label"],
                "new_index": df.conversation_id.unique().tolist().index(cid),
                "new_windex": wdx,
            } | win[-1]
        )
    ret = pd.DataFrame(ret).sort_values(["new_index", "new_windex"]).set_index("new_index")
    ret = ret.reset_index().drop(columns="new_index")
    # Set the default input text column.
    ret = ret.assign(sentence=ret.new_input_text, label=ret.new_target_text)
    return ret


def load(hlen: int) -> datasets.Dataset:
    gstr = os.path.join(WF_DIR, "seed-annotations", "*")
    df = set_label(pd.concat([pd.read_csv(p) for p in glob(gstr)]))
    df = set_hlen(df, hlen)
    # Create dataset.
    return datasets.Dataset.from_pandas(df, preserve_index=False)


def load_kfold(
    hlen: int, fold: int, k: int = 5, seed: int = 42
) -> datasets.DatasetDict:
    assert fold >= 0 and fold <= k - 1
    wf = load(hlen)
    if k == 1:
        return datasets.DatasetDict({
            "train": wf,
            "test": wf.select([]),
        })
    kf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
    train_idxs, test_idxs = list(kf.split(wf, wf["label"]))[fold]
    return datasets.DatasetDict({
        "train": wf.select(train_idxs),
        "test": wf.select(test_idxs),
    })

def load_unannotated(hlen: int) -> datasets.Dataset:
    gstr = os.path.join(WF_DIR, "unannotated", "*")
    df = pd.concat([pd.read_csv(p) for p in sorted(glob(gstr))]).reset_index(drop=True)
    df = set_hlen(df.assign(label="other"), hlen)
    # Create dataset.
    return datasets.Dataset.from_pandas(df, preserve_index=False)
