# -*- coding: utf-8 -*
import glob
import os

import pandas as pd


def load(dirpath: str):
    ret = []
    gstr = os.path.join(dirpath, "*.csv")
    for path in glob.glob(gstr):
        if path.endswith("preds.csv"):
            continue
        rid = os.path.basename(path).split(".")[0].split("-")[0]
        ret.append(pd.read_csv(path).assign(rid=rid))
    return pd.concat(ret)


def summarize(dirpath: str):
    df = load(dirpath)
    return df.groupby("rid")[
        [c for c in df.columns if c.startswith("f1")]
    ].mean().assign(
        count=df.groupby("rid").count()["model_name_or_path"]
    ).merge(
        df[[
            "rid", "model_name_or_path", "history_length", "use_lora",
            "per_device_train_batch_size", "text_max_length", "num_train_epochs"
        ]].drop_duplicates(),
        on="rid"
    )
