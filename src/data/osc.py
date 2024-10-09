# -*- coding: utf-8 -*-
import glob
import json
import os

import datasets
import numpy as np
import pandas as pd
import re
from typing import List, Dict
from ..core.path import dirparent
from sklearn.model_selection import train_test_split

OSC_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "osc")

TEST_SIZE = 0.2
SEED = 42

def load() -> datasets.Dataset:
    df = pd.read_json(os.path.join(OSC_DIR, "osc_annotations.jsonl"), lines=True)

    # Add full audio path.
    df = df.assign(
        audio=os.path.join(OSC_DIR, "osc_audio") + os.sep + df.audio_file
    )[[
        "text", "file", "label"
    ]].astype(str)

    osc = datasets.Dataset.from_pandas(df, preserve_index=False)
    osc = osc.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    #XXX: Set train/test split of 80/20. Decide if we need to do CV here.
    train_data, test_data = train_test_split(
        osc,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    
    return datasets.DatasetDict({
        "train": train_data,
        "test": test_data,
    })
