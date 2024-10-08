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

CB_TO_CLF = {
    3.0: 'CT+',
    2.0: 'PR+',
    1.0: 'PR+',
    0.0: 'UU',
    -1.0: 'PR-',
    -2.0: 'PR-',
    -3.0: 'CT-',
}

CB_TEXTONLY_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "cb_text_only")

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?;:(){}[\]%"\'’%])', r'\1', text)
    

    contraction_suffixes = [
        r"n['’]t",  # n't
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

def process_jsonl(file_path: str) -> List[Dict]:
    processed_data = []
    with open(file_path, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            text = normalize_text(data['text'])
            
            for target in data['targets']:
                span_text = normalize_text(target['span_text'])                
                processed_data.append({
                    'sentence': text,
                    'cbto_span': span_text,
                    'label': target['label']
                })
    return processed_data

def load_df() -> pd.DataFrame:
    ret = []
    for split in ("train", "dev", "test"):
        file_path = os.path.join(CB_TEXTONLY_DIR, f"{split}.jsonl")
        ret.extend(process_jsonl(file_path))
    return pd.DataFrame(ret)

def load() -> datasets.Dataset:
    return datasets.Dataset.from_pandas(load_df(), preserve_index=False)


def load_kfold(fold: int, k: int = 5, seed: int = 42) -> datasets.DatasetDict: #XXX: CBTO has established split. Fold params are not used.
    fb = load()
    return datasets.DatasetDict({
        "train": fb.filter(lambda x: x['split'] == 'train'), #XXX: Not using dev yet.
        "test": fb.filter(lambda x: x['split'] == 'test'),
    })
