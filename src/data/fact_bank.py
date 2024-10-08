# -*- coding: utf-8 -*-
import dataclasses
import os

import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from head2span import get_text_span
import json
import re
from typing import List, Dict

from ..core.path import dirparent

FB_TO_CLF = {
    3.0: 'CT+',
    2.0: 'PR+',
    1.0: 'PR+',
    0.0: 'UU',
    -1.0: 'PR-',
    -2.0: 'PR-',
    -3.0: 'CT-',
}

FB_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "fb")

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
                full_span = get_text_span(span_text, text)
                
                if full_span:
                    processed_data.append({
                        'sentence': text,
                        'original_head': span_text,
                        'fb_span': full_span,
                        'label': FB_TO_CLF[round(target['label'])]
                    })
    return processed_data


def load_df() -> pd.DataFrame:
    ret = []
    for split in ("train", "dev", "test"):
        file_path = os.path.join(FB_DIR, f"{split}.jsonl")
        processed_data = process_jsonl(file_path)
        df = pd.DataFrame(processed_data)
        df['split'] = split
        ret.append(df)
    
    df = pd.concat(ret).reset_index(drop=True)
    return df[["sentence", "original_head", "fb_span", "label", "split"]]

def load() -> datasets.Dataset:
    return datasets.Dataset.from_pandas(load_df(), preserve_index=False)


def load_kfold(fold: int, k: int = 5, seed: int = 42) -> datasets.DatasetDict: #XXX: FB has established split. Fold params are not used.
    fb = load()
    return datasets.DatasetDict({
        "train": fb.filter(lambda x: x['split'] == 'train'), #XXX: Not using dev yet.
        "test": fb.filter(lambda x: x['split'] == 'test'),
    })