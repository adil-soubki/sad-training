# -*- coding: utf-8 -*-
#  import json
import os
#  import re

import datasets
import pandas as pd

from ..core.path import dirparent
#  from ..data.head2span import get_text_span


#  FB_TO_CLF = {
#      3.0: 'CT+',
#      2.0: 'PR+',
#      1.0: 'PR+',
#      0.0: 'UU',
#      -1.0: 'PR-',
#      -2.0: 'PR-',
#      -3.0: 'CT-',
#  }
FB_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "fb")


# XXX: Repeated code in commitment_bank_textonly.py.
#  def normalize_text(text: str) -> str:
#      text = re.sub(r'\s+', ' ', text)
#      text = re.sub(r'\s+([,.!?;:(){}[\]%"\'’%])', r'\1', text)

#      contraction_suffixes = [
#          r"n['’]t",  # n't
#          r"['’]m",    # 'm
#          r"['’]s",    # 's
#          r"['’]re",   # 're
#          r"['’]ll",   # 'll
#          r"['’]ve",   # 've
#          r"['’]d",    # 'd
#          r"['’]em",   # 'em (optional, e.g., 'em for 'them')
#      ]

#      pattern = r'\s+(' + '|'.join(contraction_suffixes) + r')\b'
#      text = re.sub(pattern, r'\1', text)
#      text = re.sub(r'\$\s+', r'$', text)
#      return text


# XXX: Repeated code in commitment_bank_textonly.py.
#  def process_jsonl(file_path: str) -> list[dict]:
#      processed_data = []
#      with open(file_path, 'r') as infile:
#          for line in infile:
#              data = json.loads(line)
#              text = normalize_text(data['text'])
#              file_name = data["article"]
#              # For some reason the file names are surrounded by single quotes.
#              if file_name.startswith("'") and file_name.endswith("'"):
#                  file_name = file_name.replace("'", "")

#              for target in data['targets']:
#                  span_text = normalize_text(target['span_text'])
#                  full_span = get_text_span(span_text, text)
#                  assert target["label"].is_integer()
#                  label_int = int(target["label"])
                
#                  if full_span:
#                      processed_data.append({
#                          "file_name": file_name,
#                          "file_index": data["file_idx"],
#                          "sentence": text,
#                          "head_span": span_text,
#                          "full_span": full_span,
#                          "label_int": label_int,
#                          "label": FB_TO_CLF[target["label"]],
#                      })
#      return processed_data


#  def load_df() -> pd.DataFrame:
#      ret = []
#      for split in ("train", "dev", "test"):
#          file_path = os.path.join(FB_DIR, f"{split}.jsonl")
#          processed_data = process_jsonl(file_path)
#          df = pd.DataFrame(processed_data)
#          df['split'] = split
#          ret.append(df)
#      return pd.concat(ret).reset_index(drop=True)
#      #  import ipdb; ipdb.set_trace()
#      #  return df[["sentence", "original_head", "fb_span", "label", "split"]]


#  def load() -> datasets.Dataset:
#      return datasets.Dataset.from_pandas(load_df(), preserve_index=False)


# XXX: FB has established split. Fold params are not used.
#  def load_kfold(fold: int, k: int = 5, seed: int = 42) -> datasets.DatasetDict:
#      fb = load()
#      return datasets.DatasetDict({
#          "train": fb.filter(lambda x: x['split'] == 'train'),  # XXX: Not using dev yet.
#          "test": fb.filter(lambda x: x['split'] == 'test'),
#      })


def load() -> datasets.DatasetDict:
    df = pd.read_csv(os.path.join(FB_DIR, "span", "corpus.csv"))
    return datasets.DatasetDict({
        split: datasets.Dataset.from_pandas(
            df[df.split == split], preserve_index=False
        ) for split in ("train", "test")  # NOTE: Not using dev yet.
    })

# NOTE: FB has established split. Fold params are not used.
def load_kfold(fold: int, k: int = 5, seed: int = 42) -> datasets.DatasetDict:
    assert fold == 0, "KFold splitting not implemented"
    return load()
