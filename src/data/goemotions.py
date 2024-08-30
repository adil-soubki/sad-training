# -*- coding: utf-8 -*-
import math
import os
import re
import string
from typing import Any, Literal

import datasets
import pandas as pd

from ..core.path import dirparent
from ..data.iemocap import EmotionDatasetLoader


GEMO_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "goemotions")


# == https://github.com/SungjoonPark/EmotionDetection/blob/master/src/data/loader.py == #
class GOEMOTIONSLoader(EmotionDatasetLoader):
    def __init__(self):
        emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
            'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        super(GOEMOTIONSLoader, self).__init__(emotion_labels, 'cat')
        self.path = os.path.join(GEMO_DIR, "original")

    def _load_split_files(self):
        splits = []
        for s_name in self.split_names:
            if s_name == 'test':
                s_name = 'test'
            if s_name == 'valid':
                s_name = 'dev'
            file_path = os.path.join(self.path, s_name + '.tsv')
            split = pd.read_csv(file_path, sep='\t', names=['Text','Label','Alpha'])
            splits.append(split)
        return splits

    def _convert_to_one_hot_label(self,label,label_list_len):
        one_hot_label = [0] * label_list_len
        for l in label:
            l_int = int(l)
            one_hot_label[l_int] = 1
        return tuple(one_hot_label)

    def _preprocessing_text(self, text):
        """
        strip " and whitespace for every text
        """
        cleaned_text = []
        for t in text:
            t = t.strip('\"').strip('\'').strip()
            t = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', t)
            t = re.sub('\s{2,}', ' ', t)  # pad punctuations for bpe
            t = t.strip()
            cleaned_text.append(t)
        return cleaned_text

    def count_label(self, label_list, count_list):
        for e in label_list:
            count_list[e] +=1
        return count_list

    def measure_imbalance(self, count_list):
        sumval =0
        for e in range(len(count_list)):
            sumval +=count_list[e]
        result = 0
        for e in range(len(count_list)):
            result +=(count_list[e]/sumval) * math.log(count_list[e]/sumval) 
        return (-1)*result

    def load_data(self, preprocessing=True):
        data = {}
        count_list = [0] * len(self.labels)
        splits = self._load_split_files()
        for s_name, s_data in zip(self.split_names, splits):
            text = s_data['Text'].to_list()
            if preprocessing:
                text = self._preprocessing_text(text)

            emotions = []
            for e in s_data['Label']:
                label_list = [int(s) for s in e.split(',')]
                count_list = self.count_label(label_list, count_list)
                emotion = self._convert_to_one_hot_label(label_list, len(self.labels))
                emotions.append(emotion)

            data[s_name] = {}
            for name, d in zip(self.data_types, [text, emotions]):
                data[s_name][name] = d
        #  print(count_list)
        h= self.measure_imbalance(count_list)
        #  print(h)
        #  print(h/math.log(len(self.labels)))
        return data


class GOEMOTIONSEkmanLoader(EmotionDatasetLoader):
    def __init__(self):
        emotion_labels = ['anger','disgust','fear','joy','neutral','sadness','surprise']
        super(GOEMOTIONSEkmanLoader, self).__init__(emotion_labels, 'cat')
        self.path = os.path.join(GEMO_DIR, "ekman")

    def _load_split_files(self):
        splits = []
        for s_name in self.split_names:
            if s_name == 'test':
                s_name = 'test'
            if s_name == 'valid':
                s_name = 'dev'
            file_path = os.path.join(self.path, s_name + '.tsv')
            split = pd.read_csv(file_path, sep='\t', names=['Text','Label','Alpha'])
            splits.append(split)
        return splits

    def _convert_to_one_hot_label(self,label,label_list_len):
        one_hot_label = [0] * label_list_len
        for l in label:
            l_int = int(l)
            one_hot_label[l_int] = 1
        return tuple(one_hot_label)

    def _preprocessing_text(self, text):
        """
        strip " and whitespace for every text
        """
        cleaned_text = []
        for t in text:
            t = t.strip('\"').strip('\'').strip()
            t = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', t)
            t = re.sub('\s{2,}', ' ', t)  # pad punctuations for bpe
            t = t.strip()
            cleaned_text.append(t)
        return cleaned_text
    
    def count_label(self, label_list, count_list):
        for e in label_list:
            count_list[e] +=1
        return count_list

    def measure_imbalance(self, count_list):
        sumval =0
        for e in range(len(count_list)):
            sumval +=count_list[e]
        result = 0
        for e in range(len(count_list)):
            result +=(count_list[e]/sumval) * math.log(count_list[e]/sumval) 
        return (-1)*result
            

    def load_data(self, preprocessing=True):
        data = {}
        splits = self._load_split_files()
        count_list = [0] * len(self.labels)
        for s_name, s_data in zip(self.split_names, splits):
            text = s_data['Text'].to_list()
            if preprocessing:
                text = self._preprocessing_text(text)

            emotions = []
            for e in s_data['Label']:
                label_list = [int(s) for s in e.split(',')]
                count_list = self.count_label(label_list, count_list)
                emotion = self._convert_to_one_hot_label(label_list, len(self.labels))
                emotions.append(emotion)

            data[s_name] = {}
            for name, d in zip(self.data_types, [text, emotions]):
                data[s_name][name] = d
        #  print(count_list)
        h= self.measure_imbalance(count_list)
        #  print(h)
        #  print(h/math.log(len(self.labels)))
        return data
# == https://github.com/SungjoonPark/EmotionDetection/blob/master/src/data/loader.py == #


LabelType = Literal["original", "ekman"]


def load(label_type: LabelType, multilabel: bool) -> datasets.DatasetDict:
    assert label_type in LabelType.__args__
    ret = datasets.DatasetDict()
    loader = GOEMOTIONSLoader() if label_type == "original" else GOEMOTIONSEkmanLoader()
    data = loader.load_data()
    for split in data:
        ret[split] = datasets.Dataset.from_dict(data[split])
    ret = ret.rename_column("label", "label_multi_hot")
    if multilabel:
        raise NotImplementedError
    # Handle single label case.
    ret = ret.filter(lambda d: sum(d["label_multi_hot"]) <= 1)
    def set_single_labels(row: dict[str, Any]) -> dict[str, Any]:
        assert sum(row["label_multi_hot"]) <= 1
        row["label_int"] = row["label_multi_hot"].index(1)
        row["label_name"] = loader.labels[row["label_int"]]
        return row
    ret = ret.map(set_single_labels)
    ret = ret.cast_column(
        "label_int",
        datasets.ClassLabel(num_classes=len(loader.labels), names=loader.labels)
    )
    return ret


def load_kfold(
    fold: int, k: int = 5, seed: int = 42
) -> datasets.DatasetDict:
    raise NotImplementedError
