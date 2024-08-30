# -*- coding: utf-8 -*-
import os
import random
import re
import string
from typing import Any

import datasets
import numpy as np

from ..core.path import dirparent


IEMOCAP_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "iemocap/")

# == https://github.com/SungjoonPark/EmotionDetection/blob/master/src/data/loader.py == #
class EmotionDatasetLoader():
    def __init__(self, emotion_labels, emotion_label_type):
        assert type(emotion_labels) == list
        assert emotion_label_type in ['cat', 'dim']
        self.rel_path = os.path.dirname(__file__)
        self.labels = emotion_labels
        self.label_type = emotion_label_type
        self.split_names = ['train', 'valid', 'test']
        self.data_types = ['text', 'label']

    def load_data(self):
        """
        load data: should return data:

        data = {
            'train':
                'text': [sent1, sent2, ..] (normalized text)
                'label: [(label1), (label2), ...] (int/floats)
            'valid':
                'text': [sent1, sent2, ..] (normalized text)
                'label: [(label1), (label2), ...] (int/floats)
            'test':
                'text': [sent1, sent2, ..] (normalized text)
                'label: [(label1), (label2), ...] (int/floats)             
        }
        """
        raise NotImplementedError

    def check_number_of_data(self):
        data = self.load_data()
        for s in data.keys():
            for t in data[s].keys():
                print(s, t, len(data[s][t]))

    def _get_emotion_label_VAD_scores(self):
        vad_score_dict = {}
        dir_path = os.path.join(self.rel_path, "./../../data/NRC-VAD/")
        vad_scores = pd.read_csv(
            dir_path + "NRC-VAD-Lexicon.txt", sep='\t', index_col='Word')
        for w, (v, a, d) in vad_scores.iterrows():
            vad_score_dict[w] = (round(v, 3), round(a, 3), round(d, 3))
        return vad_score_dict

    def get_vad_coordinates_of_labels(self):
        assert 'V' not in self.labels  # for categorical labels
        total_label_vad_score_dict = self._get_emotion_label_VAD_scores()
        label_vad_score_dict = {}
        for e in self.labels:
            label_vad_score_dict[e] = total_label_vad_score_dict[e]
        return label_vad_score_dict


class IEMOCAPCatLoader(EmotionDatasetLoader):

    def __init__(self):
        emotion_labels = [
            'anger', 'happy', 'sadness',
            'frustrate', 'excite', 'fear',
            'surprise', 'disgust', 'neutral']
        super(IEMOCAPCatLoader, self).__init__(emotion_labels, 'cat')
        self.path = IEMOCAP_DIR

    def _change_label_name_to_NRC_VAD_label(self, label):
        # Counter({'xxx': 2507, 'fru': 1849, 'neu': 1708, 'ang': 1103, 'sad': 1084, 
        #          'exc': 1041, 'hap': 595, 'sur': 107, 'fea': 40, 'oth': 3, 'dis': 2})
        # original_emotion_label = ['ang', 'hap', 'sad', 'fru', 'exc', 'fea', 'sur', 
        #                           'neu', 'dis']#, 'oth',  'xxx']
        # original_label: {angry, happy, sad, neutral, frustrated, excited, fearful, 
        #                  surprised, disgusted, other}
        map_to_NRC_VAD_label_dict = {
            'ang': 'anger', 'hap': 'happy', 'sad': 'sadness', 'fru': 'frustrate',
            'exc': 'excite', 'fea': 'fear', 'sur': 'surprise', 'dis': 'disgust',
            'neu': 'neutral'
        }
        return map_to_NRC_VAD_label_dict[label] if label in map_to_NRC_VAD_label_dict else label

    def _load_preprocess_raw_data(self):
        data = []

        sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        for session in sessions:
            path_to_emotions = self.path + session + '/dialog/EmoEvaluation/'
            path_to_transcriptions = self.path + session + '/dialog/transcriptions/'

            files = os.listdir(path_to_emotions)
            files = [f[:-4] for f in files if f.endswith(".txt")]
            for f in files:
                transcriptions = self._get_transcriptions(
                    path_to_transcriptions, f + '.txt')
                emotions = self._get_emotions(path_to_emotions, f + '.txt')

                for ie, e in enumerate(emotions):
                    e['transcription'] = self._preprocessing_text(
                        transcriptions[e['id']])
                    e['emotion'] = self._change_label_name_to_NRC_VAD_label(
                        e['emotion'])
                    if e['emotion'] in self.labels:
                        data.append(e)
        sort_key = self._get_field(data, "id")
        preprocessed_data = np.array(data)[np.argsort(sort_key)]
        return preprocessed_data

    def _get_transcriptions(self, path_to_transcriptions, filename):
        f = open(path_to_transcriptions + filename, 'r').read()
        f = np.array(f.split('\n'))
        transcription = {}
        for i in range(len(f) - 1):
            g = f[i]
            i1 = g.find(': ')
            i0 = g.find(' [')
            ind_id = g[:i0]
            ind_ts = g[i1+2:]
            transcription[ind_id] = ind_ts
        return transcription

    def _get_emotions(self, path_to_emotions, filename):
        f = open(path_to_emotions + filename, 'r').read()
        f = np.array(f.split('\n'))
        idx = f == ''
        idx_n = np.arange(len(f))[idx]
        emotion = []
        for i in range(len(idx_n) - 2):
            g = f[idx_n[i]+1:idx_n[i+1]]
            head = g[0]
            actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:
                            head.find(filename[:-4]) + len(filename[:-4]) + 5]
            emo = head[head.find('\t[') - 3:head.find('\t[')]
            vad = head[head.find('\t[') + 1:]

            v = float(vad[1:7])
            a = float(vad[9:15])
            d = float(vad[17:23])

            emotion.append({'id': filename[:-4] + '_' + actor_id,
                            'v': v,
                            'a': a,
                            'd': d,
                            'emotion': emo})
        return emotion

    def _get_field(self, data, key):
        return np.array([e[key] for e in data])

    def _preprocessing_text(self, text):
        t = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', text)
        t = re.sub('\s{2,}', ' ', t)  # pad punctuations for bpe
        t = re.sub(r"\\n", " ", t)  # remove explicit \n
        t = re.sub(" +", " ", t)  # remove duplicated whitespaces
        clean_t = t.strip()
        return clean_t

    def _split_data(self, preprocessed_whole_data):
        num_dataset = list(range(len(preprocessed_whole_data)))
        random.seed(42)
        random.shuffle(num_dataset)
        train_len = int(len(num_dataset)*0.8)
        val_len = int(len(num_dataset)*0.1)
        train = preprocessed_whole_data[num_dataset[:train_len]]
        valid = preprocessed_whole_data[num_dataset[train_len:train_len+val_len]]
        test = preprocessed_whole_data[num_dataset[train_len+val_len:]]
        return train, valid, test

    def load_data(self):
        preprocessed_data = self._load_preprocess_raw_data()
        splits = self._split_data(preprocessed_data)

        data = {}
        for s_name, s_data in zip(self.split_names, splits):
            text = [s_row['transcription'] for s_row in s_data]
            labels = [s_row['emotion'] for s_row in s_data]
            data[s_name] = {}
            for name, d in zip(self.data_types, [text, labels]):
                data[s_name][name] = d
        return data
# == https://github.com/SungjoonPark/EmotionDetection/blob/master/src/data/loader.py == #


def load() -> datasets.Dataset:
    ret = datasets.DatasetDict()
    loader = IEMOCAPCatLoader()
    data = loader.load_data()
    for split in data:
        ret[split] = datasets.Dataset.from_dict(data[split])
    ret = ret.rename_column("label", "label_name")

    label_list = sorted(loader.labels)
    def set_label_int(row: dict[str, Any]) -> dict[str, Any]:
        row["label"] = label_list.index(row["label_name"])
        return row
    ret = ret.map(set_label_int)
    ret = ret.cast_column(
        "label",
        datasets.ClassLabel(num_classes=len(label_list), names=label_list)
    )
    return ret


def load_kfold(
    fold: int, k: int = 5, seed: int = 42
) -> datasets.DatasetDict:
    raise NotImplementedError
