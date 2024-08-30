import os
import random
import hashlib
import zipfile
import tarfile
import datetime
from argparse import Namespace
from typing import Any, Literal

import datasets
import requests
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold


# == https://github.com/skywalker023/fantom == #
class DownloadableFile:
    def __init__(self, url, filename, expected_hash, version="1.0", zipped=True):
        self.url = url
        self.filename = filename
        self.expected_hash = expected_hash
        self.zipped = zipped
        self.version = version


FANTOM = DownloadableFile(
    url='https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz',
    filename='fantom.tar.gz',
    expected_hash='1d08dfa0ea474c7f83b9bc7e3a7b466eab25194043489dd618b4c5223e1253a4',
    version="1.0",
    zipped=True
)

# =============================================================================================================

def unzip_file(file_path, directory='.'):
    if file_path.endswith(".zip"):
        target_location =  os.path.join(directory, os.path.splitext(os.path.basename(file_path))[0])
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_location)
    elif file_path.endswith(".tar.gz"):
        target_location =  os.path.join(directory, os.path.basename(file_path).split(".")[0])
        with tarfile.open(file_path) as tar:
            tar.extractall(target_location)

    return target_location

def check_built(path, version_string=None):
    """
    Check if '.built' flag has been set for that task.
    If a version_string is provided, this has to match, or the version is regarded as not built.
    """
    fname = os.path.join(path, '.built')
    if not os.path.isfile(fname):
        return False
    else:
        with open(fname, 'r') as read:
            text = read.read().split('\n')
        return len(text) > 1 and text[1] == version_string

def mark_built(path, version_string="1.0"):
    """
    Mark this path as prebuilt.
    Marks the path as done by adding a '.built' file with the current timestamp plus a version description string.
    """
    with open(os.path.join(path, '.built'), 'w') as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write('\n' + version_string)

def download_and_check_hash(url, filename, expected_hash, version, directory='data', chunk_size=1024*1024*10):

    # Download the file
    response = requests.get(url, stream=True)
    try:
        total_size = int(response.headers.get('content-length', 0))
    except:
        print("Couldn't get content-length from response headers, using chunk_size instead")
        total_size = chunk_size
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    data = b''
    for chunk in response.iter_content(chunk_size=chunk_size):
        progress_bar.update(len(chunk))
        data += chunk
    progress_bar.close()

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the file to disk
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        f.write(data)

    # Calculate the hash of the downloaded data
    sha256_hash = hashlib.sha256(data).hexdigest()

    # Compare the calculated hash to the expected hash
    if sha256_hash != expected_hash:
        print('@@@ Downloaded file hash does not match expected hash!')
        raise RuntimeError

    return file_path

def build_data(resource, directory='data'):
    # check whether the file already exists
    if resource.filename.endswith('.tar.gz'):
        resource_dir = os.path.splitext(os.path.splitext(os.path.basename(resource.filename))[0])[0]
    else:
        resource_dir = os.path.splitext(os.path.basename(resource.filename))[0]
    file_path = os.path.join(directory, resource_dir)

    built = check_built(file_path, resource.version)

    if not built:
        # Download the file
        file_path = download_and_check_hash(resource.url, resource.filename, resource.expected_hash, resource.version, directory)

        # Unzip the file
        if resource.zipped:
            built_location = unzip_file(file_path, directory)
            # Delete the zip file
            os.remove(file_path)
        else:
            built_location = file_path

        mark_built(built_location, resource.version)
        print("Successfully built dataset at {}".format(built_location))
    else:
        print("Already built at {}. version {}".format(file_path, resource.version))
        built_location = file_path

    return built_location

def load_fantom():
    dpath = build_data(FANTOM)
    file = os.path.join(dpath, "fantom_v1.json")
    df = pd.read_json(file)

    return df


ConversationInputType = Literal["short", "full"]


class FanToM:
    def __init__(self, conversation_input_type: ConversationInputType):
        assert conversation_input_type in ConversationInputType.__args__
        self.conversation_input_type = conversation_input_type
        # XXX: Unused.
        self.prompt_header = (
            "This is a theory-of-mind test. Please answer the question regarding "
            "facts or beliefs, based on the following in-person conversation between "
            "individuals who have just met.\n\n"
        )
        self.load_fantom()
        self.setup_fantom()

    def load_fantom(self):
        self.fantom_df = load_fantom()

    def set_beliefQA_multiple_choices(self, qa):
        if qa['question_type'].endswith(":inaccessible"):
            option_a = qa['wrong_answer']
            option_b = qa['correct_answer']
        else:
            option_a = qa['wrong_answer']
            option_b = qa['correct_answer']

        answer_goes_last = random.choice([True, False])
        if answer_goes_last:
            choices = [option_a, option_b]
            answer = 1
        else:
            choices = [option_b, option_a]
            answer = 0

        # option letters iterate over the alphabet
        option_letters = ["(" + chr(x) + ")" for x in range(ord('a'), len(choices) + ord('a'))]
        choices_text = ""
        for letter, option in zip(option_letters, choices):
            choices_text += "{} {}\n".format(letter, option)

        return choices_text, answer

    def setup_fantom(self):
        """
        Flatten the dictionary and add short and full conversation context to each question.
        The result will be a list of questions and list of short or full inputs to be used as input for the models.
        """
        self.fantom_df_to_run = self.fantom_df

        total_num_q = 0
        for idx, _set in self.fantom_df_to_run.iterrows():
            total_num_q += len(_set['beliefQAs'])
            total_num_q += len(_set['answerabilityQAs_binary'])
            total_num_q += len(_set['infoAccessibilityQAs_binary'])
            if _set['factQA'] is not None:
                total_num_q += 1
            if _set['answerabilityQA_list'] is not None:
                total_num_q += 1
            if _set['infoAccessibilityQA_list'] is not None:
                total_num_q += 1

        inputs = []
        qas = []
        for idx, _set in self.fantom_df_to_run.iterrows():
            if self.conversation_input_type == "short":
                context = _set['short_context'].strip()
            elif self.conversation_input_type == "full":
                context = _set['full_context'].strip()
            
            set_id = _set['set_id']
            fact_q = _set['factQA']['question']
            fact_a = _set['factQA']['correct_answer']

            # Fact Question
            _set['factQA']['context'] = context
            input_text = "{}\n\nQuestion: {}\nAnswer:".format(context, fact_q)
            _set['factQA']['input_text'] = input_text
            _set['factQA']['set_id'] = set_id
            qas.append(_set['factQA'])
            inputs.append(input_text)

            for _belief_qa in _set['beliefQAs']:
                # Belief Questions
                _belief_qa['context'] = context
                input_text = "{}\n\nQuestion: {}\nAnswer:".format(context, _belief_qa['question'])
                _belief_qa['input_text'] = input_text
                _belief_qa['set_id'] = set_id
                qas.append(_belief_qa)
                inputs.append(input_text)

                # Multiple Choice Belief Questions
                _mc_belief_qa = {**_belief_qa}
                choices_text, answer = self.set_beliefQA_multiple_choices(_mc_belief_qa)
                mc_question = "{}\n{}\n\nChoose an answer from above:".format(_belief_qa['question'], choices_text.strip())
                _mc_belief_qa['question'] = mc_question
                _mc_belief_qa['question_type'] = _mc_belief_qa['question_type'] + ":multiple-choice"
                _mc_belief_qa['choices_text'] = choices_text
                _mc_belief_qa['choices_list'] = choices_text.strip().split("\n")
                _mc_belief_qa['correct_answer'] = answer
                input_text = "{}\n\nQuestion: {}".format(context, mc_question)
                _mc_belief_qa['input_text'] = input_text
                qas.append(_mc_belief_qa)
                inputs.append(input_text)

            # Answerability List Questions
            _set['answerabilityQA_list']['fact_question'] = fact_q
            _set['answerabilityQA_list']['context'] = context
            input_text = "{}\n\nTarget: {}\nQuestion: {}\nAnswer:".format(context, fact_q, _set['answerabilityQA_list']['question'])
            _set['answerabilityQA_list']['input_text'] = input_text
            _set['answerabilityQA_list']['set_id'] = set_id
            if self.conversation_input_type == "full" and len(_set['answerabilityQA_list']['wrong_answer']) > 0:
                _set['answerabilityQA_list']['missed_info_accessibility'] = 'inaccessible'
            qas.append(_set['answerabilityQA_list'])
            inputs.append(input_text)

            # Answerability Binary Questions
            if self.conversation_input_type == "full":
                missed_info_accessibility_for_full = _set['answerabilityQAs_binary'][0]['missed_info_accessibility']
                for _info_accessibility_qa in _set['answerabilityQAs_binary']:
                    if _info_accessibility_qa['correct_answer'] != "yes":
                        missed_info_accessibility_for_full = 'inaccessible'

            for _answerability_qa in _set['answerabilityQAs_binary']:
                _answerability_qa['fact_question'] = fact_q
                _answerability_qa['context'] = context
                input_text = "{}\n\nTarget: {}\nQuestion: {} Answer yes or no.\nAnswer:".format(context, fact_q, _answerability_qa['question'])
                _answerability_qa['input_text'] = input_text
                _answerability_qa['set_id'] = set_id
                if self.conversation_input_type == "full":
                    _answerability_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
                qas.append(_answerability_qa)
                inputs.append(input_text)

            # Info Accessibility List Questions
            _set['infoAccessibilityQA_list']['fact_question'] = fact_q
            _set['infoAccessibilityQA_list']['fact_answer'] = fact_a
            _set['infoAccessibilityQA_list']['context'] = context
            input_text = "{}\n\nInformation: {} {}\nQuestion: {}\nAnswer:".format(context, fact_q, fact_a, _set['infoAccessibilityQA_list']['question'])
            _set['infoAccessibilityQA_list']['input_text'] = input_text
            _set['infoAccessibilityQA_list']['set_id'] = set_id
            if self.conversation_input_type == "full" and len(_set['infoAccessibilityQA_list']['wrong_answer']) > 0:
                _set['infoAccessibilityQA_list']['missed_info_accessibility'] = 'inaccessible'
            qas.append(_set['infoAccessibilityQA_list'])
            inputs.append(input_text)

            # Info Accessibility Binary Questions
            if self.conversation_input_type == "full":
                missed_info_accessibility_for_full = _set['infoAccessibilityQAs_binary'][0]['missed_info_accessibility']
                for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
                    if _info_accessibility_qa['correct_answer'] != "yes":
                        missed_info_accessibility_for_full = 'inaccessible'

            for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
                _info_accessibility_qa['fact_question'] = fact_q
                _info_accessibility_qa['fact_answer'] = fact_a
                _info_accessibility_qa['context'] = context
                input_text = "{}\n\nInformation: {} {}\nQuestion: {} Answer yes or no.\nAnswer:".format(context, fact_q, fact_a, _info_accessibility_qa['question'])
                _info_accessibility_qa['input_text'] = input_text
                _info_accessibility_qa['set_id'] = set_id
                if self.conversation_input_type == "full":
                    _info_accessibility_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
                qas.append(_info_accessibility_qa)
                inputs.append(input_text)

        self.inputs = inputs
        self.flattened_fantom = qas
# == https://github.com/skywalker023/fantom == #


#  ['fact',
#   'tom:belief:inaccessible',
#   'tom:belief:inaccessible:multiple-choice',
#   'tom:belief:accessible',
#   'tom:belief:accessible:multiple-choice',
#   'tom:answerability:list',
#   'tom:answerability:binary',
#   'tom:info_accessibility:list',
#   'tom:info_accessibility:binary']
QuestionType = Literal["all", "belief", "answerability", "info_accessibility"]
AnswerType = Literal["binary", "multiple-choice"]


def load(
    conversation_input_type: ConversationInputType,
    question_type: QuestionType,
    answer_type: AnswerType
) -> datasets.DatasetDict:
    assert conversation_input_type in ConversationInputType.__args__
    assert question_type in QuestionType.__args__
    assert answer_type in AnswerType.__args__
    loader = FanToM(conversation_input_type)
    df = pd.DataFrame(loader.flattened_fantom)
    df = df[df.question_type.str.contains(answer_type)]
    if question_type != "all":
        df = df[df.question_type.str.contains(question_type)]
    if answer_type == "binary":
        def set_label_columns(row: dict[str, Any]) -> dict[str, Any]:
            assert row["correct_answer"] in ("no", "no:long", "yes")
            row["label_name"] = row["correct_answer"].split(":")[0]
            row["label_int"] = 1 if row["label_name"] == "yes" else 0
            return row
        data = datasets.Dataset.from_pandas(df, preserve_index=False)
        return data.map(set_label_columns)
    elif answer_type == "multiple-choice":
        def set_label_columns(row: dict[str, Any]) -> dict[str, Any]:
            assert row["correct_answer"] in (0, 1)
            row["label_name"] = "a" if row["correct_answer"] == 0 else "b"
            row["label_int"] = row["correct_answer"]
            return row
        data = datasets.Dataset.from_pandas(df, preserve_index=False)
        return data.map(set_label_columns)
    else:
        raise ValueError


def load_kfold(
    conversation_input_type: ConversationInputType,
    question_type: QuestionType,
    answer_type: AnswerType,
    fold: int, k: int = 5, seed: int = 42
) -> datasets.DatasetDict:
    assert fold >= 0 and fold <= k - 1
    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    ds = load(conversation_input_type, question_type, answer_type)
    train_idxs, test_idxs = list(kf.split(ds))[fold]
    return datasets.DatasetDict({
        "train": ds.select(train_idxs),
        "test": ds.select(test_idxs),
    })
