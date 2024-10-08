#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finetuning Hugging Face models for text classification.

Note:
    The configuration file accepts all arguments in the huggingface transformers
    TrainingArguments class as well as those defined in this program's ModelArguments
    and DataArguments classes. The file is written in json.

    The datasets (or tasks) the models are trained on are defined in a separate task
    configuration json file. The parameters included here override options set in
    the main configuration file. A single model will be trained for each task.

Usage Examples: 
    $ # Run on all tasks in default task config
    $ sad_training.py configs/sad_training/early-fusion.json
    $ # Run on just one task.
    $ sad_training.py configs/sad_training/early-fusion.json -t commitment_bank
    $ # Run on custom configs.
    $ sad_training.py path/to/config.json -c path/to/tasks.json
"""
import argparse
import contextlib
import dataclasses
import hashlib
import json
import itertools
import os
import sys
from copy import copy
from typing import Any, Optional

import datasets
import evaluate
#  import peft
import numpy as np
import pandas as pd
import transformers as tf

from src.core.context import Context, get_context
from src.core.app import harness
from src.core.path import dirparent
from src.core.evaluate import f1_per_class
from src.data import tasks, validation
from src.models.multimodal_classifier import MultimodalClassifier, ModelArguments


@dataclasses.dataclass
class DataArguments:
    data_num_folds: int
    data_fold: int = dataclasses.field(default=None)
    do_regression: bool = dataclasses.field(default=False)
    metric_for_classification: str = dataclasses.field(default="f1")
    metric_for_regression: str = dataclasses.field(default="mae")
    text_max_length: int = dataclasses.field(
        default=256,
        metadata={
            "help": (
                "The maximum total text input sequence length after tokenization. "
                "Sequences longer than this will be truncated, sequences shorter "
                "will be padded."
            )
        },
    )
    audio_max_length: int = dataclasses.field(
        default=30,
        metadata={
            "help": (
                "The maximum audio length in seconds for feature extraction. "
                "Sequences longer than this will be truncated, sequences shorter "
                "will be padded."
            )
        },
    )
    task: str = dataclasses.field(default=None)
    audio_sources: list[str] = dataclasses.field(
        default_factory=list,
        metadata={
            "help": (
                "The list of audio_source's to train on. Each source listed a "
                "will produce a separate training run and model outputs."
            )
        }
    )
    audio_source: str = dataclasses.field(
        default=None,
        metadata={
            "help": (
                "The audio source to use when training. If null, no audio "
                "source will be used. Otherwise it expects either the tts "
                "voice to use (e.g., \"nova\") or \"gold\" to indicate the "
                "original audio should be used. This should not typically "
                "be set in the config and instead be handled by the "
                "audio_sources option in a task config."
            )
        }
    )
    max_train_samples: int = dataclasses.field(
        default=None,
        metadata={"help": "for debugging, truncates the number of training examples"}
    )
    max_eval_samples: int = dataclasses.field(
        default=None,
        metadata={"help": "for debugging, truncates the number of evaluation examples"}
    )

    def __post_init__(self):
        assert self.task is not None, "Specify task"
        assert self.audio_source is None, "Set audio_sources instead"
        assert len(self.audio_sources) > 0, (
            "Should have at least one audio source. Use [None] for text-only training"
        )


def update_metrics(
    preds: list,
    refs: list,
    label_list: list[int],
    metric: str,
    trainer: tf.Trainer,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: tf.TrainingArguments
):
    logger = get_context().log
    # Get run id and output dir.
    args = vars(model_args) | vars(data_args) | training_args.to_dict()
    for key in ("data_fold", "output_dir", "logging_dir"):
        del args[key]  # Do not use in run_id.
    run_id = hashlib.md5(str(sorted(args.items())).encode("utf-8")).hexdigest()
    run_id += f"-{data_args.data_fold}"
    args["data_fold"] = data_args.data_fold
    logger.info("\nRUN_ID: %s [%s]", run_id, data_args.task)
    output_dir = os.path.join(dirparent(training_args.output_dir, 2), "runs")
    os.makedirs(output_dir, exist_ok=True)
    # Compute the new results.
    eval_kwargs = {"predictions": preds, "references": refs}
    if metric == f1_per_class:
        eval_kwargs["label_list"] = label_list
    results = evaluate.combine([metric]).compute(**eval_kwargs)
    df = pd.DataFrame([args | results])
    df["last_modified"] = pd.Timestamp.now()
    df["current_epoch"] = trainer.state.epoch
    df.to_csv(os.path.join(output_dir, f"{run_id}.csv"), index=False)
    # Write out predictions.
    pd.DataFrame({"refs": refs, "preds": preds, "run_id": run_id}).to_csv(
        os.path.join(output_dir, f"{run_id}.preds.csv"), index=False
    )


def run(
    ctx: Context,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: tf.TrainingArguments
) -> None:
    # Validation.
    if model_args.audio_model_name_or_path is None:
        assert data_args.audio_source is None
    if data_args.audio_source is None:
        # Don't load an audio model if no audio data.
        model_args.audio_model_name_or_path = None
    # Make a directory per fold.
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        f"{data_args.audio_source or 'text-only'}",
        f"fold_{data_args.data_fold}",
    )
    ctx.log.info(f"Training parameters {training_args}")
    ctx.log.info(f"Data parameters {data_args}")
    ctx.log.info(f"Model parameters {model_args}")
    # Set seed before initializing model.
    tf.set_seed(training_args.seed)
    # Configure for regression if needed.
    metric = (
        data_args.metric_for_regression
        if data_args.do_regression
        else data_args.metric_for_classification
    )
    metric_map = {"f1_per_class": f1_per_class}
    metric = metric_map.get(metric, metric)
    # NOTE: Currently needed.
    training_args.greater_is_better = metric not in ("loss", "eval_loss", "mse", "mae")
    # Load training data.
    task_config = tasks.get_config()[data_args.task]
    data = tasks.load(
        data_args.task,
        fold=data_args.data_fold,
        k=data_args.data_num_folds,
        seed=training_args.data_seed
    ).rename_columns({
        task_config.label_column: "label"
    })
    label_list = sorted(set(itertools.chain(*[data[split]["label"] for split in data])))
    model_args.num_labels = len(label_list)
    if data_args.do_regression:
        model_args.num_labels = 1  # NOTE: Just used to stratify.
    if data_args.audio_source not in (None, "gold"):
        # Remove (gold) "audio" column if it exists.
        with contextlib.suppress(ValueError):
            data = data.remove_columns("audio")
        data = data.rename_column(
            f"audio_{data_args.audio_source}", "audio"
        )
    # Check if the audio/text data exceeds the configured audio/text_max_length
    # and print a warning if it does.
    if data_args.audio_source is not None:
        validation.warn_on_audio_length(data, data_args.audio_max_length)
    if model_args.text_model_name_or_path:
        validation.warn_on_text_length(
            data, data_args.task,
            data_args.text_max_length,
            model_args.text_model_name_or_path
        )
    # Preprocess training data.
    feature_extractor = tf.AutoFeatureExtractor.from_pretrained(
        model_args.audio_model_name_or_path
    ) if model_args.audio_model_name_or_path else None
    tokenizer = tf.AutoTokenizer.from_pretrained(
        model_args.text_model_name_or_path
    ) if model_args.text_model_name_or_path else None
    assert not tokenizer or (tokenizer.model_max_length >= data_args.text_max_length)
    if data_args.audio_source is not None:
        data = data.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    def preprocess_fn(examples):
        dummy = [[0]] * len(examples[list(examples.keys())[0]])
        # Label processing.
        if not data_args.do_regression:
            # Expects labels to be strings and uses the label_list to map them to ints.
            examples["label"] = list(
                map(lambda l: label_list.index(l), examples["label"])
            )
        # Text processing.
        inputs = tokenizer(
            examples[task_config.text_column],
            padding="max_length",
            max_length=data_args.text_max_length,
            truncation=True  # XXX: Should be False?
        ) if tokenizer else {"input_ids": dummy, "attention_mask": dummy}
        # Audio processing.
        inputs |= feature_extractor(
            [x["array"] for x in examples["audio"]],
            sampling_rate=getattr(feature_extractor, "sampling_rate", 16_000),
            padding="max_length",
            max_length=data_args.audio_max_length * 16_000,
            truncation=True  # XXX: Should be False?
        ) if feature_extractor else {"input_values": dummy}
        if "input_features" in inputs:  # Whisper names them differently.
            inputs["input_values"] = inputs.pop("input_features")
        if not model_args.use_opensmile_features:
            inputs["opensmile_features"] = dummy
        # Return inputs.
        return inputs
    data = data.map(preprocess_fn, batched=True, batch_size=16)
    data = data.rename_columns({
        "input_ids": "text_input_ids",
        "attention_mask": "text_attention_mask",
        "input_values": "audio_input_values",
    })
    # TODO: Check how often text_max_length is exceeded.
    train_dataset, eval_dataset = data["train"], data["test"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    # Model training.
    model = MultimodalClassifier(model_args)
    if not data_args.do_regression:
        model.label2id={v: i for i, v in enumerate(label_list)},  # XXX: Is this needed?
        model.id2label={i: v for i, v in enumerate(label_list)},  # XXX: Is this needed?
    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    def compute_metrics(eval_pred: tf.EvalPrediction):
        # Get logits.
        if isinstance(eval_pred.predictions, tuple):
            logits = eval_pred.predictions[0]
        else:
            logits = eval_pred.predictions
        # Get predictions.
        if data_args.do_regression:
            predictions = np.squeeze(logits)
        else:
            predictions = np.argmax(logits, axis=1)
        # Save predictions to file.
        pdf = eval_dataset.to_pandas().assign(pred=predictions)
        assert np.allclose(pdf.label, eval_pred.label_ids)
        pdf.to_csv(os.path.join(training_args.output_dir, "eval_results.csv"))
        # Update aggregated evaluation results.
        if np.isnan(predictions).any() or np.isnan(eval_pred.label_ids).any():
            import ipdb; ipdb.set_trace()
        update_metrics(
            predictions, eval_pred.label_ids, label_list, metric, trainer,
            model_args, data_args, training_args
        )
        # Return metrics.
        eval_kwargs = {"predictions": predictions, "references": eval_pred.label_ids}
        if metric == f1_per_class:
            eval_kwargs["label_list"] = label_list
        return evaluate.combine([metric]).compute(**eval_kwargs)
    trainer.compute_metrics = compute_metrics
    # Training
    if training_args.do_train:
        trainer.train()
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    # Prediction
    if training_args.do_predict:
        raise NotImplementedError


def main(ctx: Context) -> None:
    default_task_config = os.path.join(
        dirparent(os.path.realpath(__file__), 2),
        "configs", "sad_training", "tasks.json"
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config")
    parser.add_argument(
        "-c", "--task-config",
        default=default_task_config,
        help="path to json file containing task specific configuration options."
    )
    parser.add_argument("-t", "--tasks", nargs="*")
    args = parser.parse_args()

    with open(args.config, "r") as fd:
        config = json.load(fd)
    with open(args.task_config, "r") as fd:
        task_config = json.load(fd)
    for task in args.tasks or list(task_config):
        if task not in task_config:
            parser.error(f"unknown task: {task} {list(task_config)}")
        if task not in tasks.get_config():
            parser.error(f"unknown task: {task} {list(tasks.get_config())}")
        cfg = copy(config) | task_config[task] | {"task": task}
        cfg["output_dir"] = os.path.join(cfg["output_dir"], task)
        hf_parser = tf.HfArgumentParser((ModelArguments, DataArguments, tf.TrainingArguments))
        model_args, data_args, training_args = hf_parser.parse_dict(cfg)
        # Run the training loop.
        for audio_source in data_args.audio_sources:
            data_args.audio_source = audio_source
            if data_args.data_fold is not None:
                run(ctx, copy(model_args), copy(data_args), copy(training_args))
                continue  # Skip the other folds since a fold was specified.
            for fold in range(data_args.data_num_folds):
                data_args.data_fold = fold
                run(ctx, copy(model_args), copy(data_args), copy(training_args))


if __name__ == "__main__":
    harness(main)
