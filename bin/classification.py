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
    $ classification.py configs/flan-t5-base.json
    $ # Run on just one task.
    $ classification.py configs/flan-t5-base.json -t iemocap
    $ # Run on custom configs.
    $ classification.py path/to/config.json -c path/to/tasks/json
"""
import argparse
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
import peft
import numpy as np
import pandas as pd
import transformers as tf

from src.core.context import Context, get_context
from src.core.app import harness
from src.core.path import dirparent
from src.core.evaluate import f1_per_class
from src.data import tasks


@dataclasses.dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = dataclasses.field(default=None)
    use_int8: bool = dataclasses.field(default=False)
    use_lora: bool = dataclasses.field(default=False)
    lora_r: int = dataclasses.field(default=64)


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
    dataset: str = dataclasses.field(default=None)
    dataset_kwargs: dict[Any, Any] = dataclasses.field(default_factory=dict)
    text_column: str = dataclasses.field(default=None)
    label_column: str = dataclasses.field(default=None)
    max_train_samples: int = dataclasses.field(
        default=None,
        metadata={"help": "for debugging, truncates the number of training examples"}
    )
    max_eval_samples: int = dataclasses.field(
        default=None,
        metadata={"help": "for debugging, truncates the number of evaluation examples"}
    )

    def __post_init__(self):
        assert self.text_column is not None
        assert self.label_column is not None


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
    logger.info("\nRUN_ID: %s [%s]", run_id, data_args.dataset)
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
    # Make a directory per fold.
    training_args.output_dir = os.path.join(
        training_args.output_dir, f"fold_{data_args.data_fold}"
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
    # XXX: Currently not needed.
    training_args.greater_is_better = metric not in ("loss", "eval_loss", "mse", "mae")
    # Load training data.
    data = tasks.load_kfold(
        data_args.dataset,
        **data_args.dataset_kwargs,
        fold=data_args.data_fold,
        k=data_args.data_num_folds,
        seed=training_args.data_seed
    ).rename_columns({
        data_args.label_column: "label"
    })
    if data_args.do_regression:
        raise NotImplementedError
    # Preprocess training data.
    label_list = sorted(set(itertools.chain(*[data[split]["label"] for split in data])))
    tokenizer = tf.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    assert tokenizer.model_max_length >= data_args.text_max_length
    def preprocess_fn(examples):
        # Label processing.
        examples["label"] = list(map(lambda l: label_list.index(l), examples["label"]))
        # Text processing.
        return tokenizer(
            examples[data_args.text_column],
            padding="max_length",
            max_length=data_args.text_max_length,
            truncation=True
        )
    data = data.map(preprocess_fn, batched=True, batch_size=16)
    train_dataset, eval_dataset = data["train"], data["test"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    # Model training.
    config = tf.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task="text-classification",
        label2id={v: i for i, v in enumerate(label_list)},  # XXX: Is this needed?
        id2label={i: v for i, v in enumerate(label_list)},  # XXX: Is this needed?
    )
    model = tf.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if model_args.use_int8:
        model.quantization_config = tf.BitsAndBytesConfig(
            load_in_8bit=model_args.use_int8
        )
        model = peft.prepare_model_for_kbit_training(model)
    if model_args.use_lora:
        peft_config = peft.LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=model_args.lora_r,
            bias="none",
            task_type=peft.TaskType.SEQ_CLS,
            use_rslora=True,
        )
        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    trainer = tf.Trainer(
        model=model,
        tokenizer=tokenizer,
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
        "configs", "classification", "tasks.json"
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
        cfg = copy(config) | task_config[task]
        cfg["output_dir"] = os.path.join(cfg["output_dir"], task)
        hf_parser = tf.HfArgumentParser((ModelArguments, DataArguments, tf.TrainingArguments))
        model_args, data_args, training_args = hf_parser.parse_dict(cfg)
        # Run the training loop.
        if data_args.data_fold is not None:
            run(ctx, model_args, data_args, training_args)
            continue  # Skip the other folds since a fold was specified.
        for fold in range(data_args.data_num_folds):
            data_args.data_fold = fold
            run(ctx, copy(model_args), copy(data_args), copy(training_args))


if __name__ == "__main__":
    harness(main)
