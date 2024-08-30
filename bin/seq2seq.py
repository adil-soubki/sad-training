#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
import dataclasses
import hashlib
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

from src.core.context import Context
from src.core.app import harness
from src.core.path import dirparent
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
    metric_for_evaluation: str = dataclasses.field(default="bleu")
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
    input_text_column: Optional[str] = dataclasses.field(default=None)
    target_text_column: Optional[str] = dataclasses.field(default=None)
    dataset: str = dataclasses.field(default=None)
    dataset_kwargs: dict[Any, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        assert self.input_text_column is not None
        assert self.target_text_column is not None


def run(
    ctx: Context,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: tf.Seq2SeqTrainingArguments
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
    # Configure evaluation metric.
    metric = data_args.metric_for_evaluation
    # XXX: Currently not needed.
    training_args.greater_is_better = metric not in ("loss", "eval_loss", "mse", "mae")
    # Load training data.
    data = tasks.load_kfold(
        data_args.dataset,
        **data_args.dataset_kwargs,
        fold=data_args.data_fold,
        k=data_args.data_num_folds,
        seed=training_args.data_seed
    )
    # XXX: For some reason Hugging Face errors when using "label" as a column name.
    #   We rename it to "target_text" in that case. If that column name is unavailable
    #   this will throw an error. If "label" is not the target text we rename it to
    #   "label_value" instead.
    column_names = set(itertools.chain(*data.column_names.values()))
    if data_args.target_text_column == "label":
        data = data.rename_column("label", "target_text")
        data_args.target_text_column = "target_text"
    elif "label" in column_names:
        data = data.rename_column("label", "label_value")
    #  for split in data:
    #      data[split] = data[split].select(range(1000))  # XXX
    # Preprocess training data.
    tokenizer = tf.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    assert tokenizer.model_max_length >= data_args.text_max_length
    def preprocess_fn(examples):
        padding = "max_length"
        # Text processing.
        ret = tokenizer(
            examples[data_args.input_text_column],
            padding=padding,
            max_length=data_args.text_max_length,
            truncation=True
        )
        # Label processing. Note use of text_target keyword argument.
        # NOTE: We cast labels to str in case they happen to be numeric.
        labels = tokenizer(
            text_target=list(map(str, examples[data_args.target_text_column])),  # XXX
            max_length=data_args.text_max_length,
            padding=padding,
            truncation=True
        )
        # If we are padding, replace all tokenizer.pad_token_id in the labels by -100
        # to ignore ignore padding when calculating the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [
                    (l if l != tokenizer.pad_token_id else -100) for l in label
                ] for label in labels["input_ids"]
            ]
        # Add target_text tokens to the model input.
        ret["labels"] = labels["input_ids"]
        return ret

    data = data.map(preprocess_fn, batched=True)
    train_dataset, eval_dataset = data["train"], data["test"]
    # Model training.
    config = tf.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
    )
    model = tf.AutoModelForSeq2SeqLM.from_pretrained(
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
    # Data collator.
    data_collator = tf.DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,  # Ignore padding for loss.
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    # Override the decoding parameters of Seq2SeqTrainer.
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.text_max_length
    )
    trainer = tf.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    # Compute evaluation metrics.
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Save predictions to file.
        pdf = eval_dataset.to_pandas().assign(
            decoded_preds=decoded_preds, decoded_labels=decoded_labels
        )
        pdf.to_csv(os.path.join(training_args.output_dir, "eval_preds.csv"))
        # Compute metric. 
        result = evaluate.load(metric).compute(
            predictions=decoded_preds, references=decoded_labels
        )
        if "precisions" in result:
            del result["precisions"]  # This clutters the results.
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
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
    # Parse arguments.
    parser = tf.HfArgumentParser(
        (ModelArguments, DataArguments, tf.Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        parser.error("No configuration passed")
    # Run the training loop.
    if data_args.data_fold is not None:
        return run(ctx, model_args, data_args, training_args)
    for fold in range(data_args.data_num_folds):
        data_args.data_fold = fold
        run(ctx, copy(model_args), copy(data_args), copy(training_args))


if __name__ == "__main__":
    harness(main)
