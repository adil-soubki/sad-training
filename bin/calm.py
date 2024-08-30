#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal Language Model Pretraining"""
import dataclasses
import hashlib
import itertools
import os
import sys
from copy import copy
from typing import Optional

import datasets
import evaluate
import torch
import numpy as np
import pandas as pd
import transformers as tf

from src.core.context import Context
from src.core.app import harness
from src.core.path import dirparent
from src.data import wsj
from src.models.calm import CALM, ModelArguments


#  @dataclasses.dataclass
#  class ModelArguments:
#      model_name_or_path: Optional[str] = dataclasses.field(default=None)
#      use_int8: bool = dataclasses.field(default=False)
#      use_lora: bool = dataclasses.field(default=False)
#      lora_r: int = dataclasses.field(default=64)


@dataclasses.dataclass
class DataArguments:
    data_num_folds: int
    data_fold: int = dataclasses.field(default=None)
    metric_for_evaluation: str = dataclasses.field(default="accuracy")
    block_size: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size "
                "for training. "
                "Default to the model max input length for single sentence inputs "
                "(take into account special tokens)."
            )
        },
    )


def get_max_block_size(model: torch.nn.Module, tokenizer: tf.PreTrainedTokenizer) -> int:
    max_pos_embeddings = getattr(model.config, "max_position_embeddings", 1024)
    block_size = tokenizer.model_max_length
    if block_size > max_pos_embeddings:
        # The tokenizer has a really big model_max_length for some reason.
        if max_pos_embeddings > 0:
            block_size = min(1024, max_pos_embeddings)
        # Fallback to max_pos_embeddings or 1024.
        else:
            block_size = 1024
    return block_size


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
    # XXX: Currently not needed.
    metric = data_args.metric_for_evaluation
    training_args.greater_is_better = metric not in ("loss", "eval_loss", "mse", "mae")
    # Load training data.
    data = wsj.load_kfold(
        version="raw",
        fold=data_args.data_fold,
        k=data_args.data_num_folds,
        seed=training_args.data_seed
    )
    # Setup the model and tokenizer.
    tokenizer = tf.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = CALM(model_args)
    # Preprocess training data.
    max_block_size = get_max_block_size(model, tokenizer)
    block_size = data_args.block_size or max_block_size
    column_names = data["train"].features
    assert block_size <= max_block_size
    def tokenize_fn(examples):
        return tokenizer(examples["text"])
    data = data.map(
        tokenize_fn, batched=True, batch_size=16, remove_columns=column_names
    )
    # Main data processing function that will concatenate all texts from our dataset 
    # and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(itertools.chain(*examples[k])) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size we exclude
        # this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        # XXX: Dummy state tokens for now.
        result["state_ids"] = torch.randint(
            0, tokenizer.vocab_size, (len(result["input_ids"]), block_size, block_size)
        )
        result["state_mask"] = torch.ones(
            len(result["input_ids"]), block_size, block_size
        )
        return result
    data = data.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    train_dataset, eval_dataset = data["train"], data["test"]
    # Model training.
    trainer = tf.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=tf.default_data_collator,
    )
    # Parse out the logits before sending to compute_metrics.
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first.
            logits = logits[0]
        return logits.argmax(dim=-1)
    trainer.preprocess_logits_for_metrics = preprocess_logits_for_metrics
    # Compute evaluation metrics.
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        # XXX: compute perplexity here.
        return evaluate.combine([metric]).compute(
            predictions=preds,
            references=labels,
        )
    trainer.compute_metrics = compute_metrics
    # Training
    if training_args.do_train:
        trainer.train()
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def main(ctx: Context) -> None:
    # Parse arguments.
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, tf.TrainingArguments))
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
