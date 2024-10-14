#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarizes eval_results.json files in the outputs/ dir.

Usage Examples:
    $ oinfo.py  # No arguments needed.
"""
import glob
import json
import os
from typing import Any

import pandas as pd

from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent


def print_summary(summary: pd.DataFrame) -> None:
    col_order = [
        "task", "modality", "fusion_strategy", "signals",
        "fold", "seed", "epoch", "metric", "value"
    ]
    prev_task = None
    lines = summary[col_order].to_string(index=False).split("\n")
    max_length = max(map(len, lines))
    for line in lines:
        curr_task = line.split()[0]
        if prev_task != curr_task:
            char = "=" if prev_task == "task" else "-"
            print(char * max_length)
        prev_task = curr_task
        print(line)
    print("=" * max_length)


def main(ctx: Context) -> None:
    default_outputs_dir = os.path.join(
        dirparent(os.path.realpath(__file__), 2),
        "outputs", "sad_training"
    )
    ctx.parser.add_argument("-d", "--outputs_dir", default=default_outputs_dir)
    args = ctx.parser.parse_args()
    
    data = []
    gstr = os.path.join(args.outputs_dir, "**", "eval_results.json")
    metrics = [
        "eval_f1_micro", # "eval_f1_macro",
        "eval_accuracy",
        "eval_mae"
    ]
    task_to_metric = {}
    for path in glob.glob(gstr, recursive=True):
        if "seed_" not in path: continue
        if "fold_" not in path: continue
        with open(path, "r") as fd:
            dct = json.load(fd)
        config, task, signals, fold_seed = path.split(os.sep)[-5:-1]
        modality = config
        if modality in ("early-fusion", "late-fusion"):
            modality = "multimodal"
        fusion_strategy = config
        if fusion_strategy not in ("early-fusion", "late-fusion"):
            fusion_strategy = "none"
        if modality == "audio-only":
            signals += "-only"
        assert sum([m in dct for m in metrics]) == 1
        for metric in metrics:
            if metric not in dct: continue
            task_to_metric[task] = metric
            data.append({
                "modality": modality,
                "fusion_strategy": fusion_strategy,
                "task": task,
                "signals": signals,
                "fold": int(fold_seed.split("_")[1]),
                "seed": int(fold_seed.split("_")[3]),
                "epoch": dct["epoch"],
                "metric": metric,
                "value": dct[metric]
            })

    def set_metric(row: pd.Series) -> pd.Series:
        row["metric"] = task_to_metric[row["task"]]
        return row

    gcols = ["task", "modality", "fusion_strategy", "signals"]
    print_summary(pd.DataFrame(data).set_index(gcols).sort_index().reset_index())
    df = pd.DataFrame(data).groupby(gcols).mean(
        numeric_only=True
    ).reset_index().apply(set_metric, axis=1)
    summary = []
    for task in sorted(df.task.unique()):
        ascending = False if task_to_metric[task] != "eval_mae" else True
        summary.append(
            df[df.task == task].sort_values(
                "value", ascending=ascending
            ).reset_index(drop=True)
        )
    print_summary(pd.concat(summary))


if __name__ == "__main__":
    harness(main)
