# -*- coding: utf-8 -*
import functools

import evaluate


class f1_per_class(evaluate.EvaluationModule):
    def compute(
        predictions: list[int],
        references: list[int],
        label_list: list[str] = None
    ):
        return compute_metric_per_class("f1", predictions, references, label_list)


def compute_metric_per_class(
    metric: str,
    predictions: list[int],
    references: list[int],
    label_list: list[str] = None
):
    label_list = label_list or sorted(set(predictions) | set(references))
    assert len(label_list) >= len(set(predictions) | set(references))

    compute_fn = functools.partial(
        evaluate.load(metric).compute,
        references=references,
        predictions=predictions
    )
    per_class_values = compute_fn(average=None)[metric]
    return {
        f"{metric}_macro": compute_fn(average="macro")[metric],
        f"{metric}_micro": compute_fn(average="micro")[metric]
    } | {f"{metric}_{label_list[idx]}": val for idx, val in enumerate(per_class_values)}
