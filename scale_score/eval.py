import json
from typing import List, Optional

from scale_score.utils import eval_metrics, get_f1s, print_f1s, print_metrics

DEFAULT_METRICS = [
    "pearson",
    "spearman",
    "kendalltau",
    "majority_class_accuracy",
    "best_accuracy",
    "best_detection_precision",
    "best_detection_recall",
    "best_detection_f1",
    "accuracy@90%",
    "accuracy@70%",
    "threshold_f1",
    "threshold_@90%",
    "threshold_@70%",
]


def evaluate_scale(
    results: List[float],
    incorrect: List[int],
    threshold: float = 0.5,
    out_file: Optional[str] = None,
    print_outputs: bool = True,
) -> dict:
    """
    Evaluate SCALE Score
    Takes in the results list from a scale_score run and runs evaluation metrics on it.
    Args:
        - :param: 'results' (list): list output by a scale_score run
        - :param: 'incorrect' (list of int): list of labels, 1 for incorrect and 0 for correct
        - :param: 'threshold' (float): float threshold used for classification
        - :param: 'out_file' (str): Optional file to dump results into
        - :param: 'print_outputs' (bool): whether to print outputs from metrics
    Return:
        - :param: 'metrics' (dict): Dictionary containing resulting metrics
    """
    if len(results) != len(incorrect):
        raise ValueError(
            """The length of incorrect labels must be equal to the number of summary sentences, i.e.
        the length of results"""
        )

    metrics = {}

    eval_out = eval_metrics(list(map(lambda x: (x + 1) % 2, incorrect)), results)
    if print_outputs:
        print_metrics(eval_out)

    metrics = dict(
        zip(
            DEFAULT_METRICS,
            eval_out,
        )
    )

    preds = list(map(lambda x: 0 if x < threshold else 1, results))
    f1s = get_f1s(preds, incorrect)
    if print_outputs:
        print_f1s(f1s)

    metrics["f1_binary"] = f1s[0]
    metrics["f1_macro"] = f1s[1]
    metrics["f1_micro"] = f1s[2]
    metrics["f1_weighted"] = f1s[3]

    if out_file is not None:
        with open(out_file, "w") as file:
            json.dump(metrics, file)

    return metrics
