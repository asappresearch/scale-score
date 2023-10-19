import torch

from scale_score.eval import evaluate_scale
from scale_score.scorer import SCALEScorer


def test_evaluate_scale() -> None:
    t1Premise = """
agent: Welcome
customer: Hi, my toaster is broken.
agent: do you like shakespeare?
customer: I dont see how that is relevant to me fixing my toaster
agent: How about monster trucks
"""
    test1 = (
        [t1Premise],
        [
            [
                "agent: Welcome",
                "customer: Hi, my toaster is broken.",
                "agent: do you like shakespeare?",
                "customer: I dont see how that is relevant to me fixing my toaster",
                "agent: How about monster trucks",
            ]
        ],
    )

    test2 = (
        [
            "",
            "",
        ],
        [
            [""],
            ["", "", ""],
        ],
    )
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    scorer = SCALEScorer(size="small", device=device)

    test1PRF = scorer.score(test1[0], test1[1])
    metrics = evaluate_scale(test1PRF, [1, 0, 1, 0, 1])
    assert all(
        [
            y in [x for x in metrics.keys()]
            for y in [
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
                "f1_binary",
                "f1_macro",
                "f1_micro",
                "f1_weighted",
            ]
        ]
    )

    test2PRF = scorer.score(test2[0], test2[1])
    metrics = evaluate_scale(test2PRF, [1, 1, 1, 1], threshold=0.2)
    assert all(
        [
            y in [x for x in metrics.keys()]
            for y in [
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
                "f1_binary",
                "f1_macro",
                "f1_micro",
                "f1_weighted",
            ]
        ]
    )
