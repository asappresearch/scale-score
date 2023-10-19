import torch

from scale_score.scorer import SCALEScorer


def test_scale_scorer() -> None:
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

    assert 5 == len(test1PRF)
    assert all([type(s_score) == float for s_score in test1PRF])

    test2PRF = scorer.score(test2[0], test2[1])

    assert len(test2PRF) == 4
    assert all([type(s_score) == float for s_score in test2PRF])


def test_scale_retrieve() -> None:
    test1 = (
        [
            [
                "agent: Welcome",
                "customer: Hi, my toaster is broken.",
                "agent: do you like shakespeare?",
                "customer: I dont see how that is relevant to me fixing my toaster",
                "agent: How about monster trucks",
            ]
        ],
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
            ["", "", ""],
            [""],
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

    test1PRF_retrieval = scorer.retrieve(
        test1[0],
        test1[1],
    )

    assert len(test1PRF_retrieval) == 5

    for x in test1PRF_retrieval:
        assert len(x) == 2
        assert type(x[1]) == float
        assert type(x[0]) == str

    test2PRF_retrieval = scorer.retrieve(
        test2[0],
        test2[1],
    )

    assert len(test2PRF_retrieval) == 4
    for x in test2PRF_retrieval:
        assert len(x) == 2
        assert type(x[1]) == float
        assert type(x[0]) == str
