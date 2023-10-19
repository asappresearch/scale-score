import torch

import scale_score
from scale_score.scorer import SCALEScorer


def test_get_chunks() -> None:
    _, tokenizer = scale_score.utils.get_flan_T5_model("small")
    prompt = "{{premise}} a"
    test1 = " ".join(["hello" for x in range(998)])
    test2 = " ".join(["hello" for x in range(997)])

    chunks_t1 = scale_score.utils.get_chunks(tokenizer, test1, prompt, 500, 0.25)
    chunks_t1_2 = scale_score.utils.get_chunks(tokenizer, test1, prompt, 1000, 0)
    chunks_t1_3 = scale_score.utils.get_chunks(tokenizer, test1, prompt, 500, 0.5)
    chunks_t1_4 = scale_score.utils.get_chunks(tokenizer, test1, prompt, 500, 0.75)

    chunks_t2 = scale_score.utils.get_chunks(tokenizer, test2, prompt, 500, 0.25)
    chunks_t2_2 = scale_score.utils.get_chunks(tokenizer, test2, prompt, 1000, 0)

    assert len(chunks_t1) == 3
    assert chunks_t1[0].shape == torch.Size([1, 500])
    assert chunks_t1[1].shape == torch.Size([1, 500])
    assert chunks_t1[2].shape == torch.Size([1, 257])

    assert len(chunks_t1_3) == 5
    assert all([chunks_t1_3[i].shape == torch.Size([1, 500]) for i in range(3)])
    assert chunks_t1_3[-2].shape == torch.Size([1, 257])
    assert chunks_t1_3[-1].shape == torch.Size([1, 9])

    assert len(chunks_t1_4) == 9
    assert all([chunks_t1_4[i].shape == torch.Size([1, 500]) for i in range(5)])
    assert chunks_t1_4[-1].shape == torch.Size([1, 9])
    assert chunks_t1_4[-2].shape == torch.Size([1, 133])
    assert chunks_t1_4[-3].shape == torch.Size([1, 257])
    assert chunks_t1_4[-4].shape == torch.Size([1, 381])

    assert len(chunks_t2) == 3
    assert chunks_t2[0].shape == torch.Size([1, 500])
    assert chunks_t2[1].shape == torch.Size([1, 500])
    assert chunks_t2[2].shape == torch.Size([1, 256])

    assert len(chunks_t1_2) == 2
    assert chunks_t1_2[0].shape == torch.Size([1, 1000])

    assert len(chunks_t2_2) == 1
    assert chunks_t2_2[0].shape == torch.Size([1, 1000])


def test_scale_score() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model, tokenizer = scale_score.utils.get_flan_T5_model("small")
    model = model.to(device).eval()

    test1 = (
        ["agent: Welcome"],
        [
            [
                "The agent said Welcome.",
                "The agent said Welcome.",
                "The agent said Welcome.",
            ]
        ],
    )

    test2 = (
        [""],
        [["The agent said Welcome."]],
    )

    test1PRF = scale_score.utils.scale_score(
        model,
        tokenizer,
        test1[0],
        test1[1],
        device=device,
    )
    assert len(test1PRF) == 3

    test2PRF = scale_score.utils.scale_score(
        model,
        tokenizer,
        test2[0],
        test2[1],
        device=device,
    )

    assert len(test2PRF) == 1


def test_scale_retrieve() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model, tokenizer = scale_score.utils.get_flan_T5_model("small")
    model = model.to(device).eval()

    test1 = (
        [["agent: Welcome"]],
        [
            [
                "The agent said Welcome.",
                "The agent said Welcome.",
                "The agent said Welcome.",
            ]
        ],
    )

    test2 = (
        [[""]],
        [["The agent said Welcome."]],
    )

    test1PRF = scale_score.utils.scale_retrieve(
        model,
        tokenizer,
        test1[0],
        test1[1],
        device=device,
    )
    assert len(test1PRF) == 3
    assert len(test1PRF[0]) == 2
    assert type(test1PRF[0][0]) == str
    assert type(test1PRF[0][1]) == float

    test2PRF = scale_score.utils.scale_retrieve(
        model,
        tokenizer,
        test2[0],
        test2[1],
        device=device,
    )

    assert len(test2PRF) == 1


def test_eval_metrics() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    scorer = SCALEScorer(size="small", device=device)
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
        ["", ""],
        [
            [""],
            ["", "", ""],
        ],
    )

    test1PRF = scorer.score(
        test1[0],
        test1[1],
    )
    metrics = scale_score.utils.eval_metrics([1, 1, 1, 1, 1], test1PRF)
    assert len(metrics) == 13

    test2PRF = scorer.score(
        test2[0],
        test2[1],
    )
    metrics = scale_score.utils.eval_metrics([1, 1, 1, 1], test2PRF)
    assert len(metrics) == 13


def test_get_f1s() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    scorer = SCALEScorer(size="small", device=device)

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
        ["", ""],
        [
            [""],
            ["", "", ""],
        ],
    )

    test1PRF = scorer.score(
        test1[0],
        test1[1],
    )
    preds = list(map(lambda x: 0 if x < 0.5 else 1, test1PRF))
    metrics = scale_score.utils.get_f1s(preds, [1, 1, 1, 1, 1])
    assert len(metrics) == 4

    test2PRF = scorer.score(
        test2[0],
        test2[1],
    )
    preds = list(map(lambda x: 0 if x < 0.5 else 1, test2PRF))
    metrics = scale_score.utils.get_f1s(preds, [1, 1, 1, 1])
    assert len(metrics) == 4


def test_print_metrics() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    scorer = SCALEScorer(size="small", device=device)

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
        ["", ""],
        [
            [""],
            ["", "", ""],
        ],
    )

    test1PRF = scorer.score(
        test1[0],
        test1[1],
    )
    metrics = scale_score.utils.eval_metrics([1, 1, 1, 1, 1], test1PRF)
    scale_score.utils.print_metrics(metrics)

    test2PRF = scorer.score(
        test2[0],
        test2[1],
    )
    metrics = scale_score.utils.eval_metrics([1, 1, 1, 1], test2PRF)
    scale_score.utils.print_metrics(metrics)


def test_print_f1s() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    scorer = SCALEScorer(size="small", device=device)

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
        ["", ""],
        [
            [""],
            ["", "", ""],
        ],
    )

    test1PRF = scorer.score(
        test1[0],
        test1[1],
    )
    preds = list(map(lambda x: 0 if x < 0.5 else 1, test1PRF))
    metrics = scale_score.utils.get_f1s(preds, [1, 1, 1, 1, 1])
    scale_score.utils.print_f1s(metrics)

    test2PRF = scorer.score(
        test2[0],
        test2[1],
    )
    preds = list(map(lambda x: 0 if x < 0.5 else 1, test2PRF))
    metrics = scale_score.utils.get_f1s(preds, [1, 1, 1, 1])
    scale_score.utils.print_f1s(metrics)
