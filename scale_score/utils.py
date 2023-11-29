import math
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


def get_chunks(
    tokenizer: T5Tokenizer,
    joined_convo: str,
    prompt: str,
    chunk_size: Optional[int] = None,
    window_size: float = 0.25,
) -> List[torch.Tensor]:
    if chunk_size is None:
        chunk_size = tokenizer.model_max_length

    fulltext = prompt.replace("{{premise}}", joined_convo)
    full_tokens = tokenizer(fulltext, return_tensors="pt").input_ids
    if len(full_tokens[0]) < chunk_size:
        return [full_tokens]

    pre_premise_prompt = prompt.split("{{premise}}")[0]
    post_premise_prompt = prompt.split("{{premise}}")[1]

    pre_premise_tokens = tokenizer(
        pre_premise_prompt, return_tensors="pt"
    ).input_ids.squeeze(0)[:-1]
    post_premise_tokens = tokenizer(
        post_premise_prompt, return_tensors="pt"
    ).input_ids.squeeze(0)

    prompt_token_len = len(pre_premise_tokens) + len(post_premise_tokens)

    convo_tokens = tokenizer(
        joined_convo, return_tensors="pt", truncation=False
    ).input_ids.squeeze(0)

    chunk_size_mod = chunk_size - prompt_token_len
    num_windows = math.ceil(
        (len(convo_tokens) - 1) / (chunk_size_mod * (1 - window_size))
    )
    chunks = []
    for i in range(num_windows):
        begin = int(chunk_size_mod * (1 - window_size)) * i
        end = begin + chunk_size_mod
        end = -1 if end >= len(convo_tokens) else end
        if len(pre_premise_tokens) == 0:
            pre = convo_tokens[begin:end]
        else:
            pre = torch.cat([pre_premise_tokens, convo_tokens[begin:end]])

        post = torch.cat([pre, post_premise_tokens]).unsqueeze(0)
        chunks.append(post)
    return chunks


def get_retrieval_chunks(
    tokenizer: T5Tokenizer, convo: List[str], prompt: str, branches: int = 2
) -> Tuple[List[Any], List[List[str]]]:
    num_sents = len(convo)
    if num_sents == 1:
        return [
            tokenizer(
                prompt.replace("{{premise}}", convo[0]), return_tensors="pt"
            ).input_ids
        ], [convo]

    interval = num_sents // branches
    if interval == 0:
        interval = 1

    cs = []
    ranges = []
    for i in range(0, num_sents, interval):
        ranges.append(convo[(i) : (interval + i)])
        pmp = " ".join(convo[(i) : (interval + i)])
        cs.append(
            tokenizer(prompt.replace("{{premise}}", pmp), return_tensors="pt").input_ids
        )

    return cs, ranges


def run_model_chunks(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    premise: List[str],
    prompt: str,
    yes_no_tokens: List[int],
    device: str,
    chunk_size: int = 1000,
    window_size: float = 0.25,
    branches: Optional[int] = None,
) -> Tuple[List[float], Optional[List[List[str]]]]:
    if branches is None:
        texts = None
        chunks = get_chunks(tokenizer, premise[0], prompt, chunk_size, window_size)
    else:
        chunks, texts = get_retrieval_chunks(tokenizer, premise, prompt, branches)
    chunk_results = []
    for chunk in chunks:
        input_ids = chunk.to(device)
        outputs = model.generate(
            input_ids,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1,
        )
        scores = outputs["scores"][0][0][yes_no_tokens]
        chunk_results.append(torch.nn.functional.softmax(scores, dim=0)[0].item())
    return (chunk_results, texts)


def eval_metrics(labels: List[int], preds: List[float]) -> Tuple[float, ...]:
    """
    Inputs:
        labels: List[int] list of binary labels - 1 : the sentence is NOT correct, 0 : the sentence is correct
        preds: List[float] scores, not necessarily between 0 and 1, the higher of the score the predicted label
            is closer to 0
    Outputs:
        major_acc: accuracy if we always predict correct
        best_acc: best accuracy
        best_prec, best_reca, best_f1: best f1 and the corresponding p/r for incorrect sentence detection
        best_acc_90p: best accuracy if we need to perserve 90% correct sentences
        best_acc_70p: best accuracy if we need to perserve 70% correct sentences
    """
    # print("- Positive examples: %d" %(len(labels) - sum(labels)))
    # print("- Negative examples: %d" %(sum(labels)))
    y_true = labels
    y_probs = preds
    pairs = zip(y_true, y_probs)
    sorted_pairs = sorted(pairs, key=lambda p: p[1])
    sorted_labels = [p[0] for p in sorted_pairs]

    accs = []
    precs, recas, fscores = [], [], []
    preserved = []  # percentage of preserved correct sentences
    trim_accs = []  # acc if removed the predicted incorrect sentences
    tholds = []
    for i in range(len(sorted_labels) + 1):
        incorrect = sorted_labels[:i]
        correct = sorted_labels[i:]
        ilist = [x == 1 for x in incorrect]
        clist = [x == 0 for x in correct]
        accs.append(sum(ilist + clist) / len(sorted_labels))

        pcnt = len(ilist)
        gcnt = sum(sorted_labels)
        ccnt = sum(ilist)
        prec, reca = 0.0, 0.0
        if pcnt > 0:
            prec = ccnt / pcnt
        if gcnt > 0:
            reca = ccnt / gcnt
        fscore = 0.0
        if prec + reca > 0:
            fscore = 2 * prec * reca / (prec + reca)
        if i >= len(sorted_pairs):
            tholds.append(sorted_pairs[-1][1])
        else:
            tholds.append(sorted_pairs[i][1])
        precs.append(prec)
        recas.append(reca)
        fscores.append(fscore)

        if (len(sorted_labels) - sum(sorted_labels)) == 0:
            preserved.append(0.0)
        else:
            preserved.append(sum(clist) / (len(sorted_labels) - sum(sorted_labels)))

        if len(clist) == 0:
            trim_accs.append(1.0)
        else:
            trim_accs.append(sum(clist) / len(clist))

    major_acc = 1.0 - sum(y_true) / len(y_true)
    best_acc = max(accs)
    idx = fscores.index(max(fscores))
    thold = tholds[idx]
    best_prec, best_reca, best_f1 = precs[idx], recas[idx], fscores[idx]
    best_acc_90p, best_acc_70p = -1.0, -1.0
    thold_90p, thold_70p = 0.0, 0.0
    for i, (psv, tacc) in enumerate(zip(preserved, trim_accs)):
        if psv < 0.9 and best_acc_90p < 0:
            best_acc_90p = tacc
            thold_90p = tholds[i]
        if psv < 0.7 and best_acc_70p < 0:
            best_acc_70p = tacc
            thold_70p = tholds[i]
    p_stat = pearsonr(preds, labels)[0]
    s_stat = spearmanr(preds, labels)[0]
    kt_stat = kendalltau(preds, labels)[0]

    return (
        p_stat,
        s_stat,
        kt_stat,
        major_acc,
        best_acc,
        best_prec,
        best_reca,
        best_f1,
        best_acc_90p,
        best_acc_70p,
        thold,
        thold_90p,
        thold_70p,
    )


def print_metrics(results: Tuple[float, ...]) -> None:
    (
        p_stat,
        s_stat,
        kt_stat,
        major_acc,
        best_acc,
        best_prec,
        best_reca,
        best_f1,
        best_acc_90p,
        best_acc_70p,
        thold,
        thold_90p,
        thold_70p,
    ) = results
    print("- Pearson: %.4f" % p_stat)  # type: ignore
    print("- Spearman: %.4f" % s_stat)  # type: ignore
    print("- KendallTau: %.4f" % kt_stat)  # type: ignore
    print("- Majority class accuracy: %.4f" % major_acc)  # type: ignore
    print("- Best accuracy: %.4f" % best_acc)  # type: ignore
    print("- Best detection precision: %.4f" % best_prec)  # type: ignore
    print("- Best detection recall: %.4f" % best_reca)  # type: ignore
    print("- Best detection F1: %.4f" % best_f1)  # type: ignore
    print("- Accuracy@90%%: %.4f" % best_acc_90p)  # type: ignore
    print("- Accuracy@70%%: %.4f" % best_acc_70p)  # type: ignore
    print("- Threshold F1: %.4f" % thold)  # type: ignore
    print("- Threshold@90%%: %.4f" % thold_90p)  # type: ignore
    print("- Threshold@70%%: %.4f" % thold_70p)  # type: ignore


def get_f1s(preds: List[int], labels: List[int]) -> List[float]:
    f1s = []
    f1s.append(f1_score(labels, preds, average="binary", pos_label=0))
    f1s.append(f1_score(labels, preds, average="macro"))
    f1s.append(f1_score(labels, preds, average="micro"))
    f1s.append(f1_score(labels, preds, average="weighted"))
    return f1s


def print_f1s(f1s: List[float]) -> None:
    print("- Binary: %.4f" % f1s[0])
    print("- Macro: %.4f" % f1s[1])
    print("- Micro: %.4f" % f1s[2])
    print("- Weighted: %.4f" % f1s[3])


def get_flan_T5_model(
    size: str, model_path: Optional[str] = None
) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    tokenizer = T5Tokenizer.from_pretrained(f"google/flan-t5-{size}")
    model = T5ForConditionalGeneration.from_pretrained(
        f"google/flan-t5-{size}", device_map="auto"
    )

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model, tokenizer


def scale_score(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    premise: List[str],
    hypothesis: List[List[str]],
    device: str = "cuda",
    chunk_size: int = 1000,
    window_size: float = 0.25,
) -> List[float]:
    """
    SCALE NLI Precision and Recall Calculation
    Args:
        - :param: 'model' (T5ForConditionalGeneration): Flan-T5 model used to score
        - :param: 'tokenizer' (T5Tokenizer): Flan-T5 tokenizer associated with model
        - :param: 'premise' (list of str): premise text, the ground truth
        - :param: 'hypothesis' (list of list of str): hypothesis text, usually the text predicted by a model
        being evaluated, split by sentence or utterance
        - :param: 'device' (str): device that model is on to send input_ids to, default is 'cuda'
        - :param: 'chunk_size' (int): Size of the chunks used to break up the premise
        - :param: 'window_size' (float): Amount of overlap between chunks (percentage)
    Return:
        - :param: 'results' (list): List of shape (N) where N = number of hypothesis of floats
    """
    yes_no_tokens = [tokenizer("Yes").input_ids[0], tokenizer("No").input_ids[0]]
    prompt = '{{premise}} Question: Does this imply that "{{hypothesis}}"? Yes or No?'

    results = []

    for i in tqdm(range(len(premise))):
        for summ_sentence in hypothesis[i]:
            prompt_part_filled = prompt.replace("{{hypothesis}}", summ_sentence)

            chunk_results, _ = run_model_chunks(
                model=model,
                tokenizer=tokenizer,
                premise=[premise[i]],
                prompt=prompt_part_filled,
                chunk_size=chunk_size,
                window_size=window_size,
                yes_no_tokens=yes_no_tokens,
                device=device,
            )

            results.append(max(chunk_results))

    return results


def scale_retrieve(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    premise: List[List[str]],
    hypothesis: List[List[str]],
    branches: int = 2,
    device: str = "cuda",
) -> List[Tuple[str, float]]:
    """
    Flan-T5 NLI Precision and Recall Calculation
    Args:
        - :param: 'model' (T5ForConditionalGeneration): Flan-T5 model used to score
        - :param: 'tokenizer' (T5Tokenizer): Flan-T5 tokenizer associated with model
        - :param: 'premise' (list of list of str): premise text, the ground truth, split by sentence or utterance
        - :param: 'hypothesis' (list of list of str): hypothesis text, usually the text predicted by a model
        being evaluated, split by sentence or utterance
        - :param: 'device' (str): device that model is on to send input_ids to, default is 'cuda'
        - :param: 'branches' (int): The number of branches to have in the search tree
    Return:
        - :param: 'results': List[Tuple[str, float]] a list of the retrieved most relevant unit from the premise for each hypothesis
    """
    yes_no_tokens = [tokenizer("Yes").input_ids[0], tokenizer("No").input_ids[0]]
    prompt = '{{premise}} Question: Does this imply that "{{hypothesis}}"? Yes or No?'
    results = []
    for i in range(len(premise)):
        # Precision calculation
        for idx, summ_sentence in enumerate(hypothesis[i]):
            prompt_part_filled = prompt.replace("{{hypothesis}}", summ_sentence)
            utts = premise[i]
            # First try
            chunk_results, texts = run_model_chunks(
                model=model,
                tokenizer=tokenizer,
                premise=utts,
                prompt=prompt_part_filled,
                branches=branches,
                yes_no_tokens=yes_no_tokens,
                device=device,
            )
            if texts is None:
                raise ValueError("texts cannot be None")
            max_chunk = np.argmax(chunk_results)
            utts = texts[max_chunk]
            # Optionally continue descending
            while len(utts) > 1:
                chunk_results, texts = run_model_chunks(
                    model=model,
                    tokenizer=tokenizer,
                    premise=utts,
                    prompt=prompt_part_filled,
                    branches=branches,
                    yes_no_tokens=yes_no_tokens,
                    device=device,
                )
                if texts is None:
                    raise ValueError("texts cannot be None")
                max_chunk = np.argmax(chunk_results)
                utts = texts[max_chunk]

            results.append((utts[0], max(chunk_results)))

    return results
