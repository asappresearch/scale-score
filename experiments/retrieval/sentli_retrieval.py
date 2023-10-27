import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Optional


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


def run_model_chunks(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    premise: str,
    prompt: str,
    chunk_size: int,
    window_size: float,
    yes_no_tokens: List[int],
    device: str,
) -> List[float]:
    chunks = get_chunks(tokenizer, premise, prompt, chunk_size, window_size)
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
    return chunk_results


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


def sentli_score(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    convo: List[List[str]],
    predicted_summary: List[List[str]],
    gold_summary: Optional[List[List[str]]] = None,
    do_reference_free_recall: bool = False,
    device: str = "cuda",
    retrieval: bool = False,
    retrieval_top_k: int = 3,
    chunk_size: int = 512,
    window_size: float = 0.25,
) -> dict:
    """
    Sentli Precision and Recall Calculation
    Args:
        - :param: 'model' (T5ForConditionalGeneration): Flan-T5 model used to score
        - :param: 'tokenizer' (T5Tokenizer): Flan-T5 tokenizer associated with model
        - :param: 'convo' (list of list of str): list of dialogues split into utterances
        - :param: 'predicted_summary' (list of list of str): list of predicted summaries split by sentence
        - :param: 'gold_summary' (list of list of str): list of gold standard summaries split by sentence
        - :param: 'do_reference_free_recall' (bool): set to true if reference free recall and f1 should be
            calculated
        - :param: 'device' (str): device that model is on to send input_ids to, default is 'cuda'
        - :param: 'retrieval' (bool): set to true if model should be run on utterance level rather than
            chunk level and retrieve top k most similar utterances
        - :param: 'retrieval_top_k' (int): Define number of top utterances to be returned
    Return:
        - :param: 'results': Dict: Dictionary containing Precision and Recall scores both structured
            (nested lists) and unstructured (one list) as well as retrieved reference sentences for
            both precision and recall. If gold_summary is None and do_reference_free_recall is False,
            F1, R, R_structured, and R_reference are None. If retrieval is False P_reference and R_reference
            are None.
    """
    yes_no_tokens = [tokenizer("Yes").input_ids[0], tokenizer("No").input_ids[0]]
    prompt = '{{premise}} Question: Does this imply that "{{hypothesis}}"? Yes or no?'

    results: dict = {"P": [], "P_structured": []}
    if retrieval:
        results["P_reference"] = []
        results["R_reference"] = []
    if gold_summary is not None or do_reference_free_recall:
        results["F1"] = []
        results["R"] = []
        results["R_structured"] = []

    for i in tqdm(range(len(convo))):
        summ_scores_precision = []
        if retrieval:
            inf_summ_reference_P = []
            inf_summ_reference_R = []
        if gold_summary is not None or do_reference_free_recall:
            summ_scores_recall = []

        # Precision calculation
        for summ_sentence in predicted_summary[i]:
            prompt_part_filled = prompt.replace("{{hypothesis}}", summ_sentence)

            if retrieval:
                chunk_results = []
                for utterance in convo[i]:
                    chunk_results.append(
                        max(
                            run_model_chunks(
                                model,
                                tokenizer,
                                utterance,
                                prompt_part_filled,
                                chunk_size,
                                window_size,
                                yes_no_tokens,
                                device,
                            )
                        )
                    )
            else:
                chunk_results = run_model_chunks(
                    model,
                    tokenizer,
                    " ".join(convo[i]),
                    prompt_part_filled,
                    chunk_size,
                    window_size,
                    yes_no_tokens,
                    device,
                )

            summ_scores_precision.append(max(chunk_results))
            results["P"].append(max(chunk_results))

            if retrieval:
                top = (
                    retrieval_top_k
                    if len(chunk_results) >= retrieval_top_k
                    else len(chunk_results)
                )
                topk_args = np.argpartition(chunk_results, -top)[-top:]
                topkutts = [(convo[i][x], chunk_results[x]) for x in topk_args]
                inf_summ_reference_P.append(topkutts)

        # Recall Calculation
        if gold_summary is not None or do_reference_free_recall:
            recall_hypothesis = (  # type: ignore
                convo[i] if do_reference_free_recall else gold_summary[i]  # type: ignore
            )  # type: ignore

            for summ_sentence in recall_hypothesis:  # type: ignore
                prompt_part_filled = prompt.replace("{{hypothesis}}", summ_sentence)
                if retrieval:
                    chunk_results = []
                    for pred_summary in predicted_summary[i]:
                        chunk_results.append(
                            max(
                                run_model_chunks(
                                    model,
                                    tokenizer,
                                    pred_summary,
                                    prompt_part_filled,
                                    chunk_size,
                                    window_size,
                                    yes_no_tokens,
                                    device,
                                )
                            )
                        )
                else:
                    chunk_results = run_model_chunks(
                        model,
                        tokenizer,
                        " ".join(predicted_summary[i]),
                        prompt_part_filled,
                        chunk_size,
                        window_size,
                        yes_no_tokens,
                        device,
                    )

                summ_scores_recall.append(max(chunk_results))
                results["R"].append(max(chunk_results))

                if retrieval:
                    top = (
                        retrieval_top_k
                        if len(chunk_results) >= retrieval_top_k
                        else len(chunk_results)
                    )
                    topk_args = np.argpartition(chunk_results, -top)[-top:]
                    topkutts = [
                        (predicted_summary[i][x], chunk_results[x]) for x in topk_args
                    ]
                    inf_summ_reference_R.append(topkutts)

        results["P_structured"].append(summ_scores_precision)
        if gold_summary is not None or do_reference_free_recall:
            results["R_structured"].append(summ_scores_recall)

            P = np.mean(summ_scores_precision)
            R = np.mean(summ_scores_recall)
            if P + R == 0:
                results["F1"].append(0.0)
            else:
                F1 = 2 * P * R / (P + R)
                results["F1"].append(F1)

        if retrieval:
            results["P_reference"].append(inf_summ_reference_P)
            if gold_summary is not None or do_reference_free_recall:
                results["R_reference"].append(inf_summ_reference_R)

    return results


def score(
    convos: List[List[str]],
    predicted_summaries: List[List[str]],
    gold_summaries: Optional[List[List[str]]] = None,
    averaging: str = "micro",
    size: str = "xl",
    do_reference_free_recall: bool = False,
    device: str = "cuda",
    model_path: Optional[str] = None,
    retrieval: bool = False,
    retrieval_top_k: int = 3,
    model: Optional[T5ForConditionalGeneration] = None,
    tokenizer: Optional[T5Tokenizer] = None,
    chunk_size: int = 512,
    window_size: float = 0.25,
) -> dict:
    """
    Sentli Score
    Takes in a list of conversations, their associated summary sentences, and their reference gold summary sentences.
    Calculates Precision, Recall, and F1-Scores.
    Args:
        - :param: 'convos' (list of list of str): dialogue sentences
        - :param: 'predicted_summaries' (list of list of str): summary sentences
        - :param: 'gold_summaries' (list of list of str): gold standard summary sentences
        - :param: 'averaging' (str): Options are 'micro', 'macro' and 'none'. 'micro' waits to
            average precision and recall until both have been computed for the entire dataset, then F1-Score
            is calculated based on the dataset average precision and recall. 'macro' averages the
            precision and recall at the conversation level before averaging on the datset level. The F1-Score
            is calculated based on the conversation averaged precision in recall.
        - :param: 'size' (str): Size of FlanT5 model, options are 'small', 'base', 'large', 'xl', 'xxl'
        - :param: 'do_reference_free_recall' (bool): flag that indicates whether to calculate recall using the
            source document. False by default.
        - :param: 'device' (str): device to send the model to, by default is 'cuda'
        - :param: 'model_path' (str): Optional path to a Flan-T5 model to load
        - :param: 'retrieval' (bool): Flag saying whether or not to do the retrieval task while scoring. Note
            this will evaluate scores on the utterance level rather than the chunk level
        - :param: 'retrieval_top_k' (int): Number of utterances to retrieve per summary sentence when the
            retrieval task is True
        - :param: 'model' (T5ForConditionalGeneration): Optional model to use for scoring
        - :param: 'tokenizer' (T5Tokenizer): Optional tokenizer to use for scoring
    Return:
        - :param: 'results' (dict): Dictionary containing P, R, F1 results. P, R, F1 correspond to a list of
            the final scores, P_structured, R_structured, and F1_structured are the same scores nested into
            the same structure as the input convos. Averaged is a tuple(float, float, float) corresponding to
            the scores averaged over the corpus according to the averaging parameter. All entries pertaining
            to R and F1 will be None if gold_summaries is None and do_reference_free_recall is False.
    """
    if len(convos) != len(predicted_summaries):
        raise ValueError(
            "There must be an equal number of convos and predicted summaries"
        )
    if gold_summaries is not None and len(predicted_summaries) != len(gold_summaries):
        raise ValueError(
            "There must be an equal number of gold and predicted summaries"
        )
    if (model is None and tokenizer is not None) or (
        model is not None and tokenizer is None
    ):
        raise ValueError("You must specify both model and tokenizer or neither")

    if model is None or tokenizer is None:
        model, tokenizer = get_flan_T5_model(size, model_path)

    model = model.eval()

    results = sentli_score(
        model,
        tokenizer,
        convos,
        predicted_summaries,
        gold_summaries,
        do_reference_free_recall,
        device,
        retrieval,
        retrieval_top_k,
        chunk_size,
        window_size,
    )

    results["average"] = [None, None, None]

    if averaging == "micro":
        results["average"][0] = float(np.mean(results["P"]))
        if gold_summaries is not None or do_reference_free_recall:
            results["average"][1] = float(np.mean(results["R"]))
            results["average"][2] = float(np.mean(results["F1"]))

    elif averaging == "macro":
        P_convo = []
        if gold_summaries is not None or do_reference_free_recall:
            R_convo = []
        for i in range(len(convos)):
            P_convo.append(np.mean(results["P_structured"][i]))
            if gold_summaries is not None or do_reference_free_recall:
                R_convo.append(np.mean(results["R_structured"][i]))

        results["average"][0] = float(np.mean(P_convo))
        if gold_summaries is not None or do_reference_free_recall:
            results["average"][1] = float(np.mean(R_convo))
            results["average"][2] = float(np.mean(results["F1"]))
    else:
        raise ValueError("Invalid averaging value")

    return results



class SentliScorer:

    def __init__(
        self, size: str = "xl", model_path: Optional[str] = None, device: "str" = "cuda"
    ) -> None:
        self.model, self.tokenizer = get_flan_T5_model(size, model_path)
        self.model = self.model.eval()
        self.model_size = size
        self.device = device

    def score(
        self,
        convos: List[List[str]],
        predicted_summaries: List[List[str]],
        gold_summaries: Optional[List[List[str]]] = None,
        averaging: str = "micro",
        do_reference_free_recall: bool = False,
        model_path: Optional[str] = None,
        retrieval: bool = False,
        retrieval_top_k: int = 3,
        chunk_size: int = 512,
        window_size: float = 0.25,
    ) -> dict:
        """
        Sentli
        Takes in a list of conversations, their associated summary sentences, and their reference
            gold summary sentences.
        Calculates Precision, Recall, and F1-Scores.
        Args:
            - :param: 'convos' (list of list of str): dialogue sentences
            - :param: 'predicted_summaries' (list of list of str): summary sentences
            - :param: 'gold_summaries' (list of list of str): gold standard summary sentences
            - :param: 'averaging' (str): Options are 'micro', 'macro' and 'none'. 'micro' waits to
                average precision and recall until both have been computed for the entire dataset, then F1-Score
                is calculated based on the dataset average precision and recall. 'macro' averages the
                precision and recall at the conversation level before averaging on the datset level. The F1-Score
                is calculated based on the conversation averaged precision in recall.
            - :param: 'size' (str): Size of FlanT5 model, options are 'small', 'base', 'large', 'xl', 'xxl'
            - :param: 'do_reference_free_recall' (bool): flag that indicates whether to calculate recall using the
                source document. False by default.
            - :param: 'device' (str): device to send the model to, by default is 'cuda'
            - :param: 'model_path' (str): Optional path to a Flan-T5 model to load
            - :param: 'retrieval' (bool): Flag saying whether or not to do the retrieval task while scoring. Note
                this will evaluate
                scores on the utterance level rather than the chunk level
            - :param: 'retrieval_top_k' (int): Number of utterances to retrieve per summary sentence when the retrieval
                task is True
        Return:
            - :param: 'results' (dict): Dictionary containing P, R, F1 results. P, R, F1 correspond to a list of the
                final scores, P_structured, R_structured, and F1_structured are the same scores nested into the same
                structure as the input convos. Averaged is a tuple(float, float, float) corresponding to the scores
                averaged over the corpus according to the averaging parameter. All entries pertaining to R and F1 will
                be None if gold_summaries is None and do_reference_free_recall is False.
        """
        results = score(
            convos,
            predicted_summaries,
            gold_summaries,
            averaging=averaging,
            do_reference_free_recall=do_reference_free_recall,
            retrieval=retrieval,
            retrieval_top_k=retrieval_top_k,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            chunk_size=chunk_size,
            window_size=window_size,
        )

        return results
