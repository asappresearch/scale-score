from typing import List, Optional
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
import math
from typing import List, Optional, Tuple
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import f1_score
from tqdm import tqdm
import nltk


def run_model(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    premise: str,
    prompt: str,
    chunk_size: int,
    window_size: float,
    yes_no_tokens: List[int],
    device: str,
) -> List[float]:
    fulltext = prompt.replace("{{premise}}", premise)
    full_tokens = tokenizer(fulltext, return_tensors="pt", truncation=True, max_length=512).input_ids
    input_ids = full_tokens.to(device)
    outputs = model.generate(
        input_ids,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=1,
    )
    scores = outputs["scores"][0][0][yes_no_tokens]
    results = torch.nn.functional.softmax(scores, dim=0)[0].item()
    return results

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
    convo: List[str],
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
    Sentli NLI Precision and Recall Calculation
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

    results: dict = {"P": [], "P_structured": [], 'label':[]}

    for i in tqdm(range(len(convo))):
        summ_scores_precision = []
        convo_list = convo[i]
        # Precision calculation
        for summ_sentence in predicted_summary[i]:
            prompt_part_filled = prompt.replace("{{hypothesis}}", summ_sentence)

            chunk_results = []
            for utterance in convo_list:
                chunk_results.append(
                        run_model(
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
            sc_max = max(chunk_results)
            sc_min = min(chunk_results)
            sc = [sc_max, sc_min][np.argmax([sc_max, 1-sc_min])]
            lab = int(np.argmax([1-sc_min, sc_max]))
            
            summ_scores_precision.append(sc)
            results["P"].append(sc)
            results['label'].append(lab)

        results["P_structured"].append(summ_scores_precision)

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

    return results



class SentliScorer:
    """
    Sentli Scorer object
    """

    def __init__(
        self, size: str = "xl", model_path: Optional[str] = None, device: "str" = "cuda"
    ) -> None:
        """
        Sentli Score Init
        Builds a scorer object
        Args:
            - :param: 'size' (str): Size of FlanT5 model, options are 'small', 'base', 'large', 'xl', 'xxl'
            - :param: 'device' (str): device to send the model to, by default is 'cuda'
            - :param: 'model_path' (str): Optional model path to load a model
        Return:
            - :param: '(P, R, F)' tuple(float, float, float): Precision, Recall, and F1-Scores. If only_precision
                is true, tuple(float, None, None)
        """
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
        Sentli Score
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
