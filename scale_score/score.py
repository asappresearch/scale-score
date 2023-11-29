from typing import List, Optional, Tuple

from transformers import T5ForConditionalGeneration, T5Tokenizer

from scale_score.utils import get_flan_T5_model, scale_retrieve, scale_score


def score(
    premise: List[str],
    hypothesis: List[List[str]],
    chunk_size: int = 1000,
    window_size: float = 0.25,
    size: str = "xl",
    device: str = "cuda",
    model_path: Optional[str] = None,
    model: Optional[T5ForConditionalGeneration] = None,
    tokenizer: Optional[T5Tokenizer] = None,
) -> List[float]:
    """
    SCALE Score
    Takes in a list of ground truth premises and their associated hypothesis.
    Calculates the SCALE metric score.
    Args:
        - :param: 'premise' (list of str): premise text, the ground truth
        - :param: 'hypothesis' (list of list of str): hypothesis text, usually the text predicted by a model being evaluated
        - :param: 'chunk_size' (int): Size of the chunks used to break up the premise
        - :param: 'window_size' (float): Amount of overlap between chunks (percentage)
        - :param: 'size' (str): Size of FlanT5 model, options are 'small', 'base', 'large', 'xl', 'xxl'
        - :param: 'device' (str): device to send the model to, by default is 'cuda'
        - :param: 'model_path' (str): Optional path to a Flan-T5 model to load
        - :param: 'model' (T5ForConditionalGeneration): Optional model to use for scoring
        - :param: 'tokenizer' (T5Tokenizer): Optional tokenizer to use for scoring

    Return:
        - :param: 'results' (list): List of shape (N) where N = number of premise hypothesis pairs
    """
    if len(premise) != len(hypothesis):
        raise ValueError(
            "There must be an equal number of premise and predicted summaries"
        )
    if (model is None and tokenizer is not None) or (
        model is not None and tokenizer is None
    ):
        raise ValueError("You must specify both model and tokenizer or neither")

    if model is None or tokenizer is None:
        model, tokenizer = get_flan_T5_model(size, model_path)

    model = model.eval()

    results = scale_score(
        model=model,
        tokenizer=tokenizer,
        premise=premise,
        hypothesis=hypothesis,
        device=device,
        chunk_size=chunk_size,
        window_size=window_size,
    )

    return results


def retrieve(
    premise: List[List[str]],
    hypothesis: List[List[str]],
    branches: int = 2,
    size: str = "xl",
    device: str = "cuda",
    model_path: Optional[str] = None,
    model: Optional[T5ForConditionalGeneration] = None,
    tokenizer: Optional[T5Tokenizer] = None,
) -> List[Tuple[str, float]]:
    """
    SCALE Score
    Takes in a list of ground truth premises and their associated hypothesis.
    Calculates the SCALE metric score.
    Args:
        - :param: 'premise' (list of list of str): premise text, the ground truth
        - :param: 'hypothesis' (list of list of str): hypothesis text, usually the text predicted by a model being evaluated
        - :param: 'branches' (int): The number of branches to have in the search tree
        - :param: 'size' (str): Size of FlanT5 model, options are 'small', 'base', 'large', 'xl', 'xxl'
        - :param: 'device' (str): device to send the model to, by default is 'cuda'
        - :param: 'model_path' (str): Optional path to a Flan-T5 model to load
        - :param: 'model' (T5ForConditionalGeneration): Optional model to use for scoring
        - :param: 'tokenizer' (T5Tokenizer): Optional tokenizer to use for scoring

    Return:
        - :param: 'results': List[Tuple[str, float]] a list of the retrieved most relevant unit from the premise for each hypothesis
    """
    if len(premise) != len(hypothesis):
        raise ValueError(
            "There must be an equal number of premise and predicted summaries"
        )
    if (model is None and tokenizer is not None) or (
        model is not None and tokenizer is None
    ):
        raise ValueError("You must specify both model and tokenizer or neither")

    if model is None or tokenizer is None:
        model, tokenizer = get_flan_T5_model(size, model_path)

    model = model.eval()

    results = scale_retrieve(
        model=model,
        tokenizer=tokenizer,
        premise=premise,
        hypothesis=hypothesis,
        device=device,
        branches=branches,
    )

    return results
