from typing import List, Optional, Tuple

from scale_score.score import retrieve, score
from scale_score.utils import get_flan_T5_model


class SCALEScorer:
    """
    SCALE Scorer object
    """

    def __init__(
        self, size: str = "xl", model_path: Optional[str] = None, device: str = "cuda"
    ) -> None:
        """
        SCALE Score Init
        Builds a scorer object
        Args:
            - :param: 'size' (str): Size of FlanT5 model, options are 'small', 'base', 'large', 'xl', 'xxl'
            - :param: 'device' (str): device to send the model to, by default is 'cuda'
            - :param: 'model_path' (str): Optional model path to load a model
        Return:
            - :param: 'results' (list): List of shape (N) where N = number of premise hypothesis pairs
        """
        self.model, self.tokenizer = get_flan_T5_model(size, model_path)
        self.model = self.model.eval()
        self.model_size = size
        self.device = device

    def score(
        self,
        premise: List[str],
        hypothesis: List[List[str]],
        chunk_size: int = 1000,
        window_size: float = 0.25,
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

        Return:
            - :param: 'results' (list): List of shape (N) where N = number of premise hypothesis pairs
        """
        results = score(
            premise,
            hypothesis,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            chunk_size=chunk_size,
            window_size=window_size,
        )

        return results

    def retrieve(
        self,
        premise: List[List[str]],
        hypothesis: List[List[str]],
        branches: int = 2,
    ) -> List[Tuple[str, float]]:
        """
        SCALE Score
        Takes in a list of ground truth premises and their associated hypothesis.
        Calculates the SCALE metric score.
        Args:
            - :param: 'premise' (list of list of str): premise text, the ground truth
            - :param: 'hypothesis' (list of list of str): hypothesis text, usually the text predicted by a model being evaluated
            - :param: 'branches' (int): The number of branches to have in the search tree

        Return:
            - :param: 'results': List[Tuple[str, float]] a list of the retrieved most relevant unit from the premise for each hypothesis
        """
        results = retrieve(
            premise=premise,
            hypothesis=hypothesis,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            branches=branches,
        )

        return results
