# Fast and Accurate Factual Inconsistency Detection Over Long Documents

## Overview
Introducing SCALE, an reference-free NLI based factual inconsistency detection method, and ScreenEval, the longest dialogue based dataset for factual inconsistency detection presently available.
Both can be found in our paper Fast and Accurate Factual Inconsistency Detection Over Long Documents.

SCALE uses a novel chunking strategy to achieve state-of-the-art factual inconsistency deteciton performance across many NLG domains, tasks, and over long documents (>6k tokens). SCALE's chunking approach enables fast relevant source text retrival over long documents. 

## SCALE

This metrics outputs the estimated probablility that a hypothesis is supported by a given premise *SCALE(premise, hypothesis)*. Commonly the hypothesis is generated text and the premise is some ground truth text. For example, a premise may be a document and the hypothesis may be a language model generated summary sentence. The score is bounded as follows 0&le;*SCALE(premise, hypothesis)*&le;1. A higher score signifies a higher probability the hypothesis is factually consistent with the premise. A lower score signifies the hypothesis is more likely to be factually inconsistent with the premise. It is recommended to use Flan_T5_XL as the base model for the best results. 

### Install
To use the evaluation metric, first pip install the python module. 
```
pip install -e .
```
### Score
#### *Running the Metric*
Import the score function and load your premises, hypothesies. For scoring, the premise is a list of entire document strings while the hypothesis are single sentences represented as is a list of list of strings. Each premise has a list of associated hypothesis with a one to one mapping based on index (premise_0 -> ['hypothesis_0_0', 'hypothesis_0_1'], premise_1-> ['hypothesis_1_0', 'hypothesis_1_1', 'hypothesis_1_2']). 
```python
from scale_score import score

premise = [
    'premise_0',
    'premise_1',
]
hypothesis = [
    ['hypothesis_0_0', 'hypothesis_0_1'],
    ['hypothesis_1_0', 'hypothesis_1_1', 'hypothesis_1_2']
]

results = score(premise, hypothesis)
```
Where the results correspond to each hypothesis scored with it's respecitve premise
```python
results = [
    SCALE(premise_0, hypothesis_0_0), 
    SCALE(premise_0, hypothesis_0_1), 
    SCALE(premise_1, hypothesis_1_0), 
    SCALE(premise_1, hypothesis_1_1),
    SCALE(premise_1, hypothesis_1_2),
]
```


You can also use the `scorer` object to prevent loading the model at every call like so,
```python
from scale_score.scorer import SCALEScorer
scorer = SCALEScorer(size='small', device='cuda')
results = scorer.score(premise, hypothesis)
```
#### *Arguments*
These arguments are the exact same for both `score` and `scorer.score` functions except `scorer.score` does not take in a *size* or *device* as that is set up when building the scorer object. 
| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| premise | List[str] | required | premise text, the ground truth |
| hypothesis | List[List[str]] | required | hypothesis text, usually the text predicted by a model being evaluated |
| chunk_size | int | 512 | The size of the chunks used to perform chunking on the premise |
| window_size | float | 0.25 | The percentage of overlap between chunks. 0&le;window_size&lt;1 |
| size | str | 'xl' | Size of Flan-T5 model, options are 'small', 'base', 'large', 'xl', 'xxl' |
| device | str | 'cuda' | torch device to send the model to. |
| model_path | str | None | Optional path to a Flan-T5 model to load. Note the corresponding size must be specified in the *size* argument. |
| model | T5ForConditionalGeneration | None | Optional model to use for scoring |
| tokenizer | T5Tokenizer | None | Optional tokenizer to use for scoring |

### Evaluation
After scoring, use the `evaluate_scale` function to evaluate the results. 
```python
from scale_score.eval import evaluate_scale
from scale_score.scorer import SCALEScorer
scorer = SCALEScorer(size='small', device='cuda')
results = scorer.score(premise, hypothesis)
metrics = evaluate_scale(results)
```
The arguments for `evaluate_scale` are as follows
| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| results | List[float] | required | Output from scale_score score or scorer run |
| incorrect | List[int] | required | List of labels for summary sentences, 1 for incorrect and 0 for correct |
| threshold | float | 0.5 | Threshold used to calculate binary, micro, macro, and weighted f1 scores |
| out_file | str | None | Optional json filepath to write the metrics to |
| print_outputs | bool | True | Whether to print the metrics |

The metrics that are output are described below. 
| Metric | Description |
| ------ |------ |
| pearson | Pearson correlation | 
| spearman | Spearman correlation | 
| kendalltau | Kendall Tau correlation |
| majority_class_accuracy | Accuracy if we always predict correct | 
| best_accuracy | Best predicted accuracy possible after threshold tuning | 
| best_detection_precision | Best predicted precision possible after threshold tuning f1 score | 
| best_detection_recall | Best predicted recall possible after threshold tuning f1 score | 
| best_detection_f1 | Best predicted f1 possible after threshold tuning | 
| accuracy@90% | Accuracy achieved if we want to keep 90% of all correct sentences | 
| accuracy@70% | Accuracy achieved if we want to keep 70% of all correct sentences | 
| threshold_f1 | Threshold used to calculate best_detection_f1 | 
| threshold_@90% | Threshold used to calculate accuracy@90% | 
| threshold_@70% | Threshold used to calculate accuracy@70% | 
| f1_binary | F1 score of incorrect sentence detection | 
| f1_macro | Average F1 score between correct and incorrect sentence detection | 
| f1_micro | Calculate F1 globally by counting the total true positives, false negatives and false positives | 
| f1_weighted | Calculate F1 for each label, and find their average weighted by support | 


### Retrieve
#### *Running Retrieval*
Import the retrieve function and load your premises, hypothesies. 

**NOTE**: Premises are lists of lists in retrieval. Both premises and hypothesis are split down to the sentence or utterance level. 

Each premise list has an associated hypothesis list with a one to one mapping based on index. 
```python
from scale_score import retrieve

premise = [
    ['premise_0_utt_0', 'premise_0_utt_1', 'premise_0_utt_2'],
    ['premise_1_utt_0', 'premise_1_utt_1'],
]
hypothesis = [
    ['hypothesis_0_0', 'hypothesis_0_1'],
    ['hypothesis_1_0', 'hypothesis_1_1', 'hypothesis_1_2']
]

results = retrieve(premise, hypothesis)
```
Where the results correspond to a list which has the most relevant premise utterance/sentence and the corresponding score.

You can also use the `scorer` object to prevent loading the model at every call like so,
```python
from scale_score.scorer import SCALEScorer
scorer = SCALEScorer(size='small', device='cuda')
results = scorer.retrieve(premise, hypothesis)
```
#### *Arguments*
These arguments are the exact same for both `retrieve` and `scorer.retrieve` functions except `scorer.retrieve` does not take in a *size* or *device* as that is set up when building the scorer object. 
| Argument | Type | Default | Description |
| ------ | ------ | ------ | ------ |
| premise | List[str] | required | premise text, the ground truth |
| hypothesis | List[List[str]] | required | hypothesis text, usually the text predicted by a model being evaluated |
| branches | int | 2 | The number of branches to have in the search tree |
| size | str | 'xl' | Size of Flan-T5 model, options are 'small', 'base', 'large', 'xl', 'xxl' |
| device | str | 'cuda' | torch device to send the model to. |
| model_path | str | None | Optional path to a Flan-T5 model to load. Note the corresponding size must be specified in the *size* argument. |
| model | T5ForConditionalGeneration | None | Optional model to use for scoring |
| tokenizer | T5Tokenizer | None | Optional tokenizer to use for scoring |


## ScreenEval