import json

import numpy as np
import pandas as pd
from bert_score import BERTScorer
from tqdm import tqdm

files = [
    "begin.csv",
    "dialfact.csv",
    "fever.csv",
    "frank.csv",
    "mnbm.csv",
    "paws.csv",
    "q2.csv",
    "qags_cnndm.csv",
    "qags_xsum.csv",
    "summeval.csv",
    "vitc.csv",
]

data = {}
for f in files:
    data[f.split(".")[0]] = pd.read_csv("../../data/true/" + f)

scorer = BERTScorer(lang="en", rescale_with_baseline=True)

results_p = {}
results_r = {}
results_f = {}
for k in tqdm(data.keys()):
    inf_summ = data[k]["generated_text"].tolist()
    convo = data[k]["grounding"].tolist()
    label = data[k]["label"].tolist()
    P, R, F1 = scorer.score(convo, inf_summ)
    results_p[k] = [P.tolist(), label]
    results_r[k] = [R.tolist(), label]
    results_f[k] = [F1.tolist(), label]
    with open("bert_score_p_true_results.json", "w") as file:
        json.dump(results_p, file)
    with open("bert_score_r_true_results.json", "w") as file:
        json.dump(results_r, file)
    with open("bert_score_f_true_results.json", "w") as file:
        json.dump(results_f, file)
