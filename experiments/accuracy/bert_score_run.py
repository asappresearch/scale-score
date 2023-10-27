import json
import time

import numpy as np
import pandas as pd
from bert_score import BERTScorer
from tqdm import tqdm

with open("../../data/screen_eval.json", "r") as file:
    data = json.load(file)

scorer = BERTScorer(lang="en", rescale_with_baseline=True)

inf_summ = [data["inferred_summary"][k] for k in data["inferred_summary"].keys()]
convo = ["\n".join(data["convo"][k]) for k in data["convo"].keys()]
label = [int(data["agg_label"][k]) for k in data["agg_label"].keys()]

t0 = time.time()
P, R, F1 = scorer.score(convo, inf_summ)
t1 = time.time()

results_p = [P.tolist(), label, t1 - t0]
results_r = [R.tolist(), label, t1 - t0]
results_f = [F1.tolist(), label, t1 - t0]
with open("results/bert_score_p_results.json", "w") as file:
    json.dump(results_p, file)
with open("results/bert_score_r_results.json", "w") as file:
    json.dump(results_r, file)
with open("results/bert_score_f_results.json", "w") as file:
    json.dump(results_f, file)
