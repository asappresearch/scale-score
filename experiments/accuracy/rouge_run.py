import json
import time

import nltk
import numpy as np
import pandas as pd
from rouge import Rouge
from tqdm import tqdm

with open("../../data/screen_eval.json", "r") as file:
    data = json.load(file)

rouge = Rouge()

label = [int(data["agg_label"][k]) for k in data["agg_label"].keys()]
inf_summ = [data["inferred_summary"][k] for k in data["inferred_summary"].keys()]
convo = ["\n".join(data["convo"][k]) for k in data["convo"].keys()]

t0 = time.time()
scores = rouge.get_scores(inf_summ, convo)
t1 = time.time()

results_zs = [scores, label, t1 - t0]
with open("results/rouge_results.json", "w") as file:
    json.dump(results_zs, file)

d = [x["rouge-1"]["f"] for x in results_zs[0]]
with open("results/rouge_1_f_results.json", "w") as file:
    json.dump([d, results_zs[1], results_zs[2]], file)
