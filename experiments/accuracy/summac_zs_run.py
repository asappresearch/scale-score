import json
import time

import nltk
import numpy as np
import pandas as pd
from summac.model_summac import SummaCConv, SummaCZS
from tqdm import tqdm

with open("../../data/screen_eval.json", "r") as file:
    data = json.load(file)

model_zs = SummaCZS(
    granularity="sentence", model_name="vitc", device="cuda", max_doc_sents=800
)  # If you have a GPU: switch to: device="cuda"


label = [int(data["agg_label"][k]) for k in data["agg_label"].keys()]
inf_summ = [data["inferred_summary"][k] for k in data["inferred_summary"].keys()]
convo = ["\n".join(data["convo"][k]) for k in data["convo"].keys()]

t0 = time.time()
zs = model_zs.score(convo, inf_summ, batch_size=1)["scores"]
t1 = time.time()

results_zs = [zs, label, t1 - t0]
with open("results/summac_zs_results.json", "w") as file:
    json.dump(results_zs, file)
