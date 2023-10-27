import json

import nltk
import numpy as np
import pandas as pd
from summac.model_summac import SummaCConv, SummaCZS
from tqdm import tqdm

with open("../../data/screen_eval.json", "r") as file:
    data = json.load(file)

model_zs = SummaCZS(
    granularity="sentence",
    model_name="vitc",
    device="cuda",
    max_doc_sents=800,
    batch_size=1,
)  # If you have a GPU: switch to: device="cuda"
model_conv = SummaCConv(
    models=["vitc"],
    bins="percentile",
    granularity="sentence",
    nli_labels="e",
    device="cuda",
    start_file="default",
    agg="mean",
    max_doc_sents=800,
    batch_size=1,
)

results_zs = []
results_conv = []

inf_summ = [data["inferred_summary"][k] for k in data["inferred_summary"].keys()]
convo = [" ".join(data["convo"][k]) for k in data["convo"].keys()]
label = [data["agg_label"][k] for k in data["agg_label"].keys()]
zs = []
conv = []
for i in tqdm(range(len(convo))):
    zs.append(model_zs.score([convo[i]], [inf_summ[i]])["scores"][0])
    conv.append(model_conv.score([convo[i]], [inf_summ[i]])["scores"][0])

results_zs.append([zs, label])
results_conv.append([conv, label])
with open("results/summac_zs_results.json", "w") as file:
    json.dump(results_zs, file)
with open("results/summac_conv_results.json", "w") as file:
    json.dump(results_conv, file)
