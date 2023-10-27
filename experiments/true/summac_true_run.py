import json

import nltk
import numpy as np
import pandas as pd
from summac.model_summac import SummaCConv, SummaCZS
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

model_zs = SummaCZS(
    granularity="sentence", model_name="vitc", device="cuda"
)  # If you have a GPU: switch to: device="cuda"
model_conv = SummaCConv(
    models=["vitc"],
    bins="percentile",
    granularity="sentence",
    nli_labels="e",
    device="cuda",
    start_file="default",
    agg="mean",
)

results_zs = {}
results_conv = {}
for k in tqdm(data.keys()):
    inf_summ = data[k]["generated_text"].tolist()
    convo = data[k]["grounding"].tolist()
    label = data[k]["label"].tolist()
    zs = []
    conv = []
    for i in tqdm(range(len(convo))):
        try:
            nltk.tokenize.sent_tokenize(inf_summ[i])
            summ = inf_summ[i]
        except:
            summ = ""
        if convo[i][0] == "." or convo[i][0] == "?" or convo[i][0] == "!":
            con = convo[i][1:]
        else:
            con = convo[i]
        zs.append(model_zs.score([con], [summ])["scores"][0])
        conv.append(model_conv.score([con], [summ])["scores"][0])

    results_zs[k] = [zs, label]
    results_conv[k] = [conv, label]
    with open("summac_zs_true_results_3.json", "w") as file:
        json.dump(results_zs, file)
    with open("summac_conv_true_results_3.json", "w") as file:
        json.dump(results_conv, file)
