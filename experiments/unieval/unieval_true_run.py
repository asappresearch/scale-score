import json

import nltk
import numpy as np
import pandas as pd
from metric.evaluator import get_evaluator
from tqdm import tqdm
from utils import convert_to_json

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

task = "summarization"
evaluator = get_evaluator("summarization")

results = {}
for k in tqdm(data.keys()):
    inf_summ = data[k]["generated_text"].tolist()
    convo = data[k]["grounding"].tolist()
    label = data[k]["label"].tolist()
    res = []
    for i in tqdm(range(len(convo))):
        try:
            nltk.tokenize.sent_tokenize(inf_summ[i])
            summ = inf_summ[i]
        except:
            res.append(0.0)
            continue
        json_data = convert_to_json(output_list=[summ], src_list=[convo[i]])
        res.append(
            evaluator.evaluate(
                json_data, dims=["consistency"], overall=False, print_result=False
            )[0]["consistency"]
        )
    results[k] = [res, label]
    with open("unieval_true_results_2.json", "w") as file:
        json.dump(results, file)
