import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from scale_score.scorer import SCALEScorer

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

size = "xl"

scorer = SCALEScorer(size=size, device="cuda")

results = {}
for k in tqdm(data.keys()):
    inf_summ = data[k]["generated_text"].to_numpy()[..., np.newaxis].tolist()
    convo = data[k]["grounding"].to_numpy()[..., np.newaxis].tolist()
    label = data[k]["label"].tolist()
    res = scorer.score(convo, inf_summ)
    results[k] = [res, label]
    with open(f"results/scale_{size}_true_results.json", "w") as file:
        json.dump(results, file)
