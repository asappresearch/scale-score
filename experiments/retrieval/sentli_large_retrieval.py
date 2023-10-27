import json
import time

import numpy as np
import pandas as pd
from sentli_retrieval import SentliScorer
from tqdm import tqdm

with open("../../data/screen_eval.json", "r") as file:
    data = json.load(file)

fetch = SentliScorer(size="xl")

rank_list = []
scores = []
# Get index and replace with '' to maintian correct idx without duplicate matching
convo = [
    data["original_convo"][k].split("\n")
    for idx, k in enumerate(data["original_convo"].keys())
    if data["agg_label"][f"{idx}"]
]
convo_for_mod = [
    data["original_convo"][k].split("\n")
    for idx, k in enumerate(data["original_convo"].keys())
    if data["agg_label"][f"{idx}"]
]
summaries = [
    data["inferred_summary"][k]
    for idx, k in enumerate(data["original_convo"].keys())
    if data["agg_label"][f"{idx}"]
]
rel_utts = [
    data["rel_utt"][k]
    for idx, k in enumerate(data["rel_utt"].keys())
    if data["agg_label"][f"{idx}"]
]
assert len(rel_utts) == len(convo)
assert len(summaries) == len(convo)

res = []
for i in tqdm(range(len(convo))):
    summary = summaries[i]
    t0 = time.time()
    results = fetch.score([convo[i]], [[summary]], retrieval=True, retrieval_top_k=1)
    t1 = time.time()
    sorted_results = [
        (convo_for_mod[i].index(x[0]), x[1]) for x in results["P_reference"][0][0]
    ]
    sorted_results.sort(key=lambda x: x[1], reverse=True)
    rank_list = [x[0] for x in sorted_results]
    scores = [x[1] for x in sorted_results]

    r1 = int(rank_list[0] in rel_utts[i])
    retrieval = [r1]
    res.append(
        {
            "rank_list": rank_list,
            "scores": scores,
            "rel_utts": rel_utts[i],
            "retrieval": retrieval,
            "time": t1 - t0,
        }
    )
    with open("results/sentli_retrieval.json", "w") as file:
        json.dump(res, file)
