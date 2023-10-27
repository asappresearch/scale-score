import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

with open("../../data/screen_eval.json", "r") as file:
    data = json.load(file)

from qafacteval import QAFactEval

kwargs = {
    "cuda_device": 0,
    "use_lerc_quip": True,
    "verbose": True,
    "generation_batch_size": 1,
    "answering_batch_size": 1,
    "lerc_batch_size": 1,
}

model_folder = (
    "../../QAFactEval/models"  # path to models downloaded with download_models.sh
)
metric = QAFactEval(
    lerc_quip_path=f"{model_folder}/quip-512-mocha",
    generation_model_path=f"{model_folder}/generation/model.tar.gz",
    answering_model_dir=f"{model_folder}/answering",
    lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
    lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
    **kwargs,
)

label = [int(data["agg_label"][k]) for k in data["agg_label"].keys()]
inf_summ = [[data["inferred_summary"][k]] for k in data["inferred_summary"].keys()]
convo = ["\n".join(data["convo"][k]) for k in data["convo"].keys()]

scores = []
t0 = time.time()
for i in range(len(convo)):
    for j in range(len(inf_summ[i])):
        c = [convo[i]]
        s = [[inf_summ[i][j]]]
        res = metric.score_batch_qafacteval(c, s, return_qa_pairs=True)
        score = res[0][0]["qa-eval"]["lerc_quip"]
        scores.append(score)
t1 = time.time()

with open(f"results/qa_fact_results.json", "w") as file:
    json.dump([scores, label, t1 - t0], file)
