import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from summac.model_summac import SummaCZS, SummaCConv
import time


with open('../../data/screen_eval.json', 'r') as file:
    data=json.load(file)
    
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean", max_doc_sents=800, imager_load_cache=False,)

inf_summ = [data['inferred_summary'][k] for k in data['inferred_summary'].keys()]
convo = ['\n'.join(data['convo'][k]) for k in data['convo'].keys()]
label = [int(data['agg_label'][k]) for k in data['agg_label'].keys()]

t0 = time.time()
conv = model_conv.score(convo, inf_summ,  batch_size=1)['scores']
t1=time.time()

results_conv = [conv, label, t1-t0]
with open('results/summac_conv_results.json', 'w') as file:
    json.dump(results_conv, file)