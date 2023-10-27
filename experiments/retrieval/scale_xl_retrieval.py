import json
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scale_score.scorer import SCALEScorer

with open('../../data/screen_eval.json', 'r') as file:
    data=json.load(file)
    
fetch = SCALEScorer(size='xl')

rank_list = []
scores = []
#Get index and replace with '' to maintian correct idx without duplicate matching
convo = [data['original_convo'][k].split('\n') for idx, k in enumerate(data['original_convo'].keys()) if data['agg_label'][f'{idx}']]
convo_for_mod = [data['original_convo'][k].split('\n') for idx, k in enumerate(data['original_convo'].keys()) if data['agg_label'][f'{idx}']]
summaries = [data['inferred_summary'][k] for idx, k in enumerate(data['original_convo'].keys()) if data['agg_label'][f'{idx}']]
rel_utts = [data['rel_utt'][k] for idx, k in enumerate(data['rel_utt'].keys()) if data['agg_label'][f'{idx}']]
assert len(rel_utts) == len(convo)
assert len(summaries) == len(convo)

res = []
for i in tqdm(range(len(convo))):
    rank_list = []
    scores = []
    summary = summaries[i]
    
    t0=time.time()
    results = fetch.retrieve([convo[i]], [[summary]], branches=2)
    t1 = time.time()
    rel_utt_idx = convo_for_mod[i].index(results['utts'][0][0])
    rank_list.append(rel_utt_idx)
    scores.append(results['scores'][0])
     
    r1 = int(rank_list[0] in rel_utts[i])
    retrieval = [r1]
    res.append({'rank_list':rank_list, 'scores':scores, 'rel_utts':rel_utts[i], 'retrieval':retrieval, 'time':t1-t0})
    with open('results/scale_xl_retrieval.json', 'w') as file:
        json.dump(res, file)
