import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from scale_score.scorer import SCALEScorer
import time

with open('../../data/screen_eval.json', 'r') as file:
    data=json.load(file)
    
size = 'base'
    
scorer = SCALEScorer(size=size, device='cuda')

label = [int(data['agg_label'][k]) for k in data['agg_label'].keys()]
inf_summ = [[data['inferred_summary'][k]] for k in data['inferred_summary'].keys()]
convo = ['\n'.join(data['convo'][k]) for k in data['convo'].keys()]

t0 = time.time()
res = scorer.score(convo, inf_summ)
t1 = time.time() 

with open(f'results/scale_{size}_results.json', 'w') as file:
    json.dump([res['P'], label, t1-t0], file)