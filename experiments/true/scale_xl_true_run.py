import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from scale_score.scorer import SCALEScorer

files = ['begin_dev_download.csv',
         'dialfact_valid_download.csv',
         'fever_dev_download.csv',
         'frank_valid_download.csv',
         'mnbm_download.csv',
         'paws_download.csv',
         'q2_download.csv',
         'qags_cnndm_download.csv',
         'qags_xsum_download.csv',
         'summeval_download.csv',
         'vitc_dev_download.csv',
        ]

data = {}
for f in files:
    data[f.split('.')[0]] = pd.read_csv(f)

size = 'xl'
    
scorer = SCALEScorer(size=size, device='cuda')

results = {}
for k in tqdm(data.keys()):
    inf_summ = data[k]['generated_text'].to_numpy()[..., np.newaxis].tolist()
    convo = data[k]['grounding'].to_numpy()[..., np.newaxis].tolist()
    label = data[k]['label'].tolist()
    res = scorer.score(convo, inf_summ)
    results[k] = [res['P'], label]
    with open(f'results/scale_{size}_true_results.json', 'w') as file:
        json.dump(results, file)