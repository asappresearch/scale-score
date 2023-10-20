import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from bert_score import BERTScorer

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
    data[f.split('_download')[0]] = pd.read_csv('../'+f)
    
scorer = BERTScorer(lang="en", rescale_with_baseline=True)

results_p = {}
results_r = {}
results_f = {}
for k in tqdm(data.keys()):
    inf_summ = data[k]['generated_text'].tolist()
    convo = data[k]['grounding'].tolist()
    label = data[k]['label'].tolist()
    P, R, F1 = scorer.score(convo, inf_summ)
    results_p[k] = [P.tolist(), label]
    results_r[k] = [R.tolist(), label]
    results_f[k] = [F1.tolist(), label]
    with open('bert_score_p_true_results.json', 'w') as file:
        json.dump(results_p, file)
    with open('bert_score_r_true_results.json', 'w') as file:
        json.dump(results_r, file)
    with open('bert_score_f_true_results.json', 'w') as file:
        json.dump(results_f, file)