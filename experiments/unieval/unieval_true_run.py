import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from utils import convert_to_json
from metric.evaluator import get_evaluator

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
    
task = 'summarization'
evaluator = get_evaluator('summarization')
    
results = {}
for k in tqdm(data.keys()):
    inf_summ = data[k]['generated_text'].tolist()
    convo = data[k]['grounding'].tolist()
    label = data[k]['label'].tolist()
    res = []
    for i in tqdm(range(len(convo))):
        try:
            nltk.tokenize.sent_tokenize(inf_summ[i])
            summ=inf_summ[i]
        except:
            res.append(0.0)
            continue
        json_data = convert_to_json(output_list=[summ], src_list=[convo[i]])
        res.append(evaluator.evaluate(json_data, dims=['consistency'], overall=False, print_result=False)[0]['consistency'])
    results[k] = [res, label]
    with open('unieval_true_results_2.json', 'w') as file:
        json.dump(results, file)