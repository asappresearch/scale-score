import functools
import torch
import time
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def superpal_setup(granularity="sentence"):
    tokenizer = AutoTokenizer.from_pretrained("biu-nlp/superpal")
    model = AutoModelForSequenceClassification.from_pretrained("biu-nlp/superpal")
    model.cuda()
    model.eval()
    return functools.partial(superpal_matrix_fn, model=model, tokenizer=tokenizer)

def superpal_matrix_fn(first_list, second_list, model, tokenizer):
    BATCH_SIZE = 1
    all_align_scores = []
    for f1 in first_list:
        curr_align_scores = []
        for i in range(0, len(second_list), BATCH_SIZE):
            all_input_str = [f"{f1.strip()} </s><s> {f2.strip()}" for f2 in second_list[i:i + BATCH_SIZE]]
            with torch.inference_mode():
                tensors = tokenizer(all_input_str, return_tensors="pt", padding=True, truncation=True, max_length=512)
                tensors.to("cuda")
                align_scores = torch.nn.functional.softmax(model(**tensors).logits, dim=1)[:, 1]
                curr_align_scores.extend(align_scores)
        all_align_scores.append(curr_align_scores)
    return torch.Tensor(all_align_scores)

with open('../../data/screen_eval.json', 'r') as file:
    data=json.load(file)
    
superpal = superpal_setup()

rank_list = []
scores = []
#Get index and replace with '' to maintian correct idx without duplicate matching
convo = [data['original_convo'][k].split('\n') for idx, k in enumerate(data['original_convo'].keys()) if data['agg_label'][f'{idx}']]
convo_for_mod = [data['original_convo'][k].split('\n') for idx, k in enumerate(data['original_convo'].keys()) if data['agg_label'][f'{idx}']]
summaries = [data['inferred_summary'][k] for idx, k in enumerate(data['original_convo'].keys()) if data['agg_label'][f'{idx}']]
rel_utts = [data['rel_utt'][k] for idx, k in enumerate(data['rel_utt'].keys()) if data['agg_label'][f'{idx}']]
assert len(rel_utts) == len(convo)
assert len(summaries) == len(convo)

res_sp = []
for i in tqdm(range(len(convo))):
    summary = summaries[i]
    t0=time.time()
    results = superpal([summary], convo[i])[0].tolist()
    t1 =time.time()
    res = pd.DataFrame(results, columns=['score']).nlargest(1, 'score').reset_index()
    scores = [x for x in res['score']]
    rank_list = [x for x in res['index']]
     
    r1 = int(rank_list[0] in rel_utts[i])
    retrieval = [r1]
    res_sp.append({'rank_list':rank_list, 'scores':scores, 'rel_utts':rel_utts[i], 'retrieval':retrieval, 'time':t1-t0})
    with open('results/superpal_retrieval.json', 'w') as file:
        json.dump(res_sp, file)
