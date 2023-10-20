import json
from utils import convert_to_json
from metric.evaluator import get_evaluator
import time

with open('../../data/screen_eval.json', 'r') as file:
    data=json.load(file)
    
task = 'summarization'
evaluator = get_evaluator('summarization', max_length=6000)

inf_summ = [data['inferred_summary'][k] for k in data['inferred_summary'].keys()]
convo = ['\n'.join(data['convo'][k]) for k in data['convo'].keys()]
label = [int(data['agg_label'][k]) for k in data['agg_label'].keys()]

json_data = convert_to_json(output_list=inf_summ, src_list=convo)

t0=time.time()
res = evaluator.evaluate(json_data, dims=['consistency'], overall=False, print_result=False)
t1=time.time()

result = [x['consistency'] for x in res]
results_conv = [result, label, t1-t0]
with open('unieval_results.json', 'w') as file:
    json.dump(results_conv, file)