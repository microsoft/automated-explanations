import pickle
import csv
from os.path import dirname, join
import os.path

repo_dir = dirname(dirname(os.path.abspath(__file__)))

factor_idx_lst = [2, 16, 33, 30, 86, 125, 13, 42, 50, 102, 184, 195, 297, 322, 386]
eval_all_ps = ['test_dl_l10_i322/6ee2fb4c7a3392836f97b834c5ce163f719a8d3ea2b463ab82e1a90dc2970580',
                'dl_l10_i322/e02f845df54536abf3499bcc319f59bf693c8b6501136b79f8f742d81a3517e5']
data = []
for p in eval_all_ps:
    with (open(join(repo_dir, 'results', p, 'results.pkl'), "rb")) as openfile:
        f = pickle.load(openfile)
    data.append(f)

assert(data[0].keys() == data[1].keys())
for k in data[0].keys():
    if data[0][k] != data[1][k]:
        print(k)
        print(data[0][k])
        print(data[1][k])
        print("="*80)
'''
dict_keys(['subsample_frac', 'checkpoint', 'checkpoint_module', 'noise_ngram_scores',
 'module_num_restrict', 'seed', 'save_dir', 'module_name', 'module_num', 'subject',
  'factor_layer', 'factor_idx', 'method_name', 'num_top_ngrams_to_use', 'num_top_ngrams_to_consider',
   'num_summaries', 'num_synthetic_strs', 'generate_template_num', 'use_cache', 'save_dir_unique',
    'explanation_init_strs', 'strs_added', 'strs_removed', 'score_synthetic'])

'''