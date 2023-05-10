import pickle
import csv
from os.path import dirname, join
import os.path

repo_dir = dirname(dirname(os.path.abspath(__file__)))

factor_idx_lst = [2, 16, 33, 30, 86, 125, 13, 42, 50, 102, 184, 195, 297, 322, 386]
eval_all_ps = ['eval_dl_l4_i2/7343e8443347292dbbc1c7de8452dded36e694afd1bc9bca56778a3f0ad2dad2',
                'eval_dl_l4_i16/44efd61ddd3a509f7366bc21724e588b6ea9de4cf3f5da7f9c2e2f13bef3b3ec',
                'eval_dl_l4_i33/fc044e23e6b9a71cc06fbc00402f2d689d939378dfc17e8dfd8348b06922277e',
                'eval_dl_l4_i30/42ab3bb254bc6e5e48a2adc1853ae8380be46c12b5c075ae6887eed5bb1977a0',
                'eval_dl_l6_i86/edda0c5c9cb456ef40780efd95d60c66e5ffd350cef5058533a9bb28d545efe7',
                'eval_dl_l8_i125/7176f576d5d897399553c1903aa47c14a23f2dc5b521a133a0d372b5efc83ec8',
                'eval_dl_l10_i13/ba9bd081cc6cff60e64d09b538efaccce81427daa71118c01b0bc3965e29abe1',
                'eval_dl_l10_i42/56703ced437364eb7f92fa503ef8be83b7fe77e8a6374cb0cd64b9eecf9f8db3',
                'eval_dl_l10_i50/daf4ed3f428c16a57940454c75f14dad7d9e8b1ef0e7ea9b4f9aa4719cbc2b67',
                'eval_dl_l10_i102/c2b764a4f772f7c2e2fdd470dbc3bfa041ca0e603b39c62aa6a7adc2b47c8839',
                'eval_dl_l10_i184/29cea206934c0b64cbf2042e4d25a2e7b5443e547df943cfb235863f67694e5d',
                'eval_dl_l10_i195/7410ac2404b07b060cb63b2a4c5e42c422992efa305620a339434fcfd67131ca',
                'eval_dl_l10_i297/3bf50a31b1326b1a005111031f6fc41f0dab3e12d239401adc162d83a3139bfa',
                'eval_dl_l10_i322/af61a05f073ac4cbbb8f407cbdc610050b681933b6953c6b76cf3609e55482d6',
                'eval_dl_l10_i386/ac6f1e90dd48fb9dd825f25a9b4bd5eaae8dab43d55bccc4e0ca026ba1fc243c']
data = []
for p in eval_all_ps:
    with (open(join(repo_dir, 'results', p, 'eval.pkl'), "rb")) as openfile:
        f = pickle.load(openfile)
    data.append(f)

csv_file = open(join(repo_dir, 'results', 'dict_learn_eval.csv'), 'w')
csv_writer = csv.writer(csv_file)
header = ['factor_idx', 'our_score_synthetic', 'baseline_score_synthetic', 'null_score_synthetic']

count = 0
for i, f in enumerate(data):
    if count == 0:
        # Writing headers of CSV file
        csv_writer.writerow(header)
        count += 1
 
    # Writing data of CSV file
    row = [factor_idx_lst[i]]
    our_score = f['score_synthetic'][:-1]
    b_score = f['score_synthetic'][-1]
    n_score = f['control_score_synthetic']
    row.extend([our_score, b_score, n_score])
    csv_writer.writerow(row)
 
csv_file.close()

'''
dict_keys(['subsample_frac', 'checkpoint', 'checkpoint_module', 'noise_ngram_scores',
 'module_num_restrict', 'seed', 'save_dir', 'module_name', 'module_num', 'subject',
  'factor_layer', 'factor_idx', 'method_name', 'num_top_ngrams_to_use', 'num_top_ngrams_to_consider',
   'num_summaries', 'num_synthetic_strs', 'generate_template_num', 'use_cache', 'save_dir_unique',
    'explanation_init_strs', 'strs_added', 'strs_removed', 'score_synthetic'])

'''