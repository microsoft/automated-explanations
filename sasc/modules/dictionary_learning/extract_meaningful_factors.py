import os
import pickle
import csv
from os.path import dirname, join
import os.path
import numpy as np

# build a synthetic score matrix in dim layer x factor_dize -> 12 x 1500
# meaningful factors: synthetic score > 0
# load meaningful facors into csv
## layer, factor_idx, synthetic score, explanation, top_n_grams, rationales

factor_layer = 12
def flatten_list(l):
    res = ''
    for i, k in enumerate(l):
        if i != len(l) - 1:
            res += k + ' __ '
        else:
            res += k
    return res

repo_dir = dirname(dirname(os.path.abspath(__file__)))
unique_p = ['null' for _ in range(1500)]
scores = [0 for _ in range(1500)]

for root, dirs, files in os.walk(join(repo_dir, 'results', f'dl_l{factor_layer}')):
    for name in files:
        if name == 'results.pkl': 
            full_path = os.path.abspath(os.path.join(root, name))
            segs = full_path.split('/')
            factor_idx = int(segs[6][1:])
            unique_p[factor_idx] = full_path


for factor_idx in range(1500):
    result_p = unique_p[factor_idx]
    with (open(result_p, "rb")) as openfile:
        f = pickle.load(openfile)
        scores[factor_idx] = f['top_score_synthetic']

'''
collect = []

for factor_layer in range(12):
    for factor_idx in range(1500):
        if scores[factor_layer][factor_idx] > 0:
            collect.append(scores[factor_layer][factor_idx])
print('mean: ', np.mean(collect)) #0.3591
print('median: ', np.median(collect)) #0.1492
print('std: ', np.std(collect)) #0.544

'''

csv_file = open(join(repo_dir, 'results', f'extracted_factors_{factor_layer}.csv'), 'w')
csv_writer = csv.writer(csv_file)
header = ['layer', 'factor_idx', 'top_score_synthetic', 'top_explanation_init_strs',
         'explanation_init_strs', 'explanation_init_ngrams', 'explanation_init_rationales']

summary = []

# Writing headers of CSV file
csv_writer.writerow(header)

for factor_idx in range(1500):
    if scores[factor_idx] > 0.15:
        summary.append(factor_idx)
        row_data = [factor_layer, factor_idx, scores[factor_idx]]
        with (open(unique_p[factor_idx], "rb")) as openfile:
            f = pickle.load(openfile)
            f['explanation_init_ngrams'] = flatten_list(f['explanation_init_ngrams'][:25])
            f['explanation_init_rationales'] = flatten_list(f['explanation_init_rationales'])
            f['explanation_init_strs'] = flatten_list(f['explanation_init_strs'])
            row_data.extend([f['top_explanation_init_strs'], f['explanation_init_strs'], f['explanation_init_ngrams'], f['explanation_init_rationales']])
        csv_writer.writerow(row_data)
csv_file.close()

print(len(summary))

            

