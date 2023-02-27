import argparse
import sys
import os.path
from os.path import dirname, join
repo_dir = dirname(dirname(os.path.abspath(__file__)))
import pandas as pd
import mprompt.data.data
import mprompt.methods.m4_evaluate as m4_evaluate
import numpy as np

def process_and_add_scores(r: pd.DataFrame, add_bert_scores=False):
    # metadata
    r['task_str'] = r.apply(lambda row: mprompt.data.data.get_task_str(row['module_name'], row['module_num']), axis=1)
    r['task_keyword'] = r['task_str'].apply(lambda task_str: mprompt.data.data.get_groundtruth_keyword(task_str))
    r['task_name (groundtruth)'] = r['task_str'].apply(lambda s: s.split('_')[-1])
    r['ngrams_restricted'] = ~(r['module_num_restrict'] == -1)
    r['num_generated_explanations'] = r['explanation_init_strs'].apply(lambda x: len(x))

    # recompute recovery metrics
    r['score_contains_keywords'] = r.apply(lambda row: m4_evaluate.compute_score_contains_keywords(row, row['explanation_init_strs']), axis=1)
    if add_bert_scores:
        r['score_bert'] = r.progress_apply(lambda row: m4_evaluate.compute_score_bert(row, row['explanation_init_strs']), axis=1)

    # metrics
    # for met_suff in ['contains_keywords']:
    for met_suff in ['contains_keywords', 'bert']:
        if 'score_' + met_suff in r.columns:
            met_val = r['score_' + met_suff]
            r['top_' + met_suff] = met_val.apply(lambda x: x[0])
            r['any_' + met_suff] = met_val.apply(lambda x: np.max(x))
            r['mean_' + met_suff] = met_val.apply(lambda x: np.mean(x))
            r[f'mean_{met_suff}_weighted'] = r[f'mean_{met_suff}'] * r['num_generated_explanations']
    r['row_count_helper'] = 1
    r = r.sort_values(by='top_score_synthetic', ascending=False).round(3)
    return r