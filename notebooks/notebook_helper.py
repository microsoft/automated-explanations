import argparse
from copy import deepcopy
import sys
import os.path
from os.path import dirname, join

from mprompt.modules.fmri_module import fMRIModule
repo_dir = dirname(dirname(os.path.abspath(__file__)))
import pandas as pd
import mprompt.data.data
import mprompt.methods.m4_evaluate as m4_evaluate
import numpy as np
import imodelsx.util
from tqdm import tqdm

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


def compute_expl_data_match_heatmap(val, expls, paragraphs):
    n = len(expls)
    scores = np.zeros((n, n))
    for i in tqdm(range(n)):
        expl = expls[i]
        for j in range(n):
            text = paragraphs[j].lower()
            words = text.split()

            ngrams = imodelsx.util.generate_ngrams_list(text, ngrams=3)
            ngrams = [words[0], words[0] + ' ' + words[1]] + ngrams

            # validator-based viz
            probs = np.array(val.validate_w_scores(expl, ngrams)) > 0.5
            scores[i, j] = probs.mean()
    return scores

def compute_expl_module_match_heatmap(expls, paragraphs, voxels):
    n = len(expls)
    scores = np.zeros((n, n))
    scores_max = np.zeros((n, n))
    all_scores = {}
    all_ngrams = {}
    mod = fMRIModule(voxel_num_best=voxel_num_best, subject=subject)
    for i in tqdm(range(n)):
        # expl = expls[i]
        (expl, subject, voxel_num_best) = voxels[i]
        mod._init_fmri(subject, voxel_num_best)
        ngrams_list = []
        ngrams_scores_list = []
        for j in range(n):
            text = paragraphs[j].lower()
            words = text.split()
            ngrams_story = imodelsx.util.generate_ngrams_list(text, ngrams=3)
            ngrams_story = [words[0], words[0] + ' ' + words[1]] + ngrams_story
            ngrams_list.append(ngrams_story)

            # get mean score for each story
            ngrams_scores_story = mod(ngrams_story)
            ngrams_scores_list.append(ngrams_scores_story)
            scores[i, j] = ngrams_scores_story.mean()
            scores_max[i, j] = ngrams_scores_story.max()
        
        all_scores[i] = deepcopy(ngrams_scores_list)
        all_ngrams[i] = deepcopy(ngrams_list)
    return scores, scores_max, all_scores, all_ngrams