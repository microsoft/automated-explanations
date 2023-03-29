import os
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from tqdm import tqdm
import pandas as pd
import sys
from IPython.display import display, HTML
from typing import List
from mprompt.modules.emb_diff_module import EmbDiffModule
import numpy as np
import matplotlib
import imodelsx.util
from copy import deepcopy
import re
import notebook_helper
import mprompt.viz
import scipy.special
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from mprompt.methods.m4_evaluate import D5_Validator
import openai
from mprompt.modules.fmri_module import fMRIModule
from pprint import pprint
import joblib
from mprompt.config import RESULTS_DIR
import torch.cuda


def explanation_story_match():
    if os.path.exists(join(EXPT_DIR, 'story_data_match.pdf')):
        return
    print('Computing expl<>story match', EXPT_NAME)
    val = D5_Validator()

    # visualize single story
    scores_data_story = mprompt.viz.get_story_scores(val, expls, paragraphs)
    joblib.dump(scores_data_story, join(EXPT_DIR, 'scores_data_story.pkl'))
    s_data = notebook_helper.viz_paragraphs(
        paragraphs, scores_data_story, expls, prompts,
        normalize_to_range=True, moving_average=True, shift_to_range=True)
    with open(join(EXPT_DIR, 'story.html'), 'w') as f:
        f.write(s_data)

    # compute scores heatmap
    scores_mean, scores_all = notebook_helper.compute_expl_data_match_heatmap(
        val, expls, paragraphs)
    joblib.dump(
        {'scores_mean': scores_mean,
        'scores_all': scores_all},
        join(EXPT_DIR, 'scores_data.pkl'))
    mprompt.viz.heatmap(scores_mean.T, expls, ylab='Story', xlab='Explanation')
    plt.savefig(join(EXPT_DIR, 'story_data_match.png'), dpi=300)
    plt.savefig(join(EXPT_DIR, 'story_data_match.pdf'), bbox_inches='tight')


def module_story_match():
    if os.path.exists(join(EXPT_DIR, f'scores_mod_ngram_length={0}.pkl')):
        return
    print('Computing module<>story match', EXPT_NAME)
    if not 'module_num' in rows.columns:
        raise ValueError('module_num not in rows.columns!')
    voxel_nums = rows.module_num.values
    subjects = rows.subject.values

    # basic with no overlaps
    if 'uts0' in EXPT_NAME:
        compute_func = notebook_helper.compute_expl_module_match_heatmap_cached_single_subject
        subjects = subjects[0]
    else:
        compute_func = notebook_helper.compute_expl_module_match_heatmap
    scores_mod, scores_max_mod, all_scores = \
        compute_func(expls, paragraphs, voxel_nums, subjects)
    joblib.dump({
        'scores_mean': scores_mod,
        'scores_all': all_scores,
    }, join(EXPT_DIR, f'scores_mod_ngram_length={0}.pkl'))

    # with overlaps
    # ngram_lengths = [10, 50, 100, 384]
    # for i, ngram_length in enumerate(ngram_lengths):
    #     print(i, '/', len(ngram_lengths), 'ngram length', ngram_length)
    #     scores_mod, scores_max_mod, all_scores, all_ngrams = \
    #         notebook_helper.compute_expl_module_match_heatmap_running(
    #             expls, paragraphs, voxel_nums, subjects,
    #             ngram_length=ngram_length, paragraph_start_offset_max=50,
    #         )
    #     joblib.dump({
    #         'scores_mean': scores_mod,
    #         'scores_all': all_scores,
    #     }, join(EXPT_DIR, f'scores_mod_ngram_length={ngram_length}.pkl'))

if __name__ == '__main__':
    # EXPT_NAME = 'huth2016clusters_mar21_i_time_traveled'
    # EXPT_NAME = 'voxels_mar21_hands_arms_emergency'

    # iterate over seeds
    seeds = [1, 2, 3, 4, 5, 6, 7]
    for seed in seeds:
        for version in ['v4_noun', 'v5_noun']:
            
            EXPT_NAME = f'uts02_pilot_gpt4_mar28___ver={version}___seed={seed}'
            # EXPT_NAME = f'uts02_concepts_pilot_selected_mar28___ver={version}___seed={seed}'
            # EXPT_NAME = f'uts02_concepts_pilot_mar22_seed={seed}'
            # EXPT_NAME = f'uts02_concepts_pilot_selected_mar24_seed={seed}'
            EXPT_DIR = join(RESULTS_DIR, 'stories', EXPT_NAME)
            rows = joblib.load(join(EXPT_DIR, 'rows.pkl'))
            expls = rows.expl.values
            paragraphs = rows.paragraph.values
            prompts = rows.prompt.values

            module_story_match()
            torch.cuda.empty_cache()
            explanation_story_match()
            torch.cuda.empty_cache()
