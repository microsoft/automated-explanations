import os
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from tqdm import tqdm
import pandas as pd
import pickle as pkl
import notebook_helper
import imodelsx.process_results
import sys
import bert_score
import numpy as np
from typing import List
tqdm.pandas()
from mprompt.modules.fmri_module import get_roi
import mprompt.methods.m4_evaluate as m4_evaluate
from mprompt.config import RESULTS_DIR
import joblib
from mprompt.modules.fmri_module import SAVE_DIR_FMRI
import imodelsx.util
from mprompt.modules.emb_diff_module import EmbDiffModule
import scipy.stats
from mprompt.methods.m4_evaluate import D5_Validator
import torch.cuda
from mprompt.config import CACHE_DIR

def add_expl_preds_and_save(r, fname='results_fmri_full.pkl'):
    # Calculate match between expl and test resp
    dsets = joblib.load(join(SAVE_DIR_FMRI, 'stories', 'running_words.jbl'))
    r['neg_dists_expl_test'] = [[] for _ in range(r.shape[0])]
    r['rankcorr_expl_test'] = np.nan

    mod = EmbDiffModule(task_str='')
    for i in tqdm(range(r.shape[0])):
        row = r.iloc[i]
        expl = row['top_explanation_init_strs']
        subject = row['subject']

        # get resp
        dset = dsets[subject]
        resp = dset['resp'][:, row['module_num']]

        # check cache
        cache_fname = join(CACHE_DIR, 'expl_preds', f'expl_test_{subject}_{expl.replace("/", "__")}.jbl')
        os.makedirs(join(CACHE_DIR, 'expl_preds'), exist_ok=True)
        loaded = False
        if os.path.exists(cache_fname):
            try:
                neg_dists = joblib.load(cache_fname)
                loaded = True
            except:
                pass
            
        if not loaded:
            mod._init_task(task_str=expl)
            strs_list = dset['words']
            neg_dists = [
                mod(
                    imodelsx.util.generate_ngrams_list(strs_list[j], ngrams=3, all_ngrams=True),
                    verbose=False,
                ).mean()
                for j in range(len(strs_list))
            ]
            joblib.dump(neg_dists, cache_fname)

        r['neg_dists_expl_test'].iloc[i] = neg_dists
        r['rankcorr_expl_test'].iloc[i] = scipy.stats.spearmanr(
            neg_dists, resp, nan_policy='omit', alternative='greater').statistic
        r.to_pickle(join(RESULTS_DIR, fname))


    # mod = None
    # torch.cuda.empty_cache()
    # val = D5_Validator()
    # frac_valid = [
    #     np.mean(
    #         val.validate_w_scores(
    #             expl,
    #             imodelsx.util.generate_ngrams_list(strs_list[j], ngrams=3, all_ngrams=False)))
    #     for j in tqdm(range(len(strs_list)))
    # ]

if __name__ == '__main__':
    # results_dir = '../results/feb12_fmri_sweep/'
    # results_dir = '/home/chansingh/mntv1/mprompt/feb12_fmri_sweep_gen_template1/'
    # results_dir = '/home/chansingh/mntv1/mprompt/mar7_test/'
    # results_dir = '/home/chansingh/mntv1/mprompt/mar8/'
    # results_dir = '/home/chansingh/mntv1/mprompt/mar9/'
    results_dir = '/home/chansingh/mntv1/mprompt/mar13/'

    r = imodelsx.process_results.get_results_df(results_dir, use_cached=False)
    print(f'Loaded {r.shape[0]} results')
    for num in [25, 50, 75, 100]:
        r[f'top_ngrams_module_{num}'] = r['explanation_init_ngrams'].apply(lambda x: x[:num])
        # r[f'top_ngrams_test_{num}'] = r.apply(lambda row: get_test_ngrams(voxel_num_best=row.module_num)[:num], axis=1)
        
    print(f'Adding roi...')
    r['roi_anat'] = r.progress_apply(lambda row: get_roi(voxel_num_best=row.module_num, roi_type='anat', subject=row.subject), axis=1)
    r['roi_func'] = r.progress_apply(lambda row: get_roi(voxel_num_best=row.module_num, roi_type='func', subject=row.subject), axis=1)

    # Calculate train ngram correctness
    print(f'Finding matching ngrams_module...')
    num_top_ngrams_expl = 75
    correct_ngrams_module_scores, correct_ngrams_module_list = m4_evaluate.calc_frac_correct_score(r, col_ngrams=f'top_ngrams_module_{num_top_ngrams_expl}')    
    r['top_ngrams_module_correct'] = correct_ngrams_module_list
    r['frac_top_ngrams_module_correct'] = r['top_ngrams_module_correct'].apply(lambda x: len(x) / num_top_ngrams_expl)
    
    # Save results
    r.to_pickle(join(RESULTS_DIR, 'results_fmri.pkl'))

    # Add explanation<>test response match
    torch.cuda.empty_cache()
    print('Saved original results, now computing expl<>resp match...')
    add_expl_preds_and_save(r, fname='results_fmri_full.pkl')

    # Unnecessary metrics
    # r['top_ngrams_test_correct_score'] = correct_ngrams_module_scores # these scores are basically just 0/1 for each ngram
    # r['expl_test_bert_score'] = r.progress_apply(lambda row: m4_evaluate.test_ngrams_bert_score(
        # row['top_explanation_init_strs'], row['top_ngrams_test'].tolist()), axis=1)

    # Calculate test ngram correctness
    # num_top_ngrams_test = 75
    # test_correct_score_list, correct_ngrams_test_list = m4_evaluate.calc_frac_correct_score(r, col_ngrams=f'top_ngrams_test_{num_top_ngrams_test}')    
    # r['top_ngrams_test_correct'] = correct_ngrams_test_list
    # r['frac_top_ngrams_test_correct'] = r['top_ngrams_test_correct'].apply(lambda x: len(x) / num_top_ngrams_test)