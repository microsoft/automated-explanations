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
from typing import List
tqdm.pandas()
from mprompt.modules.fmri_module import get_roi
import mprompt.methods.m4_evaluate as m4_evaluate
from mprompt.config import RESULTS_DIR

# results_dir = '../results/feb12_fmri_sweep/'
# results_dir = '/home/chansingh/mntv1/mprompt/feb12_fmri_sweep_gen_template1/'
results_dir = '/home/chansingh/mntv1/mprompt/mar7_test/'


r = imodelsx.process_results.get_results_df(results_dir, use_cached=False)
for num in [25, 50, 75, 100]:
    r[f'top_ngrams_module_{num}'] = r['explanation_init_ngrams'].apply(lambda x: x[:num])
    # r[f'top_ngrams_test_{num}'] = r.apply(lambda row: get_test_ngrams(voxel_num_best=row.module_num)[:num], axis=1)
    
r['roi_anat'] = r.apply(lambda row: get_roi(voxel_num_best=row.module_num, roi_type='anat', subject=row.subject), axis=1)
r['roi_func'] = r.apply(lambda row: get_roi(voxel_num_best=row.module_num, roi_type='func', subject=row.subject), axis=1)

# Calculate train ngram correctness
num_top_ngrams_expl = 75
test_correct_score_list, correct_ngrams_test_list = m4_evaluate.calc_frac_correct_score(r, col_ngrams=f'top_ngrams_module_{num_top_ngrams_expl}')    
r['top_ngrams_module_correct'] = correct_ngrams_test_list
r['frac_top_ngrams_module_correct'] = r['top_ngrams_module_correct'].apply(lambda x: len(x) / num_top_ngrams_expl)

# Calculate match between expl and test resp


# Unnecessary metrics
# r['top_ngrams_test_correct_score'] = test_correct_score_list # these scores are basically just 0/1 for each ngram
# r['expl_test_bert_score'] = r.progress_apply(lambda row: m4_evaluate.test_ngrams_bert_score(
    # row['top_explanation_init_strs'], row['top_ngrams_test'].tolist()), axis=1)

# Calculate test ngram correctness
# num_top_ngrams_test = 75
# test_correct_score_list, correct_ngrams_test_list = m4_evaluate.calc_frac_correct_score(r, col_ngrams=f'top_ngrams_test_{num_top_ngrams_test}')    
# r['top_ngrams_test_correct'] = correct_ngrams_test_list
# r['frac_top_ngrams_test_correct'] = r['top_ngrams_test_correct'].apply(lambda x: len(x) / num_top_ngrams_test)


r.to_pickle(join(RESULTS_DIR, 'results_fmri.pkl'))