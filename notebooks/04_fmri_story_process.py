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

EXPT_NAME = 'relationships_mar9'
rows = joblib.load(join(RESULTS_DIR, 'stories', f'{EXPT_NAME}_rows.pkl'))
expls = rows.expl.values
voxel_nums = rows.module_num.values
subjects = rows.subject.values
paragraphs = rows.paragraph.values
prompts = rows.prompt.values


# Test Data<>Concept match
val = D5_Validator()

# visualize single story
s = mprompt.viz.visualize_story_html(
    val, expls, paragraphs, prompts, fname=join(RESULTS_DIR, 'stories', f'{EXPT_NAME}_story.html'))
# display(HTML(s))

# compute scores heatmap
scores_data = notebook_helper.compute_expl_data_match_heatmap(
    val, expls, paragraphs)
joblib.dump(scores_data, join(RESULTS_DIR, 'stories',
            f'{EXPT_NAME}_scores_data.pkl'))

# s = scores_data
# s = scipy.special.softmax(scores, axis=1)
# s = (s - s.min()) / (s.max() - s.min())
# mprompt.viz.heatmap(scores_data, expls)


# Test Module<>Concept match
expls = rows.expl.values
voxel_nums = rows.module_num.values
subjects = rows.subject.values
scores_mod, scores_max_mod, all_scores, all_ngrams = \
    notebook_helper.compute_expl_module_match_heatmap(
        expls, paragraphs, voxel_nums, subjects)

s = scores_mod
joblib.dump({
    'scores_mod': scores_mod,
    'scores_max_mod': scores_max_mod,
    'all_scores': all_scores,
    'all_ngrams': all_ngrams,
}, join(RESULTS_DIR, 'stories', f'{EXPT_NAME}_scores_mod.pkl'))
# s = scipy.special.softmax(s, axis=1)
# s = (s - s.min()) / (s.max() - s.min())
# mprompt.viz.heatmap(scores_data, expls, xlab='Explanation of voxel used for evaluation', clab='Mean voxel response')
