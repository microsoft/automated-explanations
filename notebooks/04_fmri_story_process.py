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


# Test Explanation<>Story match
val = D5_Validator()

# visualize single story
scores_data_story = mprompt.viz.get_story_scores(val, expls, paragraphs)
joblib.dump(scores_data_story, join(RESULTS_DIR, 'stories', f'{EXPT_NAME}_scores_data_story.pkl'))

# compute scores heatmap
scores_mean, scores_all = notebook_helper.compute_expl_data_match_heatmap(
    val, expls, paragraphs)
joblib.dump(
    {'scores_mean': scores_mean,
     'scores_all': scores_all},
     join(RESULTS_DIR, 'stories', f'{EXPT_NAME}_scores_data.pkl'))

# Test Module<>Story match
expls = rows.expl.values
voxel_nums = rows.module_num.values
subjects = rows.subject.values
scores_mod, scores_max_mod, all_scores, all_ngrams = \
    notebook_helper.compute_expl_module_match_heatmap(
        expls, paragraphs, voxel_nums, subjects)
joblib.dump({
    'scores_mean': scores_mod,
    # 'scores_max': scores_max_mod,
    'scores_all': all_scores,
    # 'ngrams_all': all_ngrams,
}, join(RESULTS_DIR, 'stories', f'{EXPT_NAME}_scores_mod.pkl'))

