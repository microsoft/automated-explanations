import sasc.modules.fmri_module
from sasc.config import CACHE_DIR, RESULTS_DIR
from numpy.linalg import norm
from copy import deepcopy
import pickle as pkl
import sasc.viz
import imodelsx.util
import joblib
import numpy as np
from typing import List
import sys
import pandas as pd
from tqdm import tqdm
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
import os

cache_ngrams_dir = '/home/chansingh/mntv1/deep-fMRI/sasc/mprompt/cache/cache_ngrams'
regions_idxs_dir = '/home/chansingh/mntv1/deep-fMRI/sasc/brain_regions'
ngrams_list = joblib.load(join(cache_ngrams_dir, 'fmri_UTS02_ngrams.pkl'))

mod = sasc.modules.fmri_module.fMRIModule(
    # subject="UTS02", # doesnt matter for embs
    # checkpoint="facebook/opt-30b",)
    checkpoint="huggyllama/llama-30b",)
embs = mod._get_embs(ngrams_list)
joblib.dump(embs, join(cache_ngrams_dir, 'fmri_embs_llama.pkl'))
