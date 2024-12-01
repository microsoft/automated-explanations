import sasc.modules.fmri_module
from sasc.config import CACHE_DIR, RESULTS_DIR, cache_ngrams_dir, regions_idxs_dir, FMRI_DIR
from math import ceil
from numpy.linalg import norm
from copy import deepcopy
import json
import img2pdf
from PIL import Image
import pickle as pkl
import sasc.viz
import imodelsx.util
from pprint import pprint
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


subject = 'S01'
# subject = 'S02'
# subject = 'S03'
# suffix_setting = '_fedorenko'
suffix_setting = '_spotlights'

if suffix_setting == '':
    # rois_dict = joblib.load(join(regions_idxs_dir, f'rois_{subject}.jbl'))
    # rois = joblib.load(join(FMRI_DIR, 'brain_tune/voxel_neighbors_and_pcs/', 'communication_rois_UTS02.jbl'))
    rois = joblib.load(join(FMRI_DIR, 'brain_tune/voxel_neighbors_and_pcs/',
                            f'communication_rois_v2_UT{subject}.jbl'))
    rois_dict_raw = {i: rois[i] for i in range(len(rois))}
    if subject == 'S02':
        raw_idxs = [
            [0, 7],
            [3, 4],
            [1, 5],
            [2, 6],
        ]
    elif subject == 'S03':
        raw_idxs = [
            [0, 7],
            [3, 4],
            [2, 5],
            [1, 6],
        ]
    rois_dict = {
        i: np.vstack([rois_dict_raw[j] for j in idxs]).sum(axis=0)
        for i, idxs in enumerate(raw_idxs)
    }
elif suffix_setting == '_fedorenko':
    if subject == 'S03':
        rois_fedorenko = joblib.load('lang_localizer_UTS03.jbl')
    rois_dict = {
        i: rois_fedorenko[i] for i in range(len(rois_fedorenko))
    }
    # rois_dict = rois_dict_raw
elif suffix_setting == '_spotlights':
    rois_spotlights = joblib.load(f'all_spotlights_UT{subject}.jbl')
    rois_dict = {i: rois_spotlights[i][-1]
                 for i in range(len(rois_spotlights))}


# ngrams are same for both models
ngrams_list = joblib.load(join(cache_ngrams_dir, 'fmri_UTS02_ngrams.pkl'))
for embs_fname, checkpoint, out_suffix in tqdm(zip(
    ['fmri_embs.pkl', 'fmri_embs_llama.pkl'],
    ['facebook/opt-30b', 'huggyllama/llama-30b'],
    ['_opt', '_llama'],
)):
    # if embs_fname == 'fmri_embs.pkl':
    # continue
    print(f'Running for {embs_fname}')
    embs = joblib.load(join(cache_ngrams_dir, embs_fname))
    assert len(embs) == len(ngrams_list)
    # embs = joblib.load(join(cache_ngrams_dir, 'fmri_embs_llama.pkl'))
    mod = sasc.modules.fmri_module.fMRIModule(
        subject=f"UT{subject}",
        checkpoint=checkpoint,
        init_model=False,
        restrict_weights=False,
    )
    voxel_preds = mod(embs=embs, return_all=True)

    print('Saving outputs')
    outputs_dict = {
        k: voxel_preds[:, np.array(rois_dict[k].astype(bool))].mean(axis=1)
        for k in rois_dict
    }

    joblib.dump(outputs_dict, join(
        cache_ngrams_dir, f'rois_communication_ngram_outputs_dict_{subject}{suffix_setting}{out_suffix}.pkl'))

    outputs_dict_voxels = {
        k: [voxel_preds[:, i] for i in np.where(rois_dict[k])[0]]
        for k in rois_dict
    }

    joblib.dump(outputs_dict_voxels, join(
        cache_ngrams_dir, f'rois_communication_ngram_outputs_dict_voxels_{subject}{suffix_setting}{out_suffix}.pkl'))
