import pandas as pd
import logging
from typing import List
import datasets
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import sklearn.preprocessing
from spacy.lang.en import English
import imodelsx
import imodelsx.util
import pickle as pkl
from os.path import dirname, join
import os.path
modules_dir = dirname(os.path.abspath(__file__))
SAVE_DIR_FMRI = join(modules_dir, 'fmri')


class OldFMRIModule():

    def __init__(self, voxel_num_best: int = 0):
        """
        Params
        ------
        voxel_num_best: int
            Which voxel to predict (0 for best-predicted voxel, then 1, 2, ...1000)
        """
        self.voxel_num_best = voxel_num_best

        # hyperparams for loaded model
        self.checkpoint = 'bert-base-uncased'
        self.model = 'bert-10__ndel=4'
        self.save_dir_fmri = SAVE_DIR_FMRI
        self.ndel = 4

        # load weights
        self.weights = pkl.load(
            open(join(self.save_dir_fmri, 'weights.pkl'), 'rb'))
        self.preproc = pkl.load(
            open(join(self.save_dir_fmri, 'preproc.pkl'), 'rb'))

        # load test corrs for each voxel
        # this has all voxels, so can infer voxel locations from this
        self.corrs = np.sort(
            np.load(join(self.save_dir_fmri, 'corrs.npz'))['arr_0'])[::-1]
        self.corr = self.corrs[voxel_num_best]

    def __call__(self, X: List[str], return_all=False) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        """
        # get bert embeddings
        embs = imodelsx.util.get_embs_llm(X, self.checkpoint)

        # apply StandardScaler (pre-trained)
        embs = self.preproc.transform(embs)

        # apply fMRI transform
        preds_fMRI = embs @ self.weights

        
        if return_all:
            return preds_fMRI[:, :200]
            
        # select voxel
        else:
            pred_voxel = preds_fMRI[:, self.voxel_num_best]
            return pred_voxel

def get_test_ngrams(voxel_num_best: int = 0):
    top_ngrams = pd.read_pickle(join(SAVE_DIR_FMRI, 'top_ngrams.pkl'))
    return top_ngrams['voxel_top_' + str(voxel_num_best)].values

def get_roi(voxel_num_best: int = 0):
    rois = pd.read_pickle(join(SAVE_DIR_FMRI, 'roi_dict.pkl'))
    return rois.get(voxel_num_best, '--')

if __name__ == '__main__':
    mod = OldFMRIModule()
    X = mod.get_relevant_data()
    print(X[0][:50])
    resp = mod(X[:3])
    print(resp)
    print(mod.corrs[:20])
