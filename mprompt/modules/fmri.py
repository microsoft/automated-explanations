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


class fMRIModule():

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
        self.save_dir_fmri = join(modules_dir, 'fmri')
        self.ndel = 4

        # load weights
        self.weights = pkl.load(
            open(join(self.save_dir_fmri, 'weights.pkl'), 'rb'))

        # load test corrs for each voxel
        self.preproc = pkl.load(
            open(join(self.save_dir_fmri, 'preproc.pkl'), 'rb'))
        self.corrs = np.sort(
            np.load(join(self.save_dir_fmri, 'corrs.npz'))['arr_0'])

    def __call__(self, X: List[str]) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        """
        # get bert embeddings
        embs = imodelsx.util.get_embs_llm(X, self.checkpoint)

        # apply StandardScaler (pre-trained)
        embs = self.preproc.transform(embs)

        # apply fMRI transform
        preds_fMRI = embs @ self.weights

        # select voxel
        pred_voxel = preds_fMRI[:, self.voxel_num_best]
        return pred_voxel

    def get_relevant_data(self) -> List[str]:
        """read in full text of 26 narrative stories
        """
        with open(join(self.save_dir_fmri, 'narrative_stories.txt'), 'r') as f:
            narrative_stories = f.readlines()
        return narrative_stories


if __name__ == '__main__':
    mod = fMRIModule()
    X = mod.get_relevant_data()
    print(X[0][:50])
    resp = mod(X[:3])
    print(resp)
