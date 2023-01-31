import logging
from typing import List
import datasets
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import sklearn.preprocessing
from os.path import dirname, join
import os.path
from spacy.lang.en import English
import imodelsx
import imodelsx.util
import pickle as pkl
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
        self.weights = pkl.load(open(join(self.save_dir_fmri, 'weights.pkl'), 'rb'))

        # load test corrs for each voxel
        self.preproc = pkl.load(open(join(self.save_dir_fmri, 'preproc.pkl'), 'rb'))
        self.corrs = np.sort(np.load(join(self.save_dir_fmri, 'corrs.npz'))['arr_0'])

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


def save_mini_weights(num_top=1000, save_dir='fmri'):
    """Call this once to save only the weights for the top voxels
    (All voxels is too big)
    Requires the full file weights.npz
    """
    # load weights
    weights_npz = np.load(join(save_dir, 'weights.npz'))
    weights = weights_npz['arr_0']
    weights = weights.reshape(4, -1, 768)
    # mean over delays dimension...
    weights = weights.mean(axis=0).squeeze()
    weights = weights.T  # make it (768, n_outputs)

    # load corrs
    corrs_val = np.load(join(save_dir, 'corrs.npz'))['arr_0']
    top_idxs = np.argsort(corrs_val)[::-1]

    # save top weights only
    weights = weights[:, top_idxs[:num_top]]
    pkl.dump(weights, open(join(save_dir, 'weights.pkl'), 'wb'))


if __name__ == '__main__':
    # save_mini_weights()
    mod = fMRIModule()
    X = ['I like to eat pizza', 'I like to eat pasta']
    resp = mod(X)
    print(resp)
