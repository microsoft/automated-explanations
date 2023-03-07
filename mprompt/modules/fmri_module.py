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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import joblib
modules_dir = dirname(os.path.abspath(__file__))
SAVE_DIR_FMRI = join(modules_dir, 'fmri')


class fMRIModule():

    def __init__(self, voxel_num_best: int = 0, subject: str = 'UTS01'):
        """
        Params
        ------
        voxel_num_best: int
            Which voxel to predict (0 for best-predicted voxel, then 1, 2, ...1000)
        """

        # hyperparams for loaded model
        self.checkpoint = 'facebook/opt-30b'
        self.save_dir_fmri = SAVE_DIR_FMRI
        self.voxel_num_best = voxel_num_best
        self.subject = subject
        self.ndel = 4
        weights_file = join(SAVE_DIR_FMRI, 'model_weights', f'wt_{subject}.jbl')


        # load weights
        self.weights = joblib.load(weights_file)
        # self.preproc = pkl.load(
        #     open(join(self.save_dir_fmri, 'preproc.pkl'), 'rb'))

        # # load test corrs for each voxel
        # # this has all voxels, so can infer voxel locations from this
        # self.corrs = np.sort(
        #     np.load(join(self.save_dir_fmri, 'corrs.npz'))['arr_0'])[::-1]
        # self.corr = self.corrs[voxel_num_best]

    def _get_embs(self, X: List[str]):
        """
        Returns
        -------
        embs: np.ndarray
            (n_examples, 7168)
        """
        model = AutoModelForCausalLM.from_pretrained(self.checkpoint, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        embs = []
        for i in tqdm(range(len(X))):
            # have to to this or get some weird opt error
            text = tokenizer.encode(X[i])
            inputs = {}
            inputs['input_ids'] = torch.tensor([text]).int()
            inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape)
    
            # Ideally, you would use downsampled features instead of copying features across time delays
            emb = list(model(**inputs, output_hidden_states=True)[2])[33][0][-1].cpu().detach().numpy()
            embs.append(emb)
        return np.array(embs)

    def __call__(self, X: List[str], return_all=False) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        """
        # get bert embeddings
        embs = self._get_embs(X)
        print('embs.shape', embs.shape)

        # apply StandardScaler (pre-trained)
        # embs = self.preproc.transform(embs)

        # apply fMRI transform
        embs_delayed = np.hstack([embs] * self.ndel)
        preds_fMRI = embs_delayed @ self.weights

        
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


def cache_preprocessor(X):


    mod = fMRIModule()
    embs = mod._get_embs(X)
    preproc = sklearn.preprocessing.StandardScaler()
    preproc.fit(X)
    pkl.dump(preproc, open(join(SAVE_DIR_FMRI, 'preproc.pkl'), 'wb'))

if __name__ == '__main__':
    mod = fMRIModule()
    X = ['I am happy', 'I am sad', 'I am angry']
    print(X[0][:50])
    resp = mod(X[:3])
    print(resp)
    print(mod.corrs[:20])
