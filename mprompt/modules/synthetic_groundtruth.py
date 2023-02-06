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


class SyntheticModule():

    def __init__(self, prompt_num: int = 0):
        """
        Params
        ------
        prompt_num: int
        """


    def __call__(self, X: List[str]) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        """

    def get_relevant_data(self) -> List[str]:
        """read in full text of 26 narrative stories
        """
        with open(join(self.save_dir_fmri, 'narrative_stories.txt'), 'r') as f:
            narrative_stories = f.readlines()
        return narrative_stories


if __name__ == '__main__':
    mod = SyntheticModule()
    X = mod.get_relevant_data()
    print(X[0][:50])
    resp = mod(X[:3])
    print(resp)
